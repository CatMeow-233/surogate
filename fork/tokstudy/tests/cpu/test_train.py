"""The pilot trainer, its schedule, the prereg and the run registry (implementation step S14, and the registry part
of S16), on tiny arms of all three kinds over a SYNTHETIC stream, one CPU thread, deterministic algorithms:

- the loss falls; every arm stops at F_target within one step; the rate is the WSD of the FLOPs before each step;
  stable checkpoints sit after the step that reaches 0.9 E for every budget point E;
- a decay branch equals a run whose F_target is E (same absolute warmup), bit for bit;
- resuming after 0, 1 and 7 steps (and from a checkpoint older than the interruption) is bit-identical to an
  uninterrupted run;
- the run id changes with the arm, the stream, the config and the seed; a run directory of another id, an
  unregistered seed, a changed prereg without an amendment, a stream too short for F_target are refused.
"""

import json
import random
from pathlib import Path

import pytest
import torch
from surogate_tokstudy import registry, schedule
from surogate_tokstudy import stream as S
from surogate_tokstudy import train as T
from surogate_tokstudy.arms import load_arms
from surogate_tokstudy.canonical import TokstudyError
from surogate_tokstudy.codecs import TLC_SPECIALS
from surogate_tokstudy.docs import EvalDoc, Seg
from surogate_tokstudy.prereg import load_prereg

WORDS = ("def", "add", "x", "return", "total", "self", "value", "for", "in", "range", "if", "None")
TINY_HNET = "{d0 = 16, encoder_layers = 1, decoder_layers = 1, ratio = 4.0, head_dim = 8, state = 4, ssd_chunk = 8}"
K = 32


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(threads)


def word_docs(n=400, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        domain = "py" if i % 4 else "en"
        text = " ".join(rng.choice(WORDS) for _ in range(rng.randrange(6, 30))) + "\n"
        segs = (Seg("<|bos|>"), Seg("<|code_start|>"), Seg(text=text), Seg("<|code_end|>"), Seg("<|eos|>"))
        out.append(EvalDoc(f"{domain}/{i}", domain, segs if domain == "py" else (segs[0], segs[2], segs[4])))
    return out


def write_prereg(path, *, f_target, points=(0.25, 0.5, 1.0), warmup=0.02, extra=""):
    path.write_text(
        f'format = "tokstudy-prereg/1"\ntable = "tlc-N"\n[budget]\nconvention = "matmul"\nF_target = {f_target}\n'
        f"points = {list(points)}\n[context]\nK_train = {K}\nK_eval = 64\n[schedule]\nwarmup = {warmup}\n"
        f'decay_start = 0.9\nfloor = 0.1\nshape = "linear"\n[mixture]\npy = 0.75\nen = 0.25\n{extra}',
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def table(tmp_path_factory, bpe_writer):
    """A SYNTHETIC tlc-N table: stream, tokenizer, registry (N0 subword, N1 H-Net, N2 bytes; tiny geometry)."""
    root = tmp_path_factory.mktemp("table")
    docs = word_docs()
    st = S.build_stream(
        docs,
        root / "stream",
        seed=0,
        k_train=K,
        source={"project": "tlc", "fixture": "SYNTHETIC"},
        mixture={"py": 0.75, "en": 0.25},
    )
    path, sha = bpe_writer(
        root / "tok" / "tokenizer.json",
        specials=tuple(TLC_SPECIALS),
        pre="gpt2",
        train_on=[d.texts()[0] for d in docs],
        vocab_size=300,
    )
    arms = root / "arms.toml"
    arms.write_text(
        'format = "tokstudy-arms/1"\ntable = "tlc-N"\n'
        f'[[arm]]\nid = "N0"\nkind = "subword"\ntokenizer = {{path = "tok/tokenizer.json", sha256 = "{sha}"}}\n'
        'model = {profile = "tiny"}\nseeds = [0, 1]\nstages = ["decision"]\n'
        f'[[arm]]\nid = "N1"\nkind = "hnet"\nmodel = {{profile = "tiny", hnet = {TINY_HNET}}}\nseeds = [0]\n'
        'stages = ["decision", "screen"]\n'
        '[[arm]]\nid = "N2"\nkind = "bytes"\nmodel = {profile = "tiny"}\nseeds = [0]\nstages = ["decision"]\n',
        encoding="utf-8",
    )
    found = {a.id: a for a in load_arms(arms)}
    # F_target: about 30 steps of the byte arm at 4 rows of K positions
    probe = T.setup(
        found["N2"],
        stream=st,
        prereg=load_prereg(write_prereg(root / "probe.toml", f_target=10**9)),
        seed=0,
        config=CONFIG,
    )
    from surogate_tokstudy import flops

    f_target = 30 * 4 * flops.row_train_flops(probe.spec, K)
    return {
        "root": root,
        "stream": st,
        "arms": found,
        "registry": arms,
        "f_target": f_target,
        "prereg": write_prereg(root / "prereg.toml", f_target=f_target),
    }


CONFIG = T.TrainConfig(lr=3e-3, batch_rows=4, micro_rows=2, precision="fp32")


def run(table, arm, runs, *, prereg=None, seed=0, config=CONFIG, **kw):
    return T.train(
        table["arms"][arm],
        runs,
        stream=table["stream"],
        prereg=load_prereg(prereg or table["prereg"]),
        seed=seed,
        config=config,
        registry_path=table["registry"],
        **kw,
    )


def metrics(run_dir):
    lines = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert lines[0]["format"] == "tokstudy-metrics/1"
    return lines[1:]


def final_state(path):
    return torch.load(path, weights_only=True)


def assert_same_state(a, b):
    assert a["step"] == b["step"] and a["flops"] == b["flops"] and a["cursor"] == b["cursor"]
    assert a["model"].keys() == b["model"].keys()
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    sa, sb = a["optimizer"]["state"], b["optimizer"]["state"]
    assert sa.keys() == sb.keys()
    assert all(torch.equal(sa[i][k], sb[i][k]) for i in sa for k in sa[i])


@pytest.fixture(scope="module")
def runs(table, tmp_path_factory):
    """One uninterrupted run per arm under one runs root, with the decay branches at 0.25 and 0.5."""
    root = tmp_path_factory.mktemp("runs")
    out = {}
    for arm in ("N0", "N1", "N2"):
        result = run(table, arm, root, runtime=T.Runtime(checkpoint_every=4))
        out[arm] = result
        for e in (0.25, 0.5):
            T.branch(result["dir"], endpoint=e, runtime=T.Runtime(checkpoint_every=4))
    return root, out


def test_every_arm_learns_and_stops_at_f_target_within_one_step(table, runs):
    f_target = table["f_target"]
    for arm, result in runs[1].items():
        lines = metrics(Path(result["dir"]))
        assert result["done"] and lines[-1]["flops"] >= f_target
        step_flops = lines[-1]["flops"] - lines[-2]["flops"]
        assert lines[-2]["flops"] < f_target <= lines[-1]["flops"] < f_target + step_flops + 1, arm
        assert len(lines) >= 20, (arm, len(lines))
        first = sum(x["ce"] for x in lines[:5]) / 5
        last = sum(x["ce"] for x in lines[-5:]) / 5
        assert last < first - 0.5, (arm, first, last)
        if arm == "N1":
            assert all(0 < x["F"] <= 1 and x["chunks"] > 0 and x["positions_per_chunk"] for x in lines)


def test_the_rate_is_the_wsd_of_the_flops_before_each_step(table, runs):
    prereg = load_prereg(table["prereg"])
    wsd = prereg.wsd(CONFIG.lr)
    for result in runs[1].values():
        before = 0
        for line in metrics(Path(result["dir"])):
            assert line["lr"] == wsd.lr(before)
            # lr_at on pi, up to the integer rounding of the schedule's bounds (relative 1 / warmup FLOPs)
            assert line["lr"] == pytest.approx(
                schedule.lr_at(before / prereg.f_target, peak=CONFIG.lr), rel=2 / wsd.warmup_flops, abs=1e-15
            )
            before = line["flops"]
        assert metrics(Path(result["dir"]))[0]["lr"] == 0.0  # the first step only fills Adam's moments


def test_stable_checkpoints_follow_the_step_that_reaches_09_e(table, runs):
    prereg = load_prereg(table["prereg"])
    for result in runs[1].values():
        run_dir = Path(result["dir"])
        lines = metrics(run_dir)
        for e in prereg.points:
            threshold = round(0.9 * (prereg.f_target if e == 1 else round(e * prereg.f_target)))
            payload = final_state(run_dir / "stable" / f"E{e:g}.pt")
            crossing = next(x for x in lines if x["flops"] >= threshold)
            assert payload["step"] == crossing["step"] and payload["flops"] == crossing["flops"]
            assert payload["kind"] == f"main/stable/E{e:g}"
        assert (run_dir / "final.pt").is_file() and len(list((run_dir / "ckpt").glob("*.pt"))) <= 2


def test_a_branch_equals_a_run_whose_f_target_is_its_endpoint(table, runs, tmp_path):
    """Branch E = 0.5 of the main run against a run with F_target = E_0.5 and warmup 0.04 (the same absolute warmup
    0.02 F_target): the same final weights, optimiser state and step count, bit for bit."""
    f_half = round(0.5 * table["f_target"])
    short = write_prereg(tmp_path / "short.toml", f_target=f_half, points=(1.0,), warmup=0.04)
    for arm in ("N0", "N1"):
        branch_dir = Path(runs[1][arm]["dir"]) / "branches" / "E0.5"
        info = json.loads((branch_dir / "branch.json").read_text())
        assert info["schedule"]["end_flops"] == f_half
        other = run(table, arm, tmp_path / f"short-{arm}", prereg=short)
        assert_same_state(final_state(branch_dir / "final.pt"), final_state(Path(other["dir"]) / "final.pt"))
        rates = [x["lr"] for x in metrics(branch_dir)]
        assert rates == sorted(rates, reverse=True) and rates[-1] > CONFIG.lr * 0.1  # decaying, above the floor


class Interrupt(Exception):
    pass


def kill_after(k):
    def hook(state):
        if state["step"] == k:
            raise Interrupt

    return hook


@pytest.mark.parametrize("arm,k,every", [("N2", 0, 5), ("N1", 1, 1), ("N0", 7, 1), ("N2", 7, 3)])
def test_resume_is_bit_identical_to_an_uninterrupted_run(table, runs, tmp_path, arm, k, every):
    """k = 0: killed after the first step, before any checkpoint, so the resume starts over; (7, 3): the latest
    checkpoint is step 6, so step 7 is recomputed and its metric line replaced."""
    with pytest.raises(Interrupt):
        run(table, arm, tmp_path, runtime=T.Runtime(checkpoint_every=every, on_step=kill_after(max(k, 1))))
    resumed = run(table, arm, tmp_path, runtime=T.Runtime(checkpoint_every=every))
    reference = runs[1][arm]
    assert resumed["run_id"] == reference["run_id"]
    assert_same_state(final_state(Path(resumed["dir"]) / "final.pt"), final_state(Path(reference["dir"]) / "final.pt"))
    drop = ("seconds", "bytes_per_s", "flops_per_s", "peak_memory")
    clean = lambda lines: [{k: v for k, v in x.items() if k not in drop} for x in lines]  # noqa: E731
    assert clean(metrics(Path(resumed["dir"]))) == clean(metrics(Path(reference["dir"])))
    events = [e["event"] for e in registry.read_events(Path(resumed["dir"]))]
    assert events[0] == "main-start" and events[-1] == "main-done"
    assert ("main-resume" in events) == (k > 0)


def test_a_screening_stage_halts_at_the_stable_checkpoint_and_the_run_continues(table, tmp_path):
    screen = write_prereg(
        tmp_path / "screen.toml",
        f_target=table["f_target"],
        extra="[stages.screen]\nstop_after = 0.5\n[stages.decision]\n",
    )
    first = run(table, "N1", tmp_path / "runs", prereg=screen, stage="screen")
    threshold = round(0.9 * round(0.5 * table["f_target"]))
    assert not first["done"] and first["state"]["flops"] >= threshold
    assert metrics(Path(first["dir"]))[-2]["flops"] < threshold
    T.branch(first["dir"], endpoint=0.5)
    later = run(table, "N1", tmp_path / "runs", prereg=screen, stage="decision")
    assert later["run_id"] == first["run_id"] and later["done"]
    with pytest.raises(TokstudyError, match="takes part in stages"):
        run(table, "N2", tmp_path / "runs", prereg=screen, stage="screen")


def test_the_run_id_changes_with_each_input_and_another_id_is_refused(table, tmp_path):
    prereg = load_prereg(table["prereg"])
    base = T.setup(table["arms"]["N0"], stream=table["stream"], prereg=prereg, seed=0, config=CONFIG)
    rid = base.record["run_id"]
    other_stream = S.build_stream(
        word_docs(seed=1),
        tmp_path / "s2",
        seed=0,
        k_train=K,
        source={"project": "tlc"},
        mixture={"py": 0.75, "en": 0.25},
    )
    variants = [
        T.setup(table["arms"]["N2"], stream=table["stream"], prereg=prereg, seed=0, config=CONFIG),
        T.setup(table["arms"]["N0"], stream=other_stream, prereg=prereg, seed=0, config=CONFIG),
        T.setup(
            table["arms"]["N0"],
            stream=table["stream"],
            prereg=prereg,
            seed=0,
            config=T.TrainConfig(lr=1e-3, batch_rows=4, micro_rows=2, precision="fp32"),
        ),
        T.setup(table["arms"]["N0"], stream=table["stream"], prereg=prereg, seed=1, config=CONFIG),
        T.setup(
            table["arms"]["N0"],
            stream=table["stream"],
            seed=0,
            config=CONFIG,
            prereg=load_prereg(write_prereg(tmp_path / "p.toml", f_target=table["f_target"] + 1)),
        ),
    ]
    ids = [v.record["run_id"] for v in variants]
    assert len({rid, *ids}) == 6
    assert registry.run_id(**registry.id_inputs(base.record)) == rid
    first = run(table, "N2", tmp_path / "runs")
    with pytest.raises(TokstudyError, match="refused"):
        run(table, "N0", tmp_path / "runs", run_dir=Path(first["dir"]))
    with pytest.raises(TokstudyError, match="registered seeds"):
        run(table, "N2", tmp_path / "runs", seed=1)
    listed = registry.list_runs(tmp_path / "runs", "tlc-N")
    assert [r["run_id"] for r in listed] == [first["run_id"]]


def test_the_prereg_lock_refuses_a_change_without_an_amendment(table, tmp_path):
    runs_root = tmp_path / "runs"
    run(table, "N2", runs_root)
    changed = write_prereg(tmp_path / "changed.toml", f_target=table["f_target"], extra="[seeds]\nreference = 3\n")
    with pytest.raises(TokstudyError, match="needs --amend"):
        run(table, "N2", runs_root, prereg=changed)
    result = run(table, "N2", runs_root, prereg=changed, amend="seeds recorded")
    assert result["done"]  # same budget and schedule: the same run id, already done
    lock = json.loads((runs_root / "tlc-N" / registry.LOCK_FILE).read_text())
    assert [h["reason"] for h in lock["history"]] == ["first run", "seeds recorded"]
    run(table, "N2", runs_root)  # a prereg accepted before is accepted again


def test_a_stream_too_short_for_f_target_is_refused(table, tmp_path):
    long = write_prereg(tmp_path / "long.toml", f_target=table["f_target"] * 10**4)
    with pytest.raises(TokstudyError, match="below F_target"):
        run(table, "N2", tmp_path, prereg=long)


@pytest.mark.parametrize(
    "body,message",
    [
        ('format = "tokstudy-prereg/2"\ntable = "tlc-N"\n', "not a known version"),
        ('format = "tokstudy-prereg/1"\ntable = "x"\n', "table must be one of"),
        (
            'format = "tokstudy-prereg/1"\ntable = "tlc-N"\n[budget]\nF_target = 1e9\npoints = [0.5]\n'
            "[context]\nK_train = 32\nK_eval = 1\n",
            "end at 1.0",
        ),
        (
            'format = "tokstudy-prereg/1"\ntable = "tlc-N"\n[budget]\nF_target = 1e9\n[context]\nK_train = 4\n'
            "K_eval = 1\n",
            "K_train must be an integer >= 8",
        ),
        (
            'format = "tokstudy-prereg/1"\ntable = "tlc-N"\n[budget]\nF_target = 1e9\n[context]\nK_train = 32\n'
            "K_eval = 8\n[stages.screen]\nstop_after = 0.3\n",
            "must be a budget point",
        ),
        ('format = "tokstudy-prereg/1"\ntable = "tlc-N"\n[budget]\nF_target = 1e9\nseed = 1\n', "unknown keys"),
        ('format = "tokstudy-prereg/1"\ntable = "tlc-N"\nbudgets = 1\n', "unknown top-level keys"),
        (
            'format = "tokstudy-prereg/1"\ntable = "tlc-N"\n[budget]\nF_target = 1e9\n[context]\nK_train = 32\n'
            "K_eval = 8\n[schedule]\nwarmup = 0.95\n",
            "warmup < decay_start",
        ),
    ],
)
def test_prereg_refusals(tmp_path, body, message):
    path = tmp_path / "p.toml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(TokstudyError, match=message):
        load_prereg(path)


def test_prereg_identity_and_signature(table, tmp_path):
    p = load_prereg(table["prereg"])
    assert not p.signed and p.stages == {"decision": {"stop_after": None}} and p.points == (0.25, 0.5, 1.0)
    signed = write_prereg(
        tmp_path / "s.toml", f_target=table["f_target"], extra='[signature]\nsigned_by = "owner"\ndate = "2026-11-03"\n'
    )
    q = load_prereg(signed)
    assert q.signed and q.sha256 != p.sha256 and q.f_target == p.f_target
    w = p.wsd(1.0, endpoint=0.25)
    assert (w.warmup_flops, w.end_flops) == (round(0.02 * p.f_target), round(0.25 * p.f_target))
    assert w.decay_start_flops == round(0.9 * round(0.25 * p.f_target))


def test_wsd_and_cosine_shapes():
    w = schedule.WSD(peak=1.0, warmup_flops=10, decay_start_flops=80, end_flops=100, floor=0.1)
    assert [w.lr(f) for f in (0, 5, 10, 79, 80, 90, 100, 200)] == [0.0, 0.5, 1.0, 1.0, 1.0, 0.55, 0.1, 0.1]
    c = schedule.WSD(1.0, 10, 80, 100, 0.1, "cosine")
    assert c.lr(90) == pytest.approx(0.55) and c.lr(80) == 1.0
    assert schedule.cosine_at(0.01, peak=2.0) == pytest.approx(1.0)
    assert schedule.cosine_at(1.0, peak=2.0) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="decay start < end"):
        schedule.WSD(1.0, 10, 100, 100)


def test_train_and_branch_command_lines(table, tmp_path, capsys):
    args = [
        "--arms",
        str(table["registry"]),
        "--arm",
        "N2",
        "--seed",
        "0",
        "--stream",
        str(table["root"] / "stream"),
        "--prereg",
        str(table["prereg"]),
        "--runs",
        str(tmp_path),
        "--lr",
        "3e-3",
        "--batch-rows",
        "4",
        "--micro-rows",
        "2",
        "--precision",
        "fp32",
        "--device",
        "cpu",
    ]
    assert T.main(args) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["done"] and out["progress"] >= 1
    assert T.branch_main([out["dir"], "--at", "0.25", "--device", "cpu"]) == 0
    assert json.loads(capsys.readouterr().out)["done"]
    assert T.branch_main([out["dir"], "--at", "0.3"]) == 1
    assert "budget point below 1" in capsys.readouterr().err
    assert T.main([*args[:3], "N9", *args[4:]]) == 1
    assert "has no arm 'N9'" in capsys.readouterr().err


def test_a_branch_whose_stable_step_reaches_its_endpoint_is_final_at_once(table, tmp_path):
    tiny = write_prereg(tmp_path / "tiny.toml", f_target=table["f_target"] // 30)  # about one step of the byte arm
    result = run(table, "N2", tmp_path / "runs", prereg=tiny)
    branch = T.branch(result["dir"], endpoint=0.5)
    assert branch["done"] and (Path(branch["dir"]) / "final.pt").is_file()
    stable = final_state(Path(result["dir"]) / "stable" / "E0.5.pt")
    final = final_state(Path(branch["dir"]) / "final.pt")
    assert final["step"] == stable["step"] and final["flops"] >= round(0.5 * (table["f_target"] // 30))
    assert metrics(Path(branch["dir"])) == []  # no step was needed


def test_the_tables_preregs_load_and_their_budgets_are_the_analytic_ones(tlc_default_config):
    """prereg/tlc.toml and prereg/bilingual.toml (proposals, unsigned): F_target is A0 at 0.8B tokens (TLC IR
    formula x 3) and B0 at 0.8B tokens (6 x the bilingual MACs per token at T = 1024)."""
    from surogate_bilingual.audit.flops import macs_per_token
    from surogate_bilingual.config import profile
    from surogate_tlc import ir

    prereg_dir = Path(__file__).resolve().parents[2] / "prereg"
    tlc, bilingual = (load_prereg(prereg_dir / f"{name}.toml") for name in ("tlc", "bilingual"))
    assert tlc.table == "tlc-N" and bilingual.table == "bilingual-B" and not tlc.signed and not bilingual.signed
    cfg = {**tlc_default_config, "vocab_size": 16384}
    assert tlc.f_target == 3 * ir.flops_per_token(cfg, 512) * 800_000_000
    seed = profile("pilot", "swiglu")
    assert bilingual.f_target == 6 * macs_per_token(seed, "single", T=1024) * 800_000_000
    assert (tlc.k_train, bilingual.k_train, tlc.mixture, bilingual.measure) == (
        2048,
        4096,
        {"en": 0.3, "py": 0.7},
        "B0",
    )
    assert bilingual.stop_after("screen") == 0.5 and bilingual.stop_after("decision") is None
