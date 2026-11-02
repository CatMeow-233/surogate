"""The training loop (docs/spec.md, I5-T1 … T5, T11 … T13, T16; IMPL-01, IMPL-02, NEW-01).

Subprocess cases run the real command line (`python -m surogate_bilingual train`) with one CPU thread; the kill
cases use a small driver that exits the process with os._exit at a chosen point, which is what a crash or a kill -9
does: no finally block, no cleanup.
"""

import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from surogate_bilingual.__main__ import main as cli
from surogate_bilingual.assembly.hashing import content_hash
from surogate_bilingual.config import load_toml
from surogate_bilingual.train import metrics, state
from surogate_bilingual.train.loop import TrainError, prepare, train

SRC = Path(__file__).resolve().parents[2] / "src"
TINY = Path(__file__).resolve().parents[2] / "examples" / "tiny.toml"

# Runs the command line with one CPU thread; argv: SRC, then the command's arguments.
CLI_DRIVER = (
    "import sys; sys.path.insert(0, sys.argv[1]); import torch; torch.set_num_threads(1);"
    " from surogate_bilingual.__main__ import main; sys.exit(main(sys.argv[2:]))"
)

# Trains a stage and kills itself right after the state of step KILL_AT is on disk (mode "after_save"), or inside
# the save of step KILL_AT after its temporary directory is written and before os.replace (mode "before_replace").
KILL_DRIVER = textwrap.dedent(
    """
    import os, sys
    sys.path.insert(0, sys.argv[1])
    import torch
    torch.set_num_threads(1)
    from surogate_bilingual import canonical
    from surogate_bilingual.config import load_toml
    from surogate_bilingual.train import loop, state
    mode, kill_at, root, runs, stage = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
    if mode == "after_save":
        original = state.save_state
        def save_state(run_dir, step, **kw):
            path = original(run_dir, step, **kw)
            if step == kill_at:
                os._exit(9)
            return path
        state.save_state = save_state
    else:
        real_replace = os.replace
        def replace(src, dst):
            if os.path.basename(str(dst)) == state.step_name(kill_at):
                os._exit(7)
            return real_replace(src, dst)
        canonical.os.replace = replace
    run = load_toml(sys.argv[7], stage=stage, overrides=tuple(sys.argv[8:]))
    loop.train(run, root=root, runs=runs)
    """
)


def _env():
    return {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def losses(run_dir: Path) -> list[float]:
    return [line["loss"] for line in metrics.read(run_dir / metrics.FILE)]


T2_OVERRIDES = ("registry.allow_dirty=true", "run.steps=10", "run.save_every=5")


@pytest.fixture(scope="module")
def processes(tiny_trained, tmp_path_factory):
    """Every subprocess of this module, started together so that their torch imports overlap: two full tiny-seed
    runs (I5-T1), two runs killed right after the state of step 5 is saved (I5-T2) and one killed inside the save of
    step 10, between its temporary directory and os.replace (I5-T3)."""
    root, base = tiny_trained["root"], tmp_path_factory.mktemp("processes")
    cli_args = ["train", str(TINY), "--stage", "tiny-seed", "--root", str(root), "--set", "registry.allow_dirty=true"]
    commands = {
        "t1a": [CLI_DRIVER, str(SRC), *cli_args, "--runs", str(base / "t1a")],
        "t1b": [CLI_DRIVER, str(SRC), *cli_args, "--runs", str(base / "t1b")],
        "t2-tiny-seed": [
            KILL_DRIVER,
            str(SRC),
            "after_save",
            "5",
            str(root),
            str(base / "t2-tiny-seed"),
            "tiny-seed",
            str(TINY),
            *T2_OVERRIDES,
        ],
        "t2-tiny-zh": [
            KILL_DRIVER,
            str(SRC),
            "after_save",
            "5",
            str(root),
            str(base / "t2-tiny-zh"),
            "tiny-zh",
            str(TINY),
            *T2_OVERRIDES,
        ],
        "t3": [
            KILL_DRIVER,
            str(SRC),
            "before_replace",
            "10",
            str(root),
            str(base / "t3"),
            "tiny-seed",
            str(TINY),
            *T2_OVERRIDES,
        ],
    }
    procs = {
        key: subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", *cmd],
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for key, cmd in commands.items()
    }
    done = {}
    for key, proc in procs.items():
        out, err = proc.communicate(timeout=120)
        done[key] = (proc.returncode, err, base / key)
    return done


def test_i5_t1_two_processes_give_the_same_init_and_loss_trace(processes, tiny_trained, tiny_stage):
    for key in ("t1a", "t1b"):
        assert processes[key][0] == 0, processes[key][1]
    a, b = (json.loads((processes[k][2] / "tiny-seed" / "run.json").read_text()) for k in ("t1a", "t1b"))
    assert a["run_id"] == b["run_id"] and a["init_hash"] == b["init_hash"]
    trace_a, trace_b = (losses(processes[k][2] / "tiny-seed") for k in ("t1a", "t1b"))
    assert len(trace_a) == 20 and trace_a == trace_b  # bitwise: the JSON floats round-trip exactly
    # this process builds the same initialisation; another init_seed gives another one
    root = tiny_trained["root"]
    assert prepare(tiny_stage("tiny-seed"), root=root).fields["init_hash"] == a["init_hash"]
    assert prepare(tiny_stage("tiny-seed", "run.init_seed=99"), root=root).fields["init_hash"] != a["init_hash"]


def _final(run_dir: Path) -> dict:
    path = state.latest_state(run_dir)
    loaded = state.load_state(path, run_id=json.loads((run_dir / "run.json").read_text())["run_id"])
    return {
        "step": loaded.step,
        "trainable": loaded.trainable,
        "optimizer": loaded.optimizer,
        "scheduler": loaded.extra["scheduler"],
        "sampler": loaded.extra["sampler"],
    }


def _same_optimizer(a: dict, b: dict) -> bool:
    if a["param_groups"] != b["param_groups"] or set(a["state"]) != set(b["state"]):
        return False
    return all(torch.equal(a["state"][i][k], b["state"][i][k]) for i in a["state"] for k in a["state"][i])


@pytest.mark.parametrize("stage", ["tiny-seed", "tiny-zh"])
def test_i5_t2_killed_and_resumed_equals_uninterrupted(processes, tiny_trained, tiny_stage, tmp_path, stage):
    root = tiny_trained["root"]
    whole = train(tiny_stage(stage, *T2_OVERRIDES[1:]), root=root, runs=tmp_path / "whole")
    code, err, runs = processes[f"t2-{stage}"]
    assert code == 9, err
    run_dir = runs / stage
    assert [s for s, _ in state.list_states(run_dir)] == [5] and len(losses(run_dir)) == 5
    resumed = train(tiny_stage(stage, *T2_OVERRIDES[1:]), root=root, runs=runs)
    assert resumed.resumed_from == 5 and resumed.status == "done" and len(resumed.losses) == 5
    a, b = _final(whole.run_dir), _final(run_dir)
    assert a["step"] == b["step"] == 10
    assert a["trainable"].keys() == b["trainable"].keys()
    assert all(torch.equal(a["trainable"][k], b["trainable"][k]) for k in a["trainable"])
    assert _same_optimizer(a["optimizer"], b["optimizer"])
    assert a["scheduler"] == b["scheduler"] and a["sampler"] == b["sampler"]
    assert whole.sampler.state_dict() == resumed.sampler.state_dict()
    assert losses(whole.run_dir) == losses(run_dir)
    assert [m["step"] for m in metrics.read(run_dir / "metrics.jsonl")] == list(range(1, 11))
    trained = dict(
        zip(resumed.optimizer.param_groups[0]["names"], resumed.optimizer.param_groups[0]["params"], strict=True)
    )
    assert all(torch.equal(trained[k], a["trainable"][k]) for k in trained)


def test_i5_t3_a_crash_before_replace_leaves_the_previous_state_loadable(processes, tiny_trained, tiny_stage):
    code, err, runs = processes["t3"]
    assert code == 7, err
    run_dir = runs / "tiny-seed"
    leftovers = [p.name for p in (run_dir / "state").iterdir() if state.STEP_DIR.fullmatch(p.name) is None]
    assert len(leftovers) == 1 and leftovers[0].startswith("step_00000010.tmp-")  # the half-written state
    assert [s for s, _ in state.list_states(run_dir)] == [5]
    run_id = json.loads((run_dir / "run.json").read_text())["run_id"]
    assert state.load_state(state.latest_state(run_dir), run_id=run_id).step == 5
    resumed = train(tiny_stage("tiny-seed", *T2_OVERRIDES[1:]), root=tiny_trained["root"], runs=runs)
    assert resumed.resumed_from == 5 and resumed.status == "done"
    assert sorted(p.name for p in (run_dir / "state").iterdir()) == ["step_00000005", "step_00000010"]
    assert [m["step"] for m in metrics.read(run_dir / "metrics.jsonl")] == list(range(1, 11))


def test_i5_t3_retention_keeps_the_last_two_and_every_keep_every(tmp_path):
    p = torch.nn.Parameter(torch.zeros(3))
    opt = torch.optim.AdamW([p], lr=0.1)
    for step in range(1, 13):
        state.save_state(
            tmp_path, step, trainable={"p": p}, optimizer=opt, extra={"run_id": "r", "tokens": 0}, keep_every=4
        )
    assert [s for s, _ in state.list_states(tmp_path)] == [4, 8, 11, 12]
    assert not [p for p in (tmp_path / "state").iterdir() if "tmp" in p.name]
    state.save_state(tmp_path, 13, trainable={"p": p}, optimizer=opt, extra={"run_id": "r", "tokens": 0})
    assert [s for s, _ in state.list_states(tmp_path)] == [12, 13]  # keep_every = 0: the last two only


def test_i5_t4_resume_refuses_another_run_id_and_changed_frozen_tensors(tiny_trained, tiny_stage, tmp_path):
    root, runs = tiny_trained["root"], tmp_path / "runs"
    first = train(tiny_stage("tiny-zh", "run.steps=6", "run.save_every=3"), root=root, runs=runs, stop_after=3)
    assert first.status == "running" and first.step == 3
    path = state.latest_state(first.run_dir)
    with pytest.raises(state.StateError, match="refusing to resume another run"):
        state.load_state(path, run_id="0" * 16)
    extra = json.loads((path / state.EXTRA).read_text())
    key = sorted(extra["frozen_hashes"])[0]  # a backbone tensor
    extra["frozen_hashes"][key] = "0" * 64
    (path / state.EXTRA).write_text(json.dumps(extra))
    from surogate_bilingual.assembly.freeze import FreezeError

    with pytest.raises(FreezeError, match="frozen tensors differ"):
        train(tiny_stage("tiny-zh", "run.steps=6", "run.save_every=3"), root=root, runs=runs)
    assert json.loads((first.run_dir / "run.json").read_text())["status"] == "failed"


def test_i5_t4_a_tampered_state_file_is_refused(tiny_trained, tiny_stage, tmp_path):
    first = train(
        tiny_stage("tiny-seed", "run.steps=4", "run.save_every=2"),
        root=tiny_trained["root"],
        runs=tmp_path / "runs",
        stop_after=2,
    )
    path = state.latest_state(first.run_dir)
    weights = load_file(str(path / state.TRAINABLE))
    weights["backbone.final_norm.weight"][0] += 1
    from safetensors.torch import save_file

    save_file(weights, str(path / state.TRAINABLE))
    with pytest.raises(state.StateError, match="sha256"):
        state.load_state(path, run_id=first.run_id)


def test_i5_t5_the_seed_recipe_fits_a_fixed_batch(tiny_trained, tiny_stage, tmp_path):
    run = tiny_stage(
        "tiny-seed",
        'run.shard="shards/fixed"',
        "run.steps=300",
        "run.save_every=0",
        "train.accum=1",
        "train.global_batch_tokens=32",
    )
    result = train(run, root=tiny_trained["root"], runs=tmp_path / "runs")
    trace = result.losses
    assert len(trace) == 300 and min(trace) < 0.1 * trace[0]
    starts = {int(s) for _ in range(5) for s in result.sampler.next_batch(2)[0]}
    assert starts == {0}  # one window: every batch was the same


@pytest.mark.parametrize("stage", ["tiny-seed", "tiny-bridge"])
def test_i5_t11_checkpointing_leaves_the_loop_gradients_unchanged(tiny_trained, tiny_stage, stage):
    grads = []
    for flag in ("false", "true"):
        p = prepare(tiny_stage(stage, f"train.activation_checkpointing={flag}"), root=tiny_trained["root"])
        reader = p.resolved.shard
        windows = [reader.windows([0, 37], 16), reader.windows([101, 160], 16)]
        stats = p.trainer.gradients(windows)
        assert stats["targets"] > 0
        grads.append({n: q.grad.clone() for n, q in zip(p.plan.trainable_names, p.plan.parameters(), strict=True)})
        p.plan.check_grads()  # frozen parameters got no gradient
    assert grads[0].keys() == grads[1].keys()
    assert max(float((grads[0][k] - grads[1][k]).abs().max()) for k in grads[0]) <= 1e-7


@pytest.mark.parametrize("stage", ["tiny-seed", "tiny-zh", "tiny-bridge"])
def test_i5_t12_three_bf16_autocast_steps_on_the_cpu_have_finite_loss(tiny_trained, tiny_stage, tmp_path, stage):
    result = train(
        tiny_stage(stage, "run.steps=3", 'train.dtype="bf16"'), root=tiny_trained["root"], runs=tmp_path / "runs"
    )
    assert result.status == "done" and len(result.losses) == 3 and all(math.isfinite(x) for x in result.losses)
    doc = json.loads((result.run_dir / "run.json").read_text())
    assert doc["config"]["precision"] == "bf16"
    assert all(p.dtype == torch.float32 for p in result.model.parameters())  # parameters stay fp32


def test_i5_t13_every_metrics_line_validates_and_cuda_fields_are_null_on_cpu(tiny_trained):
    for result in tiny_trained["results"].values():
        text = (result.run_dir / "metrics.jsonl").read_text("utf-8")
        lines = [json.loads(line) for line in text.splitlines()]
        assert len(lines) == result.total_steps
        for doc in lines:
            metrics.check_line(doc)
            assert doc["peak_alloc_gib"] is None and doc["peak_reserved_gib"] is None and doc["dev"] is None
        assert [d["step"] for d in lines] == list(range(1, result.total_steps + 1))
        assert [d["tokens"] for d in lines] == [64 * s for s in range(1, result.total_steps + 1)]
    good = lines[-1]
    for bad in (
        {**good, "extra": 1},
        {k: v for k, v in good.items() if k != "lr"},
        {**good, "step": True},
        {**good, "loss": float("nan")},
        {**good, "format": "bilingual-metrics/2"},
    ):
        with pytest.raises(metrics.MetricsError):
            metrics.check_line(bad)


def test_metrics_truncate_drops_later_steps_and_a_torn_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [{"step": s, "loss": 1.0 / s} for s in range(1, 6)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"step": 6, "lo')
    assert metrics.truncate(path, 3) == 2
    assert [r["step"] for r in metrics.read(path)] == [1, 2, 3] and path.read_text().endswith("\n")


def test_i5_t16_inconsistent_batch_arithmetic_is_refused_before_training(tiny_trained, tmp_path, capsys):
    with pytest.raises(ValueError, match="global_batch_tokens"):
        load_toml(TINY, stage="tiny-seed", overrides=("train.accum=3",))
    runs = tmp_path / "runs"
    args = [
        "train",
        str(TINY),
        "--stage",
        "tiny-seed",
        "--root",
        str(tiny_trained["root"]),
        "--runs",
        str(runs),
        "--set",
        "train.accum=3",
    ]
    assert cli(args) == 1
    assert "global_batch_tokens" in capsys.readouterr().err and not runs.exists()


def test_train_refuses_eval_splits_missing_inputs_and_a_seq_beyond_max_seq(tiny_trained, tiny_stage, tmp_path):
    root = tiny_trained["root"]
    with pytest.raises(TrainError, match="never reads dev or test"):
        prepare(tiny_stage("tiny-zh", 'run.shard="shards/dev"'), root=root)
    with pytest.raises(TrainError, match="reads the packs"):
        prepare(tiny_stage("tiny-bridge", 'run.packs={zh = "data/runs/tiny-zh/packs/zh"}'), root=root)
    with pytest.raises(ValueError, match="max_seq"):
        tiny_stage("tiny-seed", "train.seq=32", "train.global_batch_tokens=128")
    with pytest.raises(ValueError, match="exactly one of run.steps and run.tokens"):
        tiny_stage("tiny-seed", "run.tokens=640")
    from surogate_bilingual.config import RunSpec, TrainConfig

    spec = RunSpec(name="x", recipe="seed", init_seed=1, data_seed=2, shard="s", tokens=650)
    assert spec.total_steps(TrainConfig(microbatch=2, seq=16, accum=2, global_batch_tokens=64)) == 11  # ceil


def test_train_dry_run_writes_nothing(tiny_trained, tmp_path, capsys):
    root = tiny_trained["root"]
    before = sorted(str(p) for p in root.rglob("*"))
    args = [
        "train",
        str(TINY),
        "--stage",
        "tiny-bridge",
        "--root",
        str(root),
        "--set",
        "registry.allow_dirty=true",
        "--dry-run",
    ]
    assert cli(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["experts"] == ["zh", "en", "bridge"] and plan["total_steps"] == 10 and len(plan["run_id"]) == 16
    assert sorted(str(p) for p in root.rglob("*")) == before
    assert cli(["train", str(TINY), "--list"]) == 0
    assert "tiny-t1p-en" in capsys.readouterr().out.split()
    assert cli(["train", str(TINY), "--root", str(root)]) == 1  # a row file with stages needs --stage
    assert "choose one" in capsys.readouterr().err


def test_a_finished_run_is_idempotent(tiny_trained, tiny_stage):
    again = train(tiny_stage("tiny-zh"), root=tiny_trained["root"])
    first = tiny_trained["results"]["tiny-zh"]
    assert again.run_id == first.run_id and again.status == "done" and again.losses == []
    assert again.packs == first.packs
    seed = json.loads((first.packs["zh"] / "pack.json").read_text())
    assert seed["train"]["run_id"] == first.run_id and seed["train"]["step"] == 20
    final = state.load_state(state.latest_state(first.run_dir), run_id=first.run_id).trainable
    assert (
        content_hash({f"ffn.{k.removeprefix('bank.experts.zh.')}": v for k, v in final.items()})
        == seed["hashes"]["expert"]
    )
