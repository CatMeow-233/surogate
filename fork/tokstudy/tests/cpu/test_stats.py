"""Statistics, the preregistered verdicts and the twin gates (implementation step S16, the verdict half) on SYNTHETIC
FIXTURE `tokstudy-nll/1` sets with known effects, written by these tests in the format's real schema:

- every rule gives its decision: replace, keep, tie on a sign flip between the views, extend on a narrowing lead
  and under M6/M10, eliminated at screening, floor-only; the native table's (a)-(c) and the bilingual Track S rule;
- the bootstrap is reproducible; the pooled seed SD equals the hand value; interpolate_at is log-log;
- mixing tables, backends or evalsets, duplicate files, an unaccepted prereg and an unsigned prereg are refused;
  an amended prereg is accepted through the lock; a failed or missing twin gate labels the table internal-only.
"""

import json
import math
import zlib
from pathlib import Path

import numpy as np
import pytest
from surogate_tokstudy import stats
from surogate_tokstudy.canonical import TokstudyError, atomic_write_json
from surogate_tokstudy.evaluate import DocNLL, write_nll

PREREG_DIR = Path(__file__).resolve().parents[2] / "prereg"
LN2 = math.log(2.0)
N_DOCS = 120


def signed(tmp_path, name, *, resamples=2000, edit=lambda s: s):
    """A copy of a committed prereg, signed (SYNTHETIC FIXTURE signature) and with fewer bootstrap resamples."""
    text = (PREREG_DIR / f"{name}.toml").read_text(encoding="utf-8")
    text = text.replace('signed_by = ""', 'signed_by = "SYNTHETIC FIXTURE"').replace('date = ""', 'date = "2026-11-03"')
    text = text.replace("bootstrap_resamples = 10000", f"bootstrap_resamples = {resamples}")
    path = tmp_path / f"{name}.toml"
    path.write_text(edit(text), encoding="utf-8")
    return path


def docs(domains, seed=0):
    """Paired documents: keys, domains, bytes and a per-document difficulty shared by every arm."""
    rng = np.random.default_rng(seed)
    out = []
    for domain in domains:
        for i in range(N_DOCS):
            out.append((f"{domain}/{i:04d}", domain, int(rng.integers(200, 600)), float(rng.normal(0, 0.05))))
    return out


def write_run(
    path, documents, *, table, arm, kind, seed, point=1.0, bpb, backend="tokstudy-torch", seed_sd=0.002, **header
):
    """One NLL file: document d of domain x gets bpb[x] + its difficulty + a seed offset (sd `seed_sd`) + a little
    per-document noise, in bits per byte; every draw comes from a generator seeded by (arm, seed, point)."""
    rng = np.random.default_rng([zlib.crc32(f"{arm}|{seed}|{point:g}".encode()), 1])
    offset = float(rng.normal(0, seed_sd))
    rows = []
    for key, domain, n_bytes, difficulty in documents:
        bits = (bpb[domain] + difficulty + offset + float(rng.normal(0, 0.004))) * n_bytes
        rows.append(DocNLL(key, domain, n_bytes, bits * LN2, n_bytes // 4))
    base = {"table": table, "backend": backend, "arm": arm, "seed": seed}
    if backend == "tokstudy-torch":
        base |= {"arm_kind": kind, "point": point, "evalset_sha256": "e" * 64, "k_eval": 2048}
    else:
        base |= {"corpus_spec_sha256": "c" * 64, "split": "val", "keys": "corpus"}
    write_nll(path, {**base, **header}, rows)
    return path


def tlc_n(tmp_path, prereg, effects, *, seeds=None, bytes_at=None, points=(0.25, 0.5, 1.0)):
    """tlc-N files: effects[arm][point] = {py, en} BPB; train bytes per point (default: equal for every arm)."""
    from surogate_tokstudy.prereg import load_prereg

    sha = load_prereg(prereg).sha256
    documents = docs(("py", "en"))
    kinds = {"N0": "subword", "N1": "hnet", "N2": "bytes"}
    files = []
    for arm, by_point in effects.items():
        for seed in range((seeds or {}).get(arm, 3 if arm != "N2" else 1)):
            for point in points:
                if point not in by_point:
                    continue
                train_bytes = (bytes_at or {}).get(arm, {}).get(point, round(1e9 * point))
                path = tmp_path / f"{arm}-s{seed}-E{point:g}.nll"
                files.append(
                    write_run(
                        path,
                        documents,
                        table="tlc-N",
                        arm=arm,
                        kind=kinds[arm],
                        seed=seed,
                        point=point,
                        bpb=by_point[point],
                        prereg_sha256=sha,
                        run_prereg_sha256=sha,
                        train_bytes=train_bytes,
                    )
                )
    return files


def flat(py, en, points=(0.25, 0.5, 1.0), slope=0.1):
    """BPB per point that falls toward the final values (py, en)."""
    return {p: {"py": py + slope * (1 - p), "en": en + slope * (1 - p)} for p in points}


EVIDENCE = {"throughput": {"N1": 0.7}, "integration_signoff": True}


def test_pooled_seed_sd_bootstrap_and_interpolation_by_hand(tmp_path):
    groups = {"a": [1.0, 1.2, 1.1], "b": [2.0, 2.4], "c": [5.0]}
    ss = (0.01 + 0.01 + 0.0) + (0.04 + 0.04)
    assert stats.pooled_seed_sd(groups) == pytest.approx(math.sqrt(ss / 3))
    assert stats.pooled_seed_sd({"c": [5.0]}) is None
    assert stats.interpolate_at([(1, 2.0), (4, 1.0)], 2) == pytest.approx(math.exp(math.log(2) - 0.5 * math.log(2)))
    assert stats.interpolate_at([(1, 2.0), (4, 1.0)], 5) is None and stats.interpolate_at([(1, 2.0)], 1) == 2.0
    prereg = signed(tmp_path, "tlc")
    rules = stats.load_rules(prereg)
    files = tlc_n(tmp_path, prereg, {"N0": flat(1.0, 1.0), "N1": flat(0.96, 1.0)})
    table = stats.build_table(rules, files)
    a, b = (stats.paired_delta(table, "N1", "N0", "py", 1.0) for _ in range(2))
    assert a == b and a.sigma_boot > 0  # the same generator for the same comparison
    seed_bpb = {arm: [r.bpb("py") for r in table.at(arm, 1.0)] for arm in ("N0", "N1")}
    assert a.pooled_sd == pytest.approx(stats.pooled_seed_sd(seed_bpb))
    assert a.sigma_seed == pytest.approx(a.pooled_sd * math.sqrt(2 / 3))
    assert a.delta == pytest.approx(np.mean(seed_bpb["N1"]) - np.mean(seed_bpb["N0"]))
    (tmp_path / "x").mkdir()
    other = stats.load_rules(
        signed(tmp_path / "x", "tlc", edit=lambda s: s.replace("bootstrap_seed = 0", "bootstrap_seed = 7"))
    )
    c = stats.paired_delta(stats.build_table(other, files_for(other, tmp_path)), "N1", "N0", "py", 1.0)
    assert c.sigma_boot != a.sigma_boot and c.delta == pytest.approx(a.delta)


def files_for(rules, tmp_path):
    """The same synthetic table again under another prereg (for the bootstrap-seed check)."""
    out = tmp_path / "again"
    out.mkdir(exist_ok=True)
    return tlc_n(out, rules.path, {"N0": flat(1.0, 1.0), "N1": flat(0.96, 1.0)})


@pytest.mark.parametrize(
    ("n1", "evidence", "decision", "reason"),
    [
        (flat(0.96, 1.0), EVIDENCE, "replace", None),  # 4 % better in Python, English equal
        (flat(0.96, 1.0), {"throughput": {"N1": 0.7}}, "keep", "integration sign-off"),
        (flat(0.96, 1.0), {"throughput": {"N1": 0.3}, "integration_signoff": True}, "keep", "throughput"),
        (flat(0.96, 1.02), EVIDENCE, "keep", "inferior"),  # English 2 % worse
        (flat(0.99, 1.0), EVIDENCE, "extend", "M10"),  # within 2 %: re-sweep the LR first
        (flat(0.99, 1.0), {**EVIDENCE, "lr_resweep_done": ["N1"]}, "keep", "no gain"),
        (flat(1.05, 1.0), EVIDENCE, "keep", "no gain"),  # worse
    ],
)
def test_the_tlc_n_rules(tmp_path, n1, evidence, decision, reason):
    prereg = signed(tmp_path, "tlc")
    files = tlc_n(tmp_path, prereg, {"N0": flat(1.0, 1.0), "N1": n1, "N2": flat(1.3, 1.3)})
    doc = stats.verdict(stats.load_rules(prereg), files, evidence=evidence, twin=passed_twin(tmp_path, "tlc-N"))
    v = doc["arms"]["N1"]
    assert v["decision"] == decision, v
    assert reason is None or reason in json.dumps(v["reasons"] + v["pending"])
    assert doc["arms"]["N2"]["decision"] == "floor-only" and doc["label"] == "decision"
    assert doc["decision"]["promote"] == (["N1"] if decision == "replace" else [])
    assert doc["prereg"]["sha256"] == stats.load_rules(prereg).sha256


def test_a_sign_flip_between_the_views_is_a_tie_and_a_narrowing_lead_extends(tmp_path):
    prereg = signed(tmp_path, "tlc")
    # N1 reads 4x the bytes per FLOP: at N0's bytes (N1's 0.25 point) it is worse, at equal FLOPs better
    n1 = {0.25: {"py": 1.10, "en": 1.0}, 0.5: {"py": 1.03, "en": 1.0}, 1.0: {"py": 0.95, "en": 1.0}}
    bytes_at = {"N1": {0.25: 10**9, 0.5: 2 * 10**9, 1.0: 4 * 10**9}}
    (tmp_path / "flip").mkdir()
    files = tlc_n(tmp_path / "flip", prereg, {"N0": flat(1.0, 1.0), "N1": n1}, bytes_at=bytes_at)
    doc = stats.verdict(stats.load_rules(prereg), files, evidence=EVIDENCE, twin=passed_twin(tmp_path, "tlc-N"))
    v = doc["arms"]["N1"]
    assert v["decision"] == "tie" and "disagree in sign" in v["reasons"][0]
    assert v["conditions"]["equal_bytes_py"]["available"] and v["conditions"]["equal_bytes_py"]["delta"] > 0
    # the lead at 0.5 is 0.08, at 1.0 0.03: it shrinks by more than half
    n1 = {0.25: {"py": 1.0, "en": 1.0}, 0.5: {"py": 0.97, "en": 1.0}, 1.0: {"py": 0.97, "en": 1.0}}
    n0 = {0.25: {"py": 1.1, "en": 1.0}, 0.5: {"py": 1.05, "en": 1.0}, 1.0: {"py": 1.0, "en": 1.0}}
    (tmp_path / "narrow").mkdir()
    files = tlc_n(tmp_path / "narrow", prereg, {"N0": n0, "N1": n1})
    doc = stats.verdict(stats.load_rules(prereg), files, evidence=EVIDENCE, twin=passed_twin(tmp_path, "tlc-N"))
    v = doc["arms"]["N1"]
    assert v["decision"] == "extend" and "shrinks by more than half" in v["reasons"][0]


def test_m6_asks_for_another_seed_when_the_difference_is_within_three_sigma(tmp_path):
    prereg = signed(tmp_path, "tlc", edit=lambda s: s.replace("lr_resweep_gap = 0.02", "lr_resweep_gap = 0.0"))
    files = tlc_n(tmp_path, prereg, {"N0": flat(1.0, 1.0), "N1": flat(1.0, 1.0)}, seeds={"N0": 2, "N1": 2})
    doc = stats.verdict(stats.load_rules(prereg), files, evidence=EVIDENCE, twin=passed_twin(tmp_path, "tlc-N"))
    reasons = " ".join(doc["arms"]["N1"]["reasons"])
    assert doc["arms"]["N1"]["decision"] == "extend" and "M6" in reasons and "preregistered seeds" in reasons


def passed_twin(tmp_path, table, passed=True):
    path = tmp_path / f"twin-{table}-{passed}.json"
    atomic_write_json(path, {"format": "tokstudy-twin/1", "table": table, "passed": passed, "by_domain": {}})
    return path


def test_the_verdict_refuses_what_it_cannot_compare(tmp_path):
    prereg = signed(tmp_path, "tlc")
    rules = stats.load_rules(prereg)
    files = tlc_n(tmp_path, prereg, {"N0": flat(1.0, 1.0), "N1": flat(0.96, 1.0)})
    documents = docs(("py", "en"))
    sha = rules.sha256
    common = {"prereg_sha256": sha, "run_prereg_sha256": sha, "train_bytes": 10**9}
    cases = {
        "the files mix tables": write_run(
            tmp_path / "b.nll",
            documents,
            table="bilingual-B",
            arm="B0",
            kind="subword",
            seed=0,
            bpb={"py": 1, "en": 1},
            **common,
        ),
        "differ in evalset_sha256": write_run(
            tmp_path / "e.nll",
            documents,
            table="tlc-N",
            arm="N2",
            kind="bytes",
            seed=0,
            bpb={"py": 1, "en": 1},
            **{**common, "evalset_sha256": "f" * 64},
        ),
        "other documents": write_run(
            tmp_path / "d.nll",
            docs(("py", "en"), seed=1),
            table="tlc-N",
            arm="N2",
            kind="bytes",
            seed=0,
            bpb={"py": 1, "en": 1},
            **common,
        ),
        "two files of arm N0 seed 0": write_run(
            tmp_path / "dup.nll",
            documents,
            table="tlc-N",
            arm="N0",
            kind="subword",
            seed=0,
            bpb={"py": 1, "en": 1},
            **common,
        ),
        "not a prereg this verdict accepts": write_run(
            tmp_path / "p.nll",
            documents,
            table="tlc-N",
            arm="N2",
            kind="bytes",
            seed=0,
            bpb={"py": 1, "en": 1},
            **{**common, "run_prereg_sha256": "0" * 64},
        ),
    }
    for message, extra in cases.items():
        with pytest.raises(TokstudyError, match=message):
            stats.verdict(rules, [*files, extra], twin=passed_twin(tmp_path, "tlc-N"))
    unsigned = tmp_path / "unsigned.toml"
    unsigned.write_text((PREREG_DIR / "tlc.toml").read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(TokstudyError, match="not signed"):
        stats.verdict(stats.load_rules(unsigned), files)
    (tmp_path / "draft").mkdir()
    draft_files = tlc_n(tmp_path / "draft", unsigned, {"N0": flat(1.0, 1.0)})
    doc = stats.verdict(stats.load_rules(unsigned), draft_files, draft=True, twin=passed_twin(tmp_path, "tlc-N"))
    assert doc["label"] == "draft"
    # a failed or missing twin gate: internal-only
    assert stats.verdict(rules, files, evidence=EVIDENCE)["label"] == "internal-only"
    assert (
        stats.verdict(rules, files, evidence=EVIDENCE, twin=passed_twin(tmp_path, "tlc-N", False))["label"]
        == "internal-only"
    )
    # a prereg without the M6/M10 rules or with a threshold missing is not a verdict's prereg
    (tmp_path / "bad").mkdir()
    for old, message in (("seed_sigma = 3.0", "extension"), ("n2_floor_only = true", "missing")):
        with pytest.raises(TokstudyError, match=message):
            stats.load_rules(signed(tmp_path / "bad", "tlc", edit=lambda s, old=old: s.replace(old, "")))


def test_an_amended_prereg_is_accepted_through_the_lock(tmp_path):
    from surogate_tokstudy import registry
    from surogate_tokstudy.prereg import load_prereg

    first = signed(tmp_path, "tlc")
    (tmp_path / "v2").mkdir()
    second = signed(
        tmp_path / "v2", "tlc", edit=lambda s: s.replace("n1_throughput_ratio = 0.5", "n1_throughput_ratio = 0.6")
    )
    runs = tmp_path / "runs"
    registry.lock_prereg(runs, load_prereg(first))
    files = tlc_n(tmp_path, first, {"N0": flat(1.0, 1.0), "N1": flat(0.96, 1.0)})
    with pytest.raises(TokstudyError, match="not a prereg this verdict accepts"):
        stats.verdict(stats.load_rules(second), files, evidence=EVIDENCE)  # no lock: only its own sha
    with pytest.raises(TokstudyError, match="not the latest prereg"):
        stats.verdict(stats.load_rules(second), files, runs_root=runs, evidence=EVIDENCE)
    registry.lock_prereg(runs, load_prereg(second), amend="throughput ratio raised")
    doc = stats.verdict(
        stats.load_rules(second), files, runs_root=runs, evidence=EVIDENCE, twin=passed_twin(tmp_path, "tlc-N")
    )
    assert doc["prereg"]["sha256"] == load_prereg(second).sha256 and doc["arms"]["N1"]["decision"] == "replace"


def bilingual(tmp_path, prereg, effects, point, *, seeds=None):
    from surogate_tokstudy.prereg import load_prereg

    sha = load_prereg(prereg).sha256
    documents = docs(("zh", "en"))
    kinds = {"B0": "subword", "B1": "subword", "B2": "subword", "B3": "subword", "B4": "hnet", "B5": "bytes"}
    files = []
    for arm, bpb in effects.items():
        for seed in range((seeds or {}).get(arm, 3 if arm == "B0" else 2 if arm != "B5" else 1)):
            for p in (point / 2, point):
                path = tmp_path / f"{arm}-s{seed}-E{p:g}.nll"
                shift = 0.05 * (1 - p)
                files.append(
                    write_run(
                        path,
                        documents,
                        table="bilingual-B",
                        arm=arm,
                        kind=kinds[arm],
                        seed=seed,
                        point=p,
                        bpb={d: v + shift for d, v in bpb.items()},
                        prereg_sha256=sha,
                        run_prereg_sha256=sha,
                        train_bytes=round(1e9 * p),
                    )
                )
    return files


def test_bilingual_screening_eliminates_and_shortlists(tmp_path):
    prereg = signed(tmp_path, "bilingual")
    effects = {
        "B0": {"zh": 1.0, "en": 1.0},
        "B1": {"zh": 1.05, "en": 1.0},
        "B2": {"zh": 0.99, "en": 0.995},
        "B3": {"zh": 1.0, "en": 1.005},
        "B4": {"zh": 1.01, "en": 1.0},
        "B5": {"zh": 1.4, "en": 1.3},
    }
    files = bilingual(tmp_path, prereg, effects, 0.5)
    doc = stats.verdict(stats.load_rules(prereg), files, stage="screen", twin=passed_twin(tmp_path, "bilingual-B"))
    arms = doc["arms"]
    assert arms["B1"]["decision"] == "eliminated" and arms["B5"]["decision"] == "floor-only"
    assert doc["decision"]["shortlist"] == ["B4", "B2"]  # the H-Net first, then the best survivor
    assert arms["B2"]["conditions"]["decision_seeds"] == 3 and arms["B4"]["conditions"]["decision_seeds"] == 3
    assert "not shortlisted" in " ".join(arms["B3"]["reasons"])


def probe_file(path, arm, seed, values, *, table="bilingual-B", scorable=None):
    from surogate_tokstudy.probes import ProbeRow, write_probe

    rows = [
        ProbeRow(f"item{i}", "zh", scorable is None or i not in scorable, None, (1.0, 2.0), (3, 3), "sum", 0, True, v)
        for i, v in enumerate(values)
    ]
    write_probe(path, {"table": table, "arm": arm, "seed": seed, "point": 1.0, "items_sha256": "i" * 64}, rows)
    return path


def test_the_bilingual_decision_needs_track_s_and_the_best_gain_wins(tmp_path):
    prereg = signed(tmp_path, "bilingual")
    effects = {"B0": {"zh": 1.0, "en": 1.0}, "B2": {"zh": 0.97, "en": 1.0}, "B3": {"zh": 0.975, "en": 1.0}}
    files = bilingual(tmp_path, prereg, effects, 1.0, seeds={"B0": 3, "B2": 3, "B3": 3})
    rules = stats.load_rules(prereg)
    twin = passed_twin(tmp_path, "bilingual-B")
    pending = stats.verdict(rules, files, stage="decision", twin=twin, evidence={"shortlist": ["B2", "B3"]})
    assert pending["arms"]["B2"]["decision"] == "keep" and "Track S" in " ".join(pending["arms"]["B2"]["pending"])
    rng = np.random.default_rng(0)
    base = rng.normal(-1.0, 0.1, 50)
    probes = [probe_file(tmp_path / f"B0-{s}.probe", "B0", s, base + rng.normal(0, 0.01, 50)) for s in range(3)]
    probes += [probe_file(tmp_path / f"B2-{s}.probe", "B2", s, base + rng.normal(0, 0.01, 50)) for s in range(3)]
    probes += [probe_file(tmp_path / f"B3-{s}.probe", "B3", s, base - 0.5 + rng.normal(0, 0.01, 50)) for s in range(3)]
    doc = stats.verdict(rules, files, stage="decision", twin=twin, probes=probes, evidence={"shortlist": ["B2", "B3"]})
    assert doc["arms"]["B2"]["decision"] == "replace" and doc["decision"]["winner"] == "B2"
    assert doc["arms"]["B3"]["decision"] == "keep" and "failed: ['track_s']" in " ".join(doc["arms"]["B3"]["reasons"])
    assert doc["arms"]["B2"]["conditions"]["track_s"]["items"] == 50
    # an item unscorable for one arm leaves every arm's comparison
    probes[-1] = probe_file(tmp_path / "B3-2.probe", "B3", 2, base - 0.5, scorable={0, 1})
    assert stats.track_s(rules, probes, "B0")["B2"]["items"] == 48


def test_the_native_table_rules(tmp_path):
    prereg = signed(tmp_path, "tlc-T")
    rules = stats.load_rules(prereg)
    documents = docs(("py", "en"))
    effects = {"T0": (0.97, 1.0), "T1": (1.0, 1.0), "T2": (1.01, 1.0), "T3": (0.96, 1.0), "F3": (0.975, 1.005)}
    seeds = {"T0": 3, "T1": 2, "T2": 2, "T3": 2, "F3": 2}
    files = [
        write_run(
            tmp_path / f"{arm}-{s}.nll",
            documents,
            table="tlc-T",
            arm=arm,
            kind="subword",
            seed=s,
            bpb={"py": py, "en": en},
            backend="native",
        )
        for arm, (py, en) in effects.items()
        for s in range(seeds[arm])
    ]
    evidence = {
        "bpm_cross_boundary": {"T0": 0.037, "T3": 0.06},
        "probe_improved": {"T2": True},
        "evbind_improved": {"F3": True},
        "lr_resweep_done": ["T0", "T3"],
    }
    doc = stats.verdict(rules, files, evidence=evidence)
    arms = doc["arms"]
    assert doc["label"] == "decision" and doc["views"]["equal_flops"].startswith("unavailable")
    assert doc["decision"]["freqtrim_arm"] == "T2"  # (b) the fine regex, 1 % worse: within 2 %, probes improved
    assert arms["T0"]["decision"] == "replace"  # (a) 3 % better than T2 in Python, BPM 3.7 %
    assert arms["T3"]["decision"] == "keep" and "control arm" in " ".join(arms["T3"]["reasons"])
    assert arms["F3"]["decision"] == "replace" and doc["decision"] == {
        "freqtrim_arm": "T2",
        "tokenizer": "F3",
        "vocab": "8k",
        "digits": "per digit (d), unchanged",
    }
    worse = dict(evidence, probe_improved={"T2": False})
    assert stats.verdict(rules, files, evidence=worse)["decision"]["freqtrim_arm"] == "T1"


def test_the_twin_gate_and_the_command_line(tmp_path, capsys):
    twin_prereg = signed(tmp_path, "tlc-twin")
    documents = docs(("py", "en"))
    ours = write_run(
        tmp_path / "n0.nll", documents, table="tlc-N", arm="N0", kind="subword", seed=0, bpb={"py": 1.0, "en": 1.2}
    )
    near = write_run(
        tmp_path / "a0.nll",
        documents,
        table="tlc-T",
        arm="T1",
        kind="subword",
        seed=1,
        bpb={"py": 1.005, "en": 1.2},
        backend="native",
    )
    far = write_run(
        tmp_path / "far.nll",
        documents,
        table="tlc-T",
        arm="T1",
        kind="subword",
        seed=1,
        bpb={"py": 1.05, "en": 1.2},
        backend="native",
    )
    out = tmp_path / "twin.json"
    argv = ["twin", "--prereg", str(twin_prereg), "--tokstudy", str(ours), "--production", str(near), "--out", str(out)]
    assert stats.verdict_main(argv) == 0 and json.loads(out.read_text())["passed"]
    assert stats.verdict_main([*argv[:-4], "--production", str(far), "--out", str(out)]) == 0
    assert not json.loads(out.read_text())["passed"]
    capsys.readouterr()
    prereg = signed(tmp_path, "tlc")
    files = tlc_n(tmp_path, prereg, {"N0": flat(1.0, 1.0), "N1": flat(0.96, 1.0)})
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(EVIDENCE), encoding="utf-8")
    verdict = tmp_path / "verdict.json"
    argv = ["--prereg", str(prereg), "--nll", *map(str, files), "--evidence", str(evidence), "--twin", str(out)]
    assert stats.verdict_main([*argv, "--out", str(verdict)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {"label": "internal-only", "arms": {"N0": "keep", "N1": "replace"}}  # the far twin failed
    assert json.loads(verdict.read_text())["format"] == "tokstudy-verdict/1"
    unsigned = tmp_path / "u.toml"
    unsigned.write_text((PREREG_DIR / "tlc.toml").read_text(encoding="utf-8"), encoding="utf-8")
    assert stats.verdict_main(["--prereg", str(unsigned), "--nll", *map(str, files), "--out", str(verdict)]) == 1
    assert "not signed" in capsys.readouterr().err
