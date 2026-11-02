"""I6-E6 (budget and resource reports), I6-E8 (the MS §7 matrix), the bench and memory accounting on the CPU, and the
`stats` command. Nothing here is a measurement of the RTX 5090: on the CPU every measured GPU field is null / TBD."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from surogate_bilingual.audit import bench, memory
from surogate_bilingual.config import profile
from surogate_bilingual.eval import stats
from surogate_bilingual.eval.report import MEASURED_FIELDS, resource_report, results_markdown
from surogate_bilingual.model.lm import RoutedLM
from surogate_bilingual.registry import matrix

GOLDEN = Path(__file__).parent / "golden" / "budget_pilot.json"


def test_i6_e6_the_resource_report_carries_the_golden_budget_and_tbd_gpu_fields(pilot):
    doc, _ = pilot  # the pilot budget, regenerated from the module trees in this session
    golden = json.loads(GOLDEN.read_text("utf-8"))
    res = json.loads(json.dumps(resource_report("pilot", cuda=False, budget=doc)))
    assert res["format"] == "bilingual-resource/1" and res["budget"] == golden
    assert all(res["measured"][k] is None for k in MEASURED_FIELDS) and res["measured"]["status"] == "TBD"
    point = {"tokens_per_s": 61234.0, "max_memory_allocated_gib": 21.3, "source": "g05.jsonl"}
    assert resource_report("pilot", point, cuda=False, budget=doc)["measured"]["status"] == "TBD"  # no CUDA here
    on_gpu = resource_report("pilot", point, cuda=True, budget=doc)["measured"]
    assert on_gpu["status"] == "measured" and on_gpu["tokens_per_s"] == 61234.0
    with pytest.raises(ValueError):
        resource_report("smoke", cuda=False, budget=doc)


def test_i6_e6_audit_writes_the_resource_report(tmp_path, capsys):
    from surogate_bilingual.audit.budget import main

    out = tmp_path / "resources.json"
    assert main(["--profile", "smoke", "--resources", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["budget"]["profile"] == "smoke" and doc["measured"]["status"] == "TBD"
    assert main(["--profile", "smoke", "--resources", "--out", str(tmp_path / "dry.json"), "--dry-run"]) == 0
    assert not (tmp_path / "dry.json").exists()


# MS §7: every ID with the status of its table ("主" minimal-primary, "次" minimal-secondary, P1, "延后" deferred, G0).
MS7 = {
    "B0": "minimal-secondary", "B1": "minimal-secondary", "B2": "minimal-secondary", "B3a": "minimal-secondary",
    "B3b": "minimal-secondary", "B3c": "minimal-secondary", "B3d": "P1", "B3e": "P1", "B3-rb": "P1",
    "B4": "minimal-secondary", "B4-src": "minimal-secondary", "B4-wrong": "minimal-secondary",
    "B4-rand": "minimal-secondary", "RtS": "minimal-secondary", "B5": "deferred", "B6": "minimal-secondary",
    "B7": "G0", "B10-I": "P1", "B10-ch": "P1", "B11a": "P1", "B11b": "P1", "B12a": "minimal-secondary",
    "B12b": "P1", "LP-4": "minimal-secondary", "FX": "P1", "ST": "P1",
    "B0'": "minimal-secondary", "B1'": "minimal-secondary", "B4-II": "minimal-primary", "B9-II": "minimal-primary",
    "B8": "minimal-primary", "B3-II": "minimal-secondary", "B9-II+aff": "minimal-secondary", "B8-lin": "P1",
    "B10-II": "P1",
    "T0": "minimal-primary", "T1'": "minimal-primary", "T2": "minimal-primary", "T3'": "minimal-secondary",
    "T3''": "minimal-secondary", "T4": "deferred", "T5": "deferred",
}  # fmt: skip


def test_i6_e8_the_matrix_has_every_ms7_row_with_its_status_and_unrun_results(capsys):
    assert {r.id: r.status for r in matrix.rows()} == MS7
    entries = matrix.table()
    assert all(e["result"] == "未运行" for e in entries)
    assert matrix.main([]) == 0
    printed = capsys.readouterr().out
    assert all(f"| {rid} |" in printed for rid in MS7) and printed.count("未运行") == len(MS7)
    # matrix rows that train name a stage of examples/pilot_rows.toml
    from surogate_bilingual.config import stage_names

    stages = set(stage_names(Path(__file__).parents[2] / "examples" / "pilot_rows.toml"))
    for row in matrix.rows():
        for stage in filter(None, (s.strip() for s in (row.stage or "").split(","))):
            assert f"{stage}-r0" in stages, (row.id, stage)


def test_i6_e8_the_matrix_shows_recorded_results(tmp_path, capsys):
    doc = {
        "format": "bilingual-result/1",
        "results": [
            {
                "row": "B4",
                "arm": "B4",
                "metric": "E3ctx/cross/acc",
                "replicate": 0,
                "value": 0.25,
                "n_items": 10,
                "ci": None,
                "status": "done",
            }
        ],
    }
    (tmp_path / "results.json").write_text(json.dumps(doc))
    entries = {e["id"]: e["result"] for e in matrix.table(matrix.read_results([tmp_path]))}
    assert entries["B4"] == "E3ctx/cross/acc r0 = 0.2500" and entries["B6"] == "未运行"
    assert "0.2500" in results_markdown(doc["results"])


def test_memory_accounting_and_the_cpu_bench():
    cfg = profile("tiny", "swiglu")
    model = RoutedLM(cfg, ["seed"])
    tokens = torch.randint(5, cfg.vocab_size, (2, cfg.max_seq), generator=torch.Generator().manual_seed(0))
    plain = memory.saved_tensor_bytes(model, tokens)
    ckpt = memory.saved_tensor_bytes(model, tokens, checkpointing=True)
    assert 0 < ckpt < plain  # checkpointing keeps fewer activations (IMPL-09: it can be turned off and on)
    with memory.CudaPeak("cpu") as peak:
        pass
    assert peak.allocated_gib is None and peak.reserved_gib is None
    point = {"family": "slot", "recipe": "expert", "seq": 8, "microbatch": 2, "checkpointing": True}
    rec = bench.bench_point(profile("tiny", "slot"), point, warmup=1, steps=2, device=torch.device("cpu"))
    assert rec["status"] == "ok" and rec["tokens_per_s"] > 0 and rec["mfu"] is None
    assert rec["max_memory_allocated_gib"] is None and rec["nvidia_smi_used_gib"] is None
    assert rec["trainable_parameters"] == sum(
        p.numel() for p in RoutedLM(profile("tiny", "slot"), ["zh"]).bank.parameters()
    )
    assert bench.sdpa_probe(cfg, torch.device("cpu")) == dict.fromkeys(bench.BACKENDS, "not probed (CUDA only)")
    choice = bench.choose(
        [
            {**rec, "seq": 16, "microbatch": 2, "checkpointing": False, "tokens_per_s": 5.0},
            {**rec, "seq": 16, "microbatch": 4, "checkpointing": True, "tokens_per_s": 9.0},
            {**rec, "seq": 16, "microbatch": 8, "status": "oom", "tokens_per_s": None},
        ],
        64,
        16,
    )
    assert choice[0]["microbatch"] == 4 and choice[0]["accum"] == 1 and choice[0]["checkpointing"] is True


def test_bench_command_refuses_without_cuda_and_runs_on_an_explicit_cpu(tmp_path, capsys):
    out = tmp_path / "g05.jsonl"
    grid = [
        "--family",
        "swiglu",
        "--recipe",
        "seed",
        "--seq",
        "8",
        "--microbatch",
        "1,2",
        "--checkpointing",
        "off",
        "--warmup",
        "1",
        "--steps",
        "1",
        "--profile",
        "tiny",
        "--out",
        str(out),
    ]
    assert bench.main([*grid, "--dry-run"]) == 0 and not out.exists()
    if not torch.cuda.is_available():
        assert bench.main(grid) == 1 and "no CUDA" in capsys.readouterr().err
    assert bench.main([*grid, "--device", "cpu", "--choice-seq", "8", "--global-batch-tokens", "16"]) == 0
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["kind"] for r in lines] == ["header", "point", "point", "choice"]
    assert lines[-1]["microbatch"] == max((r for r in lines[1:3]), key=lambda r: r["tokens_per_s"])["microbatch"]
    assert bench.main([*grid, "--seq", "64"]) == 1  # beyond the tiny max_seq


def _preds(arm, rep, correct, endpoint=("E3ctx", "cross"), value="correct"):
    fam, var = endpoint
    return [
        {
            "format": "bilingual-pred/1",
            "arm": arm,
            "replicate": rep,
            "family": fam,
            "variant": var,
            "direction": "zh>en>zh",
            "item_id": f"i{k}",
            "skip_reason": None,
            value: bool(c),
            "correct": bool(c),
        }
        for k, c in enumerate(correct)
    ]


def _write(path, preds):
    path.write_text("".join(json.dumps(p) + "\n" for p in preds))
    return path


def test_stats_command_pilot_futility_power_and_confirm(tmp_path, capsys):
    rng = np.random.default_rng(0)
    preds = []
    for r in range(3):
        for arm, p in (("B0", 0.30), ("B2", 0.40), ("B12a", 0.45), ("B4", 0.35), ("B6", 0.37)):
            preds += _preds(arm, r, list(rng.random(200) < p))
    path = _write(tmp_path / "dev.jsonl", preds)
    out = tmp_path / "pilot.json"
    assert stats.main(["pilot", "--pred", str(path), "--out", str(out)]) == 0
    pilot = json.loads(out.read_text())
    assert pilot["gate"]["best"] == "B12a" and pilot["gate"]["passed"]
    assert pilot["nuisance"]["B12a:B4"]["between_replicate_sd"] > 0
    fut = tmp_path / "futility_r0.json"
    assert stats.main(["futility", "--pred", str(path), "--out", str(fut)]) == 0
    assert json.loads(fut.read_text())["decision"] in ("continue", "futility stop: branch 1 not tested")
    assert stats.main(["power", "--sd", "0.03", "--sesoi", "0.05", "--dry-run"]) == 0
    # confirm: B9-II clearly over B4-II, B8 equal to B9-II, no Bridge predictions (H_C not testable)
    test_preds = []
    for r in range(4):
        base = rng.random(400) < 0.4
        better = base | (rng.random(400) < 0.2)
        test_preds += _preds("B4-II", r, list(base)) + _preds("B9-II", r, list(better)) + _preds("B8", r, list(better))
    tp = _write(tmp_path / "test.jsonl", test_preds)
    conf = tmp_path / "confirm.json"
    assert stats.main(["confirm", "--pred", str(tp), "--dev-pred", str(tp), "--out", str(conf)]) == 0
    doc = json.loads(conf.read_text())["hypotheses"]
    assert doc["H_A"]["gatekeeper"]["rejected"] and doc["H_A"]["decision"]["decision"] == "superior"
    assert doc["H_B"]["precondition"]["met"] is False  # identical predictions: no discordance
    assert doc["H_B"]["decision"]["decision"] == "inconclusive" and "untestable" in doc["H_B"]["status"]
    assert "not testable" in doc["H_C"]["status"]
    capsys.readouterr()
    assert stats.main(["futility", "--pred", str(tp)]) == 1  # no B12a / B6 predictions
