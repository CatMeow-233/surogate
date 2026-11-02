"""Runbook G0 step 3, (g1)–(g6) and the scipy cross-check. Skipped without CUDA; nothing here has run on a GPU yet.

Results are written to the JSON file named by BILINGUAL_G0_RECORD (when set) and go into docs/measurements.md with
host, commit and date. Every tolerance is the one MS §9.4 declares; the bf16 ones are [建议] until locked at G0.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from surogate_bilingual.assembly.assemble import Tolerances, assemble, verify_fidelity
from surogate_bilingual.assembly.packs import probe_batch, probe_positions
from surogate_bilingual.audit.bench import BACKENDS, PEAK_TFLOPS, sdpa_probe
from surogate_bilingual.config import FUSIONS, profile
from surogate_bilingual.model.lm import RoutedLM
from surogate_bilingual.routing.plan import RoutePlan
from surogate_bilingual.routing.policies import apply_gate_policy
from surogate_bilingual.train.loop import train

FUSED = ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION")
BF16 = Tolerances.bf16_cuda()  # max-abs 0.05, argmax agreement >= 99.5 % [建议], locked before G0 runs


def record(key: str, value) -> None:
    path = os.environ.get("BILINGUAL_G0_RECORD")
    if not path:
        return
    from surogate_bilingual.canonical import atomic_write_json

    p = Path(path)
    doc = json.loads(p.read_text("utf-8")) if p.is_file() else {"format": "bilingual-g0/1"}
    doc[key] = value
    atomic_write_json(p, doc)


def cuda_only(device: str) -> None:
    if device != "cuda":
        pytest.skip("CUDA only (BILINGUAL_GPU_TESTS_ON_CPU rehearses the rest of the code)")


@pytest.fixture(scope="module")
def trained(tmp_path_factory, tiny_run, cpu_helpers):
    """tiny-seed, tiny-zh, tiny-en trained on the CPU and assembled: what (g1) replays on the device."""
    root = cpu_helpers.build_tiny_root(tmp_path_factory.mktemp("g1_root"))
    packs = {}
    for stage, name in (("tiny-seed", "seed"), ("tiny-zh", "zh"), ("tiny-en", "en")):
        packs[name] = train(tiny_run(stage), root=root, device="cpu").packs[name]
    manifest = assemble(packs["seed"], {"zh": packs["zh"], "en": packs["en"]}, None, root / "zh_en.json")
    return root, manifest


def test_g1_forced_route_fidelity_bf16_autocast_and_fp32(device, trained):
    _, manifest = trained
    report = verify_fidelity(manifest, BF16, device=device, autocast=torch.bfloat16)
    record("g1_bf16_tiny", report.to_dict())
    assert report.passed, report.to_dict()
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        fp32 = verify_fidelity(manifest, Tolerances(max_abs=1e-5), device=device)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    record("g1_fp32_tiny", fp32.to_dict())
    assert fp32.passed, fp32.to_dict()


@pytest.mark.parametrize("family", ["swiglu", "slot"])
def test_g1_pilot_geometry_bf16_against_cpu_fp32(device, family):
    """The pilot geometry with random weights (no trained pilot pack exists before G1): the bf16 autocast forced
    route on the device against the fp32 CPU forward, on the probe batch. Only max-abs is asserted: an untrained
    model's logits are nearly tied, so its argmax agreement is recorded but says nothing (the CPU rehearsal with
    bf16 autocast gave max-abs 0.022 with agreement 0.875); the argmax criterion applies to trained packs (above,
    and `verify` at G3)."""
    cfg = profile("pilot", family)
    model = RoutedLM(cfg, ["zh", "en"], init_seed=3).eval()
    tokens = probe_batch(cfg)
    pos = probe_positions(cfg.max_seq)
    plan = RoutePlan.forced("en", model.expert_names, *tokens.shape)
    with torch.no_grad():
        ref = model(tokens, plan)[:, pos].float()
        dev = model.to(device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            got = (
                dev(tokens.to(device), RoutePlan(plan.names, plan.gates.to(device), plan.policy))[:, pos].float().cpu()
            )
    max_abs = float((got - ref).abs().max())
    agree = float((got.argmax(-1) == ref.argmax(-1)).double().mean())
    record(f"g1_bf16_pilot_{family}", {"max_abs": max_abs, "argmax_agreement": agree})
    assert max_abs <= BF16.max_abs, (max_abs, agree)


def _plans(names, B, T, device):
    g = torch.Generator().manual_seed(4)
    ids = torch.randint(0, len(names), (B, T), generator=g)
    one_hot = RoutePlan.one_hot(ids, names, "random")
    dual = apply_gate_policy(one_hot, "dual_const")
    mixed = apply_gate_policy(one_hot, "dual_window", window=3)
    out = {"one_hot": one_hot, "dual": dual, "mixed": mixed}
    return {k: RoutePlan(p.names, p.gates.to(device), p.policy) for k, p in out.items()}


@pytest.mark.parametrize("profile_name", ["tiny", "smoke"])
@pytest.mark.parametrize("family", ["swiglu", "slot"])
def test_g2_every_dispatch_path_runs_under_cuda_autocast(device, profile_name, family):
    names = ("zh", "en")
    cfg = profile(profile_name, family)
    T = cfg.max_seq - 3
    x = torch.randint(5, cfg.vocab_size, (2, T), generator=torch.Generator().manual_seed(5)).to(device)
    for fusion in FUSIONS[family]:
        model = RoutedLM(cfg, names, fusion, init_seed=1).to(device).eval()
        with torch.no_grad(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
            for kind, plan in _plans(names, 2, T, device).items():
                out = model(x, plan)
                assert out.shape == (2, T, cfg.vocab_size) and bool(torch.isfinite(out).all()), (fusion, kind)
            for name in names:
                alone = RoutedLM(cfg, [name], fusion).to(device).eval()
                alone.load_backbone_state(model.backbone_state())
                alone.load_expert_state(name, model.expert_state(name))
                forced = model(x, RoutePlan(names, RoutePlan.forced(name, names, 2, T).gates.to(device), "forced"))
                assert torch.equal(forced, alone(x)), (fusion, name)  # NEW-01: bitwise, not a tolerance


def test_g3_sdpa_backends_with_enable_gqa_and_no_silent_fallback(device):
    cuda_only(device)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    cfg = profile("pilot", "swiglu")
    probe = sdpa_probe(cfg, torch.device("cuda"))
    record("g3_sdpa_backends", probe)
    assert set(probe) == set(BACKENDS)
    fused = [b for b in FUSED if probe[b] == "ok"]
    if not fused:
        pytest.fail(f"no fused SDPA backend accepts enable_gqa on this GPU ({probe}); set model.gqa = 'repeat' "
                    "(runbook G0 g3)")  # fmt: skip
    x = torch.randint(5, cfg.vocab_size, (2, 256), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for gqa in ("sdpa_gqa", "repeat"):
            model = RoutedLM(cfg, ["seed"], gqa=gqa, device="cuda").eval()
            with sdpa_kernel([getattr(SDPBackend, b) for b in fused]):  # math is not allowed: no silent fallback
                assert bool(torch.isfinite(model(x)).all()), gqa
            del model
    record("g3_default", {"gqa": "sdpa_gqa", "fused_backends": fused})


def _losses(result):
    return np.array(result.losses, dtype=np.float64)


def test_g4_two_runs_with_the_same_seeds_agree_within_1e3(device, tmp_path_factory, tiny_run, cpu_helpers):
    runs = []
    for k in range(2):
        root = cpu_helpers.build_tiny_root(tmp_path_factory.mktemp(f"g4_{k}"))
        runs.append(_losses(train(tiny_run("tiny-seed"), root=root, device=device)))
    assert runs[0].shape == (20,)
    rel = np.abs(runs[0] - runs[1]) / np.maximum(np.abs(runs[0]), 1e-12)
    record("g4_max_relative_loss_difference", float(rel.max()))
    assert float(rel.max()) <= 1e-3


def test_g5_resume_on_the_device_round_trips_within_1e3(device, tmp_path_factory, tiny_run, cpu_helpers):
    full_root = cpu_helpers.build_tiny_root(tmp_path_factory.mktemp("g5_full"))
    full = _losses(train(tiny_run("tiny-seed", "run.steps=10"), root=full_root, device=device))
    root = cpu_helpers.build_tiny_root(tmp_path_factory.mktemp("g5_split"))
    first = _losses(train(tiny_run("tiny-seed", "run.steps=10"), root=root, device=device, stop_after=5))
    second = train(tiny_run("tiny-seed", "run.steps=10"), root=root, device=device)
    assert second.resumed_from == 5 and second.status == "done"
    split = np.concatenate([first, _losses(second)])
    rel = np.abs(full - split) / np.maximum(np.abs(full), 1e-12)
    record("g5_max_relative_loss_difference", float(rel.max()))
    assert split.shape == (10,) and float(rel.max()) <= 1e-3


def test_g6_records_driver_torch_build_and_peak(device):
    from surogate_bilingual.registry.env import environment

    env = environment()
    rec = {"driver": env["driver"], "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": env["cudnn"],
           "gpu": env["gpu"], "peak_tflops": PEAK_TFLOPS,
           "peak_tflops_status": "to verify against the NVIDIA spec sheet (dense BF16, fp32 accumulate)"}  # fmt: skip
    record("g6", rec)
    if device == "cuda":
        missing = [k for k, v in rec.items() if v is None]
        assert not missing, f"G0 g6 fields not readable on this host: {missing}"


def test_scipy_and_numpy_hungarian_agree_here(device):
    """I3-C10's scipy part, which the CPU lane skips: in fork/.venv-bilingual scipy is required."""
    cuda_only(device)
    import scipy.optimize  # noqa: F401  (fork/bilingual/requirements-gpu.txt)

    from surogate_bilingual.ops.merge import solve_lap

    rng = np.random.default_rng(64)
    for _ in range(20):
        score = rng.standard_normal((64, 64))
        a, b = solve_lap(score, "numpy"), solve_lap(score, "scipy")
        assert abs(score[np.arange(64), a].sum() - score[np.arange(64), b].sum()) <= 1e-9
