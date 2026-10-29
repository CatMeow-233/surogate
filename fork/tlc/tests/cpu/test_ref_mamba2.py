"""The Mamba-2 reference fork/tlc/prototypes/ref_mamba2.py (docs/model.md: 初始化, 数值参照): its three evaluation orders
agree, it reads exactly the weights init writes, its assumptions match the compiled IR, an initialised A0 starts
near the uniform loss, and its CLI writes the G1(b) goldens.
"""

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from surogate_tlc import hf, init, ir, modeldir

REF_FILE = Path(__file__).resolve().parents[2] / "prototypes" / "ref_mamba2.py"
# Two layers with a short chunk, so 40 tokens cross two chunk boundaries and end in a partial chunk.
SMALL = {"num_hidden_layers": 2, "tlc_layer_pattern": "MM", "chunk_size": 16}


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """The recurrence is thousands of small ops; on a shared CPU intra-op threads make it 100x slower."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def ref():
    spec = importlib.util.spec_from_file_location("ref_mamba2", REF_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small_dirs(tmp_path_factory, tokenizer_dir) -> dict[str, Path]:
    """Two-layer A0 and A0-split model directories initialised from the same seed."""
    dirs = {}
    for row in ("A0", "A0-split"):
        cfg = hf.row_config(row, **SMALL)
        path = modeldir.write_model_dir(tmp_path_factory.mktemp(row) / "model", cfg, tokenizer_dir, init_seed=7)
        init.init_model_dir(path)
        dirs[row] = path
    return dirs


def _ids(batch: int, length: int, seed: int = 0) -> torch.Tensor:
    return torch.randint(
        0, hf.default_config()["vocab_size"], (batch, length), generator=torch.Generator().manual_seed(seed)
    )


@torch.no_grad()
def test_scan_chunked_and_step_orders_agree(ref, small_dirs):
    model = ref.Reference(small_dirs["A0"])
    ids = _ids(2, 40)
    scan = model.logits(ids)
    assert scan.abs().max() > 0.1
    torch.testing.assert_close(model.logits(ids, "chunked"), scan, rtol=0, atol=1e-4)
    torch.testing.assert_close(model.step_logits(ids), scan, rtol=0, atol=1e-4)
    # causal: later tokens do not change earlier logits
    changed = ids.clone()
    changed[:, 20:] = _ids(2, 20, seed=1)
    assert torch.equal(model.logits(changed)[:, :20], scan[:, :20])


@torch.no_grad()
def test_a0_split_reads_the_same_weights(ref, small_dirs):
    ids = _ids(2, 24)
    assert torch.equal(ref.Reference(small_dirs["A0-split"]).logits(ids), ref.Reference(small_dirs["A0"]).logits(ids))


def test_assumptions_match_the_compiled_ir(ref, small_dirs):
    cfg = json.loads((small_dirs["A0"] / "config.json").read_text())
    ops = [op for op in ir.compile_ir(cfg)["forward"]["operations"] if op["kernel_type"] != "view"]
    layer = [op for op in ops if any(name.startswith("blocks[0].") for name in op["outputs"])]
    assert tuple(op["kernel_type"] for op in layer) == ref.LAYER_OPS
    attrs = {op["kernel_type"]: op["attrs"] for op in layer}
    heads, inner = cfg["mamba_num_heads"], cfg["mamba_num_heads"] * cfg["mamba_head_dim"]
    assert attrs["mamba_split_proj"] == {
        "intermediate_size": inner,
        "conv_dim": inner + 2 * cfg["n_groups"] * cfg["ssm_state_size"],
        "num_heads": heads,
        "head_dim": cfg["mamba_head_dim"],
    }
    assert attrs["mamba_conv1d"] == {"activation": "silu"}
    scan = attrs["mamba_ssm_scan"]
    assert scan["dt_softplus"] and (scan["dt_min"], scan["dt_max"]) == ref.TIME_STEP_LIMIT
    assert attrs["mamba_gated_rmsnorm"] == {
        "eps": cfg["rms_norm_eps"],
        "n_groups": cfg["n_groups"],
        "norm_before_gate": False,
        "gate_activation": "silu",
    }
    assert all(op["attrs"]["transpose"] == "NT" for op in layer if op["kernel_type"] == "matmul")
    assert {op["attrs"]["eps"] for op in ops if op["kernel_type"] == "fused_residual_rmsnorm"} == {cfg["rms_norm_eps"]}
    assert [op["kernel_type"] for op in ops[-2:]] == ["fused_residual_rmsnorm", "fused_lm_head_loss"]
    # the reference reads exactly what init writes, in both namings, tied or not
    for row in ("A0", "A0-split"):
        for tied in (True, False):
            row_cfg = hf.row_config(row, **SMALL, tie_word_embeddings=tied)
            assert ref.weight_names(row_cfg) == set(init.init_state_dict(row_cfg, seed=0)), (row, tied)


@torch.no_grad()
def test_initialised_a0_starts_near_the_uniform_loss(ref, model_dir):
    """Gate G2 expects a step-0 loss of about ln 16384 = 9.70; tied N(0, 0.02) embeddings add about 0.08."""
    init.init_model_dir(model_dir)
    ids = _ids(1, 65)
    loss = -ref.Reference(model_dir).logprobs(ids[:, :-1], ids[:, 1:]).mean().item()
    assert abs(loss - math.log(hf.default_config()["vocab_size"])) < 0.25, loss


def test_cli_writes_the_goldens(ref, small_dirs, tmp_path):
    out = tmp_path / "golden.npz"
    command = [sys.executable, "-I", "-B", str(REF_FILE), str(small_dirs["A0"]), str(out), "--seq-len", "24"]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)

    golden = np.load(out)
    assert sorted(golden.files) == ["input_ids", "logprobs", "logprobs_bf16", "meta", "targets", "tolerance"]
    ids, targets = golden["input_ids"], golden["targets"]
    assert ids.dtype == targets.dtype == np.int32 and ids.shape == targets.shape == (2, 24)
    assert ids.min() >= 0 and ids.max() < hf.default_config()["vocab_size"]
    assert (targets[:, :-1] == ids[:, 1:]).all() and (targets[:, -1] == ref.IGNORE).all()
    fp32, bf16 = golden["logprobs"], golden["logprobs_bf16"]
    assert fp32.dtype == bf16.dtype == np.float32
    assert (fp32[:, -1] == 0).all() and (fp32[:, :-1] < 0).all()
    gap = np.abs(bf16 - fp32).max()
    assert 0 < gap < 0.1 and golden["tolerance"] == np.float32(2 * gap)
    meta = json.loads(str(golden["meta"]))
    weights = (small_dirs["A0"] / "model.safetensors").read_bytes()
    assert meta["model_safetensors_sha256"] == hashlib.sha256(weights).hexdigest()
    with torch.no_grad():
        expected = ref.Reference(small_dirs["A0"]).logprobs(
            torch.from_numpy(ids).long(), torch.from_numpy(targets).long()
        )
    np.testing.assert_array_equal(fp32, expected.numpy())
