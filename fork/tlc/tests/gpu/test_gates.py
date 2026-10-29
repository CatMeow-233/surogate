"""Gates G0 (kernels in place), G1 and G2 on the native trainer, with the model directories and shards the runbook
prepares under $TLC_RUNS (fork/tlc/README.md, 运行手册). Select a gate with -k g0, -k g1a, ..., -k g2.

Thresholds marked [P] in the README are provisional: when one fails, record the measured value and stop at that
gate; do not loosen it to pass.
"""

import dataclasses
import gc
import hashlib
import json
import math
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from surogate_tlc.train import config, driver

TLC = Path(__file__).resolve().parents[2]
ROWS = ("A0", "A1", "A1h")
PREFIXES = (1, 2, 7, 64, 511)
IGNORE = -100


def _cfg(runs: Path, row: str) -> dict:
    return json.loads((runs / "models" / row / "config.json").read_text(encoding="utf-8"))


@contextmanager
def native(runs: Path, row: str, tmp_path: Path, *, batch: int = 2, seq_len: int = 512):
    wrapper = driver.native_trainer(
        runs / "models" / row, runs / "data", tmp_path / f"native-{row}", batch=batch, seq_len=seq_len
    )
    try:
        yield wrapper.trainer
    finally:
        del wrapper
        gc.collect()


def logprobs(trainer, ids: np.ndarray, targets: np.ndarray, positions: np.ndarray | None = None) -> np.ndarray:
    ids, targets = (np.ascontiguousarray(a, dtype=np.int32) for a in (ids, targets))
    positions = None if positions is None else np.ascontiguousarray(positions, dtype=np.int32)
    return np.asarray(trainer.compute_logprobs(ids, targets, use_lora=False, position_ids=positions)).copy()


def gradients(trainer, ids: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    """Parameter gradients of one forward and backward on a fresh trainer (gradients accumulate across steps)."""
    import torch

    positions = np.ascontiguousarray(np.broadcast_to(np.arange(ids.shape[1], dtype=np.int32), ids.shape))
    trainer.step(np.ascontiguousarray(ids), np.ascontiguousarray(targets), positions)
    return {name: torch.from_dlpack(view).float().cpu().numpy() for name, view in trainer.get_gradients(0).items()}


def random_batch(vocab: int, batch: int, seq_len: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.random.default_rng(seed).integers(0, vocab, size=(batch, seq_len), dtype=np.int32)
    return ids, np.concatenate([ids[:, 1:], np.full((batch, 1), IGNORE, np.int32)], 1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize("row", ROWS)
def test_g0_one_step_is_finite(runs, row, tmp_path):
    """BF16 GEMMs, the selective scan forward and backward (A0, A1h) and flash attention with head_dim 64 (A1,
    A1h) inside the TLC graphs; the graph builds, which also covers TLCMambaFFNBlock without a BlockSchema."""
    from surogate_tlc.ir import compile_ir, param_shapes

    cfg = _cfg(runs, row)
    expected = set(param_shapes(compile_ir(cfg))) - ({"lm_head"} if cfg["tie_word_embeddings"] else set())
    with native(runs, row, tmp_path) as trainer:
        grads = gradients(trainer, *random_batch(cfg["vocab_size"], 2, 512, seed=0))
    assert expected <= set(grads), sorted(expected - set(grads))
    assert not [name for name, g in grads.items() if not np.isfinite(g).all()]
    assert not [name for name in expected if not np.any(grads[name])]


@pytest.mark.parametrize("row", ROWS)
def test_g1a_future_tokens_do_not_change_the_past(runs, row, tmp_path):
    """Changing every token from position p on leaves the log-probabilities before p bit for bit."""
    vocab = _cfg(runs, row)["vocab_size"]
    ids, targets = random_batch(vocab, 1, 512, seed=1)
    with native(runs, row, tmp_path) as trainer:
        for prefix in PREFIXES:
            changed = ids.copy()
            changed[0, prefix:] = (changed[0, prefix:] + 1) % vocab
            out = logprobs(trainer, np.concatenate([ids, changed]), np.concatenate([targets, targets]))
            assert np.array_equal(out[0, :prefix], out[1, :prefix]), prefix
            if prefix < 511:  # and the change is seen after p
                assert not np.array_equal(out[0, prefix:511], out[1, prefix:511]), prefix


def test_g1b_matches_the_reference_goldens(runs, tmp_path):
    golden = np.load(runs / "goldens" / "A0.npz")
    weights = (runs / "models" / "A0" / "model.safetensors").read_bytes()
    assert json.loads(str(golden["meta"]))["model_safetensors_sha256"] == hashlib.sha256(weights).hexdigest()
    ids, targets = golden["input_ids"], golden["targets"]
    with native(runs, "A0", tmp_path, batch=ids.shape[0], seq_len=ids.shape[1]) as trainer:
        out = logprobs(trainer, ids, targets)
    error, tolerance = float(np.abs(out - golden["logprobs"]).max()), float(golden["tolerance"])
    print(json.dumps({"g1b_max_abs_error": error, "tolerance": tolerance}))
    assert error <= tolerance


@pytest.mark.parametrize("row, isolated", [("A1", True), ("A0", False)])
def test_g1c_packing(runs, row, isolated, tmp_path):
    """Two documents packed in one row (positions restart at the second) against each scored alone: attention
    isolates them (A1); mamba_ssm_scan does not (A0, the reason every Mamba row keeps one document per row)."""
    vocab = _cfg(runs, row)["vocab_size"]
    rng = np.random.default_rng(4)
    docs = [rng.integers(0, vocab, size=n, dtype=np.int32) for n in (200, 300)]

    def row_of(pieces):
        ids = np.zeros(512, np.int32)
        targets = np.full(512, IGNORE, np.int32)
        positions = np.zeros(512, np.int32)
        start = 0
        for doc in pieces:
            end = start + len(doc)
            ids[start:end], targets[start : end - 1] = doc, doc[1:]
            positions[start:end] = np.arange(len(doc))
            start = end
        positions[start:] = np.arange(512 - start)
        return ids, targets, positions

    packed, alone = row_of(docs), [row_of([doc]) for doc in docs]
    with native(runs, row, tmp_path) as trainer:
        both = logprobs(trainer, *(np.stack([a, a]) for a in packed))[0]
        single = logprobs(trainer, *(np.stack(parts) for parts in zip(*alone, strict=True)))
    errors = [float(np.abs(both[:199] - single[0, :199]).max()), float(np.abs(both[200:499] - single[1, :299]).max())]
    print(json.dumps({"row": row, "g1c_max_abs_error": errors}))
    if isolated:
        assert max(errors) <= 2e-2  # [P]
    else:
        assert errors[0] <= 2e-2 and errors[1] > 1e-3  # the first document is unaffected, the second leaks


def test_g1d_split_first_layer_is_equivalent(runs, tmp_path):
    """A0-split (layer 1 as model-level ops) computes A0's logits and gradients from the same weights."""
    from surogate_tlc.dsl.models import split_param_name

    seeds = {
        row: json.loads((runs / "models" / row / "tlc_meta.json").read_text())["init_seed"]
        for row in ("A0", "A0-split")
    }
    assert len(set(seeds.values())) == 1, seeds
    golden = np.load(runs / "goldens" / "A0.npz")
    ids, targets = golden["input_ids"], golden["targets"]
    results = {}
    for row in ("A0", "A0-split"):
        with native(runs, row, tmp_path, batch=ids.shape[0], seq_len=ids.shape[1]) as trainer:
            results[row] = (logprobs(trainer, ids, targets), gradients(trainer, ids, targets))
    (lp, grads), (split_lp, split_grads) = results["A0"], results["A0-split"]
    assert float(np.abs(lp - split_lp).max()) <= float(golden["tolerance"])
    renamed = {split_param_name(name): g for name, g in grads.items()}
    assert set(renamed) == set(split_grads)
    worst = min((cosine(renamed[name], split_grads[name]), name) for name in renamed)
    print(json.dumps({"g1d_min_gradient_cosine": worst}))
    assert worst[0] >= 0.999  # [P]


@pytest.mark.parametrize("row", ROWS)
def test_g2_initial_loss_is_near_ln_vocab(runs, row, tmp_path):
    """Random tokens on the initial weights: CE within 0.25 of ln V (README 初始化: 9.78 predicted for A0)."""
    vocab = _cfg(runs, row)["vocab_size"]
    ids, targets = random_batch(vocab, 2, 512, seed=2)
    with native(runs, row, tmp_path) as trainer:
        loss = -float(logprobs(trainer, ids, targets)[targets != IGNORE].mean())
    print(json.dumps({"row": row, "g2_initial_loss": loss, "ln_vocab": math.log(vocab)}))
    assert abs(loss - math.log(vocab)) < 0.25


def test_g2_gradients_point_where_the_reference_does(runs, tmp_path):
    """A0, first and last layer: native bf16 gradients against fp32 autograd through prototypes/ref_mamba2.py."""
    import torch

    sys.path.insert(0, str(TLC / "prototypes"))
    import ref_mamba2

    cfg = _cfg(runs, "A0")
    ids, targets = random_batch(cfg["vocab_size"], 2, 128, seed=3)
    with native(runs, "A0", tmp_path, batch=2, seq_len=128) as trainer:
        native_grads = gradients(trainer, ids, targets)
    reference = ref_mamba2.Reference(runs / "models" / "A0")
    for weight in reference.w.values():
        weight.requires_grad_(True)
    lp = reference.logprobs(torch.from_numpy(ids).long(), torch.from_numpy(targets).long(), mode="chunked")
    (-lp[torch.from_numpy(targets != IGNORE)].mean()).backward()
    found = {}
    for layer in (0, cfg["num_hidden_layers"] - 1):
        for field in ref_mamba2.LAYER_FIELDS:
            key = ref_mamba2.layer_key(cfg, layer, field)
            found[key] = cosine(native_grads[key], reference.w[key].grad.numpy())
    print(json.dumps({"g2_gradient_cosine": found}))
    assert min(found.values()) >= 0.99  # [P]


def test_g2_resume_reproduces_the_next_steps(runs, tmp_path, monkeypatch):
    """A0 smoke stage: a run resumed from its step-5 checkpoint repeats steps 6..10 of the uninterrupted run."""
    monkeypatch.setenv("TLC_RUNS", str(runs))
    run = config.load(TLC / "examples" / "a0.yaml", stage="smoke")

    def at(out: Path) -> config.Run:
        return dataclasses.replace(run, sft={**run.sft, "output_dir": str(out)})

    full = driver.train(at(tmp_path / "full"))
    (tmp_path / "resumed").mkdir()
    shutil.copytree(tmp_path / "full" / "step_00000005", tmp_path / "resumed" / "step_00000005")
    resumed = driver.train(at(tmp_path / "resumed"))
    assert resumed["attempts"][-1]["resumed_from_step"] == 5
    before = {step: loss for step, loss in full["attempts"][-1]["summary"]["loss"]}
    after = {step: loss for step, loss in resumed["attempts"][-1]["summary"]["loss"]}
    print(json.dumps({"g2_resume": {step: [before[step], after[step]] for step in after}}))
    assert sorted(after) == [6, 7, 8, 9, 10]
    assert max(abs(before[step] - after[step]) for step in after) <= 1e-3  # v0.7 G2.6 tolerance
