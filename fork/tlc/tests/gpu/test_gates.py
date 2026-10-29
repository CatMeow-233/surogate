"""Gates G0 (kernels in place), G1, G2 and the G3 sampler on the native trainer, with the model directories and
shards the runbook prepares under $TLC_RUNS (fork/tlc/docs/runbook.md). Select a gate with -k g0, -k g1n1, ...

Thresholds marked [P] in the runbook are provisional: when one fails, record the measured value and stop at that
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
from safetensors.torch import load_file
from surogate_tlc.data.shards import IGNORE
from surogate_tlc.native import SAMPLING_LORA, native_trainer
from surogate_tlc.train import config, driver

TLC = Path(__file__).resolve().parents[2]
ROWS = ("A0", "A1", "A1h")
CHUNK = 128  # chunk_size of the P0 Mamba geometry
PREFIXES = (1, 2, 7, CHUNK - 1, CHUNK, CHUNK + 1, 511)


def _cfg(runs: Path, row: str) -> dict:
    return json.loads((runs / "models" / row / "config.json").read_text(encoding="utf-8"))


class _Trainer:
    """The native trainer of one `with native(...)` block. The block's end drops it, and with it the GPU memory (the
    native trainer frees it only when destroyed): the caller's `as` name, and a failing test's traceback, keep only
    this handle."""

    def __init__(self, trainer) -> None:
        self._trainer = trainer

    def __getattr__(self, name: str):
        if self._trainer is None:
            raise RuntimeError("native trainer used after its with-block")
        return getattr(self._trainer, name)


@contextmanager
def native(model: Path, runs: Path, tmp_path: Path, *, batch: int = 2, seq_len: int = 512, settings=None):
    out = tmp_path / f"native-{model.name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapper = native_trainer(model, runs / "data", out, batch=batch, seq_len=seq_len, settings=settings)
    handle = _Trainer(wrapper.trainer)
    try:
        yield handle
    finally:
        handle._trainer = None
        del wrapper
        gc.collect()


def logprobs(trainer, ids: np.ndarray, targets: np.ndarray, positions: np.ndarray | None = None) -> np.ndarray:
    ids, targets = (np.ascontiguousarray(a, dtype=np.int32) for a in (ids, targets))
    positions = None if positions is None else np.ascontiguousarray(positions, dtype=np.int32)
    return np.asarray(trainer.compute_logprobs(ids, targets, use_lora=False, position_ids=positions)).copy()


def _positions(ids: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.broadcast_to(np.arange(ids.shape[1], dtype=np.int32), ids.shape))


def gradients(trainer, ids: np.ndarray, targets: np.ndarray) -> dict[str, np.ndarray]:
    """Parameter gradients of one forward and backward on a fresh trainer (gradients accumulate across steps)."""
    import torch

    trainer.step(np.ascontiguousarray(ids), np.ascontiguousarray(targets), _positions(ids))
    return {name: torch.from_dlpack(view).float().cpu().numpy() for name, view in trainer.get_gradients(0).items()}


def adamw(learning_rate: float = 1e-3):
    """The native AdamW of the G2 training checks: no weight decay, clip 1, betas 0.9 / 0.95."""
    from surogate import _surogate

    return _surogate.OptimizerConfig(
        optimizer="adamw",
        learning_rate=learning_rate,
        weight_decay=0.0,
        grad_clip=1.0,
        adamw_beta1=0.9,
        adamw_beta2=0.95,
        adamw_epsilon=1e-8,
    )


def random_batch(vocab: int, batch: int, seq_len: int, seed: int, low: int = 0) -> tuple[np.ndarray, np.ndarray]:
    ids = np.random.default_rng(seed).integers(low, vocab, size=(batch, seq_len), dtype=np.int32)
    return ids, np.concatenate([ids[:, 1:], np.full((batch, 1), IGNORE, np.int32)], 1)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def below(cosines: dict[str, float], bound: float) -> dict[str, float]:
    """The cosines under `bound`. NaN (a zero or non-finite gradient) is among them: min() would skip it."""
    return {name: c for name, c in cosines.items() if not c >= bound}


@pytest.mark.parametrize("row", ROWS)
def test_g0_one_step_is_finite(runs, row, tmp_path):
    """BF16 GEMMs, the selective scan forward and backward (A0, A1h) and flash attention with head_dim 64 (A1,
    A1h) inside the TLC graphs; the graph builds, which also covers TLCMambaFFNBlock without a BlockSchema."""
    from surogate_tlc.ir import compile_ir, param_shapes

    cfg = _cfg(runs, row)
    expected = set(param_shapes(compile_ir(cfg))) - ({"lm_head"} if cfg["tie_word_embeddings"] else set())
    with native(runs / "models" / row, runs, tmp_path) as trainer:
        grads = gradients(trainer, *random_batch(cfg["vocab_size"], 2, 512, seed=0))
    assert expected <= set(grads), sorted(expected - set(grads))
    assert not [name for name, g in grads.items() if not np.isfinite(g).all()]
    assert not [name for name in expected if not np.any(grads[name])]


@pytest.mark.parametrize("row", ROWS)
def test_g1n1_future_tokens_do_not_change_the_past(runs, row, tmp_path):
    """Changing every token from position p on leaves the log-probabilities before p bit for bit; p around the
    scan's chunk boundary."""
    vocab = _cfg(runs, row)["vocab_size"]
    ids, targets = random_batch(vocab, 1, 512, seed=1)
    with native(runs / "models" / row, runs, tmp_path) as trainer:
        for prefix in PREFIXES:
            changed = ids.copy()
            changed[0, prefix:] = (changed[0, prefix:] + 1) % vocab
            out = logprobs(trainer, np.concatenate([ids, changed]), np.concatenate([targets, targets]))
            assert np.array_equal(out[0, :prefix], out[1, :prefix]), prefix
            if prefix < 511:  # and the change is seen after p
                assert not np.array_equal(out[0, prefix:511], out[1, prefix:511]), prefix


@pytest.mark.parametrize("last", (CHUNK - 1, CHUNK))
@pytest.mark.parametrize("row", ROWS)
def test_g1n3_gradients_ignore_tokens_after_the_last_target(runs, row, last, tmp_path):
    """With targets only at positions up to `last` (both sides of the scan's chunk boundary), the parameter gradients
    do not move when the tokens after `last` change, the adjacent one at last + 1 included: step() takes the targets
    apart from the tokens, so the label at `last` stays. The native backward accumulates with atomics, so the
    comparison allows twice the spread of a repeated run."""
    vocab = _cfg(runs, row)["vocab_size"]
    ids, targets = random_batch(vocab, 2, 512, seed=9)
    targets[:, last + 1 :] = IGNORE
    changed = ids.copy()
    changed[:, last + 1 :] = (changed[:, last + 1 :] + 1) % vocab
    runs_ = []
    for batch in (ids, ids, changed):
        with native(runs / "models" / row, runs, tmp_path / str(len(runs_))) as trainer:
            runs_.append(gradients(trainer, batch, targets))
    first, repeat, other = runs_
    spread = max(float(np.abs(first[k] - repeat[k]).max()) for k in first)
    moved = max(float(np.abs(first[k] - other[k]).max()) for k in first)
    print(json.dumps({"row": row, "last_target": last, "g1n3_repeat_spread": spread, "g1n3_change": moved}))
    assert moved <= 2 * spread


def test_g1b_matches_the_reference_goldens(runs, tmp_path):
    golden = np.load(runs / "goldens" / "A0.npz")
    weights = (runs / "models" / "A0" / "model.safetensors").read_bytes()
    assert json.loads(str(golden["meta"]))["model_safetensors_sha256"] == hashlib.sha256(weights).hexdigest()
    ids, targets = golden["input_ids"], golden["targets"]
    with native(runs / "models" / "A0", runs, tmp_path, batch=ids.shape[0], seq_len=ids.shape[1]) as trainer:
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
    with native(runs / "models" / row, runs, tmp_path) as trainer:
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
        with native(runs / "models" / row, runs, tmp_path, batch=ids.shape[0], seq_len=ids.shape[1]) as trainer:
            results[row] = (logprobs(trainer, ids, targets), gradients(trainer, ids, targets))
    (lp, grads), (split_lp, split_grads) = results["A0"], results["A0-split"]
    assert float(np.abs(lp - split_lp).max()) <= float(golden["tolerance"])
    renamed = {split_param_name(name): g for name, g in grads.items()}
    assert set(renamed) == set(split_grads)
    # A0-split is not a G0 row: here every gradient of both models must exist, be finite and not be all zero.
    both = [*renamed.items(), *split_grads.items()]
    dead = sorted({name for name, g in both if not np.isfinite(g).all() or not g.any()})
    assert not dead, dead
    cosines = {name: cosine(renamed[name], split_grads[name]) for name in renamed}
    print(json.dumps({"g1d_min_gradient_cosine": min(cosines.items(), key=lambda kv: kv[1])}))
    low = below(cosines, 0.999)
    assert not low, low  # [P]


@pytest.mark.parametrize("row", ROWS)
def test_g2_initial_loss(runs, row, tmp_path):
    """Random tokens on the initial weights: CE within 0.1 of ln V + var(logits)/2, logit std below 1. The final
    norm leaves each token's hidden state with mean square 1 at initialisation, so var(logits) is hidden x the
    mean square of the LM head (the tied embedding), estimated from the weights: the std half checks the init scheme
    (0.02 x sqrt(384), about 0.39), not the native trainer. The scoring trainer returns target log-probabilities
    only; full logits need a second, LoRA-carrying trainer (as in test_g3). `expected` stays weight-derived, so the
    CE half is an independent check of the native forward."""
    cfg = _cfg(runs, row)
    weights = load_file(runs / "models" / row / "model.safetensors")
    head = weights["model.embed_tokens.weight" if cfg["tie_word_embeddings"] else "lm_head.weight"].double()
    variance = cfg["hidden_size"] * float(head.square().mean())
    expected = math.log(cfg["vocab_size"]) + variance / 2
    ids, targets = random_batch(cfg["vocab_size"], 2, 512, seed=2)
    with native(runs / "models" / row, runs, tmp_path) as trainer:
        loss = -float(logprobs(trainer, ids, targets)[targets != IGNORE].mean())
    std = math.sqrt(variance)
    print(json.dumps({"row": row, "g2_initial_loss": loss, "expected": expected, "logit_std_from_weights": std}))
    assert abs(loss - expected) <= 0.1 and std < 1


def test_g2_gradients_point_where_the_reference_does(runs, tmp_path):
    """A0, first and last layer: native bf16 gradients against fp32 autograd through prototypes/ref_mamba2.py."""
    import torch

    sys.path.insert(0, str(TLC / "prototypes"))
    import ref_mamba2

    cfg = _cfg(runs, "A0")
    ids, targets = random_batch(cfg["vocab_size"], 2, 128, seed=3)
    with native(runs / "models" / "A0", runs, tmp_path, batch=2, seq_len=128) as trainer:
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
    low = below(found, 0.99)
    assert not low, low  # [P]


def _conditional_entropy(ids: np.ndarray) -> float:
    pairs, current = {}, {}
    for a, b in zip(ids[:, :-1].ravel().tolist(), ids[:, 1:].ravel().tolist(), strict=True):
        pairs[a, b] = pairs.get((a, b), 0) + 1
        current[a] = current.get(a, 0) + 1
    total = sum(pairs.values())
    return -sum(n / total * math.log(n / current[a]) for (a, _), n in pairs.items())


@pytest.mark.parametrize("row", ("A0", "A1"))
def test_g2_overfits_a_small_alphabet_batch(runs, row, tmp_path):
    """One fixed batch over 64 token ids, whose H(next | current) is at least 1.5 nats: no model that sees only
    the current token gets below that, so fitting it shows the sequence mixer works. The control with the mixers
    ablated runs on the reference (prototypes/g2_suite.py, gate G2.3); the native graph cannot be ablated."""
    ids, targets = random_batch(5 + 64, 2, 512, seed=5, low=5)  # ids 5..68: no special token
    bound = _conditional_entropy(ids)
    assert bound >= 1.5
    losses = []
    with native(runs / "models" / row, runs, tmp_path) as trainer:
        for step in range(200):
            trainer.step(ids, targets, _positions(ids))
            losses.append(float(trainer.update_with_config(adamw(), step + 1)["loss"]))
    print(json.dumps({"row": row, "g2_overfit_bound": bound, "g2_overfit_loss": losses[::20] + losses[-1:]}))
    assert losses[-1] < 0.05  # [P]


TIED_STEPS = 10


def test_g2_tied_head_trains_as_the_embedding(runs, tmp_path):
    """A0 is tied: full training trains one matrix, read both as the embedding and as the LM head (fork ledger T10).
    After a few AdamW steps the checkpoint stores it once (no `lm_head` in the weights shard, no `lm_head.weight` in
    model.safetensors), it moved, and the trained trainer's log-probabilities match the fp32 reference on the saved
    weights with the tied embedding as the head, within twice the reference's own bf16 error (the G1(b) criterion).
    A reference whose head stayed at the initial embedding, the defect, must lie outside that bound."""
    import torch

    sys.path.insert(0, str(TLC / "prototypes"))
    import ref_mamba2

    model = runs / "models" / "A0"
    cfg = _cfg(runs, "A0")
    assert cfg["tie_word_embeddings"] is True
    ids, targets = random_batch(cfg["vocab_size"], 2, 128, seed=7)
    with native(model, runs, tmp_path, batch=2, seq_len=128) as trainer:
        for step in range(TIED_STEPS):
            trainer.step(ids, targets, _positions(ids))
            trainer.update_with_config(adamw(), step + 1)
        live = logprobs(trainer, ids, targets)
        trainer.save_checkpoint(str(tmp_path / "checkpoints"), TIED_STEPS)
    saved = tmp_path / "checkpoints" / f"step_{TIED_STEPS:08d}"
    shard = set(load_file(next(saved.glob("weights.shard_*.safetensors"))))
    weights = load_file(saved / "model.safetensors")
    assert "embedding" in shard and "lm_head" not in shard
    assert ref_mamba2.EMBEDDING in weights and ref_mamba2.LM_HEAD not in weights
    initial = load_file(model / "model.safetensors")[ref_mamba2.EMBEDDING]
    assert not torch.equal(weights[ref_mamba2.EMBEDDING], initial)

    def reference(cfg, weights, **kwargs):
        with torch.no_grad():
            scorer = ref_mamba2.Reference.from_weights(cfg, weights, **kwargs)
            return scorer.logprobs(torch.from_numpy(ids).long(), torch.from_numpy(targets).long()).numpy()

    fp32 = reference(cfg, weights)
    tolerance = 2 * float(np.abs(reference(cfg, weights, bf16=True) - fp32).max())
    stale = reference({**cfg, "tie_word_embeddings": False}, {**weights, ref_mamba2.LM_HEAD: initial})
    error, stale_gap = float(np.abs(live - fp32).max()), float(np.abs(stale - fp32).max())
    print(json.dumps({"g2_tied_head_error": error, "tolerance": tolerance, "stale_head_gap": stale_gap}))
    assert stale_gap > tolerance  # the comparison can tell the trained head from the initial one
    assert error <= tolerance  # [P]


def test_g2_resume_reproduces_the_next_steps_and_the_logits(runs, tmp_path, monkeypatch):
    """A0 smoke stage, run twice: once straight through, once from the step-5 checkpoint of the first. Steps 6..10
    lose the same (native backward atomics: 1e-3), the final weights give the same logits within the G1(b)
    tolerance, and a finalized checkpoint exported against a model directory of another seed keeps its own
    weights file, with the tied embedding stored once."""
    from surogate_tlc import checkpoint, hf, init, modeldir
    from surogate_tlc.files import file_sha256

    monkeypatch.setenv("TLC_RUNS", str(runs))
    monkeypatch.setenv("TLC_ROW", "A0")
    run = config.load(TLC / "examples" / "p0.yaml", stage="smoke")

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
    assert max(abs(before[step] - after[step]) for step in after) <= 1e-3

    golden = np.load(runs / "goldens" / "A0.npz")
    ids, targets = golden["input_ids"], golden["targets"]
    scored = {}
    for name in ("full", "resumed"):
        with native(tmp_path / name, runs, tmp_path / f"score-{name}", batch=ids.shape[0]) as trainer:
            scored[name] = logprobs(trainer, ids, targets)
    assert float(np.abs(scored["full"] - scored["resumed"]).max()) <= float(golden["tolerance"])  # [P]

    other_seed = modeldir.write_model_dir(
        tmp_path / "seed2", hf.row_config("A0"), runs / "tokenizer", init_seed=init_seed_of(runs) + 1
    )
    init.init_model_dir(other_seed)
    exported = checkpoint.export(tmp_path / "full" / "step_00000005", other_seed, tmp_path / "export")
    names = set(load_file(exported / "model.safetensors"))
    assert "lm_head.weight" not in names and "model.embed_tokens.weight" in names
    assert modeldir.check_model_dir(exported)["tie_word_embeddings"] is True
    # export() copies the checkpoint's weights file: scoring both would compare a file with its copy. The source
    # directory of another seed contributes everything but the weights.
    weights = [
        file_sha256(d / "model.safetensors") for d in (tmp_path / "full" / "step_00000005", exported, other_seed)
    ]
    assert weights[0] == weights[1] != weights[2]


def init_seed_of(runs: Path) -> int:
    return json.loads((runs / "models" / "A0" / "tlc_meta.json").read_text())["init_seed"]


@pytest.mark.parametrize("weights", ["initial", "pilot"])
@pytest.mark.parametrize("row", ROWS)
def test_g3_sampler_computes_the_base_model(runs, row, weights, tmp_path):
    """The sampler's trainer (rank-1 LoRA with B = 0, for next_token_logits) gives the next-token distribution of
    the base model: its log-softmax at each target matches compute_logprobs without LoRA. On the initial weights
    and on the pilot run's final weights: the LoRA trainer reads the tied embedding as the head, which is what full
    training trains and saves (fork ledger T10), so after training both trainers still compute one network."""
    model = runs / "models" / row if weights == "initial" else runs / "train" / row / "pilot"
    vocab = _cfg(runs, row)["vocab_size"]
    ids, targets = random_batch(vocab, 8, 512, seed=6)
    last = np.array([5, 127, 128, 129, 300, 400, 500, 510], np.int32)
    with native(model, runs, tmp_path / "base", batch=8) as trainer:
        base = logprobs(trainer, ids, targets)
    with native(model, runs, tmp_path / "lora", batch=8, settings=SAMPLING_LORA) as trainer:
        logits = np.asarray(trainer.next_token_logits(np.ascontiguousarray(ids), last), np.float64)
    shifted = logits - logits.max(-1, keepdims=True)
    log_softmax = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    sampled = log_softmax[np.arange(8), targets[np.arange(8), last]]
    error = float(np.abs(sampled - base[np.arange(8), last]).max())
    print(json.dumps({"row": row, "weights": weights, "g3_sampler_max_abs_error": error}))
    assert error <= 2e-2  # [P]
