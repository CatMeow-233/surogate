"""Gate G2 on the Mamba-2 reference (ref_mamba2.py): gradients, the initial loss and a discriminating overfit, on CPU.

- G2.1: one forward and backward; every parameter has a gradient (p.grad is not None) and it is finite.
- G2.2: the initial cross-entropy on random tokens is ln V + var(logits)/2 within 0.1, and the logits' standard
  deviation is below 1. With logits ~ N(0, s^2) over V tokens, E[logsumexp] = ln V + s^2/2; a model whose initial
  loss is off this has a scale or wiring bug before any training.
- G2.3: a fixed batch over a small alphabet, whose bound H(next | current) is at least 1.5 nats: a model that only
  sees the current token cannot go below it. The full model fits the batch, while the model with its mixers
  ablated (their outputs zeroed, so no token sees another) stays at 0.9 x the bound or above. Over a large
  vocabulary a random batch has almost every next token unique given the current one, so even the ablated model
  fits it and the overfit test would not show that the sequence mixer works.

The native G2 (tests/gpu/test_gates.py) runs G2.2 and the full-model part of G2.3 on the trainer; the ablation
needs a graph change the native trainer does not offer, so its control runs here. Resume (G2.5) and reload (G2.4)
are properties of the native checkpoint, tested on the GPU host.

Usage: python g2_suite.py   (prints the results; exit status 1 on a failure)
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import sys
import time
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import torch

if "g1_suite" not in sys.modules:  # python -I does not put this directory on sys.path
    _spec = importlib.util.spec_from_file_location("g1_suite", Path(__file__).resolve().parent / "g1_suite.py")
    sys.modules["g1_suite"] = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(sys.modules["g1_suite"])
from g1_suite import load_reference, tiny_config, tiny_weights  # noqa: E402

ALPHABET, ROWS, LENGTH = 16, 2, 64
OVERFIT_STEPS, OVERFIT_LR = 200, 1e-2


def conditional_entropy(ids: torch.Tensor) -> float:
    """H(next | current) of the batch's (current, next) pairs, in nats: the lowest mean cross-entropy of any
    prediction that sees only the current token."""
    pairs = Counter(zip(ids[:, :-1].flatten().tolist(), ids[:, 1:].flatten().tolist(), strict=True))
    current = Counter()
    for (a, _), n in pairs.items():
        current[a] += n
    total = sum(pairs.values())
    return -sum(n / total * math.log(n / current[a]) for (a, _), n in pairs.items())


def small_alphabet_batch(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, ALPHABET, (ROWS, LENGTH), generator=g)


@contextlib.contextmanager
def ablated_mixers(model) -> Iterator[None]:
    """The model with every mixer output zeroed: the residual stream of a token then never reads another token."""
    original = model._mixer

    def mixer(layer, h, mode, doc):
        out, cache = original(layer, h, mode, doc)
        return torch.zeros_like(out), cache

    model._mixer = mixer
    try:
        yield
    finally:
        model._mixer = original


def gradients_are_finite(model, ids: torch.Tensor) -> str | None:
    """G2.1."""
    params = list(model.w.values())
    for p in params:
        p.requires_grad_(True)
        p.grad = None
    logp = torch.log_softmax(model.logits(ids[:, :-1], "chunked"), -1)
    (-logp.gather(-1, ids[:, 1:, None]).mean()).backward()
    names = [name for name, p in model.w.items() if p.grad is None or not torch.isfinite(p.grad).all()]
    return f"G2.1: missing or non-finite gradients for {names}" if names else None


def initial_loss(model, ids: torch.Tensor) -> tuple[float, float, float]:
    """(cross-entropy, ln V + var(logits)/2, logit standard deviation) on `ids`."""
    with torch.no_grad():
        logits = model.logits(ids[:, :-1], "chunked")
        loss = -torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None]).mean()
    variance = float(logits.var(-1).mean())
    return float(loss), math.log(logits.shape[-1]) + variance / 2, math.sqrt(variance)


def overfit(model, ids: torch.Tensor, *, steps: int = OVERFIT_STEPS, lr: float = OVERFIT_LR) -> float:
    """Final cross-entropy after `steps` of Adam on the one batch `ids`."""
    params = [p.requires_grad_(True) for p in model.w.values()]
    optimizer = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        optimizer.zero_grad()
        logp = torch.log_softmax(model.logits(ids[:, :-1], "chunked"), -1)
        loss = -logp.gather(-1, ids[:, 1:, None]).mean()
        loss.backward()
        optimizer.step()
    return loss.item()


def _model(seed: int, **weights):
    ref = load_reference()
    cfg = tiny_config()
    return ref.Reference.from_weights(cfg, tiny_weights(ref, cfg, seed=seed, **weights))


def run() -> dict[str, str | None]:
    results: dict[str, str | None] = {}
    ids = small_alphabet_batch()
    results["G2.1 gradients"] = gradients_are_finite(_model(0), ids)

    ref = load_reference()
    cfg = {**tiny_config(), "vocab_size": 4096, "hidden_size": 64}
    g = torch.Generator().manual_seed(3)
    weights = tiny_weights(ref, cfg, seed=1)
    # TLC's initialisation of the tied embedding, N(0, 0.02): the logits' variance is about hidden x 0.02^2.
    weights[ref.EMBEDDING] = torch.randn(cfg["vocab_size"], cfg["hidden_size"], generator=g, dtype=torch.float64) * 0.02
    tokens = torch.randint(0, cfg["vocab_size"], (2, 65), generator=g)
    loss, expected, std = initial_loss(ref.Reference.from_weights(cfg, weights), tokens)
    if abs(loss - expected) > 0.1 or std >= 1:
        results["G2.2 initial loss"] = f"G2.2: CE {loss:.4f}, ln V + var/2 = {expected:.4f}, logit std {std:.3f}"
    else:
        results["G2.2 initial loss"] = None

    bound = conditional_entropy(ids)
    full = overfit(_model(2), ids)
    ablated_model = _model(2)
    with ablated_mixers(ablated_model):
        ablated = overfit(ablated_model, ids)
    detail = f"bound {bound:.3f}, full {full:.3f}, ablated {ablated:.3f}"
    if bound < 1.5:
        results["G2.3 overfit"] = f"G2.3: the batch's H(next|current) is below 1.5 ({detail})"
    elif ablated < 0.9 * bound:
        results["G2.3 overfit"] = f"G2.3: the ablated model goes below 0.9 x the bound ({detail})"
    elif full > 0.05:
        results["G2.3 overfit"] = f"G2.3: the full model does not fit the batch ({detail})"
    else:
        results["G2.3 overfit"] = None
    return results


def main() -> int:
    torch.set_num_threads(1)
    start = time.process_time()
    results = run()
    for name, failure in results.items():
        print(f"{name}: {failure or 'ok'}")
    print(f"{time.process_time() - start:.1f} CPU seconds")
    return int(any(results.values()))


if __name__ == "__main__":
    sys.exit(main())
