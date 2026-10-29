"""Gate G1 on the Mamba-2 reference (ref_mamba2.py): causality and the agreement of its evaluation paths, on CPU.

The native G1 tests (tests/gpu/test_gates.py) compare the native trainer with this reference, so the reference must
first be shown causal and self-consistent. Each check returns None when it passes, else what failed:

- G1.1 paths: chunked, scan and prefill+step logits agree for T in seq_lengths(c) (around the chunk length c and
  512), within 1e-10 x max(1, |logits|) in fp64 and 1e-5 x max(1, |logits|) in fp32;
- G1.2 prefix: forward(x[:j]) equals forward(x)[:j] (within the G1.1 bounds: a shorter input may take another
  conv1d algorithm, so not bit for bit);
- G1.3 handover: prefill k tokens, then step: the full forward's logits, for k in {1, c-1, c, c+1, T-1};
- G1.4 future forward: changing the tokens from p on leaves logits[:, :p] bit for bit;
- G1.5 future gradient: d logits_t / d embedding_s is exactly 0 for s > t, and finite even for a long chunk with
  strong decay (where exp of an unmasked segment sum overflows);
- G1.6 documents: two documents packed in a row with restarting positions give the logits of each alone; without the
  positions (no reset, as the native mamba_ssm_scan) the second document differs by more than 1e-3;
- G1.7 padding: right padding leaves the logits alone, and prefill refuses left padding.

MUTATIONS are five bugs, each in one function of ref_mamba2; G1.8 is that every one of them fails some check.

Usage: python g1_suite.py   (prints the result per dtype and per mutation; exit status 1 on a failure)
"""

from __future__ import annotations

import contextlib
import importlib.util
import math
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
TOLERANCE = {torch.float64: 1e-10, torch.float32: 1e-5}
Check = Callable[[], str | None]


def load_prototype(name: str) -> ModuleType:
    """prototypes/<name>.py as a module (the prototypes are not a package, and python -I does not put this
    directory on sys.path)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_reference() -> ModuleType:
    return load_prototype("ref_mamba2")


def tiny_config(chunk: int = 16, layers: int = 2) -> dict:
    """A two-layer Mamba model small enough for a single CPU thread, with TLC's layer structure."""
    return {
        "tlc_layer_pattern": "M" * layers,
        "num_hidden_layers": layers,
        "tie_word_embeddings": True,
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "rms_norm_eps": 1e-5,
        "mamba_num_heads": 2,
        "mamba_head_dim": 8,
        "ssm_state_size": 8,
        "n_groups": 1,
        "conv_kernel": 4,
        "chunk_size": chunk,
    }


def tiny_weights(ref: ModuleType, cfg: dict, *, seed: int = 0, strong_decay: bool = False) -> dict:
    """Random weights with the Mamba-2 initialisation; the embedding is N(0, 0.5) so that logits are of order 1.
    strong_decay sets dt to about 10 and A to -16 in every head: a chunk's log-decay sum reaches -160 per token."""
    g = torch.Generator().manual_seed(seed)
    d, ff, heads = cfg["hidden_size"], cfg["intermediate_size"], cfg["mamba_num_heads"]
    inner, state, kernel = heads * cfg["mamba_head_dim"], cfg["ssm_state_size"], cfg["conv_kernel"]
    conv_dim = inner + 2 * cfg["n_groups"] * state

    def uniform(*shape: int, bound: float) -> torch.Tensor:
        return (torch.rand(*shape, generator=g, dtype=torch.float64) * 2 - 1) * bound

    if strong_decay:
        dt = torch.full((heads,), 10.0, dtype=torch.float64)
        a_log = torch.full((heads,), math.log(16.0), dtype=torch.float64)
    else:
        dt = torch.exp(torch.rand(heads, generator=g, dtype=torch.float64) * math.log(100) + math.log(1e-3))
        a_log = torch.log(1 + 15 * torch.rand(heads, generator=g, dtype=torch.float64))
    fields = {
        "ln1_weight": torch.ones(d),
        "in_proj_weight": uniform(2 * inner + 2 * cfg["n_groups"] * state + heads, d, bound=d**-0.5),
        "conv_weight": uniform(conv_dim, kernel, bound=kernel**-0.5),
        "conv_bias": uniform(conv_dim, bound=kernel**-0.5),
        "A_log": a_log,
        "D_param": torch.ones(heads),
        "dt_bias": dt + torch.log(-torch.expm1(-dt)),
        "gated_norm_weight": torch.ones(inner),
        "out_proj_weight": uniform(d, inner, bound=inner**-0.5),
        "ln2_weight": torch.ones(d),
        "mlp_up_weight": uniform(2 * ff, d, bound=d**-0.5),
        "mlp_down_weight": uniform(d, ff, bound=ff**-0.5),
    }
    weights = {
        ref.layer_key(cfg, layer, field): value.clone()
        for layer in range(cfg["num_hidden_layers"])
        for field, value in fields.items()
    }
    weights[ref.EMBEDDING] = torch.randn(cfg["vocab_size"], d, generator=g, dtype=torch.float64) * 0.5
    weights[ref.FINAL_NORM] = torch.ones(d)
    return weights


def seq_lengths(chunk: int) -> tuple[int, ...]:
    return tuple(dict.fromkeys((1, 2, 7, chunk - 1, chunk, chunk + 1, 2 * chunk + 1, 512)))


def _ids(cfg: dict, batch: int, length: int, seed: int) -> torch.Tensor:
    return torch.randint(0, cfg["vocab_size"], (batch, length), generator=torch.Generator().manual_seed(seed))


def _error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max()) if a.numel() else 0.0


def checks(dtype: torch.dtype = torch.float64) -> dict[str, Check]:
    """The G1.1-G1.7 checks on the tiny reference model in `dtype`, cheapest first."""
    ref = load_reference()
    cfg = tiny_config()
    chunk = cfg["chunk_size"]
    model = ref.Reference.from_weights(cfg, tiny_weights(ref, cfg), dtype=dtype)

    def bound(logits: torch.Tensor) -> float:
        return TOLERANCE[dtype] * max(1.0, float(logits.abs().max()))

    def future_forward() -> str | None:
        ids = _ids(cfg, 2, 2 * chunk + 1, seed=1)
        for mode in ("scan", "chunked"):
            full = model.logits(ids, mode)
            for p in (1, 7, chunk, chunk + 1):
                changed = ids.clone()
                changed[:, p:] = (changed[:, p:] + 1) % cfg["vocab_size"]
                other = model.logits(changed, mode)
                if not torch.equal(other[:, :p], full[:, :p]):
                    return f"G1.4 {mode}: changing tokens from {p} on changes earlier logits by {_error(other[:, :p], full[:, :p]):.3g}"
        return None

    def prefix() -> str | None:
        ids = _ids(cfg, 2, 2 * chunk + 1, seed=2)
        for mode in ("scan", "chunked"):
            full = model.logits(ids, mode)
            for j in (1, 2, 7, chunk - 1, chunk, chunk + 1):
                part = model.logits(ids[:, :j], mode)
                err = _error(part, full[:, :j])
                if err > bound(full):
                    return f"G1.2 {mode}: the prefix of {j} tokens differs by {err:.3g}"
        return None

    def handover() -> str | None:
        ids = _ids(cfg, 2, 2 * chunk + 1, seed=3)
        full = model.logits(ids)
        for mode in ("scan", "chunked"):
            for k in (1, chunk - 1, chunk, chunk + 1, ids.shape[1] - 1):
                logits, caches = model.prefill(ids[:, :k], mode)
                steps = []
                for t in range(k, ids.shape[1]):
                    step, caches = model.step(ids[:, t], caches)
                    steps.append(step)
                err = max(_error(logits, full[:, :k]), _error(torch.stack(steps, 1), full[:, k:]))
                if err > bound(full):
                    return f"G1.3 {mode}: prefill {k} tokens, then step: off by {err:.3g}"
        return None

    def padding() -> str | None:
        ids = _ids(cfg, 2, chunk + 3, seed=4)
        padded = torch.cat([ids, torch.zeros(2, 9, dtype=ids.dtype)], 1)
        for mode in ("scan", "chunked"):
            err = _error(model.logits(padded, mode)[:, : ids.shape[1]], model.logits(ids, mode))
            if err > bound(model.logits(ids, mode)):
                return f"G1.7 {mode}: right padding changes the logits by {err:.3g}"
        left = torch.cat([torch.zeros(2, 3, dtype=ids.dtype), ids], 1)
        try:
            model.prefill(left, pad_id=0)
        except ValueError:
            return None
        return "G1.7: prefill accepted a left-padded row"

    def documents() -> str | None:
        first, second = _ids(cfg, 1, 9, seed=5), _ids(cfg, 1, chunk + 7, seed=6)
        packed = torch.cat([first, second], 1)
        positions = torch.cat([torch.arange(first.shape[1]), torch.arange(second.shape[1])])[None]
        for mode in ("scan", "chunked"):
            alone = torch.cat([model.logits(first, mode), model.logits(second, mode)], 1)
            together = model.logits(packed, mode, positions)
            err = _error(together, alone)
            if err > bound(alone):
                return f"G1.6 {mode}: packed documents differ from separate runs by {err:.3g}"
            leak = _error(model.logits(packed, mode)[:, first.shape[1] :], alone[:, first.shape[1] :])
            if leak <= 1e-3:
                return f"G1.6 {mode}: without a reset the second document moves by only {leak:.3g} (control)"
        return None

    def future_gradient() -> str | None:
        strong_cfg = tiny_config(chunk=64)
        cases = (
            (model, cfg, _ids(cfg, 2, 2 * chunk + 1, seed=7)),
            (
                ref.Reference.from_weights(strong_cfg, tiny_weights(ref, strong_cfg, strong_decay=True), dtype=dtype),
                strong_cfg,
                _ids(strong_cfg, 2, 64, seed=8),
            ),
        )
        for reference, case_cfg, ids in cases:
            c = case_cfg["chunk_size"]
            for mode in ("scan", "chunked"):
                x = reference.embed(ids).detach().requires_grad_(True)
                logits = reference.forward(x, mode)[0]
                for t in sorted(t for t in {0, c - 1, c, ids.shape[1] - 2} if t <= ids.shape[1] - 2):
                    (grad,) = torch.autograd.grad(logits[:, t].sum(), x, retain_graph=True)
                    if not torch.isfinite(grad).all():
                        return f"G1.5 {mode}, chunk {c}: a non-finite gradient of logits at {t}"
                    if grad[:, t + 1 :].abs().sum() != 0:
                        return f"G1.5 {mode}, chunk {c}: logits at {t} depend on later embeddings"
        return None

    def paths() -> str | None:
        for length in seq_lengths(chunk):
            ids = _ids(cfg, 2, length, seed=length)
            scan = model.logits(ids, "scan")
            for name, other in (("chunked", model.logits(ids, "chunked")), ("step", model.step_logits(ids))):
                err = _error(other, scan)
                if err > bound(scan):
                    return f"G1.1 T={length}: {name} and scan differ by {err:.3g} (bound {bound(scan):.3g})"
        return None

    return {
        "G1.4 future forward": future_forward,
        "G1.2 prefix": prefix,
        "G1.3 handover": handover,
        "G1.7 padding": padding,
        "G1.6 documents": documents,
        "G1.5 future gradient": future_gradient,
        "G1.1 paths": paths,
    }


def _symmetric_conv(xbc, weight, bias, doc):
    """Bug: the conv pads both sides, so output t also reads input t+1."""
    T, D, K = xbc.shape[1], xbc.shape[2], weight.shape[1]
    conv = F.conv1d(xbc.transpose(1, 2), weight[:, None, :], bias, padding=K // 2, groups=D)
    return conv[..., :T].transpose(1, 2)


def _shifted_ssd(original):
    """Bug: the chunked SSD output is off by one token (y_t holds y_{t+1})."""

    def ssd(*args):
        y, state = original(*args)
        return torch.cat([y[:, 1:], torch.zeros_like(y[:, :1])], 1), state

    return ssd


def _late_window(xbc, kernel, doc):
    """Bug: the conv cache after a prefill is the window one token earlier."""
    return F.pad(xbc, (0, 0, kernel + 1, 0))[:, -kernel - 1 : -1].transpose(1, 2)


def _no_reset(doc, other):
    """Bug: every position counts as the same document, so nothing resets at a document start."""
    return torch.ones(torch.broadcast_shapes(doc.shape, other.shape), dtype=torch.bool)


def _exp_before_mask(a):
    """Bug: the segment sums are cumsum differences, exponentiated before the mask. The forward stays close, but
    with strong decay exp overflows above the diagonal and the masked inf turns the gradient into NaN."""
    L = a.shape[-1]
    a_cum = a.cumsum(-1)
    lower = torch.ones(L, L, dtype=torch.bool).tril()
    return torch.exp(a_cum[..., :, None] - a_cum[..., None, :]).masked_fill(~lower, 0)


def mutations(ref: ModuleType) -> dict[str, dict[str, object]]:
    """Each bug as the ref_mamba2 functions it replaces."""
    return {
        "symmetric conv padding": {"causal_conv": _symmetric_conv},
        "SSM output off by one": {"ssd_chunked": _shifted_ssd(ref.ssd_chunked)},
        "conv handover off by one": {"conv_window": _late_window},
        "no document reset": {"same_document": _no_reset},
        "exp before mask in segsum": {"decay_matrix": _exp_before_mask},
    }


@contextlib.contextmanager
def mutated(patches: dict[str, object]) -> Iterator[None]:
    ref = load_reference()
    saved = {name: getattr(ref, name) for name in patches}
    try:
        for name, value in patches.items():
            setattr(ref, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(ref, name, value)


def run(dtype: torch.dtype = torch.float64, *, stop_at_first: bool = False) -> dict[str, str | None]:
    """Every check's result (None: passed); with stop_at_first, up to the first failure."""
    results: dict[str, str | None] = {}
    with torch.no_grad():
        for name, check in checks(dtype).items():
            with torch.enable_grad() if "gradient" in name else contextlib.nullcontext():
                results[name] = check()
            if stop_at_first and results[name]:
                break
    return results


def caught(name: str) -> str | None:
    """The first failure the fp64 checks report under mutation `name` (None would mean an uncaught bug)."""
    with mutated(mutations(load_reference())[name]):
        return next((failure for failure in run(stop_at_first=True).values() if failure), None)


def main() -> int:
    torch.set_num_threads(1)
    start, status = time.process_time(), 0
    for dtype in (torch.float64, torch.float32):
        for name, failure in run(dtype).items():
            print(f"{dtype}  {name}: {failure or 'ok'}")
            status |= failure is not None
    for name in mutations(load_reference()):
        failure = caught(name)
        print(f"mutation {name}: {'caught by ' + failure if failure else 'NOT CAUGHT'}")
        status |= failure is None
    print(f"{time.process_time() - start:.1f} CPU seconds")
    return status


if __name__ == "__main__":
    sys.exit(main())
