"""Analytic training FLOPs of every arm kind, and the H-Net configuration that matches a reference arm's FLOPs per
byte (STUDY §10). Pure Python: a model is described by a spec dict (`models.build.flop_spec` makes one from a model).

Spec dicts:

    stack: {"type": "mamba", "d", "layers", "ffn" (None: no MLP), "heads", "head_dim", "state", "groups", "chunk",
            "kernel"}
           {"type": "transformer", "d", "layers", "q_width", "kv_width", "ffn"}           (SwiGLU MLP, no biases)
    model: {"kind": "subword" | "bytes", "main": stack, "vocab": V}
           {"kind": "hnet", "main": stack, "vocab": V, "d0", "encoder": stack, "decoder": stack}

Two conventions (forward FLOPs per position; training = 3 x forward):

- `"matmul"` (the default, the one progress and F_target use): 2 x the multiply-adds of every matmul, plus the
  sequence mixers. This is surogate_tlc.ir.flops_per_token term for term (a test imports it): in_proj and
  out_proj, the chunked SSD's states and intra-chunk terms (4 inner state + 2 chunk (G state + inner)), the
  depthwise conv (2 kernel (inner + 2 G state)), SwiGLU (2 (d 2f + f d)), the tied head (2 d V), and for
  attention 2 (d q + 2 d kv + q d) + 2 ctx q, ctx the row's length (QK^T and AV over the causal context). For the
  bilingual trunk it is 6 x surogate_bilingual.audit.flops.macs_per_token (test). The H-Net adds its router
  (2 x 2 d0^2), residual projection (2 d0^2) and byte head per byte; the EMA, norms and activations cost nothing,
  as in both projects' formulas.
- `"params"` (the convention of the study's budget probe, integration/flops_budget.json): 6 x the parameters a
  position reads (every parameter of the stacks it runs, norms included, plus V d for the tied head), plus
  3 x the sequence mixers: attention always, the Mamba SSD and conv terms only with scan=True. Under it the
  bilingual seed costs 652,889,088 FLOPs per token at T = 1024 and a flat-byte A0 153,545,760 per byte with
  scan=False [verified arithmetic]. In the matmul convention the seed costs 652,738,560: norms are not matmuls.

`scan=False` drops the Mamba SSD and conv terms in either convention.

A row's FLOPs: a token arm runs main + head over its n positions; an H-Net runs its outer stage (encoder, router,
residual, decoder, byte head) over n bytes and the main network over the row's realized chunk count m (attention
context m). The training loop sums rows (padding is not counted: it is not the model's computation).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

CONVENTIONS = ("matmul", "params")
DEFAULT_GRID_LAYERS = (4, 3, 2, 1)
DEFAULT_GRID_D0 = (512, 384, 320, 256, 192, 128)
TOLERANCE = 0.03


def mamba2_mixer_fwd(d: int, heads: int, head_dim: int, state: int, groups: int, chunk: int, kernel: int,
                     *, scan: bool = True) -> int:  # fmt: skip
    """Forward FLOPs per position of one Mamba-2 mixer (surogate_tlc.ir.flops_per_token's terms)."""
    inner = heads * head_dim
    total = 2 * (d * (2 * inner + 2 * groups * state + heads) + inner * d)
    if scan:
        total += 4 * inner * state + 2 * chunk * (groups * state + inner) + 2 * kernel * (inner + 2 * groups * state)
    return total


def mamba2_scan_fwd(stack: Mapping[str, Any]) -> int:
    """The SSD and conv terms of one mixer per position."""
    inner, gs = stack["heads"] * stack["head_dim"], stack["groups"] * stack["state"]
    return 4 * inner * stack["state"] + 2 * stack["chunk"] * (gs + inner) + 2 * stack["kernel"] * (inner + 2 * gs)


def swiglu_fwd(d: int, ffn: int) -> int:
    return 2 * (d * 2 * ffn + ffn * d)


def attention_fwd(d: int, q_width: int, kv_width: int, ctx: int) -> int:
    return 2 * (d * q_width + 2 * d * kv_width + q_width * d) + 2 * ctx * q_width


def head_fwd(d: int, vocab: int) -> int:
    return 2 * d * vocab


def _check_stack(stack: Mapping[str, Any]) -> None:
    if stack.get("type") not in ("mamba", "transformer"):
        raise ValueError(f"a stack is mamba or transformer, got {stack.get('type')!r}")


def stack_params(stack: Mapping[str, Any]) -> int:
    """Every parameter of a stack: its layers (norms included) and the final norm."""
    _check_stack(stack)
    d, ffn = stack["d"], stack.get("ffn")
    if stack["type"] == "mamba":
        heads, inner, gs = stack["heads"], stack["heads"] * stack["head_dim"], stack["groups"] * stack["state"]
        conv = inner + 2 * gs
        layer = d * (2 * inner + 2 * gs + heads) + conv * stack["kernel"] + conv + 3 * heads + inner + inner * d + d
        layer += (d + 3 * d * ffn) if ffn else 0
    else:
        q, kv = stack["q_width"], stack["kv_width"]
        layer = d * q + 2 * d * kv + q * d + 3 * d * ffn + 2 * d
    return stack["layers"] * layer + d


def _matmul_layer(stack: Mapping[str, Any], ctx: int, scan: bool) -> int:
    d, ffn = stack["d"], stack.get("ffn")
    if stack["type"] == "mamba":
        mixer = mamba2_mixer_fwd(d, stack["heads"], stack["head_dim"], stack["state"], stack["groups"],
                                 stack["chunk"], stack["kernel"], scan=scan)  # fmt: skip
        return mixer + (swiglu_fwd(d, ffn) if ffn else 0)
    return attention_fwd(d, stack["q_width"], stack["kv_width"], ctx) + swiglu_fwd(d, ffn)


def _mixer_terms(stack: Mapping[str, Any], ctx: int, scan: bool) -> int:
    """The sequence-mixer FLOPs per position of a whole stack (attention scores; SSD and conv when scan)."""
    if stack["type"] == "mamba":
        return stack["layers"] * mamba2_scan_fwd(stack) if scan else 0
    return stack["layers"] * 2 * ctx * stack["q_width"]


def stack_fwd(stack: Mapping[str, Any], ctx: int, *, convention: str = "matmul", scan: bool = True) -> int:
    """Forward FLOPs per position of a stack at attention context `ctx` (ignored by Mamba), without a head."""
    _check_stack(stack)
    if convention == "matmul":
        return stack["layers"] * _matmul_layer(stack, ctx, scan)
    if convention == "params":
        return 2 * stack_params(stack) + _mixer_terms(stack, ctx, scan)
    raise ValueError(f"convention must be one of {CONVENTIONS}, got {convention!r}")


def outer_fwd_per_byte(spec: Mapping[str, Any], *, convention: str = "matmul", scan: bool = True) -> int:
    """An H-Net's outer stage per byte: encoder, router (W_q, W_k), residual projection, decoder, byte head."""
    d0 = spec["d0"]
    stacks = sum(stack_fwd(spec[k], 0, convention=convention, scan=scan) for k in ("encoder", "decoder"))
    return stacks + 2 * 3 * d0 * d0 + head_fwd(d0, spec["vocab"])


def row_train_flops(spec: Mapping[str, Any], positions: int, chunks: int | None = None, *,
                    convention: str = "matmul", scan: bool = True) -> int:  # fmt: skip
    """Training FLOPs (3 x forward) of one row of `positions` ids; an H-Net row also needs its realized `chunks`."""
    if positions < 0:
        raise ValueError("a row has a non-negative length")
    main = spec["main"]
    if spec["kind"] in ("subword", "bytes"):
        per = stack_fwd(main, positions, convention=convention, scan=scan) + head_fwd(main["d"], spec["vocab"])
        return 3 * positions * per
    if spec["kind"] != "hnet":
        raise ValueError(f"unknown model kind {spec['kind']!r}")
    if chunks is None or not 0 <= chunks <= positions:
        raise ValueError(f"an H-Net row of {positions} bytes needs its realized chunk count, got {chunks}")
    outer = positions * outer_fwd_per_byte(spec, convention=convention, scan=scan)
    return 3 * (outer + chunks * stack_fwd(main, chunks, convention=convention, scan=scan))


def train_flops(model_spec: Mapping[str, Any], *, positions_main: int, positions_byte: int,
                convention: str = "matmul", scan: bool = True) -> int:  # fmt: skip
    """One row: positions_byte ids in, positions_main of them reach the main network (equal for token arms)."""
    if model_spec["kind"] != "hnet" and positions_main != positions_byte:
        raise ValueError("a token arm's main network reads every position")
    chunks = positions_main if model_spec["kind"] == "hnet" else None
    return row_train_flops(model_spec, positions_byte, chunks, convention=convention, scan=scan)


def batch_train_flops(spec: Mapping[str, Any], rows: Iterable[tuple[int, int | None]], **kw) -> int:
    """Sum over rows of (positions, chunks or None)."""
    return sum(row_train_flops(spec, n, m, **kw) for n, m in rows)


def flops_per_byte(spec: Mapping[str, Any], *, row_bytes: int, bytes_per_position: float = 1.0,
                   ratio: float | None = None, convention: str = "matmul", scan: bool = True) -> float:  # fmt: skip
    """Training FLOPs per text byte of a full row of `row_bytes` bytes: a subword arm at c bytes per token
    (row of row_bytes / c tokens), a byte arm (c = 1), an H-Net at `ratio` bytes per chunk."""
    if spec["kind"] == "hnet":
        if ratio is None or not ratio >= 1:
            raise ValueError("an H-Net's FLOPs per byte need its bytes per chunk (ratio >= 1)")
        chunks = row_bytes / ratio
        outer = outer_fwd_per_byte(spec, convention=convention, scan=scan)
        main = stack_fwd(spec["main"], round(chunks), convention=convention, scan=scan)
        return 3 * (outer + main / ratio)
    positions = row_bytes / bytes_per_position
    main = spec["main"]
    per = stack_fwd(main, round(positions), convention=convention, scan=scan) + head_fwd(main["d"], spec["vocab"])
    return 3 * per / bytes_per_position


def outer_stack(d0: int, layers: int, *, expand: int = 2, head_dim: int = 64, state: int = 64, chunk: int = 128,
                kernel: int = 4) -> dict[str, Any]:  # fmt: skip
    """The spec of an H-Net encoder or decoder (Mamba-2 layers without an MLP, models.hnet.HNetConfig's)."""
    inner = expand * d0
    if inner % head_dim:
        raise ValueError(f"inner width {inner} is not a multiple of head_dim {head_dim}")
    return {"type": "mamba", "d": d0, "layers": layers, "ffn": None, "heads": inner // head_dim, "head_dim": head_dim,
            "state": state, "groups": 1, "chunk": chunk, "kernel": kernel}  # fmt: skip


def match_hnet_config(main_spec: Mapping[str, Any], target_flops_per_byte: float, N: float,
                      grid: Sequence[tuple[int, int]] | None = None, *, row_bytes: int, vocab: int = 261,
                      tolerance: float = TOLERANCE, convention: str = "matmul",
                      outer: Mapping[str, int] | None = None) -> dict[str, Any]:  # fmt: skip
    """The first (layers, d0) of `grid` (encoder layers = decoder layers) whose H-Net, at N bytes per chunk, costs
    target_flops_per_byte within +-tolerance, with every grid point that fits under "candidates"; refuses when none
    fits, naming the closest. The default grid runs from deep to shallow and, per depth, from wide to narrow
    (DEFAULT_GRID_LAYERS x DEFAULT_GRID_D0); d0 never exceeds the main width (widths are monotone). The choice is a
    proposal: the arm registry fixes model.hnet."""
    if not target_flops_per_byte > 0:
        raise ValueError("the target FLOPs per byte must be positive")
    grid = list(grid) if grid is not None else [(n, d) for n in DEFAULT_GRID_LAYERS for d in DEFAULT_GRID_D0]
    outer = dict(outer or {})
    tried, fits = [], []
    for layers, d0 in grid:
        if d0 > main_spec["d"]:
            continue
        try:
            stack = outer_stack(d0, layers, **outer)
        except ValueError:
            continue
        spec = {"kind": "hnet", "main": dict(main_spec), "vocab": vocab, "d0": d0, "encoder": stack, "decoder": stack}
        found = flops_per_byte(spec, row_bytes=row_bytes, ratio=N, convention=convention)
        error = found / target_flops_per_byte - 1
        tried.append((abs(error), layers, d0, error))
        if abs(error) <= tolerance:
            fits.append({"d0": d0, "encoder_layers": layers, "decoder_layers": layers, "flops_per_byte": found,
                         "error": error})  # fmt: skip
    if not tried:
        raise ValueError("no grid point fits the main network's width")
    if not fits:
        _, layers, d0, error = min(tried)
        raise ValueError(f"no grid point within {tolerance:.0%} of {target_flops_per_byte:.6g} FLOPs/byte; closest: "
                         f"{layers}+{layers} layers at d0 {d0} ({error:+.2%})")  # fmt: skip
    return {**fits[0], "ratio": N, "target_flops_per_byte": target_flops_per_byte, "tolerance": tolerance,
            "convention": convention, "candidates": fits}  # fmt: skip


# --- command line ----------------------------------------------------------------------------------------------------


def _parse_c(values: Sequence[str]) -> dict[str, float]:
    out = {}
    for value in values:
        arm, sep, c = value.partition("=")
        if not sep:
            raise ValueError(f"--bytes-per-token {value!r}: expected ARM=C")
        out[arm] = float(c)
    return out


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tokstudy flops",
                                     description="analytic training FLOPs per arm; H-Net configuration matching")  # fmt: skip
    parser.add_argument("registry", help="arm registry (tokstudy-arms/1)")
    parser.add_argument("--k-train", type=int, required=True, help="row length in byte positions (K_train)")
    parser.add_argument("--bytes-per-token", action="append", default=[], metavar="ARM=C",
                        help="measured bytes per token of a subword arm (intrinsic); needed for its FLOPs per byte")  # fmt: skip
    parser.add_argument("--match", metavar="HNET_ARM", help="solve the H-Net arm's outer stage against --reference")
    parser.add_argument("--reference", metavar="ARM", help="the subword arm whose FLOPs per byte the H-Net matches")
    parser.add_argument("--convention", choices=CONVENTIONS, default="matmul")
    args = parser.parse_args(list(argv))
    from pathlib import Path

    from .arms import codec_for, load_arms
    from .canonical import TokstudyError, json_text
    from .models.build import build_model, flop_spec

    try:
        arms = {a.id: a for a in load_arms(Path(args.registry))}
        cs = _parse_c(args.bytes_per_token)
        out: dict[str, Any] = {"k_train": args.k_train, "convention": args.convention, "arms": {}}
        specs = {}
        for arm in arms.values():
            codec = codec_for(arm)
            vocab = codec.vocab_size if arm.kind == "subword" else None
            spec = specs[arm.id] = flop_spec(build_model(arm, vocab_size=vocab, init_seed=0))
            row = {"kind": arm.kind, "main_params": stack_params(spec["main"])}
            if arm.kind == "subword" and arm.id in cs:
                row["flops_per_byte"] = flops_per_byte(spec, row_bytes=args.k_train, bytes_per_position=cs[arm.id],
                                                       convention=args.convention)  # fmt: skip
            elif arm.kind == "bytes":
                row["flops_per_byte"] = flops_per_byte(spec, row_bytes=args.k_train, convention=args.convention)
            elif arm.kind == "hnet":
                ratio = arm.model["hnet"]["ratio"]
                row["flops_per_byte_at_target_ratio"] = flops_per_byte(spec, row_bytes=args.k_train, ratio=ratio,
                                                                       convention=args.convention)  # fmt: skip
            out["arms"][arm.id] = row
        if args.match:
            if args.reference not in cs:
                raise ValueError("--match needs --reference ARM with --bytes-per-token ARM=C")
            target = out["arms"][args.reference]["flops_per_byte"]
            out["match"] = match_hnet_config(specs[args.match]["main"], target, cs[args.reference],
                                             row_bytes=args.k_train, convention=args.convention)  # fmt: skip
        sys.stdout.write(json_text(out))
    except (TokstudyError, ValueError, KeyError) as err:
        print(f"flops: {err}", file=sys.stderr)
        return 1
    return 0
