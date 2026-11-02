"""Analytic multiply-accumulates per token (MS §5.1) and the FlopCounterMode cross-check (I2-A14).

Weight MACs per token = the LM head (V d) + per layer the attention projections (q, k, v, o) and the FFN matmuls
on the token's path; the embedding lookup, norms, biases and gate products cost no MACs. The attention scores add
n_layers * H * hd * T per token at sequence length T (QK^T and AV over an average causal context of T / 2).

FFN MACs per expert layer:

| row kind | swiglu | slot |
|---|---|---|
| single | 3 d f | d k + n d + k n |
| dual | 6 d f (also `concat`) | d k + 2 (n d + k n) (one Q matmul for `pre` and `post`) |

LP-k layers without an expert run the seed FFN (the single cost). `extra_per_layer` adds the MACs of per-layer
modules whose forward lands later (adapters, routers), so budget rows can price them.
"""

from __future__ import annotations

import torch
from torch.utils.flop_counter import FlopCounterMode

from ..config import ModelConfig
from ..routing.plan import RoutePlan

ROW_KINDS = ("single", "dual")
LINEAR_OPS = ("aten.mm", "aten.addmm")  # what nn.Linear and the tied head lower to on 2-D and folded inputs


def ffn_macs(cfg: ModelConfig, kind: str) -> int:
    if kind not in ROW_KINDS:
        raise ValueError(f"row kind must be one of {ROW_KINDS}: {kind!r}")
    d, f, n = cfg.d_model, cfg.ffn_width, cfg.slot_width
    if cfg.family == "swiglu":
        return 3 * d * f * (2 if kind == "dual" else 1)
    return d * f + (n * d + f * n) * (2 if kind == "dual" else 1)


def attention_projection_macs(cfg: ModelConfig) -> int:
    q = cfg.d_model * cfg.n_heads * cfg.head_dim
    kv = cfg.d_model * cfg.n_kv_heads * cfg.head_dim
    return 2 * q + 2 * kv  # q and o, k and v


def macs_per_token(cfg: ModelConfig, row: str, T: int | None = None, *, extra_per_layer: int = 0) -> int:
    expert_layers = len(cfg.layers_with_experts)
    shared_layers = cfg.n_layers - expert_layers
    total = cfg.vocab_size * cfg.d_model + cfg.n_layers * (attention_projection_macs(cfg) + extra_per_layer)
    total += expert_layers * ffn_macs(cfg, row) + shared_layers * ffn_macs(cfg, "single")
    if T is not None:
        total += cfg.n_layers * cfg.n_heads * cfg.head_dim * int(T)
    return total


def counted_flops(model, tokens: torch.Tensor, plan: RoutePlan | None) -> dict[str, int]:
    """FLOPs per aten op of one forward under FlopCounterMode (no grad)."""
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(tokens, plan)
    return {str(op): int(n) for op, n in counter.get_flop_counts()["Global"].items()}


def linear_flops(model, tokens: torch.Tensor, plan: RoutePlan | None) -> int:
    """FLOPs of every Linear and the tied head in one forward: the mm/addmm ops. On CPU, SDPA's fused kernel is not
    counted at all; on the meta device it lowers to bmm, which this sum leaves out."""
    flops = counted_flops(model, tokens, plan)
    return sum(n for op, n in flops.items() if op in LINEAR_OPS)
