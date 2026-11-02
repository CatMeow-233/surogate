"""Causal grouped-query self-attention with RoPE; route-free.

`gqa="sdpa_gqa"` passes the n_kv_heads K/V to SDPA with `enable_gqa=True`; `"repeat"` copies them to n_heads first
(the v1.0 path, kept as the fallback for SDPA backends without GQA, IMPL-10). Which backend sm_120 actually runs is
measured at G0 (docs/runbook.md); on CPU both paths agree to fp32 rounding (I2-A6).
"""

from __future__ import annotations

import functools

import torch
from torch import nn
from torch.nn import functional as F

from ..config import GQA_MODES, ModelConfig
from .rope import RopeCache, apply_rope


@functools.cache
def sdpa_supports_gqa() -> bool:
    """True when this torch build's scaled_dot_product_attention accepts `enable_gqa`."""
    q, kv = torch.zeros(1, 2, 1, 2), torch.zeros(1, 1, 1, 2)
    try:
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
    except (TypeError, RuntimeError):
        return False
    return True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, gqa: str = "sdpa_gqa"):
        super().__init__()
        if gqa not in GQA_MODES:
            raise ValueError(f"gqa must be one of {GQA_MODES}: {gqa!r}")
        self.n_heads, self.n_kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.gqa = gqa
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * self.hd, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.o = nn.Linear(cfg.n_heads * self.hd, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, rope: RopeCache) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        cos, sin = rope.get(T, x.device, q.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.n_kv == self.n_heads:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.gqa == "sdpa_gqa":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            groups = self.n_heads // self.n_kv
            k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, T, self.n_heads * self.hd))
