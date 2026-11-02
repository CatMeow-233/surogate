"""Rotary position embedding, interleaved pairs (2i, 2i+1) as in v1.0 `moe_pilot.rotary`.

The rotation runs in the promoted dtype (fp32 for bf16/fp16/fp32 activations, fp64 for fp64; MA-13). The angle
table pos * theta^(-2i/hd) is built in float64 and only cos/sin are cast to the promoted dtype: an fp32 table
carries the rounding of theta^(-2i/hd) times the position, about 5e-6 at T = 128 and 4e-5 at T = 1024, which v1.0
had; a float64 table keeps the fp32 rotation within 1e-6 of the exact one (I2-A6). Tables are cached per
(T, device, dtype), once for all layers (IMPL-10). Positions start at 0 in every forward (no KV cache in P0).
"""

from __future__ import annotations

import torch

from .norm import compute_dtype


class RopeCache:
    def __init__(self, head_dim: int, theta: float):
        if head_dim % 2:
            raise ValueError(f"RoPE needs an even head_dim: {head_dim}")
        self.head_dim = head_dim
        self.theta = float(theta)
        self._tables: dict[tuple[int, torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}

    def get(self, T: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """cos, sin of shape [T, head_dim / 2] in compute_dtype(dtype)."""
        dtype = compute_dtype(dtype)
        key = (int(T), torch.device(device), dtype)
        if key not in self._tables:
            exponent = torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float64) / self.head_dim
            inv = 1.0 / (self.theta**exponent)
            angles = torch.outer(torch.arange(T, device=device, dtype=torch.float64), inv)
            self._tables[key] = (angles.cos().to(dtype), angles.sin().to(dtype))
        return self._tables[key]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, head_dim]; cos, sin: [T, head_dim / 2]. Returns x's dtype."""
    xx = x.to(cos.dtype)
    x0, x1 = xx[..., ::2], xx[..., 1::2]
    out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)
    return out.to(x.dtype)
