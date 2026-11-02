"""RMSNorm computed in the promoted dtype (MA-13): fp32 for bf16/fp16/fp32 inputs, fp64 for fp64 inputs."""

from __future__ import annotations

import torch
from torch import nn


def compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """The dtype norms and RoPE compute in: never below float32, float64 stays float64."""
    return torch.promote_types(dtype, torch.float32)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = x.to(compute_dtype(x.dtype))
        normed = xx * torch.rsqrt(xx.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight
