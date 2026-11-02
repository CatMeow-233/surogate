"""FFN blocks of the two families (MS §4.1, §4.3).

- Family I (`swiglu`): E(u) = W_down [SiLU(W_gate u) * (W_up u)], width f = `ffn_width`.
- Family II (`slot`, the canonical operator of B-O4): an expert holds R (n x d) and W (k x n); the readout Q (d x k)
  and b (k) are shared and belong to the backbone. The standalone expert is F(u) = Q SiLU(W SiLU(R u) + b), with
  n = `slot_width` and k = `ffn_width`.

No biases except b. Each block exposes the intermediate quantities the fusions need, so fusion math never reaches
into a block's weights.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..config import ModelConfig


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, width: int):
        super().__init__()
        self.gate = nn.Linear(d_model, width, bias=False)
        self.up = nn.Linear(d_model, width, bias=False)
        self.down = nn.Linear(width, d_model, bias=False)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        """The width-f intermediate SiLU(W_gate x) * (W_up x)."""
        return F.silu(self.gate(x)) * self.up(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.hidden(x))


class SlotExpert(nn.Module):
    """The trainable part of a family-II expert: z = SiLU(R u), a = W z (the pre-activation without b)."""

    def __init__(self, d_model: int, slot_width: int, readout_width: int):
        super().__init__()
        self.R = nn.Linear(d_model, slot_width, bias=False)
        self.W = nn.Linear(slot_width, readout_width, bias=False)

    def pre_activation(self, u: torch.Tensor) -> torch.Tensor:
        return self.W(F.silu(self.R(u)))


class SlotReadout(nn.Module):
    """The shared, frozen family-II readout: Q SiLU(a + b)."""

    def __init__(self, d_model: int, readout_width: int):
        super().__init__()
        self.Q = nn.Linear(readout_width, d_model, bias=False)
        self.b = nn.Parameter(torch.zeros(readout_width))

    def activate(self, a: torch.Tensor) -> torch.Tensor:
        return F.silu(a + self.b)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.Q(self.activate(a))


def make_expert(cfg: ModelConfig) -> nn.Module:
    if cfg.family == "swiglu":
        return SwiGLU(cfg.d_model, cfg.ffn_width)
    return SlotExpert(cfg.d_model, cfg.slot_width, cfg.ffn_width)


def standalone(expert: nn.Module, readout: SlotReadout | None, u: torch.Tensor) -> torch.Tensor:
    """One expert on its own: the Top-1 path."""
    if isinstance(expert, SwiGLU):
        return expert(u)
    if readout is None:
        raise ValueError("a family-II expert needs the shared readout")
    return readout(expert.pre_activation(u))
