"""Fusion of two experts on the tokens whose gate row has two nonzero entries (MS §4.2, §4.3). Route-free: the
gates arrive as an explicit tensor [N, E] whose rows lie on the simplex, one column per expert passed in.

Step I2 holds the four fusions the executor dispatches to:

- `mix_outputs` (B6, `output`): sum_i g_i E_i(u);
- `concat_forward` on `stack_swiglu` (B7, `concat`): the rows of gate/up and the columns of down concatenated, with a
  per-token block gate on the concatenated intermediate. It equals `mix_outputs` up to rounding (a known identity,
  not a contribution, MS §4.2);
- `slot_pre` (B8): Q SiLU(sum_i g_i W_i z_i + b), mixing before the shared nonlinearity;
- `slot_post` (B9): Q [sum_i g_i SiLU(W_i z_i + b)], mixing after it, with one Q matmul.

Cross links, channel selection and the zero-init adapter join them in step I3 (docs/spec.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .ffn import SlotExpert, SlotReadout, SwiGLU


def _gate(gates: torch.Tensor, i: int, like: torch.Tensor) -> torch.Tensor:
    """Column i of the gates as [N, 1] in a dtype that does not lower `like` (fp32 gates on bf16 activations promote
    the product to fp32)."""
    return gates[:, i : i + 1].to(torch.promote_types(gates.dtype, like.dtype))


def mix_outputs(outputs: Sequence[torch.Tensor], gates: torch.Tensor) -> torch.Tensor:
    if gates.shape[-1] != len(outputs):
        raise ValueError(f"{len(outputs)} outputs but {gates.shape[-1]} gate columns")
    total = None
    for i, out in enumerate(outputs):
        term = _gate(gates, i, out) * out
        total = term if total is None else total + term
    return total


@dataclass(frozen=True)
class StackedSwiGLU:
    """Experts of width f_1..f_E as one SwiGLU of width sum f_i: gate/up rows and down columns concatenated."""

    gate: torch.Tensor  # [sum f, d]
    up: torch.Tensor  # [sum f, d]
    down: torch.Tensor  # [d, sum f]
    widths: tuple[int, ...]


def stack_swiglu(experts: Sequence[SwiGLU]) -> StackedSwiGLU:
    return StackedSwiGLU(
        gate=torch.cat([e.gate.weight for e in experts], dim=0),
        up=torch.cat([e.up.weight for e in experts], dim=0),
        down=torch.cat([e.down.weight for e in experts], dim=1),
        widths=tuple(e.gate.weight.shape[0] for e in experts),
    )


def concat_forward(x: torch.Tensor, stacked: StackedSwiGLU, block_gates: torch.Tensor) -> torch.Tensor:
    """down_cat( diag(g_1 1_{f_1}, ..., g_E 1_{f_E}) [SiLU(gate_cat x) * (up_cat x)] ), per token."""
    hidden = F.silu(F.linear(x, stacked.gate)) * F.linear(x, stacked.up)
    widths = torch.tensor(stacked.widths, device=block_gates.device)
    scale = torch.repeat_interleave(block_gates, widths, dim=-1).to(
        torch.promote_types(block_gates.dtype, hidden.dtype)
    )
    return F.linear(hidden * scale, stacked.down)


def slot_pre(u: torch.Tensor, experts: Sequence[SlotExpert], readout: SlotReadout, gates: torch.Tensor) -> torch.Tensor:
    mixed = mix_outputs([e.pre_activation(u) for e in experts], gates)
    return readout(mixed)


def slot_post(
    u: torch.Tensor, experts: Sequence[SlotExpert], readout: SlotReadout, gates: torch.Tensor
) -> torch.Tensor:
    mixed = mix_outputs([readout.activate(e.pre_activation(u)) for e in experts], gates)
    return readout.Q(mixed)
