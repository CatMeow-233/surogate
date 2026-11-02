"""Fusion of experts on the tokens whose gate row has two nonzero entries (MS §4.2–§4.4; docs/spec.md, I3).

Route-free: the gates arrive as an explicit tensor [N, E], one column per expert passed in, and the operators never
validate them. `RoutePlan.validate` keeps them on the simplex in every model run; the operators also accept the
forbidden off-simplex gates such as g = (1, 1), so that the tests can show what those gates break (MA-04).

The four fusions the executor dispatches to (step I2):

- `mix_outputs` (B6, `output`): sum_i g_i E_i(u);
- `concat_forward` on `stack_swiglu` (B7, `concat`): the rows of gate/up and the columns of down concatenated, with a
  per-token block gate on the concatenated intermediate. It equals `mix_outputs` up to rounding (a known identity,
  not a contribution, MS §4.2);
- `slot_pre` (B8): Q SiLU(sum_i g_i W_i z_i + b), mixing before the shared nonlinearity;
- `slot_post` (B9): Q [sum_i g_i SiLU(W_i z_i + b)], mixing after it, with one Q matmul.

Operators that no P0 run executes (step I3):

- `cross_link_lowrank`: v1.0's cross connections with rank-r links, kept only to test that they reparameterise
  concat, i.e. are a gated LoRA on the experts' gate/up (MA-06); `cross_link_params` counts them;
- `channel_select`: the network made of selected channels of each expert, weights gathered before projecting
  (v1.0 check 6, the B10-ch layout; the block router is P1);
- `ZeroInitAdapter`: the B11 adapter d -> r -> d whose up projection starts at zero (the repair ladder is P1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .ffn import SlotExpert, SlotReadout, SwiGLU
from .reads import declare_partial_read


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


def _as_input(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """x in the dtype of the operator's input `like` for the projection that follows. The gate product is promoted
    (fp32 gates on bf16 activations give fp32); with bf16 parameters and no autocast the input, and so the weights,
    are bf16 and F.linear needs one dtype. Under autocast nothing is cast here (autocast casts as before); fp32 and
    fp64 runs are unchanged. The weights are not touched, so the read audit sees one read per projection."""
    if x.dtype == like.dtype:
        return x
    try:
        autocast = torch.is_autocast_enabled(x.device.type)
    except RuntimeError:  # a device type without autocast (meta)
        autocast = False
    return x if autocast else x.to(like.dtype)


def concat_forward(x: torch.Tensor, stacked: StackedSwiGLU, block_gates: torch.Tensor) -> torch.Tensor:
    """down_cat( diag(g_1 1_{f_1}, ..., g_E 1_{f_E}) [SiLU(gate_cat x) * (up_cat x)] ), per token."""
    hidden = F.silu(F.linear(x, stacked.gate)) * F.linear(x, stacked.up)
    widths = torch.tensor(stacked.widths, device=block_gates.device)
    scale = torch.repeat_interleave(block_gates, widths, dim=-1).to(
        torch.promote_types(block_gates.dtype, hidden.dtype)
    )
    return F.linear(_as_input(hidden * scale, x), stacked.down)


def slot_pre(u: torch.Tensor, experts: Sequence[SlotExpert], readout: SlotReadout, gates: torch.Tensor) -> torch.Tensor:
    mixed = mix_outputs([e.pre_activation(u) for e in experts], gates)
    return readout(_as_input(mixed, u))


def slot_post(
    u: torch.Tensor, experts: Sequence[SlotExpert], readout: SlotReadout, gates: torch.Tensor
) -> torch.Tensor:
    mixed = mix_outputs([readout.activate(e.pre_activation(u)) for e in experts], gates)
    return readout.Q(_as_input(mixed, u))


def _check_gates(gates: torch.Tensor, n_experts: int) -> None:
    if gates.dim() != 2 or gates.shape[-1] != n_experts:
        raise ValueError(f"gates must be [N, {n_experts}], got {list(gates.shape)}")


def cross_link_lowrank(
    x: torch.Tensor, experts: Sequence[SwiGLU], gates: torch.Tensor, A: torch.Tensor, B: torch.Tensor
) -> torch.Tensor:
    """v1.0 `cross_mix` with rank-r links C_ij = A_ij B_ij^T between the gate (p = 0) and up (p = 1) projections:

        a_i = G_i x + sum_{j != i} g_j C^0_ij (G_j x),   b_i = U_i x + sum_{j != i} g_j C^1_ij (U_j x),
        out = sum_i g_i D_i [SiLU(a_i) * b_i].

    A and B are [2, E, E, f, r]: projection, receiving expert i, sending expert j, channel, rank. The diagonal i = j
    is ignored. A link reads the sender's own pre-activation G_j x, never the sender's linked one. C is never formed:
    C x = A (B^T x). Every expert runs on every row; a zero gate removes a term exactly, so a one-hot row returns
    the standalone expert. Not used by any run: with a constant g the operator is concat with G_i replaced by
    G_i + g_j C_ij G_j, i.e. a gated rank-r LoRA on G_i and U_i (MA-06), which the I3 tests check.
    """
    E = len(experts)
    _check_gates(gates, E)
    widths = {e.gate.weight.shape[0] for e in experts}
    if len(widths) != 1:
        raise ValueError(f"cross links need experts of one width, got {sorted(widths)}")
    f = widths.pop()
    if A.dim() != 5 or tuple(A.shape[:4]) != (2, E, E, f) or A.shape != B.shape:
        raise ValueError(f"A and B must both be [2, {E}, {E}, {f}, r], got {list(A.shape)} and {list(B.shape)}")
    pre = [(F.linear(x, e.gate.weight), F.linear(x, e.up.weight)) for e in experts]
    out = None
    for i, expert in enumerate(experts):
        a, b = pre[i]
        for j in range(E):
            if j == i:
                continue
            g_j = _gate(gates, j, x)
            a = a + g_j * ((pre[j][0] @ B[0, i, j]) @ A[0, i, j].T)
            b = b + g_j * ((pre[j][1] @ B[1, i, j]) @ A[1, i, j].T)
        term = _gate(gates, i, x) * F.linear(F.silu(a) * b, expert.down.weight)
        out = term if out is None else out + term
    return out


def cross_link_params(ffn_width: int, n_layers: int, rank: int | None = None, n_experts: int = 2) -> int:
    """Parameters of the cross links: one link per projection (gate, up), ordered expert pair and layer, with f^2
    entries at full rank and 2 f r as A B^T at rank r (MA-06: 24 * 2 * 2 * 1728^2 = 286,654,464 at the pilot)."""
    per_link = ffn_width * ffn_width if rank is None else 2 * ffn_width * rank
    return n_layers * 2 * n_experts * (n_experts - 1) * per_link


def _channel_ids(ids, width: int) -> torch.Tensor:
    ids = torch.as_tensor(ids)
    if ids.dim() != 1 or ids.dtype != torch.long or ids.numel() == 0:
        raise ValueError(f"channel ids must be a non-empty 1-D long tensor, got {ids.dtype} {list(ids.shape)}")
    if int(ids.min()) < 0 or int(ids.max()) >= width or torch.unique(ids).numel() != ids.numel():
        raise ValueError(f"channel ids must be distinct and lie in [0, {width})")
    return ids


def channel_select(
    x: torch.Tensor, experts: Sequence[SwiGLU], selection: Sequence[tuple[int, torch.Tensor]]
) -> torch.Tensor:
    """sum over (e, ids) in `selection` of D_e[:, ids] [SiLU(G_e[ids] x) * (U_e[ids] x)]: the SwiGLU made of the
    selected channels of each expert (v1.0 check 6; the B10-ch layout). The rows of gate/up and the columns of down
    are gathered before any projection, so no intermediate wider than the selected channels exists, and each
    gathered weight is declared to the parameter audit (`ops.reads`). The selection is the same for every row; the
    per-token block router of B10-ch is P1. Each expert appears at most once."""
    if not selection:
        raise ValueError("channel_select needs a non-empty selection")
    seen: set[int] = set()
    out = None
    for e, ids in selection:
        if not 0 <= e < len(experts) or e in seen:
            raise ValueError(f"selection names expert {e} twice or out of range (0..{len(experts) - 1})")
        seen.add(e)
        expert = experts[e]
        weights = (expert.gate.weight, expert.up.weight, expert.down.weight)
        ids = _channel_ids(ids, weights[0].shape[0]).to(weights[0].device)
        gate, up = weights[0].index_select(0, ids), weights[1].index_select(0, ids)
        down = weights[2].index_select(1, ids)
        for param, part in zip(weights, (gate, up, down), strict=True):
            declare_partial_read(param, part.numel())
        term = F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
        out = term if out is None else out + term
    return out


class ZeroInitAdapter(nn.Module):
    """B11: the residual delta U (D h), D: d -> r, U: r -> d, no bias, 2 d r parameters (the B11 budget rows count
    these modules). U starts at zero, so h + delta(h) = h exactly until U moves. At initialisation the gradient
    w.r.t. U is (dL/d delta)^T (D h), non-zero, and the gradient w.r.t. D is zero (it passes through U)."""

    def __init__(self, d: int, r: int, *, device=None, dtype=None):
        super().__init__()
        self.down = nn.Linear(d, r, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(r, d, bias=False, device=device, dtype=dtype)
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """The delta (not h + delta)."""
        return self.up(self.down(h))
