"""Route-dependent dispatch of one FFN layer (docs/spec.md, "模块边界"). The only place that reads a RoutePlan's
gates to decide which expert runs on which token; it owns no parameters and chooses no route.

A plan is turned into a `Dispatch` once per forward, on the plan's own device (normally the CPU, so no GPU sync),
and the token indices are moved to the activations' device. Then, per layer:

- every token on one expert: that expert runs on all tokens directly, the same call a single-expert model makes, so
  the forced route is bitwise equal to the standalone model (I2-A3);
- one-hot rows: gather the expert's tokens, run it, scatter back (Top-1);
- two-entry rows: gather the tokens of each expert pair and apply the model's fusion (`output`, `concat`, `pre`,
  `post`) with the row's two gates.

The output buffer has the dtype of the layer input and every expert output is cast to it before the scatter. v1.0
scattered bf16 expert outputs into an fp32 buffer under autocast, which `index_add` refuses (NEW-01).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..config import FUSIONS
from ..ops.ffn import SlotReadout, SwiGLU, standalone
from ..ops.fusion import concat_forward, mix_outputs, slot_post, slot_pre, stack_swiglu
from ..routing.plan import RoutePlan
from .bank import ExpertBank

ALL_FUSIONS = tuple(f for fs in FUSIONS.values() for f in fs)


@dataclass(frozen=True)
class Dispatch:
    single: str | None  # every token on this expert
    one_hot: tuple[tuple[str, torch.Tensor], ...] = ()  # (expert, token indices)
    dual: tuple[tuple[str, str, torch.Tensor, torch.Tensor], ...] = ()  # (expert a, expert b, indices, gates [n, 2])

    @staticmethod
    def single_expert(name: str) -> Dispatch:
        return Dispatch(single=name)


def make_dispatch(plan: RoutePlan, device: torch.device | str) -> Dispatch:
    """Group the tokens of a (validated) plan by expert and by expert pair. Indices refer to the flattened [B*T]."""
    names = plan.names
    gates = plan.gates.reshape(-1, len(names))
    nonzero = gates != 0
    count = nonzero.sum(-1)
    one = count == 1
    winner = gates.argmax(-1)
    if bool(one.all()):
        used = torch.unique(winner)
        if used.numel() == 1:
            return Dispatch(single=names[int(used)])
    one_hot = []
    for e, name in enumerate(names):
        idx = torch.nonzero(one & (winner == e)).flatten()
        if idx.numel():
            one_hot.append((name, idx.to(device)))
    dual = []
    rows = torch.nonzero(count == 2).flatten()
    if rows.numel():
        cols = torch.nonzero(nonzero[rows])[:, 1].reshape(-1, 2)  # the two nonzero columns per row, ascending
        pair_key = cols[:, 0] * len(names) + cols[:, 1]
        for key in torch.unique(pair_key).tolist():
            a, b = divmod(int(key), len(names))
            sel = rows[pair_key == key]
            dual.append((names[a], names[b], sel.to(device), gates[sel][:, [a, b]].to(device)))
    return Dispatch(single=None, one_hot=tuple(one_hot), dual=tuple(dual))


class Executor:
    def __init__(self, fusion: str):
        if fusion not in ALL_FUSIONS:
            raise ValueError(f"fusion must be one of {ALL_FUSIONS}: {fusion!r}")
        self.fusion = fusion

    def _fuse(self, a, b, readout: SlotReadout | None, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        if self.fusion == "output":
            return mix_outputs([standalone(a, readout, x), standalone(b, readout, x)], g)
        if self.fusion == "concat":
            return concat_forward(x, stack_swiglu([a, b]), g)
        if self.fusion == "pre":
            return slot_pre(x, [a, b], readout, g)
        return slot_post(x, [a, b], readout, g)

    def __call__(
        self,
        layer: int,
        u: torch.Tensor,
        plan: RoutePlan | None,
        bank: ExpertBank,
        readout: SlotReadout | None,
        *,
        dispatch: Dispatch | None = None,
    ) -> torch.Tensor:
        """The FFN residual update of `layer` for u [B, T, d], in u's dtype."""
        if dispatch is None:
            if plan is None:
                raise ValueError("the executor needs a plan or a dispatch")
            dispatch = make_dispatch(plan, u.device)
        B, T, d = u.shape
        flat = u.reshape(B * T, d)
        if dispatch.single is not None:
            return standalone(bank.get(dispatch.single, layer), readout, flat).to(u.dtype).reshape(B, T, d)
        out = flat.new_zeros(B * T, d)
        for name, idx in dispatch.one_hot:
            values = standalone(bank.get(name, layer), readout, flat.index_select(0, idx))
            out.index_copy_(0, idx, values.to(out.dtype))
        for a, b, idx, g in dispatch.dual:
            values = self._fuse(bank.get(a, layer), bank.get(b, layer), readout, flat.index_select(0, idx), g)
            out.index_copy_(0, idx, values.to(out.dtype))
        return out.reshape(B, T, d)


def dense_reference_ffn(
    layer: int,
    u: torch.Tensor,
    plan: RoutePlan,
    bank: ExpertBank,
    readout: SlotReadout | None,
    fusion: str,
) -> torch.Tensor:
    """Every expert of the plan on every token, combined with the gates by the fusion's closed form (tests only).

    Written independently of the executor and of ops.fusion: `output` and `concat` sum g_e-weighted expert terms
    (for `concat` the gate scales the width-f intermediate before W_down, the block-gated form of MS §4.2); `pre`
    and `post` apply MS §4.3 directly. On one-hot rows every form reduces to the standalone expert exactly.
    """
    experts: Sequence = [bank.get(name, layer) for name in plan.names]
    g = plan.gates.to(u.device)
    dt = torch.promote_types(g.dtype, u.dtype)

    def gate(e: int) -> torch.Tensor:
        return g[..., e : e + 1].to(dt)

    if fusion in ("output", "concat"):
        if not all(isinstance(e, SwiGLU) for e in experts):
            raise ValueError(f"fusion {fusion!r} needs family-I experts")
        terms = []
        for e, expert in enumerate(experts):
            if fusion == "output":
                terms.append(gate(e) * expert(u))
            else:
                terms.append(expert.down(gate(e) * expert.hidden(u)))
        return torch.stack(terms).sum(0).to(u.dtype)
    if fusion not in ("pre", "post"):
        raise ValueError(f"unknown fusion {fusion!r}")
    pre_acts = [expert.W(torch.nn.functional.silu(expert.R(u))) for expert in experts]
    if fusion == "pre":
        mixed = torch.stack([gate(e) * a for e, a in enumerate(pre_acts)]).sum(0)
        return readout.Q(torch.nn.functional.silu(mixed + readout.b)).to(u.dtype)
    mixed = torch.stack([gate(e) * torch.nn.functional.silu(a + readout.b) for e, a in enumerate(pre_acts)]).sum(0)
    return readout.Q(mixed).to(u.dtype)
