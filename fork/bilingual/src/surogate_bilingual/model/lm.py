"""`RoutedLM` = backbone + expert bank + executor (MS §4.1).

    u_l     = h_l + Attn_l(Norm(h_l))                      (shared, frozen after the seed)
    h_{l+1} = u_l + FFN_l(Norm(u_l); plan)                  (the executor, or the backbone's seed FFN under LP-k)
    logits  = FinalNorm(h_L) E^T                            (tied head)

The model never derives a route from tokens: with more than one expert, `forward` requires a RoutePlan and raises
ValueError without one (MA-14); a single-expert model runs that expert when `plan` is None.

Construction is deterministic: modules are built on the meta device, materialised on the target device, and every
parameter is drawn from one `torch.Generator` seeded with `init_seed`, in sorted parameter-name order (norm weights
1, the family-II bias b 0, every other weight N(0, 0.02) as in v1.0). On `device="meta"` nothing is allocated,
which is how the budgets of the pilot geometry are counted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import DEFAULT_FUSION, FUSIONS, ModelConfig
from ..ops.ffn import SlotReadout, standalone
from ..ops.norm import RMSNorm
from ..routing.plan import RoutePlan
from .backbone import Backbone, backbone_keys
from .bank import ExpertBank, expert_keys
from .executor import Dispatch, Executor, make_dispatch

INIT_STD = 0.02


class RoutedLM(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        expert_names: Sequence[str],
        fusion: str | None = None,
        gqa: str = "sdpa_gqa",
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        init_seed: int = 0,
    ):
        super().__init__()
        fusion = fusion or DEFAULT_FUSION[cfg.family]
        if fusion not in FUSIONS[cfg.family]:
            raise ValueError(f"fusion {fusion!r} does not belong to family {cfg.family!r}: {FUSIONS[cfg.family]}")
        self.cfg = cfg
        self.fusion = fusion
        self.gqa = gqa
        with torch.device("meta"):
            self.backbone = Backbone(cfg, gqa)
            self.bank = ExpertBank(cfg, expert_names)
        self.executor = Executor(fusion)
        if dtype != torch.float32:
            self.to(dtype)
        target = torch.device(device if device is not None else "cpu")
        if target.type != "meta":
            self.to_empty(device=target)
            self.reset_parameters(init_seed)

    @property
    def expert_names(self) -> tuple[str, ...]:
        return self.bank.names

    def reset_parameters(self, init_seed: int) -> None:
        kinds: dict[int, str] = {}
        for module in self.modules():
            if isinstance(module, RMSNorm):
                kinds[id(module.weight)] = "ones"
            elif isinstance(module, SlotReadout):
                kinds[id(module.b)] = "zeros"
            elif isinstance(module, (nn.Linear, nn.Embedding)):
                kinds[id(module.weight)] = "normal"
        generator = torch.Generator().manual_seed(int(init_seed))
        with torch.no_grad():
            for name, p in sorted(self.named_parameters()):
                kind = kinds.get(id(p))
                if kind is None:
                    raise RuntimeError(f"no initialiser for parameter {name}")
                if kind == "normal":
                    value = torch.empty(p.shape, dtype=p.dtype).normal_(0.0, INIT_STD, generator=generator)
                else:
                    value = (torch.ones if kind == "ones" else torch.zeros)(p.shape, dtype=p.dtype)
                p.copy_(value)

    # --- forward ---------------------------------------------------------------------------------------------------

    def dispatch_for(self, tokens: torch.Tensor, plan: RoutePlan | None) -> Dispatch:
        if tokens.dim() != 2:
            raise ValueError(f"tokens must be [B, T], got {list(tokens.shape)}")
        if tokens.shape[1] > self.cfg.max_seq:
            raise ValueError(f"T = {tokens.shape[1]} exceeds max_seq = {self.cfg.max_seq}")
        if plan is None:
            if len(self.expert_names) != 1:
                raise ValueError(
                    f"a model with experts {self.expert_names} needs an explicit RoutePlan; plan=None is allowed only"
                    " with exactly one expert (MA-14)"
                )
            return Dispatch.single_expert(self.expert_names[0])
        plan.validate(self.expert_names, tuple(tokens.shape))
        return make_dispatch(plan, tokens.device)

    def _block(self, index: int, h: torch.Tensor, dispatch: Dispatch, plan: RoutePlan | None) -> torch.Tensor:
        layer = self.backbone.layers[index]
        u = h + layer.attn(layer.norm_attn(h), self.backbone.rope)
        x = layer.norm_ffn(u)
        if layer.ffn is not None:
            y = standalone(layer.ffn, layer.readout, x)
        else:
            y = self.executor(index, x, plan, self.bank, layer.readout, dispatch=dispatch)
        return u + y.to(u.dtype)

    def forward(
        self, tokens: torch.Tensor, plan: RoutePlan | None = None, *, checkpointing: bool = False
    ) -> torch.Tensor:
        """Logits [B, T, V] for tokens [B, T]."""
        dispatch = self.dispatch_for(tokens, plan)
        h = self.backbone.emb(tokens)
        for index in range(self.cfg.n_layers):
            if checkpointing and torch.is_grad_enabled():
                h = checkpoint(self._block, index, h, dispatch, plan, use_reentrant=False)
            else:
                h = self._block(index, h, dispatch, plan)
        return self.backbone.head(h)

    # --- state views -----------------------------------------------------------------------------------------------

    def backbone_state(self) -> dict[str, torch.Tensor]:
        """theta_S under the pack key names (backbone.*)."""
        return {k: v for k, v in self.state_dict().items() if k.startswith("backbone.")}

    def expert_state(self, name: str) -> dict[str, torch.Tensor]:
        """One expert under the pack key names (ffn.{layer}.{param})."""
        if name not in self.expert_names:
            raise KeyError(f"no expert {name!r} in {self.expert_names}")
        return {f"ffn.{k}": v for k, v in self.bank.experts[name].state_dict().items()}

    def seed_ffn_state(self) -> dict[str, torch.Tensor]:
        """The FFN of a seed model (its one expert, named "seed")."""
        if self.expert_names != ("seed",):
            raise ValueError(f"seed_ffn_state belongs to a seed model (experts ('seed',)), not {self.expert_names}")
        return self.expert_state("seed")

    def load_backbone_state(self, state: Mapping[str, torch.Tensor]) -> None:
        _check_keys(set(state), set(backbone_keys(self.cfg)), "backbone")
        _copy_into(self.state_dict(), state, lambda key: key, "backbone")

    def load_expert_state(self, name: str, state: Mapping[str, torch.Tensor]) -> None:
        _check_keys(set(state), set(expert_keys(self.cfg)), f"expert {name!r}")
        _copy_into(
            self.bank.experts[name].state_dict(), state, lambda key: key.removeprefix("ffn."), f"expert {name!r}"
        )


def _check_keys(got: set[str], expected: set[str], what: str) -> None:
    missing, unexpected = sorted(expected - got), sorted(got - expected)
    if missing or unexpected:
        raise KeyError(
            f"{what} state: missing {missing[:5]}{'...' if len(missing) > 5 else ''}, unexpected {unexpected[:5]}"
        )


def _copy_into(own: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor], local, what: str) -> None:
    for key, value in state.items():
        if tuple(value.shape) != tuple(own[local(key)].shape):
            raise ValueError(
                f"{what} state: {key} has shape {list(value.shape)}, expected {list(own[local(key)].shape)}"
            )
    with torch.no_grad():
        for key, value in state.items():
            own[local(key)].copy_(value)
