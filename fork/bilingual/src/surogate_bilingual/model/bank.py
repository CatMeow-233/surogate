"""Named expert parameter containers, one block per expert layer. Computes no route."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn

from ..config import NAME_PATTERN, ModelConfig
from ..ops.ffn import make_expert
from .backbone import ffn_param_names


def check_expert_names(names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(names)
    if not names or len(set(names)) != len(names):
        raise ValueError(f"expert names must be non-empty and unique: {names}")
    bad = [n for n in names if not isinstance(n, str) or not NAME_PATTERN.fullmatch(n)]
    if bad:
        raise ValueError(f"expert names must match {NAME_PATTERN.pattern}: {bad}")
    return names


class ExpertBank(nn.Module):
    def __init__(self, cfg: ModelConfig, names: Sequence[str]):
        super().__init__()
        self.names = check_expert_names(names)
        self.layers = cfg.layers_with_experts
        self.experts = nn.ModuleDict(
            {name: nn.ModuleDict({str(layer): make_expert(cfg) for layer in self.layers}) for name in self.names}
        )

    def get(self, name: str, layer: int) -> nn.Module:
        return self.experts[name][str(layer)]


def expert_keys(cfg: ModelConfig) -> list[str]:
    """The keys of one expert in a pack: ffn.{layer}.{param} for every expert layer."""
    return sorted(f"ffn.{layer}.{p}" for layer in cfg.layers_with_experts for p in ffn_param_names(cfg))
