"""`RoutePlan`: per-token gates over named experts (docs/spec.md, "路由 plan").

`gates` is [B, T, E] with one column per name. Every row lies on the simplex (entries >= 0, sum 1 within 1e-6) and
has at most two nonzero entries in P0. One-hot rows run the Top-1 gather/scatter path; two-entry rows run the fusion
the model names. Plans are values: they never look at tokens, tokenizers or data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..config import NAME_PATTERN

SIMPLEX_ATOL = 1e-6
MAX_NONZERO = 2
GATE_DTYPES = (torch.float32, torch.float64)  # float32 by default; float64 for the fp64 identity tests


class PlanError(ValueError):
    pass


@dataclass(frozen=True)
class RoutePlan:
    names: tuple[str, ...]
    gates: torch.Tensor
    policy: str

    def __post_init__(self):
        object.__setattr__(self, "names", tuple(self.names))

    @staticmethod
    def one_hot(route_ids: torch.Tensor, names: Sequence[str], policy: str) -> RoutePlan:
        """route_ids [B, T] (int) indexes `names`."""
        route_ids = torch.as_tensor(route_ids)
        if route_ids.dim() != 2 or route_ids.dtype.is_floating_point:
            raise PlanError(
                f"route_ids must be an integer [B, T] tensor, got {route_ids.dtype} {list(route_ids.shape)}"
            )
        if route_ids.numel() and (int(route_ids.min()) < 0 or int(route_ids.max()) >= len(names)):
            raise PlanError(f"route ids must index {len(names)} names")
        gates = torch.nn.functional.one_hot(route_ids.long(), len(names)).to(torch.float32)
        return RoutePlan(tuple(names), gates, policy)

    @staticmethod
    def forced(name: str, names: Sequence[str], batch: int, length: int, device=None) -> RoutePlan:
        """Every token on one expert."""
        ids = torch.full((batch, length), list(names).index(name), dtype=torch.long, device=device)
        return RoutePlan.one_hot(ids, names, f"forced({name})")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.gates.shape[0]), int(self.gates.shape[1])

    def validate(self, names: Sequence[str] | None = None, shape: tuple[int, int] | None = None) -> None:
        """Raise PlanError unless the plan is well formed; `names` are the experts that exist, `shape` is (B, T)."""
        if len(set(self.names)) != len(self.names) or not self.names:
            raise PlanError(f"plan names must be non-empty and unique: {self.names}")
        bad = [n for n in self.names if not NAME_PATTERN.fullmatch(n)]
        if bad:
            raise PlanError(f"plan names must match {NAME_PATTERN.pattern}: {bad}")
        if names is not None:
            unknown = [n for n in self.names if n not in names]
            if unknown:
                raise PlanError(f"plan names {unknown} are not experts of the model {tuple(names)}")
        g = self.gates
        if not isinstance(g, torch.Tensor) or g.dim() != 3 or g.shape[-1] != len(self.names):
            got = list(g.shape) if isinstance(g, torch.Tensor) else type(g).__name__
            raise PlanError(f"gates must be [B, T, {len(self.names)}], got {got}")
        if shape is not None and tuple(g.shape[:2]) != tuple(shape):
            raise PlanError(f"gates cover {list(g.shape[:2])} tokens, the batch is {list(shape)}")
        if g.dtype not in GATE_DTYPES:
            raise PlanError(f"gates must be float32 (or float64), got {g.dtype}")
        if g.device.type == "meta":
            raise PlanError("gates must hold values (not on the meta device)")
        if not bool(torch.isfinite(g).all()):
            raise PlanError("gates must be finite")
        if bool((g < 0).any()):
            raise PlanError("gates must be >= 0")
        if bool(((g != 0).sum(-1) > MAX_NONZERO).any()):
            raise PlanError(f"a gate row has more than {MAX_NONZERO} nonzero entries")
        if g.numel() and float((g.sum(-1, dtype=torch.float64) - 1.0).abs().max()) > SIMPLEX_ATOL:
            raise PlanError(f"gate rows must sum to 1 (atol {SIMPLEX_ATOL})")

    def slice(self, start: int, end: int) -> RoutePlan:
        """The plan of positions [start, end): what a window of a longer sequence must use (MA-09)."""
        return RoutePlan(self.names, self.gates[:, start:end], self.policy)

    def is_one_hot(self) -> torch.Tensor:
        return (self.gates != 0).sum(-1) == 1

    def route_ids(self) -> torch.Tensor:
        """[B, T] index of each token's expert; only for plans whose every row is one-hot."""
        if not bool(self.is_one_hot().all()):
            raise PlanError("route_ids needs a plan whose every row is one-hot")
        return self.gates.argmax(-1)
