"""The three parameter budgets of MS §5.1, counted from the code rather than from formulas.

- `p_total(model)`: every parameter stored, each storage once (the tied head is the embedding).
- `p_update(opt)`: every parameter the optimizer holds, each storage once.
- `p_active(model, tokens, plan)`: the parameters the forward actually reads, found by running it under a
  `TorchFunctionMode` that records every parameter passed to a torch function. An operator that reads only part of
  a parameter (a sliced weight, e.g. `channel_select` in step I3) must declare how many elements it reads with
  `declare_partial_read`, because the mode sees the whole Parameter go into the slicing call (IMPL-21); a declared
  count replaces the full size of that parameter. With `per_token=True` the count is the largest over the distinct
  gate patterns of the plan (what one token reads); with `per_token=False` it is the union over the batch.

Storages are told apart by `untyped_storage().data_ptr()`; on the meta device, where every pointer is 0, by
parameter identity.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterable

import torch
from torch import nn
from torch.overrides import TorchFunctionMode

from ..routing.plan import RoutePlan

_RECORDER: contextvars.ContextVar[_ActiveRecorder | None] = contextvars.ContextVar("bilingual_audit", default=None)


def _key(p: torch.Tensor) -> tuple:
    if p.device.type == "meta" or p.untyped_storage().data_ptr() == 0:
        return ("id", id(p))
    return (str(p.device), p.untyped_storage().data_ptr())


def _count_unique(params: Iterable[torch.Tensor]) -> int:
    sizes: dict[tuple, int] = {}
    for p in params:
        key = _key(p)
        sizes[key] = max(sizes.get(key, 0), p.numel())
    return sum(sizes.values())


def p_total(model: nn.Module) -> int:
    return _count_unique(model.parameters())


def p_update(opt: torch.optim.Optimizer) -> int:
    return _count_unique(p for group in opt.param_groups for p in group["params"])


def declare_partial_read(param: torch.Tensor, n_elements: int) -> None:
    """Called by an operator that reads only `n_elements` of `param` (no-op outside an audit)."""
    recorder = _RECORDER.get()
    if recorder is not None:
        recorder.declare(param, int(n_elements))


def _walk(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _walk(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from _walk(item)


class _ActiveRecorder(TorchFunctionMode):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.params = {id(p): p for p in model.parameters()}
        self.seen: dict[int, torch.Tensor] = {}
        self.declared: dict[int, int] = {}

    def _param_of(self, t: torch.Tensor) -> torch.Tensor | None:
        p = self.params.get(id(t))
        if p is None and t._base is not None:
            p = self.params.get(id(t._base))
        return p

    def declare(self, param: torch.Tensor, n: int) -> None:
        p = self._param_of(param)
        if p is None:
            raise ValueError("declare_partial_read: not a parameter of the audited model")
        self.declared[id(p)] = max(self.declared.get(id(p), 0), n)

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for t in _walk((args, kwargs)):
            p = self._param_of(t)
            if p is not None:
                self.seen[id(p)] = p
        return func(*args, **kwargs)

    def count(self) -> int:
        sizes: dict[tuple, int] = {}
        for pid, p in self.seen.items():
            key = _key(p)
            sizes[key] = max(sizes.get(key, 0), self.declared.get(pid, p.numel()))
        return sum(sizes.values())


def _active_once(model: nn.Module, tokens: torch.Tensor, plan: RoutePlan | None) -> int:
    recorder = _ActiveRecorder(model)
    token = _RECORDER.set(recorder)
    try:
        with torch.no_grad(), recorder:
            model(tokens, plan)
    finally:
        _RECORDER.reset(token)
    return recorder.count()


def gate_patterns(plan: RoutePlan) -> list[torch.Tensor]:
    """One representative gate row per distinct set of nonzero experts."""
    rows = plan.gates.reshape(-1, plan.gates.shape[-1])
    patterns: dict[tuple[int, ...], torch.Tensor] = {}
    for row in rows:
        patterns.setdefault(tuple(torch.nonzero(row).flatten().tolist()), row)
    return list(patterns.values())


def p_active(model: nn.Module, tokens: torch.Tensor, plan: RoutePlan | None, *, per_token: bool = True) -> int:
    if not per_token or plan is None:
        return _active_once(model, tokens, plan)
    B, T = tokens.shape
    counts = []
    for row in gate_patterns(plan):
        uniform = RoutePlan(plan.names, row.expand(B, T, -1).clone(), plan.policy)
        counts.append(_active_once(model, tokens, uniform))
    return max(counts)
