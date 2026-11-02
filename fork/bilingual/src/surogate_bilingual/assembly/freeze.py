"""Which tensors a run may change, and the audits that prove nothing else changed (IMPL-06, MA-12).

`FreezePlan(model, trainable)` takes parameter names or name prefixes ending in "." (for example
"bank.experts.zh."); every entry must match at least one parameter. `apply()` sets requires_grad accordingly. The
audits compare by storage, not by name or object identity: `check_optimizer` requires the optimizer's parameter set
to equal the trainable set by `untyped_storage().data_ptr()`, and `frozen_hashes` gives one content hash per frozen
tensor, to be compared after training (`check_frozen`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn

from .hashing import tensor_hashes


class FreezeError(RuntimeError):
    pass


def storage_key(t: torch.Tensor) -> tuple[str, int]:
    return (str(t.device), t.untyped_storage().data_ptr())


class FreezePlan:
    def __init__(self, model: nn.Module, trainable: Iterable[str]):
        self.model = model
        params = dict(model.named_parameters())
        entries = list(trainable)
        selected: set[str] = set()
        for entry in entries:
            matched = {n for n in params if n == entry or (entry.endswith(".") and n.startswith(entry))}
            if not matched:
                raise FreezeError(f"trainable entry {entry!r} matches no parameter")
            selected |= matched
        self.entries = tuple(entries)
        self.trainable_names = tuple(sorted(selected))
        self.frozen_names = tuple(sorted(set(params) - selected))

    def apply(self) -> FreezePlan:
        trainable = set(self.trainable_names)
        for name, p in self.model.named_parameters():
            p.requires_grad_(name in trainable)
        return self

    def parameters(self) -> list[nn.Parameter]:
        params = dict(self.model.named_parameters())
        return [params[n] for n in self.trainable_names]

    def numel(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def frozen_state(self) -> dict[str, torch.Tensor]:
        params = dict(self.model.named_parameters())
        return {n: params[n].detach() for n in self.frozen_names}

    def frozen_hashes(self) -> dict[str, str]:
        return tensor_hashes(self.frozen_state())

    def check_frozen(self, hashes: Mapping[str, str]) -> None:
        now = self.frozen_hashes()
        changed = sorted(n for n in set(now) | set(hashes) if now.get(n) != hashes.get(n))
        if changed:
            raise FreezeError(f"{len(changed)} frozen tensors changed, e.g. {changed[:3]}")

    def check_optimizer(self, opt: torch.optim.Optimizer) -> None:
        held = {storage_key(p) for group in opt.param_groups for p in group["params"]}
        allowed = {storage_key(p) for p in self.parameters()}
        if held != allowed:
            extra, missing = len(held - allowed), len(allowed - held)
            raise FreezeError(f"optimizer holds {extra} storages outside the trainable set and lacks {missing} of it")

    def check_grads(self) -> None:
        params = dict(self.model.named_parameters())
        with_grad = [n for n in self.frozen_names if params[n].grad is not None]
        if with_grad:
            raise FreezeError(f"{len(with_grad)} frozen parameters have a .grad, e.g. {with_grad[:3]}")
