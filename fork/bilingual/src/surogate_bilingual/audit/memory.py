"""Memory accounting: saved-tensor bytes of one training forward (CPU or CUDA) and the CUDA peak recorder.

`saved_tensor_bytes` counts, under `torch.autograd.graph.saved_tensors_hooks`, the bytes of the unique storages
autograd keeps for the backward (what activation checkpointing trades away). It is an accounting of live activation
storage, not of allocator traffic or of the peak: those are only measured on CUDA (`CudaPeak`, G0.5) and are null
elsewhere. Parameters saved for the backward are left out, since they live anyway.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _storage_key(t: torch.Tensor) -> tuple[str, int]:
    s = t.untyped_storage()
    return (str(t.device), s.data_ptr())


def saved_tensor_bytes(model, tokens: torch.Tensor, plan=None, *, checkpointing: bool = False) -> int:
    """Bytes of the unique non-parameter storages autograd saves during one forward with gradients."""
    params = {_storage_key(p) for p in model.parameters()}
    seen: dict[tuple[str, int], int] = {}

    def pack(t: torch.Tensor):
        key = _storage_key(t)
        if key not in params:
            seen[key] = max(seen.get(key, 0), t.untyped_storage().nbytes())
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with torch.enable_grad():
            logits = model(tokens, plan, checkpointing=checkpointing)
    del logits
    return int(sum(seen.values()))


@dataclass
class CudaPeak:
    """`with CudaPeak(device) as peak:` resets and then reads max_memory_allocated / max_memory_reserved (GiB).
    On a CPU device both stay None."""

    device: torch.device | str = "cuda"
    allocated_gib: float | None = None
    reserved_gib: float | None = None

    def __enter__(self) -> CudaPeak:
        self._cuda = torch.device(self.device).type == "cuda" and torch.cuda.is_available()
        if self._cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def __exit__(self, *exc) -> None:
        if self._cuda:
            torch.cuda.synchronize(self.device)
            self.allocated_gib = torch.cuda.max_memory_allocated(self.device) / 2**30
            self.reserved_gib = torch.cuda.max_memory_reserved(self.device) / 2**30
