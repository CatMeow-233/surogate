"""The one device resolver of training, evaluation and the bench: "auto" is CUDA when it is available, else the CPU;
an explicit "cuda" without CUDA is refused here, with a clear message, instead of failing later inside torch."""

from __future__ import annotations

import torch


class DeviceError(ValueError):
    pass


def resolve_device(name: str | torch.device) -> torch.device:
    if isinstance(name, torch.device):
        name = str(name)
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.split(":")[0] == "cuda" and not torch.cuda.is_available():
        raise DeviceError(f"device {name!r} was asked for, but CUDA is not available")
    return torch.device(name)
