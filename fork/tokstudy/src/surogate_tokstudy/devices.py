"""Device choice, precision and the environment record of a run."""

from __future__ import annotations

import platform
import sys
from typing import Any

import torch

PRECISIONS = {"bf16": torch.float32, "fp32": torch.float32, "fp64": torch.float64}  # weight dtype per precision


def device(name: str | None = None) -> torch.device:
    """The named device; for "auto" or None, CUDA when available, else the CPU."""
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def autocast(dev: torch.device, precision: str):
    """bf16: autocast to bfloat16 over fp32 weights (fp32 master weights); fp32 and fp64: no autocast."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
    return torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=precision == "bf16")


def environment(dev: torch.device) -> dict[str, Any]:
    import numpy
    import tokenizers

    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "tokenizers": tokenizers.__version__,
        "device": str(dev),
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }
    if dev.type == "cuda":
        out |= {"cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(dev),
                "capability": list(torch.cuda.get_device_capability(dev))}  # fmt: skip
    return out


def peak_memory(dev: torch.device) -> int | None:
    return int(torch.cuda.max_memory_allocated(dev)) if dev.type == "cuda" else None
