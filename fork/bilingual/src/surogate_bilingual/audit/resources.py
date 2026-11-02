"""The resource report (`bilingual-resource/1`): the code-counted budget (`audit.budget.report`, passed in unchanged,
so it equals the I2 golden file) plus the measured resources MS §9.1 lists. On a host without CUDA, and until G0.5
and the runs measure them, every measured field is null with status "TBD"; a measurement document (a bench point,
or the compute/storage records of run.json, see `registry budget`) fills them. The report layer of `audit`: it
imports nothing from `eval`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

RESOURCE_FORMAT = "bilingual-resource/1"
TBD = "TBD"
# MS §9.1 resources: counted by code, or measured (TBD until G0.5 / the runs record them)
MEASURED_FIELDS = (
    "tokens_per_s",
    "step_s",
    "max_memory_allocated_gib",
    "max_memory_reserved_gib",
    "nvidia_smi_used_gib",
    "gpu_hours",
    "tuning_gpu_hours",
    "checkpoint_bytes",
)


def resource_report(
    profile: str = "pilot",
    measured: Mapping[str, Any] | None = None,
    *,
    budget: Mapping[str, Any],
    cuda: bool | None = None,
) -> dict[str, Any]:
    """The budget report of `profile` (from `audit.budget.report`) with the measured resources: null and "TBD"
    until a CUDA measurement fills them."""
    import torch

    cuda = torch.cuda.is_available() if cuda is None else cuda
    budget = dict(budget)
    if budget.get("profile") != profile:
        raise ValueError(f"the budget is of profile {budget.get('profile')!r}, not {profile!r}")
    fields = {k: None for k in MEASURED_FIELDS}
    status = TBD
    source = None
    if measured and cuda:
        fields.update({k: measured.get(k) for k in MEASURED_FIELDS})
        status = "measured" if all(fields[k] is not None for k in ("tokens_per_s", "max_memory_allocated_gib")) else TBD
        source = measured.get("source")
    return {
        "format": RESOURCE_FORMAT,
        "profile": profile,
        "budget": budget,
        "measured": {
            **fields,
            "status": status,
            "source": source,
            "note": "MS §9.1: tokens/s, memory peaks, GPU hours are measured on the RTX 5090 (G0.5); "
            "until then they are TBD",
        },
    }
