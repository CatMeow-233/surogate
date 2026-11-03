"""Learning-rate schedules on training-FLOP progress (spec §5.4; STUDY M5, M8).

**WSD** (warmup, stable, decay), in cumulative realized training FLOPs F, with integer bounds so that two runs whose
bounds agree compute bit-identical rates:

    F <  W:          peak * F / W                   (linear warmup from 0; the first step runs at lr 0 and only
                                                     fills Adam's moments)
    W <= F < D:      peak
    D <= F < E:      decay from peak to floor * peak, linear or cosine in (F - D) / (E - D)
    F >= E:          floor * peak

A run with F_target has W = round(warmup * F_target), D = round(decay_start * F_target), E = F_target. The decay
branch at budget point e keeps W and has E_e = round(e * F_target), D_e = round(decay_start * E_e): it leaves the
run's trajectory exactly at its stable checkpoint (the state after the first step that reaches D_e), so a branch
equals a run whose F_target is E_e with the same absolute warmup (tests/cpu/test_train.py). The rate of a step is
read at its F before the step.

**Cosine** (`cosine_at`): warmup then a cosine to floor * peak over the rest, the production schedule of the
bilingual loop (surogate_bilingual.train.schedule) on FLOP progress, for the twin gates TW-2 and TW-3.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

SHAPES = ("linear", "cosine")


@dataclass(frozen=True)
class WSD:
    peak: float
    warmup_flops: int
    decay_start_flops: int
    end_flops: int
    floor: float = 0.1
    shape: str = "linear"

    def __post_init__(self):
        if not self.peak > 0:
            raise ValueError(f"the peak learning rate must be positive, got {self.peak}")
        if not 0 <= self.warmup_flops <= self.decay_start_flops < self.end_flops:
            raise ValueError(f"WSD needs 0 <= warmup <= decay start < end, got {self.to_json()}")
        if not 0 <= self.floor <= 1:
            raise ValueError(f"the floor is a fraction of the peak in [0, 1], got {self.floor}")
        if self.shape not in SHAPES:
            raise ValueError(f"the decay shape is one of {SHAPES}, got {self.shape!r}")

    def lr(self, flops: int) -> float:
        if flops < self.warmup_flops:
            return self.peak * flops / self.warmup_flops
        if flops < self.decay_start_flops:
            return self.peak
        if flops >= self.end_flops:
            return self.peak * self.floor
        frac = (flops - self.decay_start_flops) / (self.end_flops - self.decay_start_flops)
        low = self.peak * self.floor
        if self.shape == "linear":
            return self.peak - (self.peak - low) * frac
        return low + (self.peak - low) * (1 + math.cos(math.pi * frac)) / 2

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def wsd(
    f_target: int,
    *,
    peak: float,
    warmup: float = 0.02,
    decay_start: float = 0.9,
    floor: float = 0.1,
    shape: str = "linear",
    endpoint: float = 1.0,
) -> WSD:
    """The WSD of a run with F_target (endpoint 1) or of its decay branch at budget point `endpoint`."""
    if not 0 < endpoint <= 1:
        raise ValueError(f"a budget point lies in (0, 1], got {endpoint}")
    end = f_target if endpoint == 1 else round(endpoint * f_target)
    return WSD(peak, round(warmup * f_target), round(decay_start * end), end, floor, shape)


def lr_at(
    progress: float,
    *,
    peak: float,
    warmup: float = 0.02,
    decay_start: float = 0.9,
    floor: float = 0.1,
    shape: str = "linear",
) -> float:
    """WSD on progress pi = F / F_target (the same rates as `wsd`, in real numbers)."""
    if progress < warmup:
        return peak * progress / warmup
    if progress < decay_start:
        return peak
    if progress >= 1:
        return peak * floor
    frac = (progress - decay_start) / (1 - decay_start)
    if shape == "linear":
        return peak - peak * (1 - floor) * frac
    if shape == "cosine":
        return peak * floor + peak * (1 - floor) * (1 + math.cos(math.pi * frac)) / 2
    raise ValueError(f"the decay shape is one of {SHAPES}, got {shape!r}")


def cosine_at(progress: float, *, peak: float, warmup: float = 0.02, floor: float = 0.1) -> float:
    """Warmup to the peak, then a cosine to floor * peak at progress 1."""
    if progress < warmup:
        return peak * progress / warmup
    if progress >= 1:
        return peak * floor
    frac = (progress - warmup) / (1 - warmup)
    return peak * floor + peak * (1 - floor) * (1 + math.cos(math.pi * frac)) / 2
