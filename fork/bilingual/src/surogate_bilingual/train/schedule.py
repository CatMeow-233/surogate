"""Learning-rate schedule, AdamW parameter groups and the decoupled decay to the seed (B-O9, B-O10; SCI-12, IMPL-09).

**Schedule.** Over N optimizer steps (0-based index s) with peak lr P, W = round(warmup_frac * N) warmup steps
(at least 1 when warmup_frac > 0) and floor m = min_lr_frac * P:

    s < W:   lr(s) = P * (s + 1) / W                                  (linear warmup; step 0 is not wasted at 0)
    s >= W:  lr(s) = m + (P - m) * (1 + cos(pi * (s - W) / (N - 1 - W))) / 2

so lr(W - 1) = lr(W) = P, the cosine midpoint is (P + m) / 2 and the last step runs at m (MS §8.1: warmup 2 %,
cosine to 10 % of the peak).

**Parameter groups.** The seed recipe decays every matrix but the embedding (which is also the tied head) with
`train.weight_decay`; norms, the embedding and the family-II bias b are never decayed. Every expert-type recipe
(expert, joint-wide, Bridge, target copy) puts its one trained expert in one group, decayed as
`train.expert_decay` says: "none" (wd 0, the B-O9 default), "to_zero" (AdamW's own decoupled decay with
`expert_decay_lambda`) or "to_seed".

**Decay to the seed.** `DecayToSeed` applies w <- w - lr * lambda * (w - w_seed) to every trained tensor before the
optimizer step, the same position and lr scaling as AdamW's built-in decoupled decay; it is not an L2 term under
Adam's preconditioner (SCI-12). With zero gradients Adam moves nothing, so w - w_seed shrinks by exactly
(1 - lr * lambda) per step (I5-T7).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

SCHEDULE_FORMAT = "bilingual-schedule/1"
NO_DECAY_NAMES = ("backbone.emb.weight",)


class ScheduleError(ValueError):
    pass


def warmup_steps(total: int, warmup_frac: float) -> int:
    if warmup_frac <= 0:
        return 0
    return min(total, max(1, round(warmup_frac * total)))


@dataclass(frozen=True)
class Schedule:
    total: int
    peak: float
    warmup_frac: float
    min_lr_frac: float

    def __post_init__(self):
        if self.total <= 0 or not self.peak > 0:
            raise ScheduleError(f"a schedule needs total > 0 steps and a positive peak lr: {self}")

    @property
    def warmup(self) -> int:
        return warmup_steps(self.total, self.warmup_frac)

    def lr(self, step: int) -> float:
        if not 0 <= step < self.total:
            raise ScheduleError(f"step {step} lies outside 0..{self.total - 1}")
        w = self.warmup
        if step < w:
            return self.peak * (step + 1) / w
        low = self.min_lr_frac * self.peak
        span = self.total - 1 - w
        progress = 1.0 if span <= 0 else (step - w) / span
        return low + (self.peak - low) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def state_dict(self, step: int) -> dict[str, Any]:
        return {"format": SCHEDULE_FORMAT, "step": int(step), "warmup": self.warmup, **asdict(self)}

    def check_state(self, state: Mapping[str, Any]) -> int:
        """The step a saved schedule state resumes at; refuses a state taken under another schedule."""
        if state.get("format") != SCHEDULE_FORMAT:
            raise ScheduleError(f"not a {SCHEDULE_FORMAT} state: {state.get('format')!r}")
        mine = {"warmup": self.warmup, **asdict(self)}
        theirs = {k: state.get(k) for k in mine}
        if theirs != mine:
            raise ScheduleError(f"the saved schedule {theirs} differs from this run's {mine}")
        return int(state["step"])


# --- parameter groups -----------------------------------------------------------------------------------------------


def seed_groups(named: Sequence[tuple[str, torch.nn.Parameter]], weight_decay: float) -> list[dict[str, Any]]:
    """Decay every matrix except the embedding / tied head; never decay norms, the embedding or the bias b."""
    decay = [(n, p) for n, p in named if p.dim() >= 2 and n not in NO_DECAY_NAMES]
    keep = [(n, p) for n, p in named if not (p.dim() >= 2 and n not in NO_DECAY_NAMES)]
    groups = [("decay", decay, weight_decay), ("no_decay", keep, 0.0)]
    return [
        {"params": [p for _, p in items], "names": [n for n, _ in items], "group": label, "weight_decay": wd}
        for label, items, wd in groups
        if items
    ]


def expert_groups(
    named: Sequence[tuple[str, torch.nn.Parameter]], expert_decay: str, lam: float
) -> list[dict[str, Any]]:
    wd = lam if expert_decay == "to_zero" else 0.0
    return [
        {
            "params": [p for _, p in named],
            "names": [n for n, _ in named],
            "group": f"expert:{expert_decay}",
            "weight_decay": wd,
        }
    ]


def adamw(groups: list[dict[str, Any]], lr: float, betas: tuple[float, float], eps: float) -> torch.optim.AdamW:
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


class DecayToSeed:
    """w <- w - lr * lam * (w - w_seed) for each (parameter, anchor) pair; anchors are detached copies."""

    def __init__(self, params: Sequence[torch.Tensor], anchors: Sequence[torch.Tensor], lam: float):
        if len(params) != len(anchors):
            raise ValueError("one anchor per parameter")
        for p, a in zip(params, anchors, strict=True):
            if p.shape != a.shape:
                raise ValueError(f"anchor shape {list(a.shape)} differs from parameter {list(p.shape)}")
        self.params = list(params)
        self.anchors = [
            a.detach().to(device=p.device, dtype=p.dtype).clone() for p, a in zip(params, anchors, strict=True)
        ]
        self.lam = float(lam)

    @torch.no_grad()
    def apply(self, lr: float) -> None:
        if self.lam == 0.0:
            return
        for p, a in zip(self.params, self.anchors, strict=True):
            p.sub_(p - a, alpha=lr * self.lam)
