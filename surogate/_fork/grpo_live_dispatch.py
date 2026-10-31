"""Decode rounds sized by live sessions (ledger topic T12).

`DecodeScheduler._run` gathers submissions for up to 1 ms, until `max_batch`
of them are pending. Once fewer than `max_batch` sessions are live (the tail of
every rollout phase, or a server configured above its load) that count is never
reached, so every round waits the whole millisecond. A session is live from
`new_session` to `release`; each one has at most one step outstanding, and the
request handlers create a session before its first step. With
`SUROGATE_GRPO_LIVE_DISPATCH=1` a round starts as soon as every live session
has a step pending: `min(max_batch, live)`, at least 1. A round never waits
longer than it does today (the deadline and the early exits on releases and
close are unchanged); it only stops waiting for steps that cannot arrive,
except a step from a session created during the wait, which joins the next
round.

Off by default. Its effect on results is the kind T11 has on this path: rounds
form at different times, so rounds and their batch composition differ, and
batched decode kernels need not give bitwise-equal results for different
batches; upstream's round composition already depends on timing. The difference
is what changes: T11 only removes host work whose result is discarded, and its
timing effect is that of any host speedup, whereas this switch changes the rule
that decides when a round starts, a scheduling change that stays opt-in until it
is measured on the target GPU. The scheduler reads the switch once, when it is
constructed.
"""

from __future__ import annotations

import os

ENV = "SUROGATE_GRPO_LIVE_DISPATCH"


def live_dispatch_enabled() -> bool:
    value = os.environ.get(ENV, "0").strip() or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{ENV} must be 0 or 1, not {value!r}")
    return value == "1"


class LiveSessions:
    """Session count, under the scheduler's condition lock, and the round size it implies."""

    __slots__ = ("enabled", "count")

    def __init__(self):
        self.enabled, self.count = live_dispatch_enabled(), 0

    def opened(self):
        self.count += 1

    def released(self):
        self.count = max(0, self.count - 1)

    def round_size(self, max_batch: int) -> int:
        return min(max_batch, max(1, self.count)) if self.enabled else max_batch
