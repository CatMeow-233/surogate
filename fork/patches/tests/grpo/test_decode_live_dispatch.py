"""Decode rounds sized by live sessions (ledger topic T12).

The scheduler's clock is frozen, so its 1 ms gather deadline never expires and
a round can start only through the round-size condition itself. Upstream waits
for `max_batch` pending steps; with SUROGATE_GRPO_LIVE_DISPATCH=1 it waits for
one step from every live session.
"""

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from surogate.grpo import decode_scheduler
from surogate.grpo.decode_scheduler import DecodeScheduler


class Trainer:
    batch_size, seq_length = 1, 64

    def __init__(self):
        self.rounds = []

    def decode_batch_logits(self, ids, tokens, offsets, resets):
        self.rounds.append(sorted(ids.tolist()))
        return np.zeros((len(ids), 4), dtype=np.float32)

    def release_decode_sessions(self, ids):
        pass


@pytest.fixture
def frozen_clock(monkeypatch):
    monkeypatch.setattr(decode_scheduler, "time", SimpleNamespace(monotonic=lambda: 0.0))


def scheduler(trainer, max_batch=8):
    return DecodeScheduler(trainer, max_batch=max_batch, prefill_chunk=8, token_budget=512, prefix_entries=0)


@contextmanager
def workers(sched, count):
    """A thread pool that cannot hang the test: on failure the scheduler fails the steps still waiting."""
    pool = ThreadPoolExecutor(count)
    try:
        yield pool
    except BaseException:
        sched.close()
        raise
    finally:
        pool.shutdown(wait=True)


def step_all(sched, sessions, reset, timeout):
    with workers(sched, len(sessions)) as pool:
        futures = [pool.submit(sched.step, s, [3], reset) for s in sessions]
        return [f.result(timeout=timeout) for f in futures]


def test_a_round_starts_once_every_live_session_has_a_step(monkeypatch, frozen_clock):
    """Fails on upstream's scheduler: two live sessions never make eight pending steps."""
    monkeypatch.setenv("SUROGATE_GRPO_LIVE_DISPATCH", "1")
    trainer = Trainer()
    sched = scheduler(trainer)
    try:
        sessions = [sched.new_session() for _ in range(2)]
        for reset in (True, False, False):
            step_all(sched, sessions, reset, timeout=5)
        assert trainer.rounds == [sorted(sessions)] * 3
        # A released session no longer holds rounds back.
        sched.release(sessions[0])
        step_all(sched, sessions[1:], False, timeout=5)
        assert trainer.rounds[-1] == [sessions[1]]
    finally:
        sched.close()


def test_rounds_still_gather_steps_from_sessions_still_thinking(monkeypatch, frozen_clock):
    monkeypatch.setenv("SUROGATE_GRPO_LIVE_DISPATCH", "1")
    trainer = Trainer()
    sched = scheduler(trainer)
    try:
        first, second = sched.new_session(), sched.new_session()
        with workers(sched, 2) as pool:
            early = pool.submit(sched.step, first, [3], True)
            time.sleep(0.2)
            assert not early.done() and not trainer.rounds  # waits for the second live session
            late = pool.submit(sched.step, second, [3], True)
            early.result(timeout=5), late.result(timeout=5)
        assert trainer.rounds == [sorted((first, second))]
    finally:
        sched.close()


def test_off_by_default_rounds_wait_for_max_batch(monkeypatch, frozen_clock):
    monkeypatch.delenv("SUROGATE_GRPO_LIVE_DISPATCH", raising=False)
    trainer = Trainer()
    sched = scheduler(trainer, max_batch=3)
    try:
        sessions = [sched.new_session() for _ in range(2)]
        with workers(sched, 3) as pool:
            futures = [pool.submit(sched.step, s, [3], True) for s in sessions]
            with pytest.raises(TimeoutError):
                futures[0].result(timeout=0.3)
            assert not trainer.rounds
            third = sched.new_session()
            futures.append(pool.submit(sched.step, third, [3], True))
            for future in futures:
                future.result(timeout=5)
        assert trainer.rounds == [sorted(sessions + [third])]
    finally:
        sched.close()


def test_the_switch_accepts_only_0_or_1(monkeypatch):
    monkeypatch.setenv("SUROGATE_GRPO_LIVE_DISPATCH", "yes")
    with pytest.raises(ValueError, match="SUROGATE_GRPO_LIVE_DISPATCH"):
        scheduler(Trainer())
