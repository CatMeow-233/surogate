"""`EpochChunkSampler`: deterministic, resumable training windows over a packed shard (IMPL-02, IMPL-07).

Each epoch e draws an offset o_e in [0, r] with r = min(seq - 1, n_tokens - 1 - seq) and cuts the shard into the
m_e = (n_tokens - 1 - o_e) // seq non-overlapping chunks that start at o_e + i * seq. Chunk i takes inputs
tokens[s : s + seq] and targets tokens[s + 1 : s + seq + 1], so within an epoch every position of the covered
range [o_e + 1, o_e + 1 + m_e * seq) is a target exactly once. The chunks are visited in a random order. Offsets
and orders come from numpy Generators seeded with (seed, epoch), so they depend on nothing but the seed, and the
whole state is (epoch, index).

The last token of the shard is a target in every epoch whose offset is congruent to n_tokens - 1 modulo seq. v1.0
drew starts from `np.random.randint(0, len - seq - 1)`, whose exclusive upper bound kept the last token from ever
being a target, and it re-seeded numpy on every start, so a resumed run replayed its first batches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

STATE_FORMAT = "bilingual-sampler/1"


class SamplerError(ValueError):
    pass


class EpochChunkSampler:
    def __init__(self, n_tokens: int, seq: int, seed: int):
        if seq <= 0 or n_tokens < seq + 1:
            raise SamplerError(f"a shard of {n_tokens} tokens holds no window of {seq} inputs and {seq} targets")
        self.n_tokens, self.seq, self.seed = int(n_tokens), int(seq), int(seed)
        self.epoch, self.index = 0, 0
        self._plan_epoch()

    def _rng(self, epoch: int, stream: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, epoch, stream])

    def epoch_offset(self, epoch: int) -> int:
        r = min(self.seq - 1, self.n_tokens - 1 - self.seq)
        return int(self._rng(epoch, 0).integers(0, r + 1))

    def _plan_epoch(self) -> None:
        self.offset = self.epoch_offset(self.epoch)
        m = (self.n_tokens - 1 - self.offset) // self.seq
        self.order = self._rng(self.epoch, 1).permutation(m)

    @property
    def chunks_per_epoch(self) -> int:
        return len(self.order)

    def epoch_starts(self, epoch: int) -> np.ndarray:
        """Every window start of an epoch, in visiting order (does not move the cursor)."""
        offset = self.epoch_offset(epoch)
        m = (self.n_tokens - 1 - offset) // self.seq
        return offset + self._rng(epoch, 1).permutation(m) * self.seq

    def next_batch(self, batch: int) -> tuple[np.ndarray, np.ndarray]:
        """(window starts [batch], epoch ids [batch]); a batch may cross an epoch boundary."""
        starts = np.empty(batch, dtype=np.int64)
        epochs = np.empty(batch, dtype=np.int64)
        for i in range(batch):
            if self.index >= len(self.order):
                self.epoch += 1
                self.index = 0
                self._plan_epoch()
            starts[i] = self.offset + int(self.order[self.index]) * self.seq
            epochs[i] = self.epoch
            self.index += 1
        return starts, epochs

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": STATE_FORMAT,
            "n_tokens": self.n_tokens,
            "seq": self.seq,
            "seed": self.seed,
            "epoch": self.epoch,
            "index": self.index,
            "offset": self.offset,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format") != STATE_FORMAT:
            raise SamplerError(f"not a {STATE_FORMAT} state: {state.get('format')!r}")
        for key in ("n_tokens", "seq", "seed"):
            if int(state[key]) != getattr(self, key):
                raise SamplerError(f"sampler state was taken with {key} = {state[key]}, this sampler has "
                                   f"{getattr(self, key)}")  # fmt: skip
        self.epoch, self.index = int(state["epoch"]), int(state["index"])
        self._plan_epoch()
        if int(state["offset"]) != self.offset or not 0 <= self.index <= len(self.order):
            raise SamplerError("sampler state does not match this sampler's epoch plan")
