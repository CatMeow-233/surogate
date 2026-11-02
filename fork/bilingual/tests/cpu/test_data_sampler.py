"""I4-D6: EpochChunkSampler (IMPL-02, IMPL-07)."""

import numpy as np
import pytest

from surogate_bilingual.data import shards as SH
from surogate_bilingual.data.docs import Doc, Segment
from surogate_bilingual.data.sampler import EpochChunkSampler, SamplerError


def test_i4_d6_same_seed_same_batches():
    a, b, c = EpochChunkSampler(1000, 16, 5), EpochChunkSampler(1000, 16, 5), EpochChunkSampler(1000, 16, 6)
    batches_a = [a.next_batch(7) for _ in range(30)]
    batches_b = [b.next_batch(7) for _ in range(30)]
    assert all((x[0] == y[0]).all() and (x[1] == y[1]).all() for x, y in zip(batches_a, batches_b, strict=True))
    assert not all((x[0] == c.next_batch(7)[0]).all() for x in batches_a)


@pytest.mark.parametrize("k", [0, 1, 9, 13])
def test_i4_d6_resume_continues_exactly(k):
    whole = EpochChunkSampler(500, 8, 9)
    reference = [whole.next_batch(5) for _ in range(40)]
    first = EpochChunkSampler(500, 8, 9)
    for _ in range(k):
        first.next_batch(5)
    state = first.state_dict()
    resumed = EpochChunkSampler(500, 8, 9)
    resumed.load_state_dict(dict(state))  # a fresh sampler, restored before the first draw
    rest = [resumed.next_batch(5) for _ in range(40 - k)]
    for (s1, e1), (s2, e2) in zip(reference[k:], rest, strict=True):
        assert (s1 == s2).all() and (e1 == e2).all()
    with pytest.raises(SamplerError, match="seed"):
        EpochChunkSampler(500, 8, 10).load_state_dict(state)
    with pytest.raises(SamplerError, match="n_tokens"):
        EpochChunkSampler(501, 8, 9).load_state_dict(state)


@pytest.mark.parametrize("n_tokens,seq", [(1000, 16), (257, 8), (65, 64), (12345, 100)])
def test_i4_d6_every_covered_position_is_a_target_once_per_epoch(n_tokens, seq):
    s = EpochChunkSampler(n_tokens, seq, 1)
    for epoch in range(6):
        offset = s.epoch_offset(epoch)
        m = (n_tokens - 1 - offset) // seq  # chunks of this epoch (the count varies with the offset)
        starts, epochs = s.next_batch(m)
        assert (epochs == epoch).all()
        targets = np.concatenate([np.arange(x + 1, x + seq + 1) for x in starts])
        assert len(targets) == len(set(targets.tolist())) == m * seq  # non-overlapping, each once
        assert targets.min() == offset + 1 and targets.max() == offset + m * seq
        assert targets.max() <= n_tokens - 1
        assert 0 <= offset <= min(seq - 1, n_tokens - 1 - seq)
        assert sorted(starts.tolist()) == list(range(offset, offset + m * seq, seq))
        assert s.epoch_starts(epoch).tolist() == starts.tolist()
    offsets = {s.epoch_offset(e) for e in range(60)}
    room = min(seq - 1, n_tokens - 1 - seq)
    assert len(offsets) > 1 if room > 0 else offsets == {0}  # redrawn per epoch whenever there is room


def test_i4_d6_the_last_token_can_be_a_target(moe_pilot):
    n, seq = 101, 10  # 100 = 10 * seq, so an epoch with offset 0 (or any offset = 100 mod 10) covers token 100
    s = EpochChunkSampler(n, seq, 0)
    covering = [e for e in range(200) if (s.epoch_starts(e) + seq).max() == n - 1]
    assert covering, "no epoch makes the last token a target"
    # v1.0: starts from np.random.randint(0, len - seq - 1) (exclusive) never reach the last token
    import torch

    mm = np.arange(n, dtype=np.uint16)
    np.random.seed(0)
    highest = max(int(moe_pilot.random_batch(mm, 64, seq, torch.device("cpu"))[1].max()) for _ in range(200))
    assert highest == n - 2


def test_i4_d6_routes_and_loss_mask_slice_in_lockstep(tmp_path, data_pipeline):
    tok = data_pipeline["tok"]
    docs = []
    for i in range(30):
        docs.append(Doc(f"d{i}", "parallel", (Segment(text=f"源{i}。\n", lang="zh", route="zh", loss=False),
                                             Segment(text=f"target {i}.", lang="en", route="bridge"))))  # fmt: skip
    path = SH.write_shard(tmp_path / "s", docs, tok, ["zh", "en", "bridge"], "parallel_train", None,
                          tokenizer_sha256=data_pipeline["tokenizer_sha256"])  # fmt: skip
    r = SH.ShardReader(path, tokenizer_sha256=data_pipeline["tokenizer_sha256"], vocab_size=tok.get_vocab_size())
    sampler = EpochChunkSampler(len(r), 12, 4)
    starts, _ = sampler.next_batch(6)
    w = r.windows(starts, 12)
    for row, s in enumerate(starts.tolist()):
        assert w["inputs"][row].tolist() == r.tokens[s : s + 12].tolist()
        assert w["targets"][row].tolist() == r.tokens[s + 1 : s + 13].tolist()
        assert w["routes"][row].tolist() == r.routes[s : s + 12].tolist()  # the expert computing each input position
        assert w["loss_mask"][row].tolist() == r.loss_mask[s + 1 : s + 13].tolist()  # whether each target counts
    # the source half (route zh) is never a target; the target half (route bridge) and <eos> are
    assert set(r.loss_mask[r.routes == 0].tolist()) == {0} and set(r.loss_mask[r.routes == 2].tolist()) == {1}
    with pytest.raises(SH.ShardError, match="outside"):
        r.windows([len(r) - 12], 12)
