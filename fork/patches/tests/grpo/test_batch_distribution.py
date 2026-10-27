"""Packing preserves sample order and equal per-rank work, including padding."""

import msgspec
import pytest

from surogate.grpo.batch import prepare_batch
from surogate.grpo.transport.types import TrainingSample


def _samples():
    samples = []
    for index, length in enumerate([3, 5, 2, 4, 6]):
        samples.append(
            TrainingSample(
                prompt_ids=[100 + index],
                prompt_mask=[False],
                completion_ids=[200 + index] * (length - 1),
                completion_mask=[True] * (length - 1),
                completion_logprobs=[-0.5] * (length - 1),
                completion_temperatures=[1.0] * (length - 1),
                advantage=float(index + 1),
            )
        )
    return samples


def _prepare(workers, samples):
    return prepare_batch(
        samples,
        seq_len=8,
        num_train_workers=workers,
        idxs=[0] * len(samples),
        num_loras=1,
        pad_to_multiple_of=4,
        single_sample_bins=True,
    )


@pytest.mark.parametrize("workers", [1, 2, 3, 8])
def test_equal_work_and_sample_order_across_ranks(workers):
    grid = _prepare(workers, _samples())
    assert len(grid) == workers
    assert all(len(rank) == -(-5 // workers) for rank in grid)
    flattened = [batch for rank in grid for batch in rank]
    real_batches = [batch for batch in flattened if any(batch.loss_mask)]
    assert [batch.input_ids[0] for batch in real_batches] == [104, 101, 103, 100, 102]
    expected = _prepare(1, _samples())[0]
    assert [msgspec.msgpack.encode(batch) for batch in real_batches] == [
        msgspec.msgpack.encode(batch) for batch in expected
    ]
    for batch in flattened[len(real_batches) :]:
        assert not any(batch.loss_mask)
        assert not any(batch.advantages)
        assert batch.input_ids == expected[0].input_ids
    assert all(len(batch.input_ids) % 4 == 0 for batch in flattened)


@pytest.mark.parametrize("workers", [1, 3])
def test_empty_rollouts_produce_an_empty_batch_list_for_every_rank(workers):
    assert _prepare(workers, []) == [[] for _ in range(workers)]
