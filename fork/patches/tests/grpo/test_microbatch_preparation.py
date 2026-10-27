"""Packed boundaries and read-only input views preserve GRPO gradients and ownership."""

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize(
    "positions,expected",
    [
        ([], [(0, 0)]),
        ([0], [(0, 1)]),
        ([0, 1, 2, 0], [(0, 3), (3, 4)]),
        ([0, 1, 2, 0, 0, 1], [(0, 3), (3, 4), (4, 6)]),
        ([0, 0, 0, 0], [(0, 1), (1, 2), (2, 3), (3, 4)]),
        ([4, 5, 8, 9, 3, 4], [(0, 2), (2, 4), (4, 6)]),
        ([3, 2, 1, 0], [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ],
)
def test_sample_boundaries_keep_unit_stride_contract(trainer_module, dtype, positions, expected):
    array = np.asarray(positions, dtype=dtype)
    array.setflags(write=False)
    assert trainer_module._find_sample_boundaries(array) == expected


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_position_rollover_keeps_numpy_integer_difference_semantics(trainer_module, dtype):
    # The old loop subtracted NumPy scalars in their original dtype. Widening
    # int32 here would change its extreme rollover behavior independently of
    # the vectorization (real text positions stay far below these limits).
    limit = np.iinfo(dtype)
    positions = np.array([limit.max - 1, limit.max, limit.min, limit.min + 1, 0, 1], dtype=dtype)
    with np.errstate(over="ignore"):
        assert trainer_module._find_sample_boundaries(positions) == [(0, 4), (4, 6)]


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("strided", [False, True])
def test_random_documents_keep_per_sample_loss_and_shift(trainer_module, dtype, strided):
    rng = np.random.default_rng(71)
    loss_config = SimpleNamespace(
        ratio_clip=7.389056,
        ipo_mask_low=0.2,
        ipo_mask_high=0.2,
        adv_tau=1.0,
        teacher_tau=0.1,
        kl_tau=0.001,
        opd_tau=0.0,
        opd_beta=1.0,
        replay_tau=0.0,
    )
    for _ in range(12):
        lengths = [1, 1, *rng.integers(1, 65, size=30).tolist(), 1]
        positions = np.concatenate([np.arange(length, dtype=dtype) for length in lengths])
        offsets = np.cumsum([0, *lengths]).tolist()
        expected_ranges = list(zip(offsets[:-1], offsets[1:], strict=True))
        if strided:
            backing = np.empty(len(positions) * 2, dtype=dtype)
            backing[::2] = positions
            positions = backing[::2]
            assert not positions.flags.c_contiguous
        positions.setflags(write=False)
        ranges = trainer_module._find_sample_boundaries(positions)
        assert ranges == expected_ranges

        n = len(positions)
        trainer_lp = rng.uniform(-6, -2, n).astype(np.float32)
        inference_lp = trainer_lp + rng.uniform(-0.2, 0.2, n).astype(np.float32)
        teacher_lp = trainer_lp + rng.uniform(-0.4, 0.4, n).astype(np.float32)
        advantages = rng.normal(size=n).astype(np.float32)
        loss_mask = rng.random(n) > 0.3
        args = (trainer_lp, inference_lp, advantages, loss_mask, loss_config)
        grads, metrics = trainer_module.compute_grpo_per_token_grads(
            *args,
            sample_ranges=ranges,
            teacher_logprobs=teacher_lp,
        )
        expected, expected_metrics = trainer_module.compute_grpo_per_token_grads(
            *args,
            sample_ranges=expected_ranges,
            teacher_logprobs=teacher_lp,
        )
        np.testing.assert_array_equal(grads, expected)
        assert metrics == pytest.approx(expected_metrics)

        # Construct target slots independently from each source document. A
        # one-token document never seeds its neighbour's last target slot.
        expected_shift = np.zeros_like(grads)
        expected_unshift = grads.copy()
        for start, end in expected_ranges:
            expected_shift[start : end - 1] = grads[start + 1 : end]
            expected_unshift[start] = 0
        shifted = trainer_module.shift_to_target(grads, ranges)
        np.testing.assert_array_equal(shifted, expected_shift)
        np.testing.assert_array_equal(trainer_module.unshift_to_logical(shifted, ranges), expected_unshift)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("optional_fields", [False, True])
def test_preparation_keeps_inputs_unchanged_and_padded_outputs_independent(
    trainer_module,
    strided,
    optional_fields,
):
    values = {
        "input_ids": np.arange(10, 17, dtype=np.int32),
        "position_ids": np.array([0, 1, 2, 0, 0, 1, 2], dtype=np.int32),
        "targets": np.arange(11, 18, dtype=np.int32),
        "loss_mask": np.array([False, True, True, False, False, True, True]),
        "temperatures": np.full(7, 0.8, dtype=np.float32),
        "inference_logprobs": np.linspace(-1, -2, 7, dtype=np.float32),
        "advantages": np.linspace(-0.5, 0.5, 7, dtype=np.float32),
    }
    if optional_fields:
        values["teacher_logprobs"] = np.linspace(-2, -3, 7, dtype=np.float32)
        values["turn_ids"] = np.array([-1, 0, 1, -1, -1, 0, 1], dtype=np.int32)
    batch = {}
    for key, array in values.items():
        if strided:
            backing = np.empty(array.size * 2, dtype=array.dtype)
            backing[::2] = array
            array = backing[::2]
        array.setflags(write=False)
        batch[key] = array.reshape(1, -1)
    batch.setdefault("teacher_logprobs", None)

    trainer = trainer_module.GRPOTrainer.__new__(trainer_module.GRPOTrainer)
    prepared = trainer._prepare_micro_batch(batch, seq_len=10)
    assert prepared["sample_ranges"] == [(0, 3), (3, 4), (4, 7)]
    np.testing.assert_array_equal(prepared["input"], [[10, 11, 12, 13, 14, 15, 16, 0, 0, 0]])
    np.testing.assert_array_equal(prepared["position_ids"], [[0, 1, 2, 0, 0, 1, 2, 3, 4, 5]])
    np.testing.assert_array_equal(prepared["targets"], [[11, 12, -100, -100, 15, 16, -100, -100, -100, -100]])
    for output, source in (
        ("temperatures", "temperatures"),
        ("inference", "inference_logprobs"),
        ("advantages", "advantages"),
        ("loss_mask", "loss_mask"),
        ("teacher", "teacher_logprobs"),
        ("turn_ids", "turn_ids"),
    ):
        if source not in values:
            assert prepared[output] is None
            continue
        np.testing.assert_array_equal(prepared[output].ravel()[:7], values[source])
    # Outputs are still private writable storage for the native call; ravel
    # only removed copies of inputs used for reads during preparation.
    for value in prepared.values():
        if isinstance(value, np.ndarray):
            assert value.flags.writeable
            value.fill(0)
    for key, original in values.items():
        np.testing.assert_array_equal(batch[key].ravel(), original)
