"""Ready batches are packed immediately; an idle packer backs off and stays cancellable."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from surogate.grpo import packer as packer_module
from surogate.grpo.packer import SinglePacker
from surogate.grpo.transport.types import TrainingBatch, TrainingSample


def _batch():
    sample = TrainingSample(
        prompt_ids=[10, 11],
        prompt_mask=[False, False],
        completion_ids=[12, 13],
        completion_mask=[True, True],
        completion_logprobs=[-0.5, -0.7],
        completion_temperatures=[1.0, 1.0],
        advantage=1.0,
    )
    return TrainingBatch(examples=[sample], step=3, run_idx=0)


def _packer(receives):
    packer = SinglePacker.__new__(SinglePacker)
    packer.multi_run_manager = SimpleNamespace(
        discover_runs=Mock(),
        ready_to_update=[False],
        progress=[SimpleNamespace(step=3)],
        max_runs=1,
    )
    packer.receiver = SimpleNamespace(receive=Mock(side_effect=receives))
    packer.sender = SimpleNamespace(send=Mock())
    packer.seq_len = 8
    packer.pad_to_multiple_of = 1
    packer.dp_world_size = 1
    return packer


def test_ready_batch_is_packed_without_sleep(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(packer_module, "time", SimpleNamespace(sleep=sleep))
    packer = _packer([[_batch()]])

    packer.pack()

    sleep.assert_not_called()
    packer.receiver.receive.assert_called_once()
    assert packer.multi_run_manager.progress[0].step == 4
    assert packer.multi_run_manager.ready_to_update == [True]
    (grid,) = packer.sender.send.call_args.args
    assert len(grid) == 1 and len(grid[0]) == 1
    assert grid[0][0].input_ids == [10, 11, 12, 13]
    assert grid[0][0].loss_mask == [False, False, True, True]


def test_only_empty_receives_back_off(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(packer_module, "time", SimpleNamespace(sleep=sleep))
    packer = _packer([[], [], [_batch()]])
    packer.check_cancelled = Mock()

    packer.pack()

    assert sleep.call_count == 2
    assert all(call.args == (0.2,) for call in sleep.call_args_list)
    assert packer.check_cancelled.call_count == 3
    assert packer.multi_run_manager.discover_runs.call_count == 3
    assert packer.multi_run_manager.progress[0].step == 4
    packer.sender.send.assert_called_once()


@pytest.mark.parametrize("cancel_after_idle", [False, True])
def test_cancellation_does_not_publish_a_batch(monkeypatch, cancel_after_idle):
    sleep = Mock()
    monkeypatch.setattr(packer_module, "time", SimpleNamespace(sleep=sleep))
    packer = _packer([[], [_batch()]])
    packer.check_cancelled = Mock(
        side_effect=([None, InterruptedError("stopped")] if cancel_after_idle else [InterruptedError("stopped")])
    )

    with pytest.raises(InterruptedError, match="stopped"):
        packer.pack()

    assert sleep.call_count == int(cancel_after_idle)
    assert packer.receiver.receive.call_count == int(cancel_after_idle)
    assert packer.multi_run_manager.progress[0].step == 3
    assert packer.multi_run_manager.ready_to_update == [False]
    packer.sender.send.assert_not_called()
