"""The trainer's local handoff preserves serialized batches and external transport semantics."""

import socket
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from surogate.grpo import data as data_module
from surogate.grpo import packer as packer_module
from surogate.grpo.data import GRPODataLoader, InProcessMicroBatchBuffer
from surogate.grpo.packer import SinglePacker, setup_grpo_packer
from surogate.grpo.transport import filesystem as filesystem_module
from surogate.grpo.transport import setup_training_batch_sender
from surogate.grpo.transport import zmq as zmq_module
from surogate.grpo.transport.types import TrainingBatch, TrainingSample
from tests.grpo.test_trainer_cycle_timing import trainer_module  # noqa: F401


def _samples(step, *, optional_fields=True):
    def sample(prompt, completion, *, turns=None, routed=False):
        n = len(prompt) + len(completion)
        rows = np.arange(n * 2 * 2, dtype="<i2").reshape(n, 2, 2) if routed else None
        return TrainingSample(
            prompt_ids=[token + step * 100 for token in prompt],
            prompt_mask=[False] * len(prompt),
            completion_ids=[token + step * 100 for token in completion],
            completion_mask=[True] * len(completion),
            completion_logprobs=[-0.5 - i * 0.1 for i in range(len(completion))],
            completion_temperatures=[0.8] * len(completion),
            teacher_logprobs=[-1.0 - i * 0.1 for i in range(n)],
            advantage=float(step),
            reward=1.0,
            completion_turn_ids=turns,
            routed_experts=rows.tobytes() if routed else None,
            routed_experts_dims=[2, 2] if routed else None,
        )

    # The first two samples pack together (one has routing, the other does
    # not), so missing turn/routing spans and the final padding are exercised.
    samples = [
        sample([1, 2], [3, 4], turns=[0, 1], routed=True),
        sample([5], [6, 7]),
        sample([8], [9], turns=[0], routed=True),
    ]
    if not optional_fields:
        for sample in samples:
            sample.teacher_logprobs = None
            sample.completion_turn_ids = None
            sample.routed_experts = None
            sample.routed_experts_dims = None
    return samples


def _manager(output_dir):
    return SimpleNamespace(
        output_dir=output_dir,
        max_runs=1,
        discover_runs=Mock(),
        progress=[SimpleNamespace(step=9)],  # The orchestrator is ahead on resume.
        ready_to_update=[False],
        used_idxs=[0],
        idx_2_id={0: "run_default"},
        get_run_dir=lambda _: output_dir / "run_default",
    )


def _wire_manager(monkeypatch, manager):
    monkeypatch.setattr(packer_module, "get_multi_run_manager", lambda: manager)
    monkeypatch.setattr(filesystem_module, "get_multi_run_manager", lambda: manager)
    monkeypatch.setattr(zmq_module, "get_multi_run_manager", lambda: manager)


def _assert_arrays_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual:
        if expected[key] is None:
            assert actual[key] is None, key
        else:
            assert actual[key].dtype == expected[key].dtype, key
            assert actual[key].shape == expected[key].shape, key
            assert actual[key].flags.c_contiguous, key
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


def _filesystem_pipeline(monkeypatch, output_dir, *, direct, single_sample_bins, optional_fields):
    manager = _manager(output_dir)
    _wire_manager(monkeypatch, manager)
    config = SimpleNamespace(type="filesystem")
    channel = InProcessMicroBatchBuffer(output_dir, start_step=6) if direct else None
    if direct:
        monkeypatch.setattr(packer_module, "setup_micro_batch_sender", Mock(side_effect=AssertionError("disk hop")))
        monkeypatch.setattr(data_module, "setup_micro_batch_receiver", Mock(side_effect=AssertionError("disk hop")))
    packer = setup_grpo_packer(
        1,
        8,
        4,
        None,
        config,
        start_step=6,
        micro_batch_sender=channel,
    )
    packer.single_sample_bins = single_sample_bins
    loader = GRPODataLoader(
        str(output_dir),
        0,
        6,
        config,
        micro_batch_receiver=channel,
    )
    sender = setup_training_batch_sender(manager.get_run_dir(0), config)
    for step in (5, 6, 7):
        sender.send(TrainingBatch(_samples(step, optional_fields=optional_fields), step))

    received = []
    for step in (6, 7):
        packer.pack()
        assert manager.ready_to_update == [True]
        loader.wait_for_batch()
        received.append(loader.get_batch())
        assert packer.receiver._received_steps[0] == step + 1
        if channel is not None:
            assert channel.current_step == step + 1
            assert not channel.can_receive()
        # The trainer's weight publication releases the next policy batch.
        manager.ready_to_update[0] = False

    assert manager.progress[0].step == 11
    assert len(list((manager.get_run_dir(0) / "rollouts").glob("step_*/rollouts.bin"))) == 3
    if direct:
        assert not (output_dir / "rollouts").exists()
    else:
        assert len(list((output_dir / "rollouts").glob("step_*/rank_0.bin"))) == 2
    return received


@pytest.mark.parametrize("single_sample_bins", [False, True])
@pytest.mark.parametrize("optional_fields", [False, True])
def test_direct_batches_match_filesystem_handoff_on_resume(monkeypatch, tmp_path, single_sample_bins, optional_fields):
    with monkeypatch.context() as legacy_patch:
        legacy = _filesystem_pipeline(
            legacy_patch,
            tmp_path / "legacy",
            direct=False,
            single_sample_bins=single_sample_bins,
            optional_fields=optional_fields,
        )
    with monkeypatch.context() as direct_patch:
        direct = _filesystem_pipeline(
            direct_patch,
            tmp_path / "direct",
            direct=True,
            single_sample_bins=single_sample_bins,
            optional_fields=optional_fields,
        )

    for actual_step, expected_step in zip(direct, legacy, strict=True):
        assert len(actual_step) == len(expected_step) == (3 if single_sample_bins else 2)
        for actual, expected in zip(actual_step, expected_step, strict=True):
            _assert_arrays_equal(actual, expected)
    assert direct[0][0]["input_ids"][0, 0] == 601  # Not stale step 5 or orchestrator step 9.
    assert direct[1][0]["input_ids"][0, 0] == 701
    # Routing views retained from step 6 must survive slot reuse for step 7.
    for field in ("routed_experts", "turn_ids", "teacher_logprobs"):
        assert (direct[0][0][field] is not None) is optional_fields


@pytest.mark.parametrize("empty", [False, True])
def test_local_slot_prevents_overwrite_and_duplicate_consumption(tmp_path, empty):
    channel = InProcessMicroBatchBuffer(tmp_path, start_step=6)
    payload = [] if empty else [object()]
    assert not channel.can_receive()
    for operation in (channel.wait, channel.receive):
        with pytest.raises(RuntimeError, match="pack step 6"):
            operation()
    assert channel.current_step == 6

    channel.send([payload])
    channel.wait()
    assert channel.can_receive()  # An empty packed batch still occupies the slot.
    with pytest.raises(RuntimeError, match="step 6.*not been consumed"):
        channel.send([[]])
    assert channel.receive() is payload
    assert channel.current_step == 7
    with pytest.raises(RuntimeError, match="pack step 7"):
        channel.receive()
    assert channel.current_step == 7
    channel.send([[]])
    assert channel.receive() == []
    assert channel.current_step == 8


def test_local_slot_rejects_multiple_workers(tmp_path):
    channel = InProcessMicroBatchBuffer(tmp_path)
    with pytest.raises(ValueError, match="one data worker"):
        channel.send([[], []])
    assert not channel.can_receive()


@pytest.mark.parametrize("workers,runs", [(2, 1), (1, 2)])
def test_local_packer_rejects_unsupported_topology_before_opening_transport(monkeypatch, tmp_path, workers, runs):
    manager = _manager(tmp_path)
    manager.max_runs = runs
    _wire_manager(monkeypatch, manager)
    receive = Mock(side_effect=AssertionError("should not open transport"))
    monkeypatch.setattr(packer_module, "setup_training_batch_receiver", receive)
    channel = InProcessMicroBatchBuffer(tmp_path)
    with pytest.raises(ValueError, match="one data worker and one run"):
        setup_grpo_packer(workers, 8, 4, None, SimpleNamespace(type="filesystem"), micro_batch_sender=channel)
    receive.assert_not_called()


def test_local_loader_rejects_wrong_rank_before_opening_transport(monkeypatch, tmp_path):
    receive = Mock(side_effect=AssertionError("should not open transport"))
    monkeypatch.setattr(data_module, "setup_micro_batch_receiver", receive)
    with pytest.raises(ValueError, match="receiver rank"):
        GRPODataLoader(
            str(tmp_path),
            1,
            0,
            SimpleNamespace(type="filesystem"),
            micro_batch_receiver=InProcessMicroBatchBuffer(tmp_path),
        )
    receive.assert_not_called()


def test_local_endpoints_reject_mismatched_resume_step_before_opening_transport(monkeypatch, tmp_path):
    _wire_manager(monkeypatch, _manager(tmp_path))
    receive = Mock(side_effect=AssertionError("should not open transport"))
    monkeypatch.setattr(packer_module, "setup_training_batch_receiver", receive)
    monkeypatch.setattr(data_module, "setup_micro_batch_receiver", receive)
    channel = InProcessMicroBatchBuffer(tmp_path, start_step=6)
    config = SimpleNamespace(type="filesystem")
    with pytest.raises(ValueError, match="sender step.*resume step"):
        SinglePacker(1, 8, 4, None, config, start_step=5, micro_batch_sender=channel)
    with pytest.raises(ValueError, match="receiver step.*resume step"):
        GRPODataLoader(str(tmp_path), 0, 5, config, micro_batch_receiver=channel)
    assert channel.current_step == 6
    receive.assert_not_called()


def test_local_packer_keeps_idle_cancellation(monkeypatch, tmp_path):
    manager = _manager(tmp_path)
    _wire_manager(monkeypatch, manager)
    monkeypatch.setattr(packer_module, "time", SimpleNamespace(sleep=Mock()))
    monkeypatch.setattr(
        packer_module, "setup_training_batch_receiver", lambda *_, **__: SimpleNamespace(receive=lambda: [])
    )
    channel = InProcessMicroBatchBuffer(tmp_path, start_step=6)
    packer = setup_grpo_packer(
        1,
        8,
        4,
        None,
        SimpleNamespace(type="filesystem"),
        start_step=6,
        micro_batch_sender=channel,
    )
    packer.check_cancelled = Mock(side_effect=[None, InterruptedError("stopped")])
    with pytest.raises(InterruptedError, match="stopped"):
        packer.pack()
    assert channel.current_step == 6 and not channel.can_receive()
    assert manager.progress[0].step == 9
    assert manager.ready_to_update == [False]


def test_local_packer_rejects_missing_advantage_before_publishing(monkeypatch, tmp_path):
    manager = _manager(tmp_path)
    _wire_manager(monkeypatch, manager)
    config = SimpleNamespace(type="filesystem")
    channel = InProcessMicroBatchBuffer(tmp_path, start_step=6)
    packer = setup_grpo_packer(1, 8, 4, None, config, start_step=6, micro_batch_sender=channel)
    samples = _samples(6)
    samples[0].advantage = None
    sender = setup_training_batch_sender(manager.get_run_dir(0), config)
    sender.send(TrainingBatch(samples, 6))
    with pytest.raises(ValueError, match="must have advantages"):
        packer.pack()
    assert not channel.can_receive()
    assert manager.progress[0].step == 9
    assert manager.ready_to_update == [False]


@pytest.mark.parametrize("transport", ["filesystem", "zmq"])
def test_trainer_setup_keeps_external_transport_but_uses_local_handoff(
    trainer_module,
    monkeypatch,
    tmp_path,
    transport,
):
    from surogate.core.config import grpo_orch_config

    manager = _manager(tmp_path)
    _wire_manager(monkeypatch, manager)
    monkeypatch.setattr(packer_module, "init_multi_run_manager", lambda **_: None)
    monkeypatch.setattr(packer_module, "setup_micro_batch_sender", Mock(side_effect=AssertionError("second sender")))
    monkeypatch.setattr(data_module, "setup_micro_batch_receiver", Mock(side_effect=AssertionError("second receiver")))
    config = SimpleNamespace(type=transport)
    if transport == "zmq":
        context = zmq_module.zmq.Context()
        monkeypatch.setattr(zmq_module.zmq.Context, "instance", classmethod(lambda cls: context))
        with socket.socket() as candidate:
            candidate.bind(("127.0.0.1", 0))
            port = candidate.getsockname()[1]
        config = SimpleNamespace(type="zmq", host="127.0.0.1", port=port, hwm=8)
        monkeypatch.setattr(grpo_orch_config, "ZMQTransportConfig", lambda _: config)

    trainer = trainer_module.GRPOTrainer.__new__(trainer_module.GRPOTrainer)
    trainer.config = SimpleNamespace(
        transport_type=transport,
        output_dir=str(tmp_path),
        sequence_len=8,
        pad_to_multiple_of=4,
        tokenizer=None,
        single_sample_bins=True,
    )
    deadline = time.monotonic() + 5

    def check_cancelled():
        if time.monotonic() > deadline:
            raise TimeoutError("external training batch did not arrive")

    trainer.phase_controller = SimpleNamespace(check_cancelled=check_cancelled)
    trainer._setup_data(start_step=6)
    sender = setup_training_batch_sender(manager.get_run_dir(0), config)
    try:
        channel = trainer.packer.sender
        assert channel is trainer.data_loader.receiver
        assert isinstance(channel, InProcessMicroBatchBuffer)
        assert channel.current_step == 6
        assert trainer.packer.single_sample_bins is True
        assert isinstance(
            trainer.packer.receiver,
            (
                zmq_module.ZMQTrainingBatchReceiver
                if transport == "zmq"
                else filesystem_module.FileSystemTrainingBatchReceiver
            ),
        )
        sender.send(TrainingBatch(_samples(6), 6))
        trainer.packer.pack()
        trainer.data_loader.wait_for_batch()
        batches = trainer.data_loader.get_batch()
        assert len(batches) == 3
        assert batches[0]["input_ids"][0, 0] == 601
        assert channel.current_step == 7 and not channel.can_receive()
        assert not (tmp_path / "rollouts").exists()
    finally:
        sender.close()
        trainer.packer.receiver.close()
