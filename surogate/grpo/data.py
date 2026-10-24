"""Data adapter: converts MicroBatch to numpy arrays for Surogate."""

from pathlib import Path

import numpy as np

from surogate.grpo.routing_replay import rows_from_bytes
from surogate.grpo.transport import MicroBatchReceiver, MicroBatchSender, setup_micro_batch_receiver
from surogate.grpo.transport.types import MicroBatch
from surogate.utils.logger import get_logger

logger = get_logger()


class InProcessMicroBatchBuffer(MicroBatchSender, MicroBatchReceiver):
    """One-slot handoff for a packer and rank-zero loader in the same thread.

    Packing completes before the trainer receives, so waiting on an empty slot
    or overwriting an unconsumed step is a wiring error, not backpressure to
    block on. Ownership of the packed list passes to the receiver without
    serializing its token arrays or routing payloads.
    """

    def __init__(self, output_dir: Path, start_step: int = 0):
        self.output_dir = Path(output_dir)
        self.data_world_size = 1
        self.data_rank = 0
        self.current_step = start_step
        self._pending: list[MicroBatch] | None = None

    def send(self, micro_batch_grid: list[list[MicroBatch]]) -> None:
        if len(micro_batch_grid) != 1:
            raise ValueError("in-process micro-batches require exactly one data worker")
        if self._pending is not None:
            raise RuntimeError(f"micro-batches for step {self.current_step} have not been consumed")
        self._pending = micro_batch_grid[0]

    def can_receive(self) -> bool:
        return self._pending is not None

    def wait(self) -> None:
        if not self.can_receive():
            raise RuntimeError(f"pack step {self.current_step} before waiting for its micro-batches")

    def receive(self) -> list[MicroBatch]:
        self.wait()
        micro_batches = self._pending
        assert micro_batches is not None
        self._pending = None
        self.current_step += 1
        return micro_batches


def microbatch_to_numpy(micro_batch) -> dict[str, np.ndarray]:
    """Convert a MicroBatch (msgspec struct with lists) to numpy arrays.

    Returns dict with:
        input_ids: int32 [1, T]
        targets: int32 [1, T]  (input_ids shifted left by 1)
        position_ids: int32 [1, T]
        advantages: float32 [1, T]
        inference_logprobs: float32 [1, T]
        loss_mask: bool [1, T]
        temperatures: float32 [1, T]
        teacher_logprobs: float32 [1, T] or None
        turn_ids: int32 [1, T] or None  (-1 = initial prompt or padding; env
            observation tokens carry the index of the turn they precede)
        routed_experts: int16 [T, layers, top_k] or None  (Rollout Routing Replay rows; -1 = no replay)
    """
    T = len(micro_batch.input_ids)

    input_ids = np.array(micro_batch.input_ids, dtype=np.int32).reshape(1, T)

    # Targets = input_ids shifted left by 1 (predict next token)
    targets = np.empty_like(input_ids)
    targets[0, :-1] = input_ids[0, 1:]
    targets[0, -1] = 0  # Last position doesn't matter (will be masked)

    position_ids = np.array(micro_batch.position_ids, dtype=np.int32).reshape(1, T)
    advantages = np.array(micro_batch.advantages, dtype=np.float32).reshape(1, T)
    inference_logprobs = np.array(micro_batch.inference_logprobs, dtype=np.float32).reshape(1, T)
    loss_mask = np.array(micro_batch.loss_mask, dtype=np.bool_).reshape(1, T)
    temperatures = np.array(micro_batch.temperatures, dtype=np.float32).reshape(1, T)

    teacher_logprobs = None
    if micro_batch.teacher_logprobs is not None:
        teacher_logprobs = np.array(micro_batch.teacher_logprobs, dtype=np.float32).reshape(1, T)

    turn_ids = None
    if micro_batch.turn_ids is not None:
        turn_ids = np.array(micro_batch.turn_ids, dtype=np.int32).reshape(1, T)

    routed_experts = rows_from_bytes(
        getattr(micro_batch, "routed_experts", None), getattr(micro_batch, "routed_experts_dims", None), T)

    return {
        "input_ids": input_ids,
        "targets": targets,
        "position_ids": position_ids,
        "advantages": advantages,
        "inference_logprobs": inference_logprobs,
        "loss_mask": loss_mask,
        "temperatures": temperatures,
        "teacher_logprobs": teacher_logprobs,
        "turn_ids": turn_ids,
        "routed_experts": routed_experts,
    }


class GRPODataLoader:
    """Wraps the transport layer to deliver numpy micro-batches."""

    def __init__(
        self, output_dir: str, dp_rank: int, start_step: int, transport_config,
        *, micro_batch_receiver: MicroBatchReceiver | None = None,
    ):
        if micro_batch_receiver is not None:
            if dp_rank != micro_batch_receiver.data_rank:
                raise ValueError("micro-batch receiver rank does not match the data loader")
            if getattr(micro_batch_receiver, "current_step", start_step) != start_step:
                raise ValueError("micro-batch receiver step does not match the data loader resume step")
            self.receiver = micro_batch_receiver
        else:
            self.receiver = setup_micro_batch_receiver(Path(output_dir), dp_rank, start_step, transport_config)
        logger.info(f"GRPO data loader initialized (dp_rank={dp_rank}, start_step={start_step})")

    def wait_for_batch(self):
        """Block until the next batch is available."""
        self.receiver.wait()

    def get_batch(self) -> list[dict[str, np.ndarray]]:
        """Receive micro-batches and convert to numpy."""
        micro_batches = self.receiver.receive()
        return [microbatch_to_numpy(mb) for mb in micro_batches]
