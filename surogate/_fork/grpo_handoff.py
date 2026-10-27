"""In-process GRPO micro-batch handoff (ledger topic T7).

The single-process trainer packs and trains in the same thread, so the packer
hands each packed step straight to the rank-zero data loader instead of
serializing it through a second filesystem or ZMQ channel.
"""

from pathlib import Path

from surogate.grpo.transport import MicroBatchReceiver, MicroBatchSender
from surogate.grpo.transport.types import MicroBatch


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
