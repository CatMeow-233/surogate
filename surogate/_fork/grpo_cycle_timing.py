"""GRPO cycle timing log line (ledger topic T6).

A cycle spans publication of the preceding update through this optimizer
update, including batch acquisition and excluding logging and checkpointing.
The existing train duration/throughput metrics keep their scope. The trainer
records the cycle metrics in its metrics row; this module prints them on the
steps the trainer logs, as a line of their own.
"""

from surogate.utils.logger import get_logger

logger = get_logger()


def format_cycle_summary(cycle_metrics: dict[str, float]) -> str:
    return (
        f"cycle={cycle_metrics['train/cycle_duration_ms'] / 1000:.3f}s"
        f" publish={cycle_metrics['train/publish_duration_ms'] / 1000:.3f}s"
        f" pack={cycle_metrics['train/pack_duration_ms'] / 1000:.3f}s"
        f" batch_wait={cycle_metrics['train/batch_wait_duration_ms'] / 1000:.3f}s"
        f" receive={cycle_metrics['train/batch_receive_duration_ms'] / 1000:.3f}s"
    )


def log_cycle_summary(step: int, logging_steps: int, cycle_metrics: dict[str, float]) -> None:
    if step % logging_steps == 0:
        logger.info(f"step={step} {format_cycle_summary(cycle_metrics)}")
