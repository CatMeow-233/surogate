"""The TLC training driver: one run configuration, one native training run (docs/training.md: 训练驱动).

The driver calls SurogateTrainerWrapper(config, train_files, eval_files).train() itself. Upstream's sft command wraps
that call in train_with_oom_recovery, which on an out-of-memory error halves the microbatch and, once gradient
accumulation has reached 16, keeps the accumulation: the tokens per update halve with only a warning. TLC compares
rows at a fixed number of nominal tokens per update, so on any failure, OOM included, the driver stops and writes a
failure report into the run record, tlc_run.json in the run directory; a smaller microbatch is a new run
(--microbatch). Rerunning the same command resumes from the latest checkpoint, and only when the weights, tokenizer,
data and training settings are the ones the run started with. After the run, finished checkpoints and the final
export are finalized as model directories.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..checkpoint import checkpoint_step, finalize_checkpoints, finalize_dir
from ..data.shards import ShardWriter, manifest, manifest_sha256, read_shard, shard_files
from ..data.tokenizer import PAD
from ..files import dump_json, file_sha256, write_text_atomic
from ..modeldir import CONFIG_FILE, META_FILE, WEIGHTS_FILE
from ..native import native_wrapper
from . import report
from .config import Run, problems, settings, settings_sha256
from .report import RUN_FILE

# 2: one attempt per invocation (no downshift); the identity holds the training settings.
RUN_VERSION = 2
FIXED_BATCH_FILE = "fixed-batch.bin"
REGRESSION_ENV = "SUROGATE_REGRESSION_ARTIFACT"
# Native allocation failures: "Cuda OOM when allocating tensor ..." (utilities/allocator.cpp) and CUDA's "out of
# memory" arrive as RuntimeError, the activation stack's std::bad_alloc as MemoryError.
OOM_MESSAGE = re.compile(r"out of memory|\bOOM\b", re.IGNORECASE)

# (SFT settings, train files, eval files) -> an object with train(), normally a SurogateTrainerWrapper.
Factory = Callable[[Mapping[str, Any], Sequence[str], Sequence[str]], Any]


class RunError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("the run cannot start:\n" + "\n".join(f"  - {p}" for p in problems))


class RunFailed(RuntimeError):
    """Training failed; the run record holds the failure report."""


def is_oom(error: BaseException) -> bool:
    return isinstance(error, MemoryError) or (isinstance(error, RuntimeError) and bool(OOM_MESSAGE.search(str(error))))


def write_fixed_batch(source: Path, out: Path, *, rows: int, seq_len: int, pad_id: int) -> dict[str, Any]:
    """Copy the first `rows` rows of shard `source` into a new shard `out`; returns its statistics."""
    shard = read_shard(source)
    tokens = shard.tokens[: rows * seq_len].reshape(rows, seq_len)
    masks = shard.masks[: rows * seq_len].reshape(rows, seq_len)
    writer = ShardWriter(out, seq_len=seq_len, vocab_size=shard.vocab_size, pad_id=pad_id)
    for ids, mask in zip(tokens, masks, strict=True):
        writer.add_row(ids[: int(mask.sum()) + 1])  # mask bit i is set iff token i + 1 is a real token of the row
    return writer.close()


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _last_checkpoint(run: Run) -> int | None:
    steps = [checkpoint_step(p) for p in run.run_dir.iterdir()] if run.run_dir.is_dir() else []
    return max((s for s in steps if s is not None), default=None)


def _write(run: Run, record: dict[str, Any]) -> None:
    write_text_atomic(run.run_dir / RUN_FILE, dump_json(record))


def _identity(run: Run) -> dict[str, Any]:
    """What a run directory belongs to; resuming with anything else is refused."""
    data = manifest(run.data)
    return {
        "model_weights_sha256": file_sha256(run.model_dir / WEIGHTS_FILE),
        "model_config_sha256": file_sha256(run.model_dir / CONFIG_FILE),
        "tokenizer_sha256": data["tokenizer"]["sha256"],
        "data_manifest_sha256": manifest_sha256(run.data),
        "settings_sha256": settings_sha256(settings(run)),
    }


def _data(run: Run) -> tuple[list[Path], list[Path], dict[str, Any]]:
    """Train and eval files, and the valid targets per update the record states."""
    data, seq_len = manifest(run.data), run.sft["sequence_len"]
    train, evals = shard_files(run.data, "train"), shard_files(run.data, "val")
    info: dict[str, Any] = {"dir": str(run.data), "eval_files": [p.name for p in evals], "fixed_batch": None}
    if run.fixed_batch:
        out, rows = run.run_dir / FIXED_BATCH_FILE, run.tokens_per_update // seq_len
        pad = data["tokenizer"]["special_ids"][PAD]
        stats = write_fixed_batch(train[0], out, rows=rows, seq_len=seq_len, pad_id=pad)
        info["fixed_batch"] = {"source": train[0].name, **stats}
        train, valid = [out], stats["valid_targets"]
    else:
        split = data["splits"]["train"]
        valid = round(run.tokens_per_update * split["valid_targets"] / split["tokens"])
    info |= {"train_files": [p.name for p in train], "valid_targets_per_update": valid}
    return train, evals, info


def _start(run: Run) -> dict[str, Any]:
    """The run record: a new one, or the existing one of a run directory being resumed."""
    identity, current = _identity(run), settings(run)
    path = run.run_dir / RUN_FILE
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("tlc_run_version") != RUN_VERSION:
            raise RunError([f"{path} has run record version {record.get('tlc_run_version')}; start a new run"])
        changed = sorted(key for key, value in identity.items() if record["identity"].get(key) != value)
        if changed:
            keys = sorted(
                k for k in current.keys() | record["settings"].keys() if current.get(k) != record["settings"].get(k)
            )
            detail = f" (settings: {keys})" if "settings_sha256" in changed else ""
            raise RunError([f"{run.run_dir} holds a run with other {changed}{detail}; use a new output_dir or stage"])
        return record
    return {
        "tlc_run_version": RUN_VERSION,
        "config": str(run.config),
        "stage": run.stage,
        "model_dir": str(run.model_dir),
        "model_meta": json.loads((run.model_dir / META_FILE).read_text(encoding="utf-8")),
        "identity": identity,
        "settings": current,
        "tokens_per_update": run.tokens_per_update,
        "attempts": [],
        "status": "running",
        "failure_report": None,
    }


def _device_memory() -> dict[str, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
    except Exception:  # best effort while reporting another failure
        return None
    return {"free_bytes": free, "total_bytes": total}


def _finalize(run: Run, record: dict[str, Any], *, done: bool) -> None:
    """Make the run's complete checkpoints, and the final export when `done`, model directories with provenance."""
    identity, environment = record["identity"], record["attempts"][-1]["environment"]
    meta = {
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "data_manifest_sha256": identity["data_manifest_sha256"],
        "shard_version": manifest(run.data)["tlc_shard_version"],
        "run": {
            "settings_sha256": identity["settings_sha256"],
            "config_sha256": record["attempts"][-1]["config_sha256"],
            "git_commit": environment["git_commit"],
            "git_dirty": environment["git_dirty"],
        },
    }
    checkpoints = finalize_checkpoints(run.run_dir, run.model_dir, meta={**meta, "kind": "checkpoint"})
    record["finalized_checkpoints"] = [p.name for p in checkpoints]
    if done:
        finalize_dir(run.run_dir, run.model_dir, meta={**meta, "kind": "run", "step": run.sft["max_steps"]})


def train(run: Run, *, factory: Factory = native_wrapper) -> dict[str, Any]:
    """Train `run`; returns the run record. Raises RunError before training, RunFailed when training fails."""
    found = problems(run)
    if found:
        raise RunError(found)
    run.run_dir.mkdir(parents=True, exist_ok=True)
    record = _start(run)
    train_files, eval_files, record["data"] = _data(run)
    microbatch, accum = run.batch
    attempt: dict[str, Any] = {
        "index": len(record["attempts"]),
        "started": _now(),
        "microbatch": microbatch,
        "grad_accum": accum,
        "tokens_per_update": microbatch * accum * run.sft["sequence_len"] * run.sft["gpus"],
        "config_sha256": file_sha256(run.config),
        "resumed_from_step": _last_checkpoint(run),
        "environment": report.environment(),
    }
    regression = run.run_dir / f"regression-{attempt['index']}.json"
    attempt["regression_file"] = regression.name
    record["attempts"].append(attempt)
    record["sft"] = run.sft
    record["status"], record["failure_report"] = "running", None
    _write(run, record)
    previous, log = os.environ.get(REGRESSION_ENV), None
    os.environ[REGRESSION_ENV] = str(regression)
    try:
        wrapper = factory(run.sft, [str(p) for p in train_files], [str(p) for p in eval_files])
        log = getattr(getattr(wrapper, "config", None), "log_file", None)
        attempt["log_file"] = log
        wrapper.train()
    except Exception as error:
        failure = _failure_report(run, attempt, error)
        attempt |= {"finished": _now(), "status": "oom" if failure["oom"] else "failed", "error": failure["message"]}
        record["status"], record["failure_report"] = "failed", failure
        _finalize(run, record, done=False)
        _write(run, record)
        raise RunFailed(f"{run.run_dir}: {failure['message'][:400]} (failure report in {RUN_FILE})") from error
    finally:
        if previous is None:
            os.environ.pop(REGRESSION_ENV, None)
        else:
            os.environ[REGRESSION_ENV] = previous
    attempt |= {"finished": _now(), "status": "completed"}
    attempt["summary"] = report.summarize(regression, Path(log) if log else None, run.tokens_per_update)
    record["status"] = "completed"
    _finalize(run, record, done=True)
    _write(run, record)
    return record


def _failure_report(run: Run, attempt: dict[str, Any], error: BaseException) -> dict[str, Any]:
    microbatch, accum = run.batch
    return {
        "attempt": attempt["index"],
        "error_type": type(error).__name__,
        "message": f"{type(error).__name__}: {error}"[:4000],
        "traceback": "".join(traceback.format_exception(error))[-8000:],
        "oom": is_oom(error),
        "microbatch": microbatch,
        "grad_accum": accum,
        "device_memory": _device_memory(),
        "last_checkpoint": _last_checkpoint(run),
        # A smaller microbatch keeps the tokens per update, but it is a new run: the loader position in a checkpoint
        # counts microbatches of the size it was saved with.
        "remedy": "a new run with a smaller --microbatch" if is_oom(error) else None,
    }
