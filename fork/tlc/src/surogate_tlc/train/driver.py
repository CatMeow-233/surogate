"""The TLC training driver: one run configuration, one native training run (README: 训练驱动).

The driver calls SurogateTrainerWrapper(config, train_files, eval_files).train() itself. Upstream's sft command wraps
that call in train_with_oom_recovery, which on an out-of-memory error halves the microbatch and, once gradient
accumulation has reached 16, keeps the accumulation: the tokens per update halve with only a warning. TLC compares
rows at a fixed number of nominal tokens per update, so on OOM the driver either fails and writes a failure report
(the default) or, when the run configuration says `on_oom: downshift`, halves the microbatch and doubles the
accumulation, at most max_downshifts times and only before the run's first checkpoint (the loader's position in a
checkpoint counts microbatches). Every attempt, downshift and failure goes into the run record, tlc_run.json in the
run directory. After the run, finished checkpoints and the final export are finalized as model directories.
"""

from __future__ import annotations

import datetime
import gc
import json
import os
import re
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..checkpoint import checkpoint_step, finalize_checkpoints, finalize_dir
from ..data.corpus import file_sha256
from ..data.shards import ShardWriter, manifest_sha256, read_shard
from ..data.tokenizer import PAD
from ..modeldir import CONFIG_FILE, META_FILE, WEIGHTS_FILE, dump_json, write_text_atomic
from . import report
from .config import FIXED, Run, manifest, problems, shard_files
from .report import RUN_FILE

RUN_VERSION = 1
FIXED_BATCH_FILE = "fixed-batch.bin"
REGRESSION_ENV = "SUROGATE_REGRESSION_ARTIFACT"
# Native allocation failures: "Cuda OOM when allocating tensor ..." (utilities/allocator.cpp) and CUDA's "out of
# memory" arrive as RuntimeError, the activation stack's std::bad_alloc as MemoryError.
OOM_MESSAGE = re.compile(r"out of memory|\bOOM\b", re.IGNORECASE)

# SFT settings of native_trainer() besides the model, output directory and shape.
SCORING = {
    "gradient_accumulation_steps": 1,
    "max_steps": 1,
    "resume_from_checkpoint": False,
    "use_cuda_graphs": False,
    "recompute": False,
    **FIXED,
}

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


def native_wrapper(sft: Mapping[str, Any], train_files: Sequence[str], eval_files: Sequence[str]) -> Any:
    """SurogateTrainerWrapper for `sft`, built the way upstream's sft command builds it, minus tokenization."""
    from surogate.core.config.sft_config import SFTConfig
    from surogate.train.trainer import SurogateTrainerWrapper
    from surogate.utils.dict import DictDefault

    from .. import register

    register()
    config = SFTConfig(DictDefault(dict(sft)))
    config.__post_init__()  # as surogate.train.tokenize.TokenizeDatasets does
    return SurogateTrainerWrapper(config, [str(f) for f in train_files], [str(f) for f in eval_files] or None)


def native_trainer(model_dir: Path, data: Path, out: Path, *, batch: int, seq_len: int) -> Any:
    """A wrapper for scoring and gate tests: model_dir's weights, no LoRA, eager, nothing resumed or trained.

    Its train loader reads data's train shards only because the wrapper needs one; `out` receives upstream's files.
    """
    sft = {
        "model": str(model_dir),
        "output_dir": str(out),
        "sequence_len": seq_len,
        "per_device_train_batch_size": batch,
    }
    return native_wrapper({**sft, **SCORING}, shard_files(data, "train"), [])


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
    identity = _identity(run)
    path = run.run_dir / RUN_FILE
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["identity"] != identity:
            raise RunError([f"{run.run_dir} holds a run of other weights, tokenizer or data: {record['identity']}"])
        return record
    return {
        "tlc_run_version": RUN_VERSION,
        "config": str(run.config),
        "stage": run.stage,
        "model_dir": str(run.model_dir),
        "model_meta": json.loads((run.model_dir / META_FILE).read_text(encoding="utf-8")),
        "identity": identity,
        "tokens_per_update": run.tokens_per_update,
        "on_oom": run.on_oom,
        "max_downshifts": run.max_downshifts,
        "attempts": [],
        "downshifts": [],
        "status": "running",
        "failure_report": None,
    }


def _resume_batch(run: Run, record: dict[str, Any]) -> tuple[int, int]:
    """The batch shape to use: once a run has checkpoints, the one its latest attempt trained with."""
    attempts = record["attempts"]
    if _last_checkpoint(run) is None or not attempts:
        return run.batch
    first, latest = attempts[0], attempts[-1]
    if (first["microbatch"], first["grad_accum"]) != run.batch:
        raise RunError([f"{run.run_dir} was started with microbatch {first['microbatch']}; start a new run"])
    return latest["microbatch"], latest["grad_accum"]


def _device_memory() -> dict[str, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
    except Exception:  # best effort while reporting another failure
        return None
    return {"free_bytes": free, "total_bytes": total}


def _release() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _finalize(run: Run, record: dict[str, Any], *, done: bool) -> None:
    identity = record["identity"]
    meta = {
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "data_manifest_sha256": identity["data_manifest_sha256"],
        "shard_version": manifest(run.data)["tlc_shard_version"],
    }
    record["finalized_checkpoints"] = [p.name for p in finalize_checkpoints(run.run_dir, run.model_dir, meta=meta)]
    if done:
        finalize_dir(run.run_dir, run.model_dir, meta={**meta, "step": run.sft["max_steps"]})


def train(run: Run, *, factory: Factory = native_wrapper) -> dict[str, Any]:
    """Train `run`; returns the run record. Raises RunError before training, RunFailed when training fails."""
    found = problems(run)
    if found:
        raise RunError(found)
    run.run_dir.mkdir(parents=True, exist_ok=True)
    record = _start(run)
    run = run.with_batch(*_resume_batch(run, record))
    train_files, eval_files, record["data"] = _data(run)
    downshifts = 0
    while True:
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
        _write(run, record)
        previous = os.environ.get(REGRESSION_ENV)
        os.environ[REGRESSION_ENV] = str(regression)
        wrapper, log, failure, error = None, None, None, None
        try:
            wrapper = factory(run.sft, [str(p) for p in train_files], [str(p) for p in eval_files])
            log = getattr(getattr(wrapper, "config", None), "log_file", None)
            attempt["log_file"] = log
            wrapper.train()
        except Exception as caught:
            failure = _failure_report(run, attempt, caught, _downshift_refusal(run, caught, downshifts))
            # The traceback's frames hold the failed trainer; clear them so that its memory can be released.
            traceback.clear_frames(caught.__traceback__)
            error = caught if failure["no_downshift_because"] is not None else None
        finally:
            if previous is None:
                os.environ.pop(REGRESSION_ENV, None)
            else:
                os.environ[REGRESSION_ENV] = previous
        wrapper = None
        if failure is None:
            attempt |= {"finished": _now(), "status": "completed"}
            attempt["summary"] = report.summarize(regression, Path(log) if log else None, run.tokens_per_update)
            record["status"] = "completed"
            _finalize(run, record, done=True)
            _write(run, record)
            return record
        attempt |= {"finished": _now(), "status": "oom" if failure["oom"] else "failed", "error": failure["message"]}
        _release()
        if error is None:
            record["downshifts"].append({"attempt": attempt["index"], "from": [microbatch, accum]})
            run, downshifts = run.with_batch(microbatch // 2, accum * 2), downshifts + 1
            record["downshifts"][-1]["to"] = list(run.batch)
            continue
        record["status"], record["failure_report"] = "failed", failure
        _finalize(run, record, done=False)
        _write(run, record)
        raise RunFailed(f"{run.run_dir}: {failure['message'][:400]} (failure report in {RUN_FILE})") from error


def _failure_report(run: Run, attempt: dict[str, Any], error: BaseException, refusal: str | None) -> dict[str, Any]:
    microbatch, accum = run.batch
    return {
        "attempt": attempt["index"],
        "error_type": type(error).__name__,
        "message": f"{type(error).__name__}: {error}"[:4000],
        "traceback": "".join(traceback.format_exception(error))[-8000:],
        "oom": is_oom(error),
        "no_downshift_because": refusal,
        "microbatch": microbatch,
        "grad_accum": accum,
        "device_memory": _device_memory(),
        "last_checkpoint": _last_checkpoint(run),
    }


def _downshift_refusal(run: Run, error: BaseException, downshifts: int) -> str | None:
    """Why this failure must not lead to a downshift, or None when the driver halves the microbatch and retries."""
    if not is_oom(error):
        return "not an out-of-memory error"
    if run.on_oom != "downshift":
        return "tlc.on_oom is fail"
    if downshifts >= run.max_downshifts:
        return f"tlc.max_downshifts ({run.max_downshifts}) reached"
    if run.batch[0] % 2:
        return f"microbatch {run.batch[0]} cannot be halved"
    if _last_checkpoint(run) is not None:
        return "the run has a checkpoint, whose loader position counts microbatches of the current size"
    return None
