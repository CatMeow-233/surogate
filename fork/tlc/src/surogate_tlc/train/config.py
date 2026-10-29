"""Run configurations for the TLC training driver (docs/training.md: 训练驱动).

A run configuration is an upstream SFT config file (keys of surogate.core.config.sft_config.SFTConfig) plus a `tlc`
table:

    tlc:
      data: DIR                  # shard directory written by `python -m surogate_tlc.data.shards`
      tokens_per_update: 16384   # nominal tokens per update = microbatch x accumulation x sequence_len x gpus
      stages:                    # named overrides of the SFT keys, selected with --stage
        overfit: {max_steps: 100, fixed_batch: true}

gradient_accumulation_steps follows from tokens_per_update and the microbatch. A stage may also set `fixed_batch:
true` (STAGE_KEYS): every step then trains on the same rows, one update's worth from the start of the first train
shard. ${VAR} in strings is replaced from the environment. The run directory is <output_dir>/<stage>, with
-mb<B> appended when the microbatch is given on the command line, so a microbatch sweep never resumes another run's
checkpoints. settings() is what a run directory's training depends on; the driver refuses to resume a run whose
settings changed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..data.shards import MANIFEST_FILE, manifest, shard_files
from ..data.tokenizer import tokenizer_sha256
from ..hf import positive_int
from ..modeldir import ModelDirError, check_model_dir
from ..native import FIXED

TOKENS_PER_UPDATE = 16384
TLC_DEFAULTS: dict[str, Any] = {"tokens_per_update": TOKENS_PER_UPDATE}
TLC_KEYS = {"data", "stages", *TLC_DEFAULTS}
STAGE_KEYS = {"fixed_batch"}  # the TLC keys a stage may set; its other keys override SFT keys
# Data comes from tlc.data only and P0 has no distillation. Upstream ignores checkpoint_dir (checkpoints always go to
# output_dir, the run directory), so a value there would mislead.
FORBIDDEN = ("datasets", "validation_datasets", "checkpoint_dir", "distillation")
# SFT keys a resumed run may change: where and how often it saves, logs and evaluates, and how long it runs. Every
# other key shapes the training trajectory. max_steps also sets the length of a non-constant learning-rate schedule,
# which settings() therefore records on its own.
RESUMABLE = (
    "output_dir",
    "max_steps",
    "save_steps",
    "save_total_limit",
    "eval_steps",
    "logging_steps",
    "log_gpu_util",
    "run_name",
    "resume_from_checkpoint",
)


@dataclass(frozen=True)
class Run:
    config: Path
    stage: str | None
    sft: dict[str, Any]  # upstream SFT settings; output_dir is the run directory
    data: Path
    tokens_per_update: int
    fixed_batch: bool = False

    @property
    def run_dir(self) -> Path:
        return Path(self.sft["output_dir"])

    @property
    def model_dir(self) -> Path:
        return Path(self.sft["model"])

    @property
    def batch(self) -> tuple[int, int]:
        """(microbatch, gradient accumulation)."""
        return self.sft["per_device_train_batch_size"], self.sft["gradient_accumulation_steps"]


def grad_accum(tokens_per_update: int, seq_len: int, gpus: int, microbatch: int) -> int:
    """The accumulation that makes microbatch x accumulation x seq_len x gpus equal tokens_per_update."""
    rows, rest = divmod(tokens_per_update, seq_len * gpus)
    if rest or microbatch < 1 or rows % microbatch:
        raise ValueError(
            f"{tokens_per_update} tokens per update are not a whole number of microbatches of {microbatch} rows "
            f"x {seq_len} tokens x {gpus} GPU(s)"
        )
    return rows // microbatch


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand(v) for key, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise ValueError(f"unset environment variable in {value!r}")
        return expanded
    return value


def load(path: Path, *, stage: str | None = None, microbatch: int | None = None) -> Run:
    """Read a run configuration, apply `stage` and an optional microbatch, and derive the accumulation."""
    path = Path(path)
    raw = _expand(yaml.safe_load(path.read_text(encoding="utf-8")))
    if not isinstance(raw, dict) or not isinstance(raw.get("tlc"), dict):
        raise ValueError(f"{path}: a TLC run configuration needs a `tlc` table")
    tlc = {**TLC_DEFAULTS, **raw["tlc"]}
    unknown = sorted(set(tlc) - TLC_KEYS)
    if unknown or "data" not in tlc:
        raise ValueError(f"{path}: the tlc table needs `data` and knows {sorted(TLC_KEYS)}; unknown: {unknown}")
    sft = {key: value for key, value in raw.items() if key != "tlc"}
    stages = tlc.get("stages") or {}
    if stage is not None and stage not in stages:
        raise ValueError(f"{path}: no stage {stage!r}; stages: {sorted(stages)}")
    overrides = dict(stages[stage] or {}) if stage is not None else {}
    stage_tlc = {key: overrides.pop(key) for key in STAGE_KEYS if key in overrides}
    if overrides.keys() & {"tlc", *FORBIDDEN}:
        raise ValueError(f"{path}: stage {stage!r} sets {sorted(overrides.keys() & {'tlc', *FORBIDDEN})}")
    sft.update(overrides)
    for key, value in FIXED.items():
        sft.setdefault(key, value)
    sft.setdefault("gpus", 1)
    if microbatch is not None:
        sft["per_device_train_batch_size"] = microbatch
    for key in ("output_dir", "model", "sequence_len", "per_device_train_batch_size"):
        if key not in sft:
            raise ValueError(f"{path}: {key} is required")
    accum = grad_accum(tlc["tokens_per_update"], sft["sequence_len"], sft["gpus"], sft["per_device_train_batch_size"])
    if sft.setdefault("gradient_accumulation_steps", accum) != accum:
        raise ValueError(
            f"{path}: gradient_accumulation_steps must be {accum} for {tlc['tokens_per_update']} tokens per update "
            "(it follows from tokens_per_update; leave it out)"
        )
    name = (stage or "run") + ("" if microbatch is None else f"-mb{microbatch}")
    sft["output_dir"] = str(Path(sft["output_dir"]) / name)
    return Run(
        config=path,
        stage=stage,
        sft=sft,
        data=Path(tlc["data"]),
        tokens_per_update=tlc["tokens_per_update"],
        fixed_batch=bool(stage_tlc.get("fixed_batch", False)),
    )


def settings(run: Run) -> dict[str, Any]:
    """What the training trajectory of `run` depends on besides the model and data: the SFT settings other than
    RESUMABLE, the schedule length (None for a constant learning rate) and the TLC batch settings."""
    constant = run.sft.get("lr_scheduler_type") == "constant"
    return {
        **{key: value for key, value in sorted(run.sft.items()) if key not in RESUMABLE},
        "schedule_total_steps": None if constant else run.sft.get("max_steps"),
        "tokens_per_update": run.tokens_per_update,
        "fixed_batch": run.fixed_batch,
    }


def settings_sha256(values: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def problems(run: Run) -> list[str]:
    """Every reason `run` must not start; an empty list means it may."""
    sft, out = run.sft, []
    out += [f"{key} is not used by TLC runs; remove it" for key in FORBIDDEN if key in sft]
    out += [f"{key} must be {value}" for key, value in FIXED.items() if sft.get(key) is not value]
    if not positive_int(sft.get("max_steps")):
        out.append("max_steps must be a positive integer: every run has a fixed length")
    microbatch, accum = run.batch
    seq_len, gpus = sft["sequence_len"], sft["gpus"]
    if microbatch * accum * seq_len * gpus != run.tokens_per_update:
        out.append(f"{microbatch} x {accum} x {seq_len} x {gpus} tokens per update, not {run.tokens_per_update}")
    cfg: Mapping[str, Any] | None = None
    try:
        cfg = check_model_dir(run.model_dir)
    except ModelDirError as error:
        out += [f"model: {problem}" for problem in error.problems]
    if not (run.data / MANIFEST_FILE).is_file():
        return [*out, f"data: {run.data / MANIFEST_FILE} is missing"]
    data = manifest(run.data)
    if data["format"]["seq_len"] != seq_len:
        out.append(f"sequence_len is {seq_len}, but the shards in {run.data} have rows of {data['format']['seq_len']}")
    if cfg is not None:
        if seq_len > cfg["max_position_embeddings"]:
            out.append(f"sequence_len {seq_len} exceeds max_position_embeddings {cfg['max_position_embeddings']}")
        if data["tokenizer"]["sha256"] != tokenizer_sha256(run.model_dir):
            out.append(f"the shards in {run.data} were built with another tokenizer than {run.model_dir}'s")
    train = shard_files(run.data, "train")
    if not train:
        out.append(f"data: no train-*.bin in {run.data}")
    elif run.fixed_batch:
        rows = next(s["rows"] for s in data["shards"] if s["file"] == train[0].name)
        if rows < run.tokens_per_update // seq_len:
            out.append(f"fixed_batch needs {run.tokens_per_update // seq_len} rows; {train[0].name} has {rows}")
    return out
