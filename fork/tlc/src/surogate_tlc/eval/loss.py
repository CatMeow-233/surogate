"""Teacher-forced loss of a model directory on a split of a TLC shard directory (README: 评测).

The rows are scored as the native DataLoader serves them (data.shards.served_rows), by the native trainer's
compute_logprobs without LoRA; the loss is the mean negative log-probability over the valid targets. The baseline
is the add-one smoothed unigram distribution of the train split's targets: v0.7's gate G3 asks that the validation
loss be below it.
"""

from __future__ import annotations

import math
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..data.shards import manifest_sha256, read_shard, served_rows
from ..data.tokenizer import tokenizer_sha256
from ..modeldir import check_model_dir
from ..train.config import manifest, shard_files

IGNORE = -100
# (inputs, targets, positions), int32 [batch, seq_len] -> log-probabilities of the targets, 0 where IGNORE.
Scorer = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def unigram_loss(data: Path, targets: np.ndarray, vocab_size: int) -> float:
    """Mean -log p(target) over the valid `targets` under the train split's add-one smoothed target unigram."""
    seq_len = manifest(data)["format"]["seq_len"]
    counts = np.ones(vocab_size, dtype=np.float64)
    for path in shard_files(data, "train"):
        shard = read_shard(path)
        rows = len(shard.tokens) // seq_len
        for start in range(0, rows, 65536):
            span = slice(start * seq_len, min(rows, start + 65536) * seq_len)
            tokens = np.asarray(shard.tokens[span]).reshape(-1, seq_len)
            masks = shard.masks[span].reshape(-1, seq_len)
            # Position i's target is token i + 1 when mask bit i is set, as served_rows reads it.
            counts += np.bincount(tokens[:, 1:][masks[:, :-1]], minlength=vocab_size)
    valid = targets[targets != IGNORE]
    return float(-np.log(counts[valid] / counts.sum()).mean())


def _native_scorer(model_dir: Path, data: Path, out: Path, *, batch: int, seq_len: int) -> Scorer:
    from ..train.driver import native_trainer

    trainer = native_trainer(model_dir, data, out, batch=batch, seq_len=seq_len).trainer

    def score(inputs: np.ndarray, targets: np.ndarray, positions: np.ndarray) -> np.ndarray:
        return np.asarray(trainer.compute_logprobs(inputs, targets, use_lora=False, position_ids=positions))

    return score


def evaluate(
    model_dir: Path,
    data: Path,
    *,
    split: str = "val",
    batch: int = 32,
    max_rows: int | None = None,
    scorer: Scorer | None = None,
) -> dict[str, Any]:
    """Loss of `model_dir` on `split` of shard directory `data`; `scorer` defaults to the native trainer."""
    model_dir, data = Path(model_dir), Path(data)
    cfg, info = check_model_dir(model_dir), manifest(data)
    if info["tokenizer"]["sha256"] != tokenizer_sha256(model_dir):
        raise ValueError(f"the shards in {data} were built with another tokenizer than {model_dir}'s")
    seq_len = info["format"]["seq_len"]
    parts = [served_rows(read_shard(path), seq_len) for path in shard_files(data, split)]
    if not parts:
        raise ValueError(f"no {split}-*.bin in {data}")
    inputs, targets, positions = (np.concatenate(arrays)[:max_rows].astype(np.int32) for arrays in zip(*parts))
    total, count = 0.0, 0
    with tempfile.TemporaryDirectory(prefix="tlc-eval-") as tmp:
        score = scorer or _native_scorer(model_dir, data, Path(tmp), batch=batch, seq_len=seq_len)
        for start in range(0, len(inputs), batch):
            rows = slice(start, start + batch)
            n = len(inputs[rows])
            # The trainer scores whole batches: the last one is filled with copies of its first row, unscored.
            fill = batch - n
            ids, tgt, pos = (
                np.concatenate([a[rows], np.repeat(a[rows][:1], fill, 0)]) for a in (inputs, targets, positions)
            )
            tgt[n:] = IGNORE
            logprobs = score(np.ascontiguousarray(ids), np.ascontiguousarray(tgt), np.ascontiguousarray(pos))[:n]
            valid = targets[rows] != IGNORE
            total -= float(logprobs[valid].sum())
            count += int(valid.sum())
    if not count:
        raise ValueError(f"{split} of {data} has no valid targets")
    loss = total / count
    unigram = unigram_loss(data, targets, cfg["vocab_size"])
    return {
        "model_dir": str(model_dir),
        "data": str(data),
        "tokenizer_sha256": info["tokenizer"]["sha256"],
        "data_manifest_sha256": manifest_sha256(data),
        "split": split,
        "rows": len(inputs),
        "valid_targets": count,
        "loss": loss,
        "perplexity": math.exp(loss),
        "unigram_loss": unigram,
        "below_unigram": loss < unigram,
    }
