"""Teacher-forced loss of a model directory on a split of a TLC shard directory (docs/training.md: 评测).

The rows are scored as the native DataLoader serves them (data.shards.served_rows), by the native trainer's
compute_logprobs without LoRA; the loss is the mean negative log-probability over the valid targets. The baseline
is the add-one smoothed unigram distribution of the train split's targets: gate G3 asks that the validation loss
be below it, since a model that has not learnt even the token frequencies has not trained. The loss is also given
per source and per source kind (code, text) from the row ranges the shard manifest records; the text kind's
perplexity is the English perplexity.
"""

from __future__ import annotations

import math
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..data.shards import IGNORE, check_tokenizer, manifest, manifest_sha256, read_shard, served_rows, shard_files
from ..modeldir import check_model_dir
from ..native import Scorer, native_scorer


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


def _row_sources(info: dict[str, Any], files: list[Path]) -> list[str | None]:
    """The source of every served row of `files`, in order, from the manifest's per-shard row ranges."""
    shards = {entry["file"]: entry for entry in info["shards"]}
    out: list[str | None] = []
    for path in files:
        entry = shards[path.name]
        names: list[str | None] = [None] * entry["rows"]
        for part in entry.get("sources", ()):
            names[part["first_row"] : part["first_row"] + part["rows"]] = [part["name"]] * part["rows"]
        out += names
    return out


def _losses(totals: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    return {
        name: {"valid_targets": int(count), "loss": total / count, "perplexity": math.exp(total / count)}
        for name, (total, count) in sorted(totals.items())
        if count
    }


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
    cfg, info = check_model_dir(model_dir), check_tokenizer(data, model_dir)
    seq_len = info["format"]["seq_len"]
    files = shard_files(data, split)
    parts = [served_rows(read_shard(path), seq_len) for path in files]
    if not parts:
        raise ValueError(f"no {split}-*.bin in {data}")
    inputs, targets, positions = (np.concatenate(arrays)[:max_rows].astype(np.int32) for arrays in zip(*parts))
    sources = _row_sources(info, files)[: len(inputs)]
    kinds = {source["name"]: source["kind"] for source in info["sources"]}
    by_source: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    by_kind: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    total, count = 0.0, 0
    with tempfile.TemporaryDirectory(prefix="tlc-eval-") as tmp:
        score = scorer or native_scorer(model_dir, data, Path(tmp), batch=batch, seq_len=seq_len)
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
            for row, source in enumerate(sources[rows]):
                if source is None:
                    continue
                row_loss, row_count = -float(logprobs[row][valid[row]].sum()), int(valid[row].sum())
                for table, key in ((by_source, source), (by_kind, kinds.get(source, "unknown"))):
                    table[key][0] += row_loss
                    table[key][1] += row_count
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
        "by_source": _losses(by_source),
        "by_kind": _losses(by_kind),
    }
