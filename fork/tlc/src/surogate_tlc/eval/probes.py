"""Binding probes: forced-choice scoring of suite items with teacher-forced log-probabilities (README: 绑定探针).

For every candidate of an item (data.suites.Item) the text prompt + candidate + tail is encoded as one document
would be in the shards (data.shards: no special ids from the text, `<|bos|> <|code_start|>` before a prompt that
starts its document). The candidate's score is the sum of the log-probabilities of the tokens from the first one
that reaches past the shared context to the one holding the first character of the tail, so a candidate is
scored as a whole word that ends there; the shared context ends where the earliest candidate's first token
starts, and an item whose candidates do not share it is not scored. A synthetic item (T1, T2) longer than the
model's context is not scored either, since cutting it could cut the binding asked about; a natural one (T3) keeps
its last seq_len tokens, as a shard window would. The item is correct when the gold scores best.

rows_of(), score_rows() and summarize() need no native code; evaluate_suite() runs them with the native trainer's
compute_logprobs, the same scorer as eval.loss. summarize() reports per task the accuracy, the chance level, each
heuristic's accuracy and the floors per stratum (v0.7 §12.2): `floor` is the best heuristic of the stratum,
`shortcut_floor` the best of those that are not exact there (a heuristic right on every item of the stratum is
the task's algorithm, as same_name is where nothing shadows); `above_floor` and `above_shortcut_floor` average
each scored item's correctness minus its stratum's floor. The `disagreement` subset holds the items every
shortcut heuristic gets wrong; `stale_rate` is the share of choices equal to the reader's previous value of the
expression when that is a wrong candidate (the salience memo's EV-BIND).
"""

from __future__ import annotations

import math
import tempfile
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..data.corpus import file_sha256
from ..data.shards import manifest_sha256
from ..data.suites import Item, read_items
from ..data.tokenizer import BOS, CODE_START, PAD, load, special_ids, tokenizer_sha256
from ..modeldir import check_model_dir
from ..train.config import manifest
from .loss import IGNORE, Scorer, native_scorer

# text -> (token ids, (start, end) character offsets per token)
Encoder = Callable[[str], tuple[list[int], list[tuple[int, int]]]]
DISTANCE_BUCKETS = ((128, "<=128"), (512, "128-512"), (2048, "512-2048"), (math.inf, ">2048"))
CELLS = {
    "T1": ("pattern", "live", "hops", "overwrites", "depth"),
    "T2": ("pattern", "formals", "distractors", "actuals"),
    "T3": (),
}


def rows_of(item: Item, encode: Encoder, frame: Sequence[int], seq_len: int) -> dict[str, Any]:
    """Token rows of an item's candidates: {"rows", "first" (index of each row's first scored token), "distance",
    "pieces"}, or {"skip": reason} with reason "context" or "too_long"."""
    p = len(item.prompt)
    encoded = [encode(item.prompt + c + item.tail) for c in item.candidates]
    starts = [next(s for s, e in offsets if e > p) for _, offsets in encoded]
    cut = min(starts)
    rows, firsts, contexts = [], [], set()
    for (ids, offsets), cand in zip(encoded, item.candidates, strict=True):
        n_context = sum(e <= cut for _, e in offsets)
        stop = p + len(cand) + 1
        n_scored = sum(s < stop for s, _ in offsets[n_context:])
        contexts.add(tuple(ids[:n_context]))
        head = list(frame) if item.start == 0 else []
        rows.append(head + list(ids[: n_context + n_scored]))
        firsts.append(len(head) + n_context)
    if len(contexts) != 1:
        return {"skip": "context"}
    drop = max(len(r) for r in rows) - seq_len
    if drop > 0:
        if item.program is not None or drop > min(firsts) - 1:  # a whole program, or a scored token without context
            return {"skip": "too_long"}
        rows, firsts = [r[drop:] for r in rows], [f - drop for f in firsts]
    ids, offsets = encoded[item.candidates.index(item.gold)]
    n_context = sum(e <= cut for _, e in offsets)
    span = item.query or (p, p + len(item.gold))
    return {
        "rows": rows,
        "first": firsts,
        "distance": None if item.anchor is None else sum(e > item.anchor for _, e in offsets[:n_context]),
        "pieces": sum(s < span[1] and e > span[0] for s, e in offsets),
    }


def score_rows(
    rows: Sequence[Sequence[int]], firsts: Sequence[int], scorer: Scorer, *, batch: int, seq_len: int, pad_id: int
) -> list[float]:
    """Sum of the scored tokens' log-probabilities per row, `batch` rows of `seq_len` per scorer call; the last call
    is filled with padding rows that have no targets."""
    out = []
    positions = np.ascontiguousarray(np.broadcast_to(np.arange(seq_len, dtype=np.int32), (batch, seq_len)))
    for start in range(0, len(rows), batch):
        part = list(zip(rows[start : start + batch], firsts[start : start + batch], strict=True))
        inputs = np.full((batch, seq_len), pad_id, np.int32)
        targets = np.full((batch, seq_len), IGNORE, np.int32)
        for b, (row, first) in enumerate(part):
            inputs[b, : len(row)] = row
            targets[b, first - 1 : len(row) - 1] = row[first:]
        logprobs = np.asarray(scorer(inputs, targets, positions), dtype=np.float64)
        out += [float(logprobs[b, first - 1 : len(row) - 1].sum()) for b, (row, first) in enumerate(part)]
    return out


def probe(
    items: Sequence[Item],
    encode: Encoder,
    scorer: Scorer,
    *,
    frame: Sequence[int],
    batch: int,
    seq_len: int,
    pad_id: int,
) -> list[dict[str, Any]]:
    """One result per item: {"id", "status", "choice", "correct", "scores", "distance", "pieces"}."""
    prepared = [rows_of(item, encode, frame, seq_len) for item in items]
    owners, rows, firsts = [], [], []
    for i, p in enumerate(prepared):
        for row, first in zip(p.get("rows", ()), p.get("first", ()), strict=True):
            owners.append(i)
            rows.append(row)
            firsts.append(first)
    scores = score_rows(rows, firsts, scorer, batch=batch, seq_len=seq_len, pad_id=pad_id)
    per_item: dict[int, list[float]] = defaultdict(list)
    for i, score in zip(owners, scores, strict=True):
        per_item[i].append(score)
    results = []
    for i, (item, p) in enumerate(zip(items, prepared, strict=True)):
        if "skip" in p:
            results.append({"id": item.id, "status": p["skip"]})
            continue
        best = max(range(len(item.candidates)), key=lambda c: per_item[i][c])
        choice = item.candidates[best]
        results.append(
            {
                "id": item.id,
                "status": "scored",
                "choice": choice,
                "correct": choice == item.gold,
                "scores": per_item[i],
                "distance": p["distance"],
                "pieces": p["pieces"],
            }
        )
    return results


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _distance(tokens: int | None) -> str | None:
    return None if tokens is None else next(label for bound, label in DISTANCE_BUCKETS if tokens <= bound)


def _groups(item: Item, result: dict[str, Any], key: str) -> list[Any]:
    if key == "heldout":
        return item.strata.get("heldout") or ["none"]
    if key == "distance":
        return [_distance(result["distance"])]
    if key == "pieces":
        return [min(result["pieces"], 4)]  # 4 means 4 or more
    return [item.strata.get(key)]


def _table(pairs: Sequence[tuple[Item, dict[str, Any]]]) -> dict[str, Any]:
    names = sorted({h for item, _ in pairs for h in item.heuristics})
    return {
        "items": len(pairs),
        "accuracy": _mean([r["correct"] for _, r in pairs]),
        "chance": _mean([1 / len(item.candidates) for item, _ in pairs]),
        "heuristics": {h: _mean([item.heuristics.get(h) == item.gold for item, _ in pairs]) for h in names},
    }


def _floors(task: str, scored: Sequence[tuple[Item, dict[str, Any]]]) -> tuple[dict[str, tuple], list[dict]]:
    """Per item id (floor, shortcut floor, in the disagreement subset), and one row per cell of the task's grid."""
    cells: dict[tuple, list] = defaultdict(list)
    for item, r in scored:
        cells[tuple(item.strata.get(k) for k in CELLS[task])].append((item, r))
    floors, rows = {}, []
    for key, pairs in sorted(cells.items(), key=str):
        table = _table(pairs)
        exact = sorted(h for h, a in table["heuristics"].items() if a == 1)
        shortcuts = [h for h in table["heuristics"] if h not in exact]
        floor = max(table["heuristics"].values(), default=0.0)
        shortcut = max((table["heuristics"][h] for h in shortcuts), default=0.0)
        for item, _ in pairs:
            floors[item.id] = (floor, shortcut, all(item.heuristics.get(h) != item.gold for h in shortcuts))
        row = {**dict(zip(CELLS[task], key, strict=True)), **table}
        rows.append({**row, "floor": floor, "shortcut_floor": shortcut, "exact": exact})
    return floors, rows


def _metrics(pairs: Sequence[tuple[Item, dict[str, Any]]], floors: dict[str, tuple]) -> dict[str, Any]:
    disagreement = [r["correct"] for item, r in pairs if floors[item.id][2]]
    stale = [(item, r) for item, r in pairs if item.heuristics.get("stale") not in (None, item.gold)]
    return {
        **_table(pairs),
        "above_floor": _mean([r["correct"] - floors[item.id][0] for item, r in pairs]),
        "above_shortcut_floor": _mean([r["correct"] - floors[item.id][1] for item, r in pairs]),
        "disagreement": {"items": len(disagreement), "accuracy": _mean(disagreement)},
        "stale_rate": _mean([r["choice"] == item.heuristics["stale"] for item, r in stale]),
    }


def summarize(items: Sequence[Item], results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per task: the overall metrics, the per-cell floors (`cells`) and the metrics by each stratum key, held-out
    axis, distance bucket (tokens from the gold's binding to the end of the prompt) and identifier pieces (`by`)."""
    by_id = {item.id: item for item in items}
    out = {}
    for task in sorted({item.task for item in items}):
        mine = [(by_id[r["id"]], r) for r in results if by_id[r["id"]].task == task]
        scored = [(item, r) for item, r in mine if r["status"] == "scored"]
        floors, cells = _floors(task, scored)
        keys = sorted({k for item, _ in scored for k in item.strata} - {"heldout"}) + ["heldout", "distance", "pieces"]
        by = {}
        for key in keys:
            groups: dict[Any, list] = defaultdict(list)
            for item, r in scored:
                for value in _groups(item, r, key):
                    groups[value].append((item, r))
            ordered = sorted(groups.items(), key=lambda kv: str(kv[0]))
            by[key] = {str(value): _metrics(pairs, floors) for value, pairs in ordered}
        skipped: dict[str, int] = defaultdict(int)
        for _, r in mine:
            if r["status"] != "scored":
                skipped[r["status"]] += 1
        out[task] = {"items": len(mine), "skipped": dict(skipped), **_metrics(scored, floors), "cells": cells, "by": by}
    return out


def evaluate_suite(
    model_dir: Path,
    data: Path,
    suite: Path,
    *,
    batch: int = 32,
    max_items: int | None = None,
    scorer: Scorer | None = None,
) -> dict[str, Any]:
    """Probe results of `model_dir` on a suite file; the rows have the shard directory `data`'s length, and
    `scorer` defaults to the native trainer."""
    model_dir, data, suite = Path(model_dir), Path(data), Path(suite)
    check_model_dir(model_dir)
    info = manifest(data)
    if info["tokenizer"]["sha256"] != tokenizer_sha256(model_dir):
        raise ValueError(f"the shards in {data} were built with another tokenizer than {model_dir}'s")
    seq_len = info["format"]["seq_len"]
    tokenizer = load(model_dir, ordinary=True)
    special = special_ids(tokenizer)

    def encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
        encoding = tokenizer.encode(text, add_special_tokens=False)
        return encoding.ids, encoding.offsets

    items = read_items(suite)[:max_items]
    with tempfile.TemporaryDirectory(prefix="tlc-probe-") as tmp:
        score = scorer or native_scorer(model_dir, data, Path(tmp), batch=batch, seq_len=seq_len)
        frame = [special[BOS], special[CODE_START]]
        results = probe(items, encode, score, frame=frame, batch=batch, seq_len=seq_len, pad_id=special[PAD])
    return {
        "model_dir": str(model_dir),
        "data": str(data),
        "suite": str(suite),
        "suite_sha256": file_sha256(suite),
        "tokenizer_sha256": info["tokenizer"]["sha256"],
        "data_manifest_sha256": manifest_sha256(data),
        "seq_len": seq_len,
        "summary": summarize(items, results),
        "results": results,
    }
