"""`metrics.jsonl` (`bilingual-metrics/1`): one JSON object per logged optimizer step (docs/spec.md, I5; IMPL-09).

    {"format": "bilingual-metrics/1", "step": 100, "tokens": 6553600, "targets": 6553500, "loss": 3.21, "kl": null,
     "lr": 0.0006, "grad_norm": 0.83, "tokens_per_s": 61234.0, "step_s": 1.07, "peak_alloc_gib": 21.3,
     "peak_reserved_gib": 23.0, "dev": null, "time": "..."}

`step` counts completed optimizer steps and `lr` is the rate that step used; `tokens` counts input tokens so far,
`targets` the loss positions of this step, `loss` the step's objective per target (cross entropy, plus anchor_kl x
KL when the anchor is on, with the KL itself in `kl`). The CUDA fields are null on the CPU. Lines are appended and
flushed; on resume the file is cut back atomically to the steps the restored state covers (a torn last line from a
crash is dropped), so a resumed run never logs a step twice.
"""

from __future__ import annotations

import datetime
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..canonical import atomic_write_bytes

METRICS_FORMAT = "bilingual-metrics/1"
FILE = "metrics.jsonl"
NUMBER = (int, float)
FIELDS: dict[str, tuple[tuple[type, ...], bool]] = {  # name -> (types, may be null)
    "format": ((str,), False),
    "step": ((int,), False),
    "tokens": ((int,), False),
    "targets": ((int,), False),
    "loss": (NUMBER, False),
    "kl": (NUMBER, True),
    "lr": (NUMBER, False),
    "grad_norm": (NUMBER, False),
    "tokens_per_s": (NUMBER, False),
    "step_s": (NUMBER, False),
    "peak_alloc_gib": (NUMBER, True),
    "peak_reserved_gib": (NUMBER, True),
    "dev": ((dict,), True),
    "time": ((str,), False),
}


class MetricsError(ValueError):
    pass


def line(**values: Any) -> dict[str, Any]:
    doc = {"format": METRICS_FORMAT, "kl": None, "dev": None,
           "time": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"), **values}  # fmt: skip
    check_line(doc)
    return doc


def check_line(doc: Mapping[str, Any]) -> None:
    """Raise MetricsError unless `doc` is a bilingual-metrics/1 line: exactly the known fields, each of its type."""
    if doc.get("format") != METRICS_FORMAT:
        raise MetricsError(f"not a {METRICS_FORMAT} line: {doc.get('format')!r}")
    missing, unknown = sorted(set(FIELDS) - set(doc)), sorted(set(doc) - set(FIELDS))
    if missing or unknown:
        raise MetricsError(f"metrics line: missing {missing}, unknown {unknown}")
    for name, (types, nullable) in FIELDS.items():
        value = doc[name]
        if value is None and nullable:
            continue
        if isinstance(value, bool) or not isinstance(value, types):
            raise MetricsError(f"metrics field {name} = {value!r} is not {'/'.join(t.__name__ for t in types)}")
        if isinstance(value, float) and not math.isfinite(value):
            raise MetricsError(f"metrics field {name} is not finite: {value}")


def append(path: Path, doc: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(doc, sort_keys=True, allow_nan=False) + "\n")
        f.flush()


def read(path: Path) -> list[dict[str, Any]]:
    """Every complete line; a torn last line (a crash mid-write) is skipped, a bad line elsewhere raises."""
    path = Path(path)
    if not path.is_file():
        return []
    lines = path.read_text("utf-8").split("\n")
    out = []
    for i, text in enumerate(lines):
        if not text:
            continue
        try:
            out.append(json.loads(text))
        except json.JSONDecodeError:
            if i == len(lines) - 1:  # no trailing newline: the write was cut
                break
            raise MetricsError(f"{path}: line {i + 1} is not JSON") from None
    return out


def truncate(path: Path, step: int) -> int:
    """Keep the lines of steps <= `step`; returns how many were dropped."""
    path = Path(path)
    if not path.is_file():
        return 0
    lines = read(path)
    kept = [doc for doc in lines if doc["step"] <= step]
    if len(kept) != len(lines) or not path.read_text("utf-8").endswith("\n"):
        atomic_write_bytes(path, "".join(json.dumps(d, sort_keys=True) + "\n" for d in kept).encode("utf-8"))
    return len(lines) - len(kept)
