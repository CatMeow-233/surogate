"""The overlap report (`bilingual-overlap/1`) as its readers use it: load it, take its flagged training-document keys,
filter documents. `data.decontam` writes the report; `shard --exclude` and `split materialize --exclude` apply it
through these functions only, so the filter they apply is the one the tests check."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..canonical import check_format
from .docs import Doc

FORMAT = "bilingual-overlap/1"


def load_report(path: Path) -> dict[str, Any]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    check_format(doc, "overlap")
    return doc


def flagged_keys(report: Mapping[str, Any]) -> frozenset[str]:
    """The keys of the training documents the report flags (whatever rule matched)."""
    return frozenset(report["flagged"])


def filter_docs(docs: Iterable[Doc], report: Mapping[str, Any]) -> list[Doc]:
    """The documents the report did not flag, in order."""
    flagged = flagged_keys(report)
    return [d for d in docs if d.key not in flagged]
