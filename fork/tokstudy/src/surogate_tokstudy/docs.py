"""Evaluation documents: the study's one document type, and readers of plain sample files.

An EvalDoc is a key, a domain (TLC: "py" | "en"; bilingual: the segment language) and a sequence of segments, each
either a special token (by name) or text. Every arm encodes a document segment by segment, so tokens never merge
across a special (README: 测量契约). The project adapters that frame the projects' own splits exactly as their
shards do (tlc_documents, bilingual_documents) land with step S9; until then the CPU screens read plain samples:

- `.jsonl`: one JSON object per line, the text in `field` (default "text"): fork/corpora export parts and TLC
  corpus JSONL files; key `<file name>:<line number>`.
- `.json`: a datasets-server rows response, `{"rows": [{"row_idx": i, "row": {...}}]}`, the text in `field`
  (default "content", the UltraData column); key `<file name>:<row_idx>`.

Text is taken as it is: no normalisation, no newline translation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .canonical import TokstudyError

DOMAIN_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")


@dataclass(frozen=True)
class Seg:
    special: str | None = None
    text: str | None = None

    def __post_init__(self):
        if (self.special is None) == (self.text is None):
            raise TokstudyError("a segment is exactly one of a special token or a text")


@dataclass(frozen=True)
class EvalDoc:
    key: str
    domain: str
    segs: tuple[Seg, ...]

    def texts(self) -> list[str]:
        return [s.text for s in self.segs if s.text is not None]


def check_domain(domain: str) -> str:
    if not DOMAIN_PATTERN.fullmatch(domain):
        raise TokstudyError(f"domain {domain!r}: lower-case letters, digits, '-' and '_', starting with a letter")
    return domain


def plain_documents(
    path: Path, domain: str, *, field: str | None = None, limit: int | None = None
) -> Iterator[EvalDoc]:
    """One single-text-segment EvalDoc per record of a `.jsonl` or datasets-server `.json` sample file."""
    path, domain = Path(path), check_domain(domain)
    if path.suffix == ".jsonl":
        field = field or "text"
        with path.open(encoding="utf-8") as f:
            records = ((f"{path.name}:{n}", line) for n, line in enumerate(f, 1) if line.strip())
            yield from _take(
                ((key, _record(path, key, json.loads(line), field)) for key, line in records), domain, limit
            )
    elif path.suffix == ".json":
        field = field or "content"
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("rows") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise TokstudyError(f"{path}: not a datasets-server rows response (no 'rows' list)")
        items = ((f"{path.name}:{r.get('row_idx', n)}", r.get("row")) for n, r in enumerate(rows))
        yield from _take(((key, _record(path, key, row, field)) for key, row in items), domain, limit)
    else:
        raise TokstudyError(f"{path}: a sample file is .jsonl or a datasets-server .json")


def _record(path: Path, key: str, record, field: str) -> str:
    text = record.get(field) if isinstance(record, dict) else None
    if not isinstance(text, str):
        raise TokstudyError(f"{path}: {key} has no string field {field!r}")
    return text


def _take(items, domain: str, limit: int | None) -> Iterator[EvalDoc]:
    for n, (key, text) in enumerate(items):
        if limit is not None and n >= limit:
            return
        yield EvalDoc(key, domain, (Seg(text=text),))


def parse_docs_arg(arg: str) -> tuple[str, Path]:
    """`DOMAIN=PATH` from the command line."""
    domain, sep, path = arg.partition("=")
    if not sep or not path:
        raise TokstudyError(f"--docs {arg!r}: expected DOMAIN=PATH")
    return check_domain(domain), Path(path)
