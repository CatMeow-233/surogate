"""The corpus spec: which sources make up the TLC corpus, where they came from, and the train/val split.

A spec is a TOML file (README: "语料"). Its top level holds the split settings, `seed` and `val_fraction`; each
`[[source]]` table names one source with its provenance and its input files, given as globs relative to the spec
file. A `.jsonl` file holds one document per line in the field "text", plus an optional "group" (for example the
repository) that keeps related documents in the same split; any other file is one document.

Documents are normalised the same way for the tokenizer and the shards: newlines follow Python's universal
newlines (\r\n and a lone \r become \n), so that later label alignment by UTF-8 byte offsets sees the text the
model trains on. Empty documents and documents that are not valid UTF-8 are dropped and counted.
"""

from __future__ import annotations

import glob
import hashlib
import itertools
import json
import math
import tomllib
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

KINDS = ("code", "text")
SOURCE_KEYS = ("name", "kind", "files", "url", "license", "downloaded", "filters")
OPTIONAL_SOURCE_KEYS = ("revision", "stage")
SPLITS = ("train", "val")
# Builder rules recorded with the provenance, next to each source's own filters.
RULES = (
    "newlines normalised: \\r\\n and a lone \\r become \\n",
    "documents empty after normalisation are dropped",
    "documents that are not valid UTF-8 are dropped",
    "split by sha256(seed, group): val when the hash, read as a fraction of 2**64, is below val_fraction",
)


@dataclass(frozen=True)
class Source:
    name: str
    kind: str
    files: tuple[Path, ...]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Corpus:
    path: Path
    seed: int
    val_fraction: float
    sources: tuple[Source, ...]

    def relative(self, path: Path) -> str:
        """`path` relative to the spec's directory where possible: the form recorded in manifests and keys."""
        return path.relative_to(self.path.parent).as_posix() if path.is_relative_to(self.path.parent) else str(path)


@dataclass(frozen=True)
class Document:
    source: Source
    key: str  # unique: "<source>/<file>" or "<source>/<file>:<line>"
    group: str  # the split unit
    text: str


def _source(table: dict[str, Any], base: Path) -> Source:
    name = table.get("name")
    missing = [key for key in SOURCE_KEYS if key not in table]
    unknown = sorted(set(table) - set(SOURCE_KEYS) - set(OPTIONAL_SOURCE_KEYS))
    if missing or unknown:
        raise ValueError(f"source {name!r}: missing keys {missing}, unknown keys {unknown}")
    if table["kind"] not in KINDS:
        raise ValueError(f"source {name!r}: kind must be one of {KINDS}, got {table['kind']!r}")
    for key in ("name", "url", "license", *(k for k in OPTIONAL_SOURCE_KEYS if k in table)):
        if not isinstance(table[key], str) or not table[key]:
            raise ValueError(f"source {name!r}: {key} must be a non-empty string")
    if not isinstance(table["filters"], list) or not all(isinstance(f, str) for f in table["filters"]):
        raise ValueError(f"source {name!r}: filters must be a list of strings (may be empty)")
    downloaded = table["downloaded"]
    downloaded = downloaded.isoformat() if isinstance(downloaded, date) else date.fromisoformat(downloaded).isoformat()
    files: set[Path] = set()
    for pattern in table["files"]:
        matched = [Path(p) for p in glob.glob(str(base / pattern), recursive=True) if Path(p).is_file()]
        if not matched:
            raise ValueError(f"source {name!r}: {pattern!r} matches no file")
        files.update(p.resolve() for p in matched)
    provenance = {key: table.get(key) for key in ("url", "revision", "license")}
    provenance |= {"downloaded": downloaded, "stage": table.get("stage"), "filters": list(table["filters"])}
    return Source(name, table["kind"], tuple(sorted(files)), provenance)


def load_corpus(path: Path) -> Corpus:
    """Read and check a corpus spec; every glob must match at least one file."""
    path = Path(path).resolve()
    spec = tomllib.loads(path.read_text(encoding="utf-8"))
    unknown = sorted(set(spec) - {"seed", "val_fraction", "source"})
    if unknown:
        raise ValueError(f"{path}: unknown top-level keys {unknown}")
    seed, val_fraction = spec.get("seed"), spec.get("val_fraction")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"{path}: seed must be a non-negative integer")
    if not isinstance(val_fraction, int | float) or isinstance(val_fraction, bool) or not 0 <= val_fraction < 1:
        raise ValueError(f"{path}: val_fraction must be a number in [0, 1)")
    sources = tuple(_source(table, path.parent) for table in spec.get("source", []))
    names = [s.name for s in sources]
    if not sources or len(set(names)) != len(names):
        raise ValueError(f"{path}: needs at least one [[source]], with distinct names (got {names})")
    return Corpus(path, seed, float(val_fraction), sources)


def normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _decode(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _valid_utf8(text: str) -> bool:
    """False for strings with lone surrogates, which a JSON escape such as "\\ud800" can produce."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _records(path: Path) -> Iterator[tuple[int | None, str | None, str | None]]:
    """(line, group, text) per document of `path`; text None when the document is not valid UTF-8."""
    if path.suffix != ".jsonl":
        yield None, None, _decode(path.read_bytes())
        return
    with path.open("rb") as f:
        for line_no, raw in enumerate(f, 1):
            line = _decode(raw)
            if line is None:
                yield line_no, None, None
                continue
            if not line.strip():
                continue
            record = json.loads(line)
            text, group = record.get("text"), record.get("group")
            if not isinstance(text, str) or not (group is None or isinstance(group, str)):
                raise ValueError(f"{path}:{line_no}: needs a string 'text' and an optional string 'group'")
            yield line_no, group, text


def source_documents(corpus: Corpus, source: Source, dropped: Counter | None = None) -> Iterator[Document]:
    """The documents of one source in file order, normalised; dropped ones are counted by (source, reason)."""
    for path in source.files:
        name = f"{source.name}/{corpus.relative(path)}"
        for line_no, group, text in _records(path):
            key = name if line_no is None else f"{name}:{line_no}"
            if text is None or not _valid_utf8(text):
                reason = "not_utf8"
            else:
                text = normalize(text)
                reason = None if text else "empty"
            if reason:
                if dropped is not None:
                    dropped[source.name, reason] += 1
                continue
            yield Document(source, key, f"group:{group}" if group is not None else key, text)


def documents(corpus: Corpus, dropped: Counter | None = None) -> Iterator[Document]:
    for source in corpus.sources:
        yield from source_documents(corpus, source, dropped)


def sample(corpus: Corpus, limit: int) -> list[Document]:
    """Up to `limit` documents, the first ones of each source in equal shares."""
    share = math.ceil(limit / len(corpus.sources))
    out = [doc for s in corpus.sources for doc in itertools.islice(source_documents(corpus, s), share)]
    return out[:limit]


def unit_hash(seed: int, key: str) -> int:
    """64 uniform bits from (seed, key)."""
    return int.from_bytes(hashlib.sha256(f"{seed}\0{key}".encode()).digest()[:8], "little")


def split_of(corpus: Corpus, doc: Document) -> str:
    return "val" if unit_hash(corpus.seed, doc.group) < corpus.val_fraction * 2**64 else "train"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(corpus: Corpus) -> list[dict[str, Any]]:
    """Per source: kind, provenance and the input files with their size and sha256."""
    out = []
    for s in corpus.sources:
        files = [{"path": corpus.relative(p), "bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in s.files]
        out.append({"name": s.name, "kind": s.kind, **s.provenance, "files": files})
    return out
