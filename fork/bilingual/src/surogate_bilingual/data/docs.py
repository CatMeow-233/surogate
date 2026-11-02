"""The document model of the data pipeline and its JSON Lines form (docs/spec.md, I4 落地记录).

A document is a sequence of segments. A text segment holds text in one language (`zh`, `en`, or none) and is
encoded as user text, so a literal "<|en|>" inside it stays plain text (IMPL-03). A tag segment is one control
tag (`<|zh|>`, `<|en|>`, `<|bridge|>`) and is encoded as its special id; only the D_seed builder writes tag
segments (B-O3: tags are ordinary tokens of D_seed and never part of user text). A segment may name the route of
its tokens and whether they are prediction targets; the shard writer resolves both.

Corpus files are `.jsonl` (one JSON object per non-empty line, with `text` or `segments`) or `.txt` (documents
separated by one or more blank lines). A document's key is its `key` field if it has one, else
`<source>/<id>` from its `id` field, else `<source>/<relative path>#<index>`; keys are unique within a manifest.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..canonical import canonical_json

KINDS = ("zh", "en", "parallel", "codeswitch", "synthetic")
LANGS = ("zh", "en")
CONTROL_TAGS = ("<|zh|>", "<|en|>", "<|bridge|>")
TAG_OF_LANG = {"zh": "<|zh|>", "en": "<|en|>"}
CORPUS_SUFFIXES = (".jsonl", ".txt")

# Han ideographs: the CJK Unified Ideographs block, extensions A-H and the compatibility ideographs. Used by the
# zh decontamination rule (Han 25-grams) and by the script segmentation of code-switched text.
HAN_RANGES = (
    (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF), (0x2A700, 0x2EBEF),
    (0x2EBF0, 0x2EE5F), (0x2F800, 0x2FA1F), (0x30000, 0x3134F), (0x31350, 0x323AF),
)  # fmt: skip


class DocError(ValueError):
    """A corpus file or document that the pipeline refuses."""


def is_han(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in HAN_RANGES)


def is_latin_letter(ch: str) -> bool:
    return ch.isascii() and ch.isalpha() or (ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN"))


@dataclass(frozen=True)
class Segment:
    text: str | None = None
    tag: str | None = None
    lang: str | None = None
    role: str | None = None
    route: str | None = None
    loss: bool = True

    def __post_init__(self):
        if (self.text is None) == (self.tag is None):
            raise DocError("a segment holds exactly one of text or tag")
        if self.tag is not None and self.tag not in CONTROL_TAGS:
            raise DocError(f"tag segments hold a control tag {CONTROL_TAGS}, not {self.tag!r}")
        if self.text is not None and not isinstance(self.text, str):
            raise DocError(f"segment text must be a string: {self.text!r}")
        if self.lang is not None and self.lang not in LANGS:
            raise DocError(f"segment lang must be one of {LANGS} or null: {self.lang!r}")
        if not isinstance(self.loss, bool):
            raise DocError(f"segment loss must be a bool: {self.loss!r}")

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"tag": self.tag} if self.tag is not None else {"text": self.text}
        for key in ("lang", "role", "route"):
            if getattr(self, key) is not None:
                out[key] = getattr(self, key)
        if not self.loss:
            out["loss"] = False
        return out

    @classmethod
    def from_json(cls, data: Mapping[str, Any], default_lang: str | None = None) -> Segment:
        unknown = set(data) - {"text", "tag", "lang", "role", "route", "loss"}
        if unknown:
            raise DocError(f"unknown segment fields {sorted(unknown)}")
        lang = data.get("lang", default_lang if "text" in data else None)
        return cls(data.get("text"), data.get("tag"), lang, data.get("role"), data.get("route"), data.get("loss", True))


@dataclass(frozen=True)
class Doc:
    key: str
    kind: str
    segments: tuple[Segment, ...]
    source: str | None = None
    pair_id: str | None = None
    entity_id: str | None = None
    route: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.key:
            raise DocError("a document needs a non-empty key")
        if self.kind not in KINDS:
            raise DocError(f"{self.key}: kind must be one of {KINDS}: {self.kind!r}")
        object.__setattr__(self, "segments", tuple(self.segments))
        if not any(s.text for s in self.segments):
            raise DocError(f"{self.key}: a document needs at least one non-empty text segment")

    @property
    def text(self) -> str:
        """The text segments joined by newlines (tags left out): what clustering and decontamination read."""
        return "\n".join(s.text for s in self.segments if s.text is not None)

    @property
    def langs(self) -> tuple[str, ...]:
        return tuple(sorted({s.lang for s in self.segments if s.lang is not None}))

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"key": self.key, "kind": self.kind, "segments": [s.to_json() for s in self.segments]}
        for name in ("source", "pair_id", "entity_id", "route"):
            if getattr(self, name) is not None:
                out[name] = getattr(self, name)
        if self.meta:
            out["meta"] = dict(self.meta)
        return out

    def replace(self, **changes) -> Doc:
        values = {f: getattr(self, f) for f in self.__dataclass_fields__}
        values.update(changes)
        return Doc(**values)


def doc_line(doc: Doc) -> bytes:
    """One canonical JSON line (sorted keys, UTF-8): byte-identical for equal documents."""
    return canonical_json(doc.to_json()) + b"\n"


def doc_from_json(data: Mapping[str, Any], *, source: str | None, kind: str | None, fallback_key: str) -> Doc:
    known = {"key", "id", "kind", "text", "segments", "lang", "source", "pair_id", "entity_id", "route", "meta"}
    unknown = set(data) - known
    if unknown:
        raise DocError(f"{fallback_key}: unknown document fields {sorted(unknown)}")
    kind = data.get("kind", kind)
    if kind is None:
        raise DocError(f"{fallback_key}: the document has no kind and its source declares none")
    default_lang = data.get("lang", kind if kind in LANGS else None)
    if ("text" in data) == ("segments" in data):
        raise DocError(f"{fallback_key}: a document holds exactly one of text or segments")
    if "text" in data:
        segments = (Segment(text=data["text"], lang=default_lang),)
    else:
        segments = tuple(Segment.from_json(s, default_lang) for s in data["segments"])
    source = source if source is not None else data.get("source")  # the manifest's source name wins
    if "key" in data:
        key = str(data["key"])
    elif "id" in data:
        key = f"{source}/{data['id']}" if source else str(data["id"])
    else:
        key = fallback_key
    return Doc(
        key=key,
        kind=kind,
        segments=segments,
        source=source,
        pair_id=data.get("pair_id"),
        entity_id=data.get("entity_id"),
        route=data.get("route"),
        meta=data.get("meta", {}),
    )


def iter_file_docs(path: Path, *, source: str | None, kind: str | None, rel: str | None = None) -> Iterator[Doc]:
    """The documents of one corpus file, in file order."""
    path = Path(path)
    rel = rel or path.name
    prefix = f"{source}/{rel}" if source else rel
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            index = 0
            for number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as err:
                    raise DocError(f"{path}:{number}: not JSON: {err}") from None
                if not isinstance(data, dict):
                    raise DocError(f"{path}:{number}: a document line must be a JSON object")
                yield doc_from_json(data, source=source, kind=kind, fallback_key=f"{prefix}#{index}")
                index += 1
    elif path.suffix == ".txt":
        if kind not in KINDS:
            raise DocError(f"{path}: a .txt corpus file needs its source's kind")
        text = path.read_text(encoding="utf-8")
        blocks = [b.strip("\n") for b in _blank_line_blocks(text)]
        for index, block in enumerate(b for b in blocks if b.strip()):
            lang = kind if kind in LANGS else None
            yield Doc(f"{prefix}#{index}", kind, (Segment(text=block, lang=lang),), source=source)
    else:
        raise DocError(f"{path}: corpus files are {CORPUS_SUFFIXES}, not {path.suffix!r}")


def _blank_line_blocks(text: str) -> list[str]:
    blocks, current = [], []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def write_docs(path: Path, docs: Sequence[Doc] | Iterator[Doc]) -> int:
    """Write documents as canonical JSON lines (atomically); return how many."""
    from ..canonical import atomic_write_bytes

    lines = [doc_line(d) for d in docs]
    atomic_write_bytes(Path(path), b"".join(lines))
    return len(lines)


def script_segments(text: str, *, lang: str | None = None) -> list[Segment]:
    """Split code-switched text into zh and en runs by script: a Han character starts or continues a zh run, a
    Latin letter an en run; everything else (spaces, digits, punctuation) stays in the current run. Text before
    the first letter of either script joins the first run. [建议] a script heuristic, not language identification."""
    runs: list[list[Any]] = []
    for ch in text:
        script = "zh" if is_han(ch) else "en" if is_latin_letter(ch) else None
        if script is None or (runs and runs[-1][0] == script):
            if runs:
                runs[-1][1].append(ch)
            else:
                runs.append([None, [ch]])
            continue
        if runs and runs[-1][0] is None:
            runs[-1][0] = script
            runs[-1][1].append(ch)
        else:
            runs.append([script, [ch]])
    if not runs:
        return [Segment(text=text, lang=lang)]
    return [Segment(text="".join(chars), lang=script if script is not None else lang) for script, chars in runs]
