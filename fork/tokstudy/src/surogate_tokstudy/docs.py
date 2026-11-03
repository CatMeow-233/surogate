"""Documents: the study's one document type, the project adapters that frame each project's own split exactly as
its shards do, truncation, and the encoding of a document under a codec.

An EvalDoc is a key, a domain (TLC: "py" | "en"; bilingual: the document's one language, or "mixed") and a sequence
of segments, each either a special token (by name) or text (with its language for bilingual text). Every arm encodes
a document segment by segment, so tokens never merge across a special (README: 测量契约).

- `tlc_documents(spec, split)`: the documents of a TLC corpus spec (surogate_tlc.data.corpus: load_corpus with its
  decontamination report, source_documents, split_of, fences), framed as surogate_tlc.data.shards frames them: a
  code document is <|bos|> <|code_start|> text <|code_end|> <|eos|>; a text document is <|bos|> text <|eos|> with
  its fence lines replaced by <|code_start|> / <|code_end|> (a block still open at the end is closed there). Text is
  TLC's normalised text (universal newlines), the bytes TLC trains on. Domain "py" for code sources, "en" for text.
- `bilingual_documents(manifest, split)`: the split-materialized files of a bilingual derived manifest (format
  bilingual-manifest/1 with `derived_from`), each checked against its sha256 and read with
  surogate_bilingual.data.docs.iter_file_docs, framed as one document of a bilingual shard: <eos> (the end of the
  previous document) + the segments (tag segments as specials, text as text with its language) + <eos>. Keys are
  the project's. A segment with `loss = false` is refused: the study trains and scores every text token.
- Plain sample files (`plain_documents`, for the CPU screens): `.jsonl` (the text in `field`, default "text";
  fork/corpora export parts and TLC corpus JSONL files) or a datasets-server rows `.json` (default "content"), one
  text segment per record; `frame_plain` gives them a project's framing for evaluation.

Text is taken as it is: no normalisation beyond the project's own, no newline translation.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import TokstudyError, sha256_file
from .codecs import TLC_SPECIALS, Codec, utf8

DOMAIN_PATTERN = re.compile(r"[a-z][a-z0-9_-]*")
TLC_PAD, TLC_BOS, TLC_EOS, TLC_CODE_START, TLC_CODE_END = TLC_SPECIALS
BILINGUAL_EOS = "<eos>"
BILINGUAL_LANGS = ("zh", "en")
MIXED = "mixed"
TLC_SPLITS = ("train", "val")
BILINGUAL_MANIFEST = "bilingual-manifest/1"


@dataclass(frozen=True)
class Seg:
    special: str | None = None
    text: str | None = None
    lang: str | None = None  # the language of a bilingual text segment (route changes force H-Net boundaries)

    def __post_init__(self):
        if (self.special is None) == (self.text is None):
            raise TokstudyError("a segment is exactly one of a special token or a text")
        if self.lang is not None and self.text is None:
            raise TokstudyError("only a text segment has a language")

    def to_json(self) -> dict[str, Any]:
        if self.special is not None:
            return {"special": self.special}
        return {"text": self.text} if self.lang is None else {"text": self.text, "lang": self.lang}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Seg:
        unknown = sorted(set(data) - {"special", "text", "lang"})
        if unknown:
            raise TokstudyError(f"unknown segment fields {unknown}")
        return cls(data.get("special"), data.get("text"), data.get("lang"))


@dataclass(frozen=True)
class EvalDoc:
    key: str
    domain: str
    segs: tuple[Seg, ...]

    def texts(self) -> list[str]:
        return [s.text for s in self.segs if s.text is not None]

    def text_bytes(self) -> int:
        return sum(len(utf8(t)) for t in self.texts())

    def to_json(self) -> dict[str, Any]:
        return {"key": self.key, "domain": self.domain, "segs": [s.to_json() for s in self.segs]}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> EvalDoc:
        try:
            return cls(str(data["key"]), check_domain(data["domain"]), tuple(Seg.from_json(s) for s in data["segs"]))
        except (KeyError, TypeError) as err:
            raise TokstudyError(f"not a document record: {err!r}") from None


def check_domain(domain: str) -> str:
    if not isinstance(domain, str) or not DOMAIN_PATTERN.fullmatch(domain):
        raise TokstudyError(f"domain {domain!r}: lower-case letters, digits, '-' and '_', starting with a letter")
    return domain


# --- TLC -------------------------------------------------------------------------------------------------------------


def tlc_segments(text: str, kind: str) -> tuple[Seg, ...]:
    """The framing of surogate_tlc.data.shards.segments, with special names instead of ids (a test compares the
    two, and the served rows of TLC's own shards)."""
    from surogate_tlc.data.corpus import fences

    if kind == "code":
        return (Seg(TLC_BOS), Seg(TLC_CODE_START), Seg(text=text), Seg(TLC_CODE_END), Seg(TLC_EOS))
    if kind != "text":
        raise TokstudyError(f"a TLC source kind is code or text, not {kind!r}")
    pieces = fences(text)
    out: list[Seg] = [Seg(TLC_BOS)]
    for i, (in_code, chunk) in enumerate(pieces):
        if chunk:
            out.append(Seg(text=chunk))
        if i + 1 < len(pieces):  # the fence line after this piece
            out.append(Seg(TLC_CODE_END if in_code else TLC_CODE_START))
    if pieces[-1][0]:
        out.append(Seg(TLC_CODE_END))
    out.append(Seg(TLC_EOS))
    return tuple(out)


def tlc_documents(spec: Path, split: str) -> Iterator[EvalDoc]:
    """The `split` ("train" | "val") documents of a TLC corpus spec, in the corpus's order, framed as its shards."""
    from surogate_tlc.data.corpus import documents, load_corpus, split_of

    if split not in TLC_SPLITS:
        raise TokstudyError(f"a TLC split is one of {TLC_SPLITS}, not {split!r}")
    try:
        corpus = load_corpus(Path(spec))
        for doc in documents(corpus):
            if split_of(corpus, doc) == split:
                domain = "py" if doc.source.kind == "code" else "en"
                yield EvalDoc(doc.key, domain, tlc_segments(doc.text, doc.source.kind))
    except (OSError, ValueError) as err:
        if isinstance(err, TokstudyError):
            raise
        raise TokstudyError(f"{spec}: {err}") from None


def tlc_source(spec: Path, split: str) -> dict[str, Any]:
    """What a stream or evalset records of a TLC corpus: the spec's file name and sha256, its decontamination record,
    the split. No directory: moving the tree keeps the identity, changing a byte of the spec does not."""
    from surogate_tlc.data.corpus import decontam_record, load_corpus

    try:
        corpus = load_corpus(Path(spec))
    except (OSError, ValueError) as err:
        raise TokstudyError(f"{spec}: {err}") from None
    return {"project": "tlc", "spec": Path(spec).name, "spec_sha256": sha256_file(Path(spec)), "split": split,
            "seed": corpus.seed, "val_fraction": corpus.val_fraction, "decontam": decontam_record(corpus)}  # fmt: skip


# --- bilingual -------------------------------------------------------------------------------------------------------


def _derived_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise TokstudyError(f"{path}: cannot read a bilingual manifest: {err}") from None
    if not isinstance(doc, dict) or doc.get("format") != BILINGUAL_MANIFEST:
        raise TokstudyError(f"{path}: not a {BILINGUAL_MANIFEST} document")
    if not doc.get("derived_from"):
        raise TokstudyError(f"{path}: not a derived manifest (`split materialize`); its files carry no split")
    return doc


def bilingual_documents(manifest: Path, split: str) -> Iterator[EvalDoc]:
    """The documents of one split of a bilingual derived manifest, in the manifest's file order, framed as one
    document of a bilingual shard (module docstring)."""
    from surogate_bilingual.data.docs import iter_file_docs

    manifest = Path(manifest)
    doc = _derived_manifest(manifest)
    sources = {s["name"]: s for s in doc.get("sources", [])}
    root = manifest.parent / doc.get("root", ".")
    entries = [f for f in doc.get("files", []) if f.get("split") == split]
    if not entries:
        known = sorted({f.get("split") for f in doc.get("files", [])} - {None})
        raise TokstudyError(f"{manifest}: no files of split {split!r} (splits present: {known})")
    for entry in entries:
        path = root / entry["path"]
        found = sha256_file(path)
        if found != entry["sha256"]:
            raise TokstudyError(f"{path}: sha256 {found}, the manifest says {entry['sha256']}")
        source = sources[entry["source"]]
        try:
            for d in iter_file_docs(path, source=source["name"], kind=source["kind"], rel=entry["path"]):
                yield bilingual_doc(d)
        except ValueError as err:
            if isinstance(err, TokstudyError):
                raise
            raise TokstudyError(f"{path}: {err}") from None


def bilingual_doc(doc) -> EvalDoc:
    """One surogate_bilingual.data.docs.Doc as an EvalDoc: <eos> + segments + <eos>; domain its one language."""
    segs = [Seg(BILINGUAL_EOS)]
    for s in doc.segments:
        if not s.loss:
            raise TokstudyError(f"{doc.key}: a segment with loss = false; the study scores every text token")
        segs.append(Seg(s.tag) if s.tag is not None else Seg(text=s.text, lang=s.lang))
    segs.append(Seg(BILINGUAL_EOS))
    langs = doc.langs or ((doc.kind,) if doc.kind in BILINGUAL_LANGS else ())
    return EvalDoc(doc.key, langs[0] if len(langs) == 1 else MIXED, tuple(segs))


def bilingual_source(manifest: Path, split: str) -> dict[str, Any]:
    manifest = Path(manifest)
    doc = _derived_manifest(manifest)
    return {"project": "bilingual", "manifest": manifest.name, "manifest_sha256": sha256_file(manifest),
            "split": split, "derived_from": doc["derived_from"]}  # fmt: skip


# --- plain sample files ----------------------------------------------------------------------------------------------


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


def frame_plain(doc: EvalDoc, project: str) -> EvalDoc:
    """A one-text-segment sample document in a project's framing: TLC code ("py") or text framing (any other
    domain); bilingual <eos> text <eos> with the domain as the language when it is zh or en."""
    texts = doc.texts()
    if len(texts) != 1 or len(doc.segs) != 1:
        raise TokstudyError(f"{doc.key}: only a one-text-segment document can be framed")
    if project == "tlc":
        return EvalDoc(doc.key, doc.domain, tlc_segments(texts[0], "code" if doc.domain == "py" else "text"))
    if project == "bilingual":
        lang = doc.domain if doc.domain in BILINGUAL_LANGS else None
        return EvalDoc(doc.key, doc.domain, (Seg(BILINGUAL_EOS), Seg(text=texts[0], lang=lang), Seg(BILINGUAL_EOS)))
    raise TokstudyError(f"unknown project {project!r}")


def parse_docs_arg(arg: str) -> tuple[str, Path]:
    """`DOMAIN=PATH` from the command line."""
    domain, sep, path = arg.partition("=")
    if not sep or not path:
        raise TokstudyError(f"--docs {arg!r}: expected DOMAIN=PATH")
    return check_domain(domain), Path(path)


# --- truncation and encoding -----------------------------------------------------------------------------------------


def utf8_prefix(text: str, max_bytes: int) -> str:
    """The longest prefix of `text` whose UTF-8 encoding has at most max_bytes bytes (never splits a character)."""
    raw = utf8(text)
    if len(raw) <= max_bytes:
        return text
    cut = max(0, max_bytes)
    while cut > 0 and (raw[cut] & 0xC0) == 0x80:  # raw[cut] continues the character that starts before the cut
        cut -= 1
    return raw[:cut].decode("utf-8")


def truncate(doc: EvalDoc, max_bytes: int) -> EvalDoc:
    """The first max_bytes text bytes of `doc` (cut at a character boundary): a document of at most max_bytes text
    bytes is returned unchanged; otherwise the specials at text offsets below the cut are kept and every later
    special is dropped."""
    if max_bytes < 0:
        raise TokstudyError(f"cannot truncate to {max_bytes} bytes")
    if doc.text_bytes() <= max_bytes:
        return doc
    out: list[Seg] = []
    used = 0
    for seg in doc.segs:
        if used >= max_bytes:
            break
        if seg.special is not None:
            out.append(seg)
            continue
        piece = utf8_prefix(seg.text, max_bytes - used)
        if piece:
            out.append(Seg(text=piece, lang=seg.lang))
        used += len(utf8(piece))
        if piece != seg.text:
            break
    return EvalDoc(doc.key, doc.domain, tuple(out))


@dataclass(frozen=True)
class Encoded:
    """A document under one codec: every position's id, the text bytes its token carries (0 for a special),
    whether it is a special, and whether it starts a text segment whose language differs from the previous text
    segment's (a route change of bilingual mixed-language data)."""

    ids: np.ndarray  # int64 [n]
    token_bytes: np.ndarray  # int64 [n]
    special: np.ndarray  # bool [n]
    route_start: np.ndarray  # bool [n]

    @property
    def text_bytes(self) -> int:
        return int(self.token_bytes.sum())

    def forced(self) -> np.ndarray:
        """The positions an H-Net must treat as boundaries: every special and every route change."""
        return self.special | self.route_start


def encode_doc(codec: Codec, doc: EvalDoc | Sequence[Seg]) -> Encoded:
    """Segment by segment: a special is its id, a text its codec encoding; tokens never merge across segments."""
    segs = doc.segs if isinstance(doc, EvalDoc) else tuple(doc)
    ids: list[np.ndarray] = []
    lengths: list[np.ndarray] = []
    specials: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    lang = None
    seen_text = False
    for seg in segs:
        if seg.special is not None:
            if seg.special not in codec.special_ids:
                raise TokstudyError(f"{codec.name}: no special token {seg.special!r}")
            ids.append(np.array([codec.special_ids[seg.special]], dtype=np.int64))
            lengths.append(np.zeros(1, dtype=np.int64))
            specials.append(np.ones(1, dtype=bool))
            starts.append(np.zeros(1, dtype=bool))
            continue
        piece, ends = codec.encode_text(seg.text)
        if not piece.size:
            continue
        start = np.zeros(piece.size, dtype=bool)
        start[0] = seen_text and seg.lang != lang
        lang, seen_text = seg.lang, True
        ids.append(piece.astype(np.int64))
        lengths.append(np.diff(ends, prepend=0))
        specials.append(np.zeros(piece.size, dtype=bool))
        starts.append(start)
    if not ids:
        empty = np.zeros(0, dtype=np.int64)
        return Encoded(empty, empty.copy(), np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))
    return Encoded(*(np.concatenate(parts) for parts in (ids, lengths, specials, starts)))
