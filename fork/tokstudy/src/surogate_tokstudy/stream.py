"""The training stream of a table (`tokstudy-stream/1`): one document order, one byte mixture, shared by every arm,
and the rows each arm trains on.

**Order.** Every document gets the rank key sha256("<seed>\\0<key>"); within a domain the stream takes documents in
ascending rank (a seeded shuffle that does not depend on the input order). With a mixture {domain: share}, the
domains are interleaved so that the measured units taken so far stay at their shares: the next document comes from
the domain whose units / share is lowest (ties by domain name), until each domain reaches share x total (the last
document of a domain may overshoot by less than one document). Units are UTF-8 text bytes, or the tokens of a
reference codec (STUDY M9: the reference tokenizer fixes the document list). `max_units` bounds the total; without
it the scarcest domain bounds it. Documents of a domain outside the mixture are left out and counted. Without a
mixture all documents are taken in rank order.

**Storage.** `build_stream` reads the documents once, spills them into rank-prefix buckets on disk and writes the
chosen order as `docs-NNNNN.jsonl` parts (one canonical EvalDoc per line) and `stream.json`, the manifest with the
source's identity, the split, the seed, the mixture (declared and achieved), the sha256 of the ordered key list
and of every part. Memory holds one bucket per domain at a time. The stream's identity (`StreamManifest.sha256`) is
the sha256 of the manifest's canonical JSON; it enters every run id.

**Rows.** A document is cut into windows of at most k_train positions, where a text byte and a special each take
one position (what the byte codec spends on them), at character boundaries; every arm encodes the same windows, so
the rows of all arms cover the same byte spans in the same order, one document per row (no packing, as TLC's
shards). A subword row then holds on average t_arm = round8(k_train / c_arm) tokens, c_arm the arm's measured bytes
per token; byte and H-Net rows hold the window's positions. The first token of a continuation window is input only
(its target would need the previous window). An H-Net row's forced boundaries are its specials and, for bilingual
mixed-language documents, every route (language) change.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import TokstudyError, atomic_write_json, canonical_json, check_format, sha256_file, sha256_json
from .codecs import Codec, utf8
from .docs import EvalDoc, Seg, check_domain, encode_doc, utf8_prefix

FORMAT = "tokstudy-stream/1"
MANIFEST = "stream.json"
PART_BYTES = 64 << 20  # a part closes at 64 MiB of lines
BUCKETS = 256
FLUSH_BYTES = 1 << 20
MIN_K = 8  # a window must hold a 4-byte character and the specials around it


def rank_key(seed: int, key: str) -> str:
    return hashlib.sha256(f"{seed}\0{key}".encode()).hexdigest()


def check_mixture(mixture: Mapping[str, float] | None) -> dict[str, float] | None:
    if mixture is None:
        return None
    try:
        out = {check_domain(d): float(s) for d, s in sorted(mixture.items())}
    except (TypeError, ValueError):
        raise TokstudyError(f"a mixture is {{domain: share}} with numeric shares, got {dict(mixture)}") from None
    if not out or any(not s > 0 for s in out.values()) or abs(sum(out.values()) - 1.0) > 1e-9:
        raise TokstudyError(f"a mixture is {{domain: share}} with positive shares summing to 1, got {dict(mixture)}")
    return out


@dataclass(frozen=True)
class StreamManifest:
    root: Path
    doc: dict[str, Any]

    @property
    def sha256(self) -> str:
        return sha256_json(self.doc)

    @property
    def k_train(self) -> int:
        return int(self.doc["k_train"])

    @property
    def n_docs(self) -> int:
        return int(self.doc["docs"])

    @property
    def project(self) -> str:
        return str(self.doc["source"]["project"])


def _units(doc: EvalDoc, measure: Codec | None) -> int:
    if measure is None:
        return doc.text_bytes()
    return int(sum(measure.encode_text(t)[0].size for t in doc.texts()))


class _Buckets:
    """Spill files <tmp>/<group>/<bucket>.jsonl, written through per-bucket buffers."""

    def __init__(self, root: Path):
        self.root = root
        self.buffers: dict[tuple[str, int], list[bytes]] = {}
        self.sizes: Counter = Counter()

    def add(self, group: str, rank: str, line: bytes) -> None:
        bucket = (group, int(rank[:2], 16))
        self.buffers.setdefault(bucket, []).append(line)
        self.sizes[bucket] += len(line)
        if self.sizes[bucket] >= FLUSH_BYTES:
            self._flush(bucket)

    def _flush(self, bucket: tuple[str, int]) -> None:
        path = self.root / bucket[0] / f"{bucket[1]:03d}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as f:
            f.write(b"".join(self.buffers.pop(bucket, [])))
        self.sizes[bucket] = 0

    def close(self) -> None:
        for bucket in list(self.buffers):
            self._flush(bucket)

    def records(self, group: str) -> Iterator[dict[str, Any]]:
        """The group's records in ascending rank."""
        for b in range(BUCKETS):
            path = self.root / group / f"{b:03d}.jsonl"
            if path.is_file():
                with path.open("rb") as f:
                    records = [json.loads(line) for line in f]
                yield from sorted(records, key=lambda r: r["rank"])


def build_stream(
    docs: Iterable[EvalDoc],
    out: Path,
    *,
    seed: int,
    k_train: int,
    source: Mapping[str, Any],
    mixture: Mapping[str, float] | None = None,
    max_units: int | None = None,
    measure: Codec | None = None,
) -> StreamManifest:
    """Write the stream of `docs` into `out` (new or empty) and return its manifest (module docstring)."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise TokstudyError(f"{out} is not empty")
    if k_train < MIN_K:
        raise TokstudyError(f"k_train must be at least {MIN_K} positions, got {k_train}")
    if max_units is not None and max_units <= 0:
        raise TokstudyError(f"max_units must be positive, got {max_units}")
    mixture = check_mixture(mixture)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / ".build"
    try:
        return _build(docs, out, tmp, seed=seed, k_train=k_train, source=dict(source), mixture=mixture,
                      max_units=max_units, measure=measure)  # fmt: skip
    except BaseException:
        if not (out / MANIFEST).exists():
            shutil.rmtree(out, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _build(docs, out, tmp, *, seed, k_train, source, mixture, max_units, measure) -> StreamManifest:
    buckets = _Buckets(tmp)
    seen: set[str] = set()
    available: Counter = Counter()
    kept_docs = 0
    dropped: Counter = Counter()
    for doc in docs:
        if doc.key in seen:
            raise TokstudyError(f"document key {doc.key!r} occurs twice")
        seen.add(doc.key)
        if mixture is not None and doc.domain not in mixture:
            dropped[f"domain {doc.domain} not in the mixture"] += 1
            continue
        units = _units(doc, measure)
        if units <= 0:
            dropped["no text"] += 1
            continue
        group = doc.domain if mixture is not None else "all"
        rank = rank_key(seed, doc.key)
        available[group] += units
        kept_docs += 1
        line = canonical_json({"rank": rank, "units": units, "doc": doc.to_json()}) + b"\n"
        buckets.add(group, rank, line)
    buckets.close()
    shares = mixture if mixture is not None else {"all": 1.0}
    missing = sorted(set(shares) - set(available))
    if missing:
        raise TokstudyError(f"no documents of domain(s) {missing} for the mixture {mixture}")
    if max_units is None:
        total = min(available[d] / s for d, s in shares.items())
    else:
        total = float(max_units)
        short = {d: (available[d], s * total) for d, s in shares.items() if available[d] < s * total}
        if short:
            raise TokstudyError(f"too few units for {max_units} at the mixture {mixture}: (have, need) {short}")
    quota = {d: s * total for d, s in shares.items()}
    order = hashlib.sha256()
    taken: Counter = Counter()
    counts = {k: Counter() for k in ("docs", "bytes", "units")}
    writer = _PartWriter(out)
    iterators = {d: buckets.records(d) for d in shares}
    heads = {d: next(it, None) for d, it in iterators.items()}
    while True:
        open_ = [d for d in shares if heads[d] is not None and taken[d] < quota[d]]
        if not open_:
            break
        d = min(open_, key=lambda g: (taken[g] / shares[g], g))
        record = heads[d]
        heads[d] = next(iterators[d], None)
        doc = EvalDoc.from_json(record["doc"])
        taken[d] += record["units"]
        order.update(doc.key.encode("utf-8") + b"\n")
        writer.add(doc)
        for name, value in (("docs", 1), ("bytes", doc.text_bytes()), ("units", record["units"])):
            counts[name][doc.domain] += value
    parts = writer.close()
    total_units = sum(counts["units"].values())
    manifest = {
        "format": FORMAT,
        "source": source,
        "seed": seed,
        "k_train": k_train,
        "order_rule": "rank sha256(seed\\0key) within a domain; domains interleaved by units/share (lowest first)",
        "mixture": mixture,
        "measure": {"unit": "bytes"} if measure is None else {"unit": "tokens", "codec": measure.identity()},
        "max_units": max_units,
        "docs": sum(counts["docs"].values()),
        "bytes": sum(counts["bytes"].values()),
        "units": total_units,
        "by_domain": {d: {k: counts[k][d] for k in counts} for d in sorted(counts["docs"])},
        "achieved_mixture": {d: counts["units"][d] / total_units for d in sorted(counts["units"])}
        if total_units
        else {},
        "order_sha256": order.hexdigest(),
        "parts": parts,
        "dropped": dict(sorted(dropped.items())),
        "unused_docs": kept_docs - sum(counts["docs"].values()),
    }
    atomic_write_json(out / MANIFEST, manifest)
    return StreamManifest(out, manifest)


class _PartWriter:
    def __init__(self, root: Path):
        self.root, self.parts, self.file, self.size = root, [], None, 0

    def add(self, doc: EvalDoc) -> None:
        line = canonical_json(doc.to_json()) + b"\n"
        if self.file is None or (self.size + len(line) > PART_BYTES and self.parts[-1]["docs"]):
            self._next()
        self.file.write(line)
        self.size += len(line)
        self.parts[-1]["docs"] += 1
        self.parts[-1]["bytes"] += doc.text_bytes()

    def _next(self) -> None:
        self._finish()
        name = f"docs-{len(self.parts):05d}.jsonl"
        self.parts.append({"path": name, "docs": 0, "bytes": 0})
        self.file = (self.root / name).open("wb")
        self.size = 0

    def _finish(self) -> None:
        if self.file is not None:
            self.file.close()
            self.parts[-1]["sha256"] = sha256_file(self.root / self.parts[-1]["path"])
            self.file = None

    def close(self) -> list[dict[str, Any]]:
        self._finish()
        return self.parts


def open_stream(path: Path, *, verify: bool = True) -> StreamManifest:
    """A stream directory; with verify, every part's sha256 is checked."""
    root = Path(path)
    try:
        doc = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise TokstudyError(f"{root}: cannot read {MANIFEST}: {err}") from None
    check_format(doc, "stream")
    if verify:
        for part in doc["parts"]:
            found = sha256_file(root / part["path"])
            if found != part["sha256"]:
                raise TokstudyError(f"{root / part['path']}: sha256 {found}, the stream says {part['sha256']}")
    return StreamManifest(root, doc)


def iter_docs(stream: StreamManifest, start: int = 0) -> Iterator[tuple[int, EvalDoc]]:
    """(index, document) in stream order from document `start`."""
    first = 0
    for part in stream.doc["parts"]:
        if start >= first + part["docs"]:
            first += part["docs"]
            continue
        with (stream.root / part["path"]).open("rb") as f:
            for n, line in enumerate(f):
                if first + n >= start:
                    yield first + n, EvalDoc.from_json(json.loads(line))
        first += part["docs"]


# --- windows and rows ------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    index: int
    segs: tuple[Seg, ...]
    span: tuple[int, int]  # text bytes [start, end) of the document
    last: bool


def windows(doc: EvalDoc, k: int) -> list[Window]:
    """The document in windows of at most k positions (a text byte or a special is one), cut at characters."""
    if k < MIN_K:
        raise TokstudyError(f"windows hold at least {MIN_K} positions, got {k}")
    out: list[tuple[tuple[Seg, ...], tuple[int, int]]] = []
    current: list[Seg] = []
    used, offset, start = 0, 0, 0

    def close():
        nonlocal current, used, start
        if current:
            out.append((tuple(current), (start, offset)))
        current, used, start = [], 0, offset

    for seg in doc.segs:
        if seg.special is not None:
            if used == k:
                close()
            current.append(seg)
            used += 1
            continue
        text = seg.text
        while text:
            if used == k:
                close()
            piece = utf8_prefix(text, k - used)
            if not piece:  # the next character does not fit in what is left of this window
                if not used:
                    raise TokstudyError(f"{doc.key}: a character longer than {k} bytes")
                close()
                continue
            n = len(utf8(piece))
            current.append(Seg(text=piece, lang=seg.lang))
            used += n
            offset += n
            text = text[len(piece) :]
    close()
    return [Window(i, segs, span, i == len(out) - 1) for i, (segs, span) in enumerate(out)]


@dataclass(frozen=True)
class Row:
    doc: int
    window: int
    key: str
    domain: str
    span: tuple[int, int]
    ids: np.ndarray  # int64 [n]
    forced: np.ndarray  # bool [n]: specials and route changes
    last: bool  # the document's last window

    @property
    def next_cursor(self) -> tuple[int, int]:
        return (self.doc + 1, 0) if self.last else (self.doc, self.window + 1)

    @property
    def text_bytes(self) -> int:
        return self.span[1] - self.span[0]


def rows_for_arm(stream: StreamManifest, codec: Codec, *, start: Sequence[int] = (0, 0)) -> Iterator[Row]:
    """The rows of one arm from cursor (document, window), in stream order (module docstring)."""
    first_doc, first_window = int(start[0]), int(start[1])
    for index, doc in iter_docs(stream, first_doc):
        for w in windows(doc, stream.k_train):
            if index == first_doc and w.index < first_window:
                continue
            encoded = encode_doc(codec, w.segs)
            if encoded.text_bytes != w.span[1] - w.span[0]:
                raise TokstudyError(f"{codec.name}: the tokens of {doc.key} window {w.index} miss bytes")
            yield Row(index, w.index, doc.key, doc.domain, w.span, encoded.ids, encoded.forced(), w.last)


def bytes_per_token(stream: StreamManifest, codec: Codec, *, max_docs: int = 2000) -> float:
    """c_arm: text bytes per text token over the stream's first documents (the train split)."""
    n_bytes = n_tokens = 0
    for index, doc in iter_docs(stream):
        if index >= max_docs:
            break
        for text in doc.texts():
            n_tokens += codec.encode_text(text)[0].size
            n_bytes += len(utf8(text))
    if not n_tokens:
        raise TokstudyError(f"{stream.root}: no text to measure bytes per token on")
    return n_bytes / n_tokens


def t_arm(k_train: int, c_arm: float) -> int:
    """round(k_train / c_arm) rounded to a multiple of 8 (at least 8)."""
    return max(8, 8 * round(k_train / c_arm / 8))


# --- command line ----------------------------------------------------------------------------------------------------


def main(argv: Sequence[str]) -> int:
    import argparse
    import sys

    from .canonical import sha256_file as file_sha256
    from .docs import (
        bilingual_documents,
        bilingual_source,
        frame_plain,
        parse_docs_arg,
        plain_documents,
        tlc_documents,
        tlc_source,
    )

    parser = argparse.ArgumentParser(prog="python -m surogate_tokstudy stream",
                                     description="build a table's training stream (order, mixture, parts)")  # fmt: skip
    parser.add_argument("--prereg", type=Path, required=True, help="the table's prereg: K_train and the mixture")
    parser.add_argument("--tlc", type=Path, help="a TLC corpus spec (split --split, default train)")
    parser.add_argument("--bilingual", type=Path, help="a bilingual derived manifest (default split seed_train)")
    parser.add_argument("--docs", action="append", default=[], help="DOMAIN=PATH sample files (framed per --project)")
    parser.add_argument("--project", choices=["tlc", "bilingual"])
    parser.add_argument("--split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-units", type=int, help="total units (bytes, or tokens with --measure)")
    parser.add_argument("--measure", metavar="REGISTRY:ARM", help="count the mixture in this subword arm's tokens")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(list(argv))
    try:
        from .arms import codec_for, load_arms
        from .prereg import load_prereg

        prereg = load_prereg(args.prereg)
        given = [x for x in (args.tlc, args.bilingual) if x is not None] + (["plain"] if args.docs else [])
        if len(given) != 1:
            raise TokstudyError("give exactly one of --tlc SPEC, --bilingual MANIFEST or --docs DOMAIN=PATH")
        project = "tlc" if prereg.table == "tlc-N" else "bilingual"
        if args.tlc is not None:
            split = args.split or "train"
            docs, source = tlc_documents(args.tlc, split), tlc_source(args.tlc, split)
        elif args.bilingual is not None:
            split = args.split or "seed_train"
            docs, source = bilingual_documents(args.bilingual, split), bilingual_source(args.bilingual, split)
        else:
            if args.project != project:
                raise TokstudyError(f"--docs needs --project {project} (the prereg's table)")
            parsed = [parse_docs_arg(a) for a in args.docs]
            docs = [frame_plain(d, project) for domain, path in parsed for d in plain_documents(path, domain)]
            source = {"project": project, "plain": [{"domain": d, "file": p.name, "sha256": file_sha256(p)}
                                                    for d, p in parsed]}  # fmt: skip
        if source["project"] != project:
            raise TokstudyError(f"the prereg is table {prereg.table}'s; these are {source['project']} documents")
        measure = None
        if (args.measure is None) != (prereg.measure == "bytes"):
            raise TokstudyError(f"the prereg measures the mixture in {prereg.measure!r}: "
                                + ("pass --measure REGISTRY:" + prereg.measure if args.measure is None
                                   else "--measure does not apply"))  # fmt: skip
        if args.measure:
            registry, _, arm_id = args.measure.rpartition(":")
            if arm_id != prereg.measure:
                raise TokstudyError(f"--measure names arm {arm_id!r}, the prereg {prereg.measure!r}")
            arms = {a.id: a for a in load_arms(Path(registry))}
            if arm_id not in arms or arms[arm_id].kind != "subword":
                raise TokstudyError(f"--measure {args.measure!r}: not a subword arm of that registry")
            measure = codec_for(arms[arm_id])
        manifest = build_stream(docs, args.out, seed=args.seed, k_train=prereg.k_train, source=source,
                                mixture=prereg.mixture, max_units=args.max_units, measure=measure)  # fmt: skip
    except (TokstudyError, ValueError) as err:
        print(f"stream: {err}", file=sys.stderr)
        return 1
    d = manifest.doc
    print(json.dumps({"out": str(args.out), "sha256": manifest.sha256, "docs": d["docs"], "bytes": d["bytes"],
                      "achieved_mixture": d["achieved_mixture"], "dropped": d["dropped"]}))  # fmt: skip
    return 0
