"""Teacher-forced loss and bits per byte of a model directory on a split of a TLC shard directory (docs/training.md:
评测).

The rows are scored as the native DataLoader serves them (data.shards.served_rows), by the native trainer's
compute_logprobs without LoRA; the loss is the mean negative log-probability over the valid targets. The baseline
is the add-one smoothed unigram distribution of the train split's targets: gate G3 asks that the validation loss
be below it, since a model that has not learnt even the token frequencies has not trained. The loss is also given
per source and per source kind (code, text) from the row ranges the shard manifest records; the text kind's
perplexity is the English perplexity.

Bits per byte (the tokenizer study's measure): every document is a run of rows that starts with a row
whose first token is <|bos|> (a continuation window starts with the token it shares with the previous window, never
with <|bos|>), and every token of a document but its first is a valid target of exactly one of its rows. Per
document, the targets whose token stands for text (data.tokenizer.token_bytes) give the numerator, the summed NLL
in nats, and their bytes the denominator: the UTF-8 bytes of the document's text pieces, the bytes the model is
trained on (normalised newlines, fence lines of text documents replaced by the code markers), whatever the
tokenizer. The special targets are kept apart: <|eos|> (the sensitivity variant that counts it) and the others.

    BPB(domain) = sum nll_nats / (ln 2 * sum bytes) over the domain's complete documents (byte-weighted)

Domains are the tokenizer study's TLC domains: "py" for code sources, "en" for text sources. A document cut by
--max-rows is partial and left out of BPB. `--dump-nll` writes one line per complete document in the study's neutral
format `tokstudy-nll/1` (header backend "native"), keyed "<shard file>:<n>" (the n-th document of the file) or, with
the corpus spec, by the corpus's document keys, which are checked against the shards. fork/tokstudy reads that format
without either package importing the other.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..data.shards import (
    IGNORE,
    check_tokenizer,
    manifest,
    manifest_sha256,
    read_shard,
    segments,
    served_rows,
    shard_files,
    shard_index,
)
from ..data.tokenizer import BOS, EOS, SPECIAL_TOKENS, load, special_ids, token_bytes
from ..files import file_sha256, read_json
from ..modeldir import META_FILE, WEIGHTS_FILE, check_model_dir
from ..native import Scorer, native_scorer

NLL_FORMAT = "tokstudy-nll/1"
BACKEND = "native"
NATIVE_TABLE = "tlc-T"  # the native TOK-AB table; never one table with fork/tokstudy's tlc-N
DOMAINS = {"code": "py", "text": "en"}  # source kind -> the TLC domain of fork/tokstudy's documents
LN2 = math.log(2.0)


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


# --- per-document NLL and bits per byte ------------------------------------------------------------------------------


@dataclass(frozen=True)
class DocNLL:
    """One line of tokstudy-nll/1: the fields and their order are the format's (DocNLL of fork/tokstudy's evaluate)."""

    key: str
    domain: str
    bytes: int
    nll_nats: float
    n_targets: int
    eos_nats: float = 0.0
    n_eos: int = 0
    special_nats: float = 0.0
    n_special: int = 0


@dataclass(frozen=True)
class NllDump:
    """Where and as what `evaluate` writes the per-document lines: the table and the arm label of the model."""

    path: Path
    arm: str
    table: str = NATIVE_TABLE


class _RowSums:
    """Per scored row: the NLL sums (text targets, <|eos|> targets, other special targets) and the counts (text
    targets, their bytes, <|eos|> targets, other special targets); a document sums its rows."""

    def __init__(self, nbytes: np.ndarray, special: np.ndarray, eos: int):
        self.nbytes, self.special, self.eos = nbytes, special, eos
        self.nats: list[tuple[float, float, float]] = []
        self.counts: list[tuple[int, int, int, int]] = []

    def add(self, targets: np.ndarray, logprobs: np.ndarray) -> None:
        for tgt, lp in zip(targets, logprobs, strict=True):
            valid = tgt != IGNORE
            t, nll = tgt[valid], -lp[valid].astype(np.float64)
            special = self.special[t]
            eos = t == self.eos
            other = special & ~eos
            self.nats.append((float(nll[~special].sum()), float(nll[eos].sum()), float(nll[other].sum())))
            self.counts.append((int((~special).sum()), int(self.nbytes[t].sum()), int(eos.sum()), int(other.sum())))

    def document(self, key: str, domain: str, first: int, end: int) -> DocNLL:
        text, eos, other = (math.fsum(column) for column in zip(*self.nats[first:end], strict=True))
        n_text, n_bytes, n_eos, n_other = (sum(column) for column in zip(*self.counts[first:end], strict=True))
        return DocNLL(key, domain, n_bytes, text, n_text, eos, n_eos, other, n_other)


def _documents(inputs: np.ndarray, starts_after: bool, files: list[Path], file_rows: list[int], bos: int):
    """(file name, number in file, first row, end row, complete) of every document of the scored rows `inputs`;
    `starts_after` tells whether the first unscored row (if any) starts a document."""
    out = []
    first = 0
    for path, n_rows in zip(files, file_rows, strict=True):
        end = min(first + n_rows, len(inputs))
        if first >= end:
            break
        starts = [r for r in range(first, end) if inputs[r, 0] == bos]
        if not starts or starts[0] != first:
            raise ValueError(f"{path.name}: its first row does not start a document with <|bos|>")
        cut = end < first + n_rows  # --max-rows ends the scored rows inside this file
        for n, (a, b) in enumerate(zip(starts, [*starts[1:], end], strict=True)):
            complete = b < end or not cut or starts_after
            out.append((path.name, n, a, b, complete))
        first += n_rows
    return out


def content_bytes(text: str, kind: str) -> int:
    """The UTF-8 bytes of a document's text pieces as data.shards frames it (fence lines of text documents are
    replaced by the code markers): the denominator of its bits per byte under every tokenizer."""
    ids = dict(zip(SPECIAL_TOKENS, range(len(SPECIAL_TOKENS)), strict=True))
    return sum(len(piece.encode("utf-8")) for piece in segments(text, kind, ids) if isinstance(piece, str))


def corpus_keys(spec: Path, info: Mapping[str, Any], split: str) -> dict[str, list[tuple[str, int]]]:
    """Per shard file, the (key, content bytes) of its documents in written order, from the corpus spec the shard
    manifest was built from (its sha256 and decontamination record must be the manifest's)."""
    from ..data.corpus import decontam_record, documents, load_corpus, split_of

    corpus = load_corpus(Path(spec))
    recorded = info["corpus"]
    if file_sha256(corpus.path) != recorded["spec_sha256"]:
        raise ValueError(f"{spec} is not the corpus spec the shards were built from (sha256 differs)")
    if decontam_record(corpus) != recorded["decontam"]:
        raise ValueError(f"{spec}: its decontamination record is not the shard manifest's")
    out: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for doc in documents(corpus):
        if split_of(corpus, doc) == split:
            name = f"{split}-{shard_index(corpus, doc, split, recorded['train_files']):05d}.bin"
            out[name].append((doc.key, content_bytes(doc.text, doc.source.kind)))
    return out


def _bpb(rows: Sequence[DocNLL]) -> dict[str, Any]:
    n_bytes = sum(r.bytes for r in rows)
    nats = sum(r.nll_nats for r in rows)
    eos = sum(r.eos_nats for r in rows)
    return {
        "bpb": nats / (LN2 * n_bytes) if n_bytes else None,
        "bpb_with_eos": (nats + eos) / (LN2 * n_bytes) if n_bytes else None,
        "documents": len(rows),
        "bytes": n_bytes,
        "targets": sum(r.n_targets for r in rows),
    }


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def write_nll(path: Path, header: Mapping[str, Any], rows: Sequence[DocNLL]) -> Path:
    """tokstudy-nll/1: the header line, then one line per document (canonical JSON). A non-finite NLL (a diverged
    model) is refused rather than written as NaN."""
    try:
        lines = [_canonical({"format": NLL_FORMAT, **header})] + [_canonical(asdict(r)) for r in rows]
    except ValueError as err:
        raise ValueError(f"{path}: a document's NLL is not finite ({err})") from None
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


# --- the command -----------------------------------------------------------------------------------------------------


def evaluate(
    model_dir: Path,
    data: Path,
    *,
    split: str = "val",
    batch: int = 32,
    max_rows: int | None = None,
    scorer: Scorer | None = None,
    corpus: Path | None = None,
    dump: NllDump | None = None,
) -> dict[str, Any]:
    """Loss and bits per byte of `model_dir` on `split` of shard directory `data`; `scorer` defaults to the native
    trainer. `corpus`: the spec the shards were built from, for document keys; `dump`: also write tokstudy-nll/1."""
    model_dir, data = Path(model_dir), Path(data)
    cfg, info = check_model_dir(model_dir), check_tokenizer(data, model_dir)
    seq_len = info["format"]["seq_len"]
    files = shard_files(data, split)
    parts = [served_rows(read_shard(path), seq_len) for path in files]
    if not parts:
        raise ValueError(f"no {split}-*.bin in {data}")
    full_inputs, full_targets, full_positions = (np.concatenate(arrays).astype(np.int32) for arrays in zip(*parts))
    inputs, targets, positions = (a[:max_rows] for a in (full_inputs, full_targets, full_positions))
    sources = _row_sources(info, files)[: len(inputs)]
    kinds = {source["name"]: source["kind"] for source in info["sources"]}
    tokenizer = load(model_dir)
    special = special_ids(tokenizer)
    pieces = token_bytes(tokenizer)
    nbytes = np.array([0 if b is None else len(b) for b in pieces], dtype=np.int64)
    is_special = np.array([b is None for b in pieces], dtype=bool)
    per_row = _RowSums(nbytes, is_special, special[EOS])
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
            per_row.add(targets[rows], logprobs)
            for row, source in enumerate(sources[rows]):
                if source is None:
                    continue
                row_loss, row_count = -float(logprobs[row][valid[row]].sum()), int(valid[row].sum())
                for totals, key in ((by_source, source), (by_kind, kinds.get(source, "unknown"))):
                    totals[key][0] += row_loss
                    totals[key][1] += row_count
    if not count:
        raise ValueError(f"{split} of {data} has no valid targets")
    file_rows = [len(p[0]) for p in parts]
    starts_after = len(inputs) == len(full_inputs) or full_inputs[len(inputs), 0] == special[BOS]
    found = _documents(inputs, starts_after, files, file_rows, special[BOS])
    keys = corpus_keys(corpus, info, split) if corpus is not None else None
    docs: list[DocNLL] = []
    doc_sources: list[str | None] = []
    partial = 0
    for name, n, a, b, complete in found:
        n_bytes = sum(c[1] for c in per_row.counts[a:b])
        if keys is not None:
            listed = keys.get(name, [])
            if n >= len(listed):
                raise ValueError(f"{name} holds more documents than the corpus spec puts in it")
            key, expected = listed[n]
            if complete and expected != n_bytes:
                raise ValueError(f"{name}: document {n} ({key}) covers {n_bytes} bytes, the corpus says {expected}")
        else:
            key = f"{name}:{n}"
        if not complete:
            partial += 1
            continue
        source = sources[a]
        docs.append(per_row.document(key, DOMAINS.get(kinds.get(source, ""), "unknown"), a, b))
        doc_sources.append(source)
    if keys is not None and max_rows is None:
        counted = defaultdict(int)
        for name, *_ in found:
            counted[name] += 1
        short = [name for name in keys if counted.get(name, 0) != len(keys[name])]
        if short:
            raise ValueError(f"the corpus spec puts another number of documents in {short}")
    loss = total / count
    unigram = unigram_loss(data, targets, cfg["vocab_size"])
    result = {
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
        "bpb": {d: _bpb([r for r in docs if r.domain == d]) for d in sorted({r.domain for r in docs})},
        "bpb_by_source": {
            s: _bpb([r for r, src in zip(docs, doc_sources, strict=True) if src == s])
            for s in sorted({s for s in doc_sources if s is not None})
        },
        "documents": {"complete": len(docs), "partial": partial, "keys": "shard" if keys is None else "corpus"},
    }
    if dump is not None:
        meta_path = model_dir / META_FILE
        weights = model_dir / WEIGHTS_FILE
        header = {
            "table": dump.table,
            "backend": BACKEND,
            "arm": dump.arm,
            "seed": read_json(meta_path).get("init_seed") if meta_path.is_file() else None,
            "model_dir": model_dir.name,
            "weights_sha256": file_sha256(weights) if weights.is_file() else None,
            "tokenizer_sha256": result["tokenizer_sha256"],
            "data_manifest_sha256": result["data_manifest_sha256"],
            "corpus_spec_sha256": info["corpus"]["spec_sha256"],
            "split": split,
            "seq_len": seq_len,
            "rows": len(inputs),
            "max_rows": max_rows,
            "keys": result["documents"]["keys"],
            "partial_documents": partial,
            "domains": {s["name"]: DOMAINS.get(s["kind"], "unknown") for s in info["sources"]},
        }
        write_nll(dump.path, header, docs)
        result["nll"] = str(dump.path)
    return result
