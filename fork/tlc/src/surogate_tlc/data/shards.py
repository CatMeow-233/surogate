"""Token shards for the native trainer: BIN.TOK v3 files with one document per row, and their manifest.

`mamba_ssm_scan` does not reset its state between packed documents, so no row holds more than one document, for
every P0 row alike (README: "数据 shard"):

- Framing: a code document is <|bos|> <|code_start|> text <|code_end|> <|eos|>; a text document is <|bos|> text
  <|eos|>, with its markdown code fences turned into the code markers (segments()). Text is encoded with
  special-token strings as plain text, so a document never yields a special id by itself.
- A framed document longer than a row is cut into windows of seq_len tokens. Consecutive windows share one token,
  so every token but a document's first is a target exactly once. Each window is a row padded at the end with
  <|pad|>; positions run 0..seq_len-1 in every row.
- The mask bit of input position i is set when its target, token i + 1, is a real token of the same row. The
  header marks the file non-overlapping, so the native DataLoader serves exactly the rows (num_tokens / seq_len
  chunks) and never takes a target from the next row. A row with n real tokens has n - 1 valid targets; the
  manifest sums them per shard (`valid_targets`, the epoch total of the trainer's ValidTokenCount).

The file layout is that of upstream surogate.train.tokenize.TokenizedDataFileWriter (masking=True,
non_overlapping=True), byte for byte (CPU test). That module imports the native extension, so it is not used here.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from .. import hf
from ..modeldir import dump_json
from .corpus import (
    RULES,
    Corpus,
    Document,
    decontam_record,
    documents,
    file_sha256,
    load_corpus,
    provenance,
    split_of,
    unit_hash,
)
from .tokenizer import BOS, CODE_END, CODE_START, EOS, PAD, load, special_ids, tokenizer_sha256

MAGIC = b"BIN.TOK\n"
FORMAT_VERSION = 3
HEADER_BYTES = 1024
MANIFEST_FILE = "manifest.json"
FENCE = "```"
# Bump when framing, windowing or the row layout changes.
SHARD_VERSION = 1
ENCODE_BATCH = 256


def segments(text: str, kind: str, special: dict[str, int]) -> list[str | int]:
    """A framed document: text pieces to encode, and special ids between them.

    In a text document a line that starts with ``` (after leading whitespace) opens a code block, and inside a block
    a line that is exactly ``` closes it; the fence lines are replaced by <|code_start|> and <|code_end|>. A block
    still open at the end of the document is closed there.
    """
    bos, eos, start, end = (special[token] for token in (BOS, EOS, CODE_START, CODE_END))
    if kind == "code":
        return [bos, start, text, end, eos]
    out: list[str | int] = [bos]
    lines: list[str] = []
    in_code = False
    for line in re.split(r"(?<=\n)", text):  # str.splitlines would also split at U+2028 and friends
        fence = line.strip()
        if (in_code and fence == FENCE) or (not in_code and fence.startswith(FENCE)):
            out += ["".join(lines), end if in_code else start]
            lines, in_code = [], not in_code
        else:
            lines.append(line)
    out += ["".join(lines), *([end] if in_code else []), eos]
    return [piece for piece in out if piece != ""]


def windows(ids: Sequence[int], seq_len: int) -> list[Sequence[int]]:
    """Rows of at most seq_len tokens; consecutive rows share one token."""
    if len(ids) <= seq_len:
        return [ids]
    return [ids[start : start + seq_len] for start in range(0, len(ids) - 1, seq_len - 1)]


def encode(
    tokenizer: Tokenizer, docs: Iterable[Document], special: dict[str, int]
) -> Iterator[tuple[Document, list[int]]]:
    """Framed token ids per document. `tokenizer` must encode special-token strings as text (load(ordinary=True))."""
    for batch in itertools.batched(docs, ENCODE_BATCH):
        framed = [segments(doc.text, doc.source.kind, special) for doc in batch]
        texts = [piece for pieces in framed for piece in pieces if isinstance(piece, str)]
        encoded = iter(e.ids for e in tokenizer.encode_batch(texts, add_special_tokens=False))
        for doc, pieces in zip(batch, framed, strict=True):
            yield doc, [i for piece in pieces for i in (next(encoded) if isinstance(piece, str) else [piece])]


class ShardWriter:
    """Writes padded rows to a BIN.TOK v3 file: header, tokens, positions, then mask bits (little-endian bits)."""

    def __init__(self, path: Path, *, seq_len: int, vocab_size: int, pad_id: int) -> None:
        if seq_len < 8 or seq_len % 8:
            raise ValueError(f"seq_len must be a positive multiple of 8, got {seq_len}")
        self.path, self.seq_len, self.vocab_size, self.pad_id = Path(path), seq_len, vocab_size, pad_id
        self.lengths: list[int] = []
        self._tmp = self.path.with_name(f".{self.path.name}.tmp")
        self._file = self._tmp.open("wb")
        self._file.write(b"*" * (HEADER_BYTES - 1) + b"\n")  # the header goes in last, as upstream does it

    def add_row(self, ids: Sequence[int]) -> None:
        if not 1 <= len(ids) <= self.seq_len:
            raise ValueError(f"a row holds 1..{self.seq_len} tokens, got {len(ids)}")
        row = np.full(self.seq_len, self.pad_id, dtype="<i4")
        row[: len(ids)] = ids
        if row.min() < 0 or row.max() >= self.vocab_size:
            raise ValueError(f"token ids must lie in [0, {self.vocab_size})")
        if (len(self.lengths) + 1) * self.seq_len >= 2**31:
            raise ValueError(f"{self.path} would reach 2**31 tokens; use more train files")
        self._file.write(row.tobytes())
        self.lengths.append(len(ids))

    def close(self) -> dict[str, Any]:
        """Finish the file and return its statistics."""
        lengths = np.asarray(self.lengths, dtype=np.int64)
        for chunk in np.array_split(lengths, max(1, len(lengths) // 4096)):
            self._file.write(np.tile(np.arange(self.seq_len, dtype="<i4"), len(chunk)).tobytes())
        for chunk in np.array_split(lengths, max(1, len(lengths) // 4096)):
            bits = np.arange(self.seq_len)[None, :] < (chunk - 1)[:, None]
            self._file.write(np.packbits(bits, axis=1, bitorder="little").tobytes())
        n_tokens = len(lengths) * self.seq_len
        self._file.seek(0)
        self._file.write(MAGIC + np.array([FORMAT_VERSION, 4, n_tokens, self.vocab_size, 1, 1], "<i4").tobytes())
        self._file.close()
        os.replace(self._tmp, self.path)
        real = int(lengths.sum())
        return {
            "file": self.path.name,
            "sha256": file_sha256(self.path),
            "rows": len(lengths),
            "tokens": n_tokens,
            "real_tokens": real,
            "valid_targets": real - len(lengths),
            "pad_fraction": _fraction(n_tokens - real, n_tokens),
        }


def _fraction(part: int, whole: int) -> float:
    return round(part / whole, 6) if whole else 0.0


@dataclass(frozen=True)
class Shard:
    vocab_size: int
    non_overlapping: bool
    tokens: np.ndarray  # int32 [n_tokens]
    positions: np.ndarray  # int32 [n_tokens]
    masks: np.ndarray | None  # bool [n_tokens]; None without mask bits


def read_shard(path: Path) -> Shard:
    """Read a BIN.TOK v3 file (memory-mapped)."""
    header = np.fromfile(path, dtype="<i4", count=HEADER_BYTES // 4)
    if header[:2].tobytes() != MAGIC or header[2] != FORMAT_VERSION or header[3] != 4:
        raise ValueError(f"{path} is not a BIN.TOK v{FORMAT_VERSION} file with 4-byte tokens")
    n, vocab_size, has_masks, non_overlapping = (int(v) for v in header[4:8])
    data = np.memmap(path, dtype="<i4", mode="r", offset=HEADER_BYTES, shape=(2 * n,))
    masks = None
    if has_masks:
        bits = np.memmap(path, dtype=np.uint8, mode="r", offset=HEADER_BYTES + 8 * n, shape=((n + 7) // 8,))
        masks = np.unpackbits(bits, bitorder="little")[:n].astype(bool)
    return Shard(vocab_size, bool(non_overlapping), data[:n], data[n:], masks)


def served_rows(shard: Shard, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inputs, targets and positions [rows, seq_len] of every row, as the native DataLoader serves them.

    Follows load_seq in csrc/src/runtime/training/dataloader.cpp for a non-overlapping shard with mask bits: the
    target of position i is the file's next token, -100 where the mask bit of position i is clear, and -100 at
    the last position of a row. The count of targets other than -100 is the trainer's ValidTokenCount.
    """
    if not shard.non_overlapping or shard.masks is None:
        raise ValueError("expects a non-overlapping shard with mask bits")
    rows = len(shard.tokens) // seq_len
    shape = (rows, seq_len)
    inputs = np.asarray(shard.tokens[: rows * seq_len]).reshape(shape)
    targets = np.concatenate([inputs[:, 1:], np.full((rows, 1), -100, dtype=inputs.dtype)], axis=1)
    targets[~shard.masks[: rows * seq_len].reshape(shape)] = -100
    return inputs, targets, np.asarray(shard.positions[: rows * seq_len]).reshape(shape)


def build_shards(
    corpus: Corpus, tokenizer_dir: Path, out: Path, *, seq_len: int = hf.CONTEXT_LENGTH, train_files: int = 1
) -> dict[str, Any]:
    """Write train-NNNNN.bin, val-00000.bin and, last, manifest.json into `out`, which must be new or empty.

    A train document goes to one of `train_files` files by a hash of its key, so each file samples every source:
    the native DataLoader reads files one after another and shuffles rows only within a file.
    """
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = load(tokenizer_dir, ordinary=True)
    special = special_ids(tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    writers = {
        (split, i): ShardWriter(
            out / f"{split}-{i:05d}.bin", seq_len=seq_len, vocab_size=vocab_size, pad_id=special[PAD]
        )
        for split, i in [*(("train", i) for i in range(train_files)), ("val", 0)]
    }
    dropped: Counter = Counter()
    counts: dict[str, Counter] = defaultdict(Counter)
    for doc, ids in encode(tokenizer, documents(corpus, dropped), special):
        split = split_of(corpus, doc)
        rows = windows(ids, seq_len)
        writer = writers[split, unit_hash(corpus.seed, "shard\0" + doc.key) % train_files if split == "train" else 0]
        for row in rows:
            writer.add_row(row)
        counted = {"documents": 1, "characters": len(doc.text), "rows": len(rows), "real_tokens": sum(map(len, rows))}
        for key, n in counted.items():
            counts[doc.source.name][split, key] += n
    shards = []
    for (split, _), writer in writers.items():
        stats = writer.close()
        if stats["rows"]:
            shards.append({"split": split, **stats})
        else:
            writer.path.unlink()
    splits = {}
    for split in ("train", "val"):
        totals = {
            k: sum(s[k] for s in shards if s["split"] == split)
            for k in ("rows", "tokens", "real_tokens", "valid_targets")
        }
        splits[split] = {
            **totals,
            "pad_fraction": _fraction(totals["tokens"] - totals["real_tokens"], totals["tokens"]),
        }
    sources = provenance(corpus)
    for s in sources:
        mine = counts[s["name"]]
        for key in ("documents", "characters", "rows", "real_tokens"):
            s[key] = {split: mine[split, key] for split in ("train", "val")}
        s["train_token_share"] = _fraction(mine["train", "real_tokens"], splits["train"]["real_tokens"])
        s["dropped"] = {reason: n for (name, reason), n in sorted(dropped.items()) if name == s["name"]}
    manifest = {
        "tlc_shard_version": SHARD_VERSION,
        "format": {"name": "BIN.TOK", "version": FORMAT_VERSION, "seq_len": seq_len, "non_overlapping": True},
        "tokenizer": {"sha256": tokenizer_sha256(tokenizer_dir), "vocab_size": vocab_size, "special_ids": special},
        "corpus": {
            "spec_sha256": file_sha256(corpus.path),
            "seed": corpus.seed,
            "val_fraction": corpus.val_fraction,
            "train_files": train_files,
            "decontam": decontam_record(corpus),
        },
        "rules": [
            *RULES,
            "framing: code <|bos|> <|code_start|> text <|code_end|> <|eos|>; text <|bos|> text <|eos|>",
            "in text documents, a line starting with ``` opens a code block (<|code_start|>) and, inside one, a line "
            "that is exactly ``` closes it (<|code_end|>); an open block is closed at the end of the document",
            "special-token strings inside a document are encoded as plain text",
            "one document per row: windows of seq_len tokens, consecutive windows share one token, end padding",
        ],
        "sources": sources,
        "shards": shards,
        "splits": splits,
    }
    (out / MANIFEST_FILE).write_text(dump_json(manifest), encoding="utf-8")
    return manifest


def manifest_sha256(out: Path) -> str:
    """The data's identity, which results record next to the tokenizer's (data_manifest_sha256)."""
    return file_sha256(Path(out) / MANIFEST_FILE)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tlc.data.shards", description=__doc__.split("\n")[0])
    parser.add_argument("corpus", type=Path, help="corpus spec (TOML)")
    parser.add_argument("tokenizer", type=Path, help="tokenizer (or model) directory")
    parser.add_argument("out", type=Path, help="output directory, new or empty")
    parser.add_argument("--seq-len", type=int, default=hf.CONTEXT_LENGTH)
    parser.add_argument("--train-files", type=int, default=1)
    args = parser.parse_args(argv)
    manifest = build_shards(
        load_corpus(args.corpus), args.tokenizer, args.out, seq_len=args.seq_len, train_files=args.train_files
    )
    print(json.dumps({"out": str(args.out), "manifest_sha256": manifest_sha256(args.out), **manifest["splits"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
