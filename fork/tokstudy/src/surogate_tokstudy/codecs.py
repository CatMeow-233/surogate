"""Codecs: what an arm turns text into, with the exact bytes of every token.

Every arm of a table scores the same raw UTF-8 bytes (README: 测量契约), so a codec gives, for a text, its token ids
and the end byte offset of each token, and `token_bytes(id)` such that the bytes of the ids of `encode_text(text)`
joined are exactly `text.encode("utf-8")`. `encode_text` always checks that the lengths add up; with `check=True`
(the tests, and `--check` on the command line) it compares the bytes themselves. A lone surrogate is not a Unicode
scalar value and is refused: it has no UTF-8 encoding.

- SubwordCodec: a byte-level BPE `tokenizer.json`, identified by path and sha256 (checked on load). Token bytes come
  from GPT-2's byte-to-unicode alphabet, inverted. Text is document text: special-token strings in it stay text
  (`encode_special_tokens`), and an encoding that still yields a special id is refused. A tokenizer with a
  normaliser is refused, because it would score other bytes than the other arms; a reference or teacher tokenizer
  (Qwen's carries NFC) can be loaded with `drop_normalizer=True`, which removes the normaliser and records it.
- ByteCodec: id = 5 + byte, ids 0-4 the project's special tokens, vocab_size 261 (`pad_to` for a backend that wants
  a rounder table; padding ids never occur).

The projects' special ids are fixed here (PROJECT_SPECIALS) and equal the projects' own tables (a test reads both).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .canonical import TokstudyError, sha256_bytes

TLC_SPECIALS = {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2, "<|code_start|>": 3, "<|code_end|>": 4}
BILINGUAL_SPECIALS = {"<unk>": 0, "<eos>": 1, "<|zh|>": 2, "<|en|>": 3, "<|bridge|>": 4}
PROJECT_SPECIALS = {"tlc": TLC_SPECIALS, "bilingual": BILINGUAL_SPECIALS}
BYTE_OFFSET = 5  # ByteCodec: ids 0-4 are the project's specials, 5 + b is byte b
BYTE_VOCAB = BYTE_OFFSET + 256


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's byte-to-unicode table (the alphabet of every byte-level BPE): printable bytes stand for themselves,
    the other 68 are shifted to 256 + n in byte order."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    table = {b: chr(b) for b in printable}
    shifted = 0
    for b in range(256):
        if b not in table:
            table[b] = chr(256 + shifted)
            shifted += 1
    return table


BYTE_OF_CHAR = {c: b for b, c in _bytes_to_unicode().items()}


def utf8(text: str) -> bytes:
    """The UTF-8 bytes of `text`; a lone surrogate (not a scalar value) is refused."""
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as err:
        raise TokstudyError(f"text holds a lone surrogate at character {err.start}; it has no UTF-8 bytes") from None


class Codec(Protocol):
    name: str
    vocab_size: int
    special_ids: Mapping[str, int]

    def encode_text(self, text: str) -> tuple[np.ndarray, np.ndarray]: ...

    def token_bytes(self, token_id: int) -> bytes: ...

    def identity(self) -> dict[str, Any]: ...


class _Encoded:
    """Shared by both codecs: lengths per id (-1 for ids that carry no text) and the exactness checks."""

    name: str
    check: bool
    _lengths: np.ndarray

    def _finish(self, text: str, raw: bytes, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lengths = self._lengths[ids] if ids.size else np.zeros(0, dtype=np.int64)
        if (lengths < 0).any():
            bad = int(ids[np.flatnonzero(lengths < 0)[0]])
            raise TokstudyError(f"{self.name}: text produced id {bad}, which carries no text (a special token)")
        ends = np.cumsum(lengths, dtype=np.int64)
        total = int(ends[-1]) if ends.size else 0
        if total != len(raw):
            raise TokstudyError(f"{self.name}: tokens cover {total} bytes of a {len(raw)}-byte text")
        if self.check and b"".join(self.token_bytes(int(i)) for i in ids) != raw:
            raise TokstudyError(f"{self.name}: token bytes differ from the text's UTF-8 bytes")
        return ids.astype(np.int32, copy=False), ends


class SubwordCodec(_Encoded):
    def __init__(self, tokenizer, table: list[bytes | None], *, name: str, path: Path, sha256: str,
                 special_ids: Mapping[str, int], normalizer_dropped: Any = None, dropped=None,
                 check: bool = False):  # fmt: skip
        self._tokenizer = tokenizer
        self._dropped = dropped  # the removed normaliser object, to count the texts it would have changed
        self._table = table
        self._lengths = np.array([-1 if b is None else len(b) for b in table], dtype=np.int64)
        self.name, self.path, self.sha256 = name, Path(path), sha256
        self.special_ids = dict(special_ids)
        self.vocab_size = len(table)
        self.normalizer_dropped = normalizer_dropped
        self.check = check

    @classmethod
    def from_tokenizer_json(
        cls,
        path: Path,
        sha256: str,
        specials: Mapping[str, int] | None = None,
        *,
        name: str | None = None,
        drop_normalizer: bool = False,
        check: bool = False,
    ) -> SubwordCodec:
        """Load a byte-level BPE tokenizer.json whose sha256 must be `sha256`; `specials` (name -> id), when given,
        must be exactly the file's special tokens at those ids."""
        from tokenizers import Tokenizer, models

        path = Path(path)
        raw = path.read_bytes()
        found = sha256_bytes(raw)
        if found != sha256:
            raise TokstudyError(f"{path}: sha256 {found}, expected {sha256}")
        data = json.loads(raw)
        try:
            tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
        except Exception as err:  # tokenizers raises a bare Exception for a file it cannot read
            raise TokstudyError(f"{path}: not a tokenizer.json: {err}") from None
        if not isinstance(tokenizer.model, models.BPE):  # older files (GPT-2's) omit model.type
            raise TokstudyError(f"{path}: model {type(tokenizer.model).__name__}, not a byte-level BPE")
        _check_byte_level(path, data.get("pre_tokenizer"))
        normalizer = data.get("normalizer")
        if normalizer is not None and not drop_normalizer:
            raise TokstudyError(f"{path}: has a normaliser {normalizer}; every arm must score the raw bytes")
        dropped = tokenizer.normalizer if normalizer is not None else None
        tokenizer.normalizer = None
        tokenizer.encode_special_tokens = True  # document text: special-token strings are text
        table = _token_table(path, data["model"]["vocab"], data.get("added_tokens", []), tokenizer.get_vocab_size())
        special_ids = {t["content"]: int(t["id"]) for t in data.get("added_tokens", []) if t.get("special")}
        if specials is not None and special_ids != dict(specials):
            raise TokstudyError(f"{path}: special tokens {special_ids}, expected {dict(specials)}")
        return cls(tokenizer, table, name=name or path.parent.name, path=path, sha256=sha256,
                   special_ids=special_ids, normalizer_dropped=normalizer, dropped=dropped, check=check)  # fmt: skip

    def encode_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        raw = utf8(text)
        if not text:
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
        ids = np.asarray(self._tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.int64)
        return self._finish(text, raw, ids)

    def changed_by_dropped_normalizer(self, text: str) -> bool:
        """Whether the normaliser removed on load would have changed `text` (False when there was none)."""
        return self._dropped is not None and self._dropped.normalize_str(text) != text

    def token_bytes(self, token_id: int) -> bytes:
        b = self._table[token_id]
        return b"" if b is None else b

    def identity(self) -> dict[str, Any]:
        return {"kind": "subword", "name": self.name, "path": str(self.path), "sha256": self.sha256,
                "vocab_size": self.vocab_size, "normalizer_dropped": self.normalizer_dropped}  # fmt: skip


def _check_byte_level(path: Path, pre_tokenizer: Any) -> None:
    steps = (pre_tokenizer or {}).get("pretokenizers") if (pre_tokenizer or {}).get("type") == "Sequence" else None
    steps = steps if steps is not None else [pre_tokenizer or {}]
    byte_level = [s for s in steps if s.get("type") == "ByteLevel"]
    if len(byte_level) != 1:
        raise TokstudyError(f"{path}: the pre-tokenizer has no ByteLevel step; not a byte-level BPE")
    if byte_level[0].get("add_prefix_space"):
        raise TokstudyError(f"{path}: ByteLevel adds a prefix space, so tokens would not cover the text's bytes")


def _token_table(path: Path, vocab: Mapping[str, int], added: list, size: int) -> list[bytes | None]:
    """The bytes of every id: byte-level vocabulary strings mapped back to bytes; added tokens as their UTF-8 text,
    special ones as None (they carry no text)."""
    table: list[bytes | None] = [None] * size
    for token, i in vocab.items():
        try:
            table[i] = bytes(BYTE_OF_CHAR[c] for c in token)
        except KeyError:
            raise TokstudyError(f"{path}: vocabulary entry {token!r} is not in the byte-level alphabet") from None
    for entry in added:
        table[int(entry["id"])] = None if entry.get("special") else entry["content"].encode("utf-8")
    return table


class ByteCodec(_Encoded):
    def __init__(self, special_ids: Mapping[str, int], *, name: str = "bytes", pad_to: int | None = None,
                 check: bool = False):  # fmt: skip
        if sorted(special_ids.values()) != list(range(BYTE_OFFSET)):
            raise TokstudyError(f"ByteCodec needs special ids 0-4, got {dict(special_ids)}")
        self.name = name
        self.special_ids = dict(special_ids)
        self.vocab_size = max(BYTE_VOCAB, pad_to or 0)
        self.pad_to = pad_to
        self.check = check
        self._lengths = np.array([-1] * BYTE_OFFSET + [1] * 256 + [-1] * (self.vocab_size - BYTE_VOCAB))

    def encode_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        raw = utf8(text)
        ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int64) + BYTE_OFFSET
        return self._finish(text, raw, ids)

    def token_bytes(self, token_id: int) -> bytes:
        if BYTE_OFFSET <= token_id < BYTE_VOCAB:
            return bytes([token_id - BYTE_OFFSET])
        if 0 <= token_id < BYTE_OFFSET:
            return b""
        raise TokstudyError(f"{self.name}: id {token_id} is padding, never a token")

    def identity(self) -> dict[str, Any]:
        return {"kind": "bytes", "name": self.name, "vocab_size": self.vocab_size, "pad_to": self.pad_to}


def parse_tokenizer_arg(arg: str) -> tuple[str, Path, str]:
    """`NAME=PATH@SHA256` from the command line."""
    name, sep, rest = arg.partition("=")
    path, at, sha = rest.rpartition("@")
    if not sep or not at or not name or not path:
        raise TokstudyError(f"--tokenizer {arg!r}: expected NAME=PATH@SHA256")
    return name, Path(path), sha
