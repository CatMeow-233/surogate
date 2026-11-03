"""Intrinsic metrics per arm and domain (`tokstudy-intrinsic/1`): a CPU screen and an explanation, never the decision.

`measure(codec, docs)` encodes every text segment of every document on its own (tokens never cross a special) and
counts, per domain, with exact integers:

- bytes (UTF-8), chars (code points), tokens; bytes/token, chars/token;
- Han characters (Unicode Script=Han, the `\\p{Han}` of the tokenizers regex engine) and Han chars/token;
- words (maximal runs of non-whitespace, `str.split()`) and words/token, in the word domains only (default "en");
- Rényi efficiency H_α(p) / ln|V| with α = 2.5, p the token unigram distribution of the domain, |V| the codec's
  vocabulary size; distinct tokens and vocabulary utilisation (distinct / (|V| - specials));
- invalid-UTF-8 tokens: tokens whose bytes alone are not valid UTF-8 (a byte-level BPE splits characters);
- digit runs (maximal runs of Unicode decimal digits, `\\d`), their characters, and digit pieces: the tokens that
  overlap a run, summed over runs;
- context: over the documents with at least k text tokens, the bytes covered by their first k tokens, for k = 512
  and 2048: the number of such documents and the upper median (sorted list at index n // 2);
- Python domains only (default "py"), on segments that `ast.parse` and the stdlib `tokenize` both accept, whose
  lexeme spans the self-check finds exact: lexemes are the tokenize tokens NAME, OP, NUMBER, STRING, COMMENT and the
  f-string (and t-string) parts, as the notebook's lexeme check counts them (whitespace, newlines and indentation
  are not lexemes). A cross-lexeme token overlaps two or more lexemes. Identifiers are NAME lexemes that are not
  keywords; identifier pieces are the tokens overlapping each identifier, summed. The stdlib lexer is that of the
  running Python, recorded in the output, because f-string tokenization changed in 3.12.

Ratios are computed from the integers and are None when their denominator is 0.
"""

from __future__ import annotations

import argparse
import ast
import io
import keyword
import math
import re
import sys
import tokenize
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import TokstudyError, atomic_write_json, sha256_file
from .codecs import PROJECT_SPECIALS, ByteCodec, Codec, SubwordCodec, parse_tokenizer_arg
from .docs import EvalDoc, parse_docs_arg, plain_documents

FORMAT = "tokstudy-intrinsic/1"
RENYI_ALPHA = 2.5
CONTEXTS = (512, 2048)
DIGIT_RUN = re.compile(r"\d+")
LEXEME_TYPES = frozenset(
    getattr(tokenize, name)
    for name in (
        "NAME",
        "OP",
        "NUMBER",
        "STRING",
        "COMMENT",
        "FSTRING_START",
        "FSTRING_MIDDLE",
        "FSTRING_END",
        "TSTRING_START",
        "TSTRING_MIDDLE",
        "TSTRING_END",
    )  # fmt: skip
    if hasattr(tokenize, name)
)
_HAN_REMOVED = None


def han_chars(text: str) -> int:
    """Characters of Unicode Script=Han in `text` (the regex engine the bilingual han-isolate arm splits with)."""
    global _HAN_REMOVED
    if _HAN_REMOVED is None:
        from tokenizers import Regex, pre_tokenizers

        _HAN_REMOVED = pre_tokenizers.Split(Regex(r"\p{Han}+"), behavior="removed")
    return len(text) - sum(len(piece) for piece, _ in _HAN_REMOVED.pre_tokenize_str(text))


def byte_offsets(text: str) -> np.ndarray:
    """offsets[i] = UTF-8 byte offset of character i; offsets[len(text)] = the byte length."""
    code = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    width = 1 + (code >= 0x80) + (code >= 0x800) + (code >= 0x10000)
    return np.concatenate([[0], np.cumsum(width, dtype=np.int64)])


def overlapping(ends: Sequence[int], start: int, end: int) -> tuple[int, int]:
    """The range [first, last] of tokens (token i spans [ends[i-1], ends[i])) that overlap bytes [start, end)."""
    return bisect_right(ends, start), bisect_left(ends, end)


def lexemes(text: str) -> list[tuple[int, int, int, str]] | None:
    """Character spans (start, end, type, string) of the lexemes of a Python text, or None when it does not parse or
    a span does not reproduce its token string."""
    try:
        ast.parse(text)
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (SyntaxError, ValueError, RecursionError, MemoryError, tokenize.TokenError):
        return None
    starts = [0]
    for line in io.StringIO(text).readlines():
        starts.append(starts[-1] + len(line))
    out = []
    for tok in tokens:
        if tok.type not in LEXEME_TYPES or not tok.string:
            continue
        s, e = starts[tok.start[0] - 1] + tok.start[1], starts[tok.end[0] - 1] + tok.end[1]
        if text[s:e] != tok.string:
            return None
        out.append((s, e, tok.type, tok.string))
    return out


class _Domain:
    def __init__(self, vocab_size: int, words: bool, python: bool):
        self.vocab_size = vocab_size
        self.n = dict(docs=0, bytes=0, chars=0, tokens=0, han_chars=0, digit_runs=0, digit_chars=0, digit_pieces=0)
        self.words = 0 if words else None
        self.python = (
            dict(
                parsed_segments=0,
                unparsed_segments=0,
                tokens=0,
                cross_lexeme_tokens=0,
                identifiers=0,
                identifier_pieces=0,
                single_piece_identifiers=0,
            )  # fmt: skip
            if python
            else None
        )
        self.counts = np.zeros(vocab_size, dtype=np.int64)
        self._pending: list[np.ndarray] = []
        self._pending_size = 0
        self.cover: dict[int, list[int]] = {k: [] for k in CONTEXTS}

    def add_ids(self, ids: np.ndarray) -> None:
        self._pending.append(ids)
        self._pending_size += ids.size
        if self._pending_size >= 1 << 22:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            ids = np.concatenate(self._pending).astype(np.int64)
            self.counts += np.bincount(ids, minlength=self.vocab_size)[: self.vocab_size]
            self._pending, self._pending_size = [], 0


def _segment(acc: _Domain, text: str, ids: np.ndarray, ends: np.ndarray) -> None:
    acc.n["chars"] += len(text)
    acc.n["han_chars"] += han_chars(text)
    if acc.words is not None:
        acc.words += len(text.split())
    offsets = None
    end_list = ends.tolist()
    for match in DIGIT_RUN.finditer(text):
        offsets = byte_offsets(text) if offsets is None else offsets
        first, last = overlapping(end_list, int(offsets[match.start()]), int(offsets[match.end()]))
        acc.n["digit_runs"] += 1
        acc.n["digit_chars"] += match.end() - match.start()
        acc.n["digit_pieces"] += last - first + 1
    if acc.python is None:
        return
    spans = lexemes(text)
    if spans is None:
        acc.python["unparsed_segments"] += 1
        return
    offsets = byte_offsets(text) if offsets is None else offsets
    acc.python["parsed_segments"] += 1
    acc.python["tokens"] += len(end_list)
    per_token = np.zeros(len(end_list) + 1, dtype=np.int64)
    for s, e, kind, string in spans:
        first, last = overlapping(end_list, int(offsets[s]), int(offsets[e]))
        per_token[first] += 1
        per_token[last + 1] -= 1
        if kind == tokenize.NAME and not keyword.iskeyword(string):
            acc.python["identifiers"] += 1
            acc.python["identifier_pieces"] += last - first + 1
            acc.python["single_piece_identifiers"] += first == last
    acc.python["cross_lexeme_tokens"] += int((np.cumsum(per_token[:-1]) >= 2).sum())


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def renyi_efficiency(counts: np.ndarray, vocab_size: int, alpha: float = RENYI_ALPHA) -> float | None:
    """H_α(p) / ln|V| for the unigram distribution p of `counts`; None without tokens."""
    total = int(counts.sum())
    if total == 0 or vocab_size < 2:
        return None
    p = counts[counts > 0] / total
    return math.log(float(np.sum(p**alpha))) / (1 - alpha) / math.log(vocab_size)


def _report(codec: Codec, acc: _Domain) -> dict[str, Any]:
    acc.flush()
    n, tokens = acc.n, acc.n["tokens"]
    observed = np.flatnonzero(acc.counts)
    invalid = 0
    for i in observed.tolist():
        try:
            codec.token_bytes(i).decode("utf-8")
        except UnicodeDecodeError:
            invalid += int(acc.counts[i])
    usable = codec.vocab_size - len(codec.special_ids)
    if isinstance(codec, ByteCodec):
        usable = 256
    out = dict(n)
    out |= {
        "bytes_per_token": _ratio(n["bytes"], tokens),
        "chars_per_token": _ratio(n["chars"], tokens),
        "han_chars_per_token": _ratio(n["han_chars"], tokens),
        "words": acc.words,
        "words_per_token": None if acc.words is None else _ratio(acc.words, tokens),
        "renyi_alpha": RENYI_ALPHA,
        "renyi_efficiency": renyi_efficiency(acc.counts, codec.vocab_size),
        "distinct_tokens": int(observed.size),
        "vocab_utilisation": _ratio(int(observed.size), usable),
        "invalid_utf8_tokens": invalid,
        "invalid_utf8_token_share": _ratio(invalid, tokens),
        "digit_pieces_per_run": _ratio(n["digit_pieces"], n["digit_runs"]),
        "context": {
            str(k): {"docs": len(v), "median_bytes": sorted(v)[len(v) // 2] if v else None}
            for k, v in acc.cover.items()
        },
    }
    if acc.python is not None:
        py = dict(acc.python)
        py["cross_lexeme_rate"] = _ratio(py["cross_lexeme_tokens"], py["tokens"])
        py["pieces_per_identifier"] = _ratio(py["identifier_pieces"], py["identifiers"])
        py["single_piece_share"] = _ratio(py["single_piece_identifiers"], py["identifiers"])
        out["python"] = py
    else:
        out["python"] = None
    return out


def measure(
    codec: Codec,
    docs: Iterable[EvalDoc],
    *,
    python_domains: Sequence[str] = ("py",),
    word_domains: Sequence[str] = ("en",),
) -> dict[str, dict[str, Any]]:
    """The intrinsic metrics of `codec` on `docs`, per domain (sorted)."""
    domains: dict[str, _Domain] = {}
    for doc in docs:
        acc = domains.get(doc.domain)
        if acc is None:
            acc = domains[doc.domain] = _Domain(codec.vocab_size, doc.domain in word_domains,
                                                doc.domain in python_domains)  # fmt: skip
        acc.n["docs"] += 1
        lengths = []
        for text in doc.texts():
            ids, ends = codec.encode_text(text)
            acc.n["bytes"] += int(ends[-1]) if ends.size else 0
            acc.n["tokens"] += int(ids.size)
            acc.add_ids(ids)
            lengths.append(np.diff(ends, prepend=0))
            _segment(acc, text, ids, ends)
        covered = np.cumsum(np.concatenate(lengths)) if lengths else np.zeros(0, dtype=np.int64)
        for k in CONTEXTS:
            if covered.size >= k:
                acc.cover[k].append(int(covered[k - 1]))
    return {d: _report(codec, domains[d]) for d in sorted(domains)}


# --- command line --------------------------------------------------------------------------------------------------


def add_codec_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--arms", type=Path, help="an arm registry: every subword and bytes arm (H-Net arms skipped)")
    parser.add_argument("--tokenizer", action="append", default=[], metavar="NAME=PATH@SHA256",
                        help="a byte-level BPE tokenizer.json, its sha256 checked (repeatable)")  # fmt: skip
    parser.add_argument("--bytes", action="append", default=[], metavar="NAME", help="a byte codec arm (repeatable)")
    parser.add_argument("--project", choices=sorted(PROJECT_SPECIALS),
                        help="check --tokenizer files for this project's special ids 0-4 (default: no check)")  # fmt: skip
    parser.add_argument("--drop-normalizer", action="store_true",
                        help="load --tokenizer files that carry a normaliser without it (reference tokenizers)")  # fmt: skip
    parser.add_argument("--check", action="store_true", help="compare token bytes with the text on every encode")


def codecs_from(args: argparse.Namespace) -> tuple[dict[str, Codec], list[str]]:
    """The codecs the command line names, and the H-Net arms skipped."""
    from .arms import codec_for, load_arms

    out: dict[str, Codec] = {}
    skipped: list[str] = []

    def put(name: str, codec: Codec) -> None:
        if name in out:
            raise TokstudyError(f"two codecs are named {name!r}")
        out[name] = codec

    for arm in load_arms(args.arms) if args.arms else []:
        codec = codec_for(arm, check=args.check)
        if codec is None:
            skipped.append(arm.id)
        else:
            put(arm.id, codec)
    specials = PROJECT_SPECIALS[args.project] if args.project else None
    for arg in args.tokenizer:
        name, path, sha = parse_tokenizer_arg(arg)
        put(name, SubwordCodec.from_tokenizer_json(path, sha, specials, name=name,
                                                   drop_normalizer=args.drop_normalizer, check=args.check))  # fmt: skip
    for name in args.bytes:
        put(name, ByteCodec(PROJECT_SPECIALS[args.project or "tlc"], name=name, check=args.check))
    if not out:
        raise TokstudyError("no codec: give --arms, --tokenizer or --bytes")
    return out, skipped


def add_docs_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--docs", action="append", required=True, metavar="DOMAIN=PATH",
                        help=".jsonl or datasets-server .json sample file of one domain (repeatable)")  # fmt: skip
    parser.add_argument("--field", help="text field (default: text in .jsonl, content in .json)")
    parser.add_argument("--limit", type=int, help="documents per --docs file")


def docs_from(args: argparse.Namespace) -> tuple[list[EvalDoc], list[dict[str, Any]]]:
    docs, inputs = [], []
    for arg in args.docs:
        domain, path = parse_docs_arg(arg)
        part = list(plain_documents(path, domain, field=args.field, limit=args.limit))
        docs += part
        inputs.append({"domain": domain, "path": str(path), "sha256": sha256_file(path), "docs": len(part)})
    return docs, inputs


def lexer_record() -> dict[str, Any]:
    return {"python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "lexeme_types": sorted(tokenize.tok_name[t] for t in LEXEME_TYPES)}  # fmt: skip


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tokstudy intrinsic", description=__doc__.split("\n")[0])
    add_codec_arguments(parser)
    add_docs_arguments(parser)
    parser.add_argument("--python-domain", action="append", help="domains with Python metrics (default: py)")
    parser.add_argument("--word-domain", action="append", help="domains with word counts (default: en)")
    parser.add_argument("--out", type=Path, required=True, help="the tokstudy-intrinsic/1 JSON to write")
    args = parser.parse_args(list(argv))
    try:
        codecs, skipped = codecs_from(args)
        docs, inputs = docs_from(args)
        python_domains, word_domains = args.python_domain or ["py"], args.word_domain or ["en"]
        arms = {
            name: {"codec": codec.identity(),
                   "domains": measure(codec, docs, python_domains=python_domains, word_domains=word_domains)}
            for name, codec in codecs.items()
        }  # fmt: skip
    except (TokstudyError, OSError, ValueError) as err:
        print(f"intrinsic: {err}", file=sys.stderr)
        return 1
    report = {"format": FORMAT, "inputs": inputs, "arms": arms, "skipped_hnet_arms": skipped, "lexer": lexer_record(),
              "python_domains": sorted(python_domains), "word_domains": sorted(word_domains)}  # fmt: skip
    atomic_write_json(args.out, report)
    for name, arm in arms.items():
        for domain, m in arm["domains"].items():
            bpt = m["bytes_per_token"]
            print(f"{name}\t{domain}\t{m['docs']} docs\t{m['bytes']} bytes\t{m['tokens']} tokens\t"
                  f"{'-' if bpt is None else f'{bpt:.3f}'} bytes/token")  # fmt: skip
    return 0
