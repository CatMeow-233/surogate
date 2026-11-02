"""Train/held-out overlap by language-appropriate n-grams, and the filter (`bilingual-overlap/1`; SCI-10, MS §6.1).

Rules (a training document is flagged when any of its n-grams under any rule occurs in any held-out document):

- `en_word13`: 13-grams of words, where words are the whitespace-separated pieces of the NFKC-casefolded text with
  leading and trailing punctuation stripped. Chinese has no spaces between words, so this rule finds almost
  nothing in Chinese text; that is the v1.0 gap this module closes.
- `zh_han25`: 25-grams of Han characters, taken from the text with every non-Han character removed.
- `token13`: 13-grams of token ids (the text encoded as user text); needs the tokenizer, so it runs after the
  tokenizer is trained.

n-grams are compared by 64-bit hashes (a polynomial rolling hash over per-element splitmix64 hashes, numpy
uint64 arithmetic). A hash collision can only flag a clean document, never miss an overlap, so the filter errs on
the side of removing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..canonical import atomic_write_json, check_format
from ..config import DEFAULTS
from .docs import Doc, is_han

FORMAT = "bilingual-overlap/1"
RULES = {"en_word13": 13, "zh_han25": 25, "token13": 13}
_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_BASE = np.uint64(0x100000001B3)


class DecontamError(ValueError):
    pass


def _splitmix64(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def ngram_hashes(elements: np.ndarray, n: int) -> np.ndarray:
    """uint64 hashes of every n-gram of a 1-D uint64 element array (empty when it is shorter than n)."""
    if len(elements) < n:
        return np.zeros(0, dtype=np.uint64)
    mixed = _splitmix64(np.asarray(elements, dtype=np.uint64))
    windows = np.lib.stride_tricks.sliding_window_view(mixed, n)
    powers = np.ones(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(n - 2, -1, -1):
            powers[i] = powers[i + 1] * _BASE
        return (windows * powers).sum(axis=1, dtype=np.uint64)


def words(text: str) -> list[str]:
    out = []
    for piece in unicodedata.normalize("NFKC", text).casefold().split():
        start, end = 0, len(piece)
        while start < end and unicodedata.category(piece[start]).startswith("P"):
            start += 1
        while end > start and unicodedata.category(piece[end - 1]).startswith("P"):
            end -= 1
        if start < end:
            out.append(piece[start:end])
    return out


def _word_ids(text: str) -> np.ndarray:
    return np.array(
        [int.from_bytes(hashlib.blake2b(w.encode("utf-8"), digest_size=8).digest(), "little") for w in words(text)],
        dtype=np.uint64,
    )


def elements(doc_text: str, rule: str, tok=None) -> np.ndarray:
    if rule == "en_word13":
        return _word_ids(doc_text)
    if rule == "zh_han25":
        return np.array([ord(c) for c in doc_text if is_han(c)], dtype=np.uint64)
    if rule == "token13":
        if tok is None:
            raise DecontamError("rule token13 needs the tokenizer")
        from .tokenizer import encode_user_text

        return np.array(encode_user_text(tok, doc_text), dtype=np.uint64)
    raise DecontamError(f"unknown rule {rule!r}; known: {sorted(RULES)}")


def doc_grams(doc: Doc, rule: str, tok=None) -> np.ndarray:
    # segments are n-grammed separately: an n-gram never spans a segment boundary that the text does not have
    parts = [ngram_hashes(elements(s.text, rule, tok), RULES[rule]) for s in doc.segments if s.text]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.uint64)


@dataclass(frozen=True)
class OverlapReport:
    rules: tuple[str, ...]
    n_train: int
    n_heldout: int
    flagged: dict[str, list[str]]  # train doc key -> rules that matched
    per_rule: dict[str, int]

    def to_json(self) -> dict:
        return {
            "format": FORMAT,
            "rules": list(self.rules),
            "n": {r: RULES[r] for r in self.rules},
            "n_train": self.n_train,
            "n_heldout": self.n_heldout,
            "per_rule": dict(self.per_rule),
            "flagged": {k: list(v) for k, v in sorted(self.flagged.items())},
        }


def overlap_report(
    train: Iterable[Doc], heldout: Iterable[Doc], tok=None, rules: Sequence[str] | None = None
) -> OverlapReport:
    """Flag every training document that shares an n-gram with any held-out document, per rule."""
    if rules is None:  # DEFAULTS["data"]["decontam_rules"]; token13 only once the tokenizer exists
        rules = [r for r in DEFAULTS["data"]["decontam_rules"] if r != "token13" or tok is not None]
    rules = tuple(rules)
    for rule in rules:
        if rule not in RULES:
            raise DecontamError(f"unknown rule {rule!r}; known: {sorted(RULES)}")
    heldout = list(heldout)
    held = {r: np.unique(np.concatenate([doc_grams(d, r, tok) for d in heldout] or [np.zeros(0, np.uint64)]))
            for r in rules}  # fmt: skip
    flagged: dict[str, list[str]] = {}
    per_rule = {r: 0 for r in rules}
    n_train = 0
    for doc in train:
        n_train += 1
        hits = []
        for rule in rules:
            grams = doc_grams(doc, rule, tok)
            if len(grams) and len(held[rule]):
                pos = np.searchsorted(held[rule], grams)
                pos[pos == len(held[rule])] = 0
                if bool((held[rule][pos] == grams).any()):
                    hits.append(rule)
                    per_rule[rule] += 1
        if hits:
            flagged[doc.key] = hits
    return OverlapReport(rules, n_train, len(heldout), flagged, per_rule)


def filter_docs(docs: Iterable[Doc], report: OverlapReport | Mapping) -> list[Doc]:
    """The documents the report did not flag, in order."""
    flagged = report.flagged if isinstance(report, OverlapReport) else report["flagged"]
    return [d for d in docs if d.key not in flagged]


def load_report(path: Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    check_format(doc, "overlap")
    return doc


def main(argv: Sequence[str]) -> int:
    from .manifest import load_manifest
    from .split import HELDOUT_SPLITS, TRAIN_SPLITS, load_split

    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual decontam", description="overlap report")
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--split", type=Path, help="bilingual-split/1; omit for a derived manifest whose files carry it")
    ap.add_argument("--train", default=",".join(TRAIN_SPLITS))
    ap.add_argument("--heldout", default=",".join(HELDOUT_SPLITS))
    ap.add_argument("--rules", help="comma-separated; default every rule (token13 only with --tokenizer)")
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(list(argv))
    if not args.dry_run and args.out is None:
        ap.error("--out is required unless --dry-run")
    manifest = load_manifest(args.manifest)
    train_names, held_names = args.train.split(","), args.heldout.split(",")
    if args.split is not None:
        where = {k: n for n, keys in load_split(args.split)["splits"].items() for k in keys}
        docs = list(manifest.iter_docs())
        train = [d for d in docs if where.get(d.key) in train_names]
        held = [d for d in docs if where.get(d.key) in held_names]
    else:
        train = [d for name in train_names for d in manifest.iter_docs(split=name)]
        held = [d for name in held_names for d in manifest.iter_docs(split=name)]
    tok = None
    if args.tokenizer is not None:
        from .tokenizer import load_tokenizer

        tok = load_tokenizer(args.tokenizer)
    report = overlap_report(train, held, tok, args.rules.split(",") if args.rules else None)
    if not args.dry_run:
        atomic_write_json(args.out, report.to_json())
    print(f"decontam: {len(report.flagged)} of {report.n_train} training documents overlap {report.n_heldout} "
          f"held-out documents; per rule {report.per_rule}" + (" (dry run)" if args.dry_run else ""))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
