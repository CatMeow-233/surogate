"""freqtrim (GAP-18): TLC's vocabulary as a subset of a Qwen byte-level BPE, chosen by frequency on the train split.

The Qwen teacher of the teacher-led training scores student text by byte-prefix marginals (BPM); a student whose
every token is a Qwen token, merged in Qwen's rank order, keeps that scoring nearly exact. Construction, as the
experiment notebook specifies it (docs/data.md: Tokenizer):

0. The subset starts with the 256 ByteLevel alphabet tokens (all Qwen tokens), so every byte encodes.
1. Qwen's own BPE (its vocabulary and merges, without its normaliser) encodes the train split inside the pre-tokens
   of the *arm's* regex, and its tokens are counted. Counting under one regex and serving another would spend slots
   on tokens the served regex never produces.
2. Tokens are taken by count, highest first (ties: lower Qwen id first); each brings its merge parents, recursively
   (the parents of a token are those of the lowest-rank Qwen merge producing it). A token whose closure no longer
   fits is skipped and the next one tried, until the subset holds vocab_size - 5 tokens. A train split that reaches
   fewer tokens is refused, as train_tokenizer refuses a corpus too small for its vocabulary.
3. The merges are Qwen's merges whose result and both parents are in the subset, in Qwen's rank order.
4. The special tokens are the student's own, ids 0-4 as in P0 (<|pad|> <|bos|> <|eos|> <|code_start|>
   <|code_end|>); Qwen's special tokens (<|im_end|> and the rest) never enter. The ordinary tokens follow in Qwen
   id order, from id 5.
5. The regex is "qwen", Qwen2's pattern verbatim (the native tokenizer's hand-written splitter; the teacher file
   must carry exactly this pattern), or a "fine" form for the std::regex fallback: "fine-a" lets letters take only
   an optional leading space (the notebook's form A); "fine-b" also keeps "_" + letters together (form B). The
   notebook fixes one form before TOK-AB; both exist here so that the owner can.

Qwen's tokenizer.json carries an NFC normaliser; the student has none, so decode(encode(x)) == x, and the manifest
records the teacher's normaliser for the BPM side, which must see the same bytes. The manifest records the teacher's
repository, revision, licence and file sha256, and the Qwen vocabulary identity: the sha256 of the ordinary
vocabulary, merges, pre-tokenizer and normaliser, which two teacher files differing only in added special tokens
share: the notebook freezes this identity, not a file hash. check() runs the notebook's CPU checks on a built
directory against its teacher.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers

from ..files import dump_json, file_sha256
from .corpus import Corpus
from .tokenizer import (
    BYTE_OF,
    EDGE_CASES,
    MANIFEST_FILE,
    PRETOKENIZERS,
    SPECIAL_TOKENS,
    TOKENIZER_FILE,
    TOKENIZER_VERSION,
    VOCAB_SIZE,
    TrainTexts,
    finish_tokenizer,
    split_pre_tokenizer,
    tokenizer_sha256,
    write_config,
)

QWEN2_PATTERN = PRETOKENIZERS["qwen2"]
# The fine forms run on the native std::regex fallback, which has no inline flags: (?i:'s|...) is spelt '[sS]|....
_CONTRACTIONS = r"'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]"
_QWEN_TAIL = r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
REGEXES = {
    "qwen": QWEN2_PATTERN,
    "fine-a": rf"{_CONTRACTIONS}| ?\p{{L}}+|{_QWEN_TAIL}",
    "fine-b": rf"{_CONTRACTIONS}| ?\p{{L}}+|_\p{{L}}+|{_QWEN_TAIL}",
}
KIND = "freqtrim"
ENCODE_BATCH = 1024  # documents per encode_batch call while counting


class FreqtrimError(ValueError):
    pass


@dataclass(frozen=True)
class TeacherRef:
    """Where the teacher file came from; recorded in the manifest. `sha256`, when given, must match the file."""

    repo: str
    revision: str
    licence: str
    sha256: str | None = None


@dataclass(frozen=True)
class Teacher:
    path: Path
    file_sha256: str
    vocab: dict[str, int]  # ordinary tokens only (no added tokens)
    merges: list[tuple[str, str]]  # in rank order
    pattern: str
    normalizer: Any  # the JSON of the teacher's normaliser, or None
    identity: str  # sha256 of vocab + merges + pre-tokenizer + normaliser

    def bpe(self, pattern: str) -> Tokenizer:
        """The teacher's BPE, without its normaliser, inside the pre-tokens of `pattern`."""
        tokenizer = Tokenizer(models.BPE(vocab=self.vocab, merges=self.merges))
        tokenizer.pre_tokenizer = split_pre_tokenizer(pattern)
        return tokenizer

    def parents(self) -> dict[str, tuple[str, str]]:
        """Each merged token's parents: those of the lowest-rank merge producing it."""
        out: dict[str, tuple[str, str]] = {}
        for a, b in self.merges:
            out.setdefault(a + b, (a, b))
        return out


def _canonical_sha256(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _merge_pair(entry: Any) -> tuple[str, str]:
    if isinstance(entry, str):
        a, sep, b = entry.partition(" ")
        if not sep or " " in b:
            raise FreqtrimError(f"merge {entry!r} is not two byte-level tokens")
        return a, b
    if isinstance(entry, list) and len(entry) == 2 and all(isinstance(x, str) for x in entry):
        return entry[0], entry[1]
    raise FreqtrimError(f"merge {entry!r} is neither 'a b' nor [a, b]")


def _split_pattern(pre_tokenizer: Any) -> str:
    """The Split pattern of a Qwen-style pre-tokenizer: Sequence[Split(Regex, isolated), ByteLevel(no regex)]."""
    steps = pre_tokenizer.get("pretokenizers") if isinstance(pre_tokenizer, dict) else None
    if pre_tokenizer is None or pre_tokenizer.get("type") != "Sequence" or not steps or len(steps) != 2:
        raise FreqtrimError("the teacher's pre-tokenizer is not Sequence[Split, ByteLevel]")
    split, byte_level = steps
    regex = split.get("pattern", {}).get("Regex") if split.get("type") == "Split" else None
    if regex is None or split.get("behavior") != "Isolated" or split.get("invert"):
        raise FreqtrimError("the teacher's first pre-tokenizer step is not an isolating Split on a regex")
    if byte_level.get("type") != "ByteLevel" or byte_level.get("use_regex") or byte_level.get("add_prefix_space"):
        raise FreqtrimError("the teacher's second pre-tokenizer step is not ByteLevel without regex or prefix space")
    return regex


def load_teacher(path: Path, ref: TeacherRef) -> Teacher:
    """Read a Qwen-format tokenizer.json: a byte-level BPE whose split pattern is Qwen2's, verbatim."""
    path = Path(path)
    sha = file_sha256(path)
    if ref.sha256 is not None and ref.sha256 != sha:
        raise FreqtrimError(f"{path}: sha256 {sha}, expected {ref.sha256}")
    if len(ref.revision) != 40 or any(c not in "0123456789abcdef" for c in ref.revision):
        raise FreqtrimError(f"teacher revision {ref.revision!r} is not a full 40-hex commit sha")
    if not ref.repo or not ref.licence:
        raise FreqtrimError("the teacher's repository and licence are required: the manifest records them")
    data = json.loads(path.read_text(encoding="utf-8"))
    model = data.get("model", {})
    if model.get("type") != "BPE":
        raise FreqtrimError(f"{path}: model type {model.get('type')!r}, not BPE")
    settings = {k: model.get(k) for k in ("dropout", "continuing_subword_prefix", "end_of_word_suffix")}
    if (
        settings["dropout"] not in (None, 0.0)
        or settings["continuing_subword_prefix"]
        or settings["end_of_word_suffix"]
    ):
        raise FreqtrimError(f"{path}: unsupported BPE settings {settings}")
    if model.get("byte_fallback") or model.get("ignore_merges"):
        raise FreqtrimError(f"{path}: byte_fallback / ignore_merges are not Qwen2's settings")
    pattern = _split_pattern(data.get("pre_tokenizer"))
    if pattern != QWEN2_PATTERN:
        raise FreqtrimError(f"{path}: split pattern is not Qwen2's verbatim: {pattern!r}")
    added = {entry["content"] for entry in data.get("added_tokens", [])}
    vocab = {token: int(i) for token, i in model["vocab"].items() if token not in added}
    merges = [_merge_pair(entry) for entry in model["merges"]]
    missing = [c for c in pre_tokenizers.ByteLevel.alphabet() if c not in vocab]
    if missing:
        raise FreqtrimError(f"{path}: {len(missing)} ByteLevel alphabet tokens are not teacher tokens")
    identity = _canonical_sha256(
        {
            "vocab": sorted(vocab.items(), key=lambda kv: kv[1]),
            "merges": [list(m) for m in merges],
            "pre_tokenizer": data.get("pre_tokenizer"),
            "normalizer": data.get("normalizer"),
        }
    )
    return Teacher(path, sha, vocab, merges, pattern, data.get("normalizer"), identity)


def _encodings(tokenizer: Tokenizer, texts: Iterable[str]):
    """Encodings of `texts` without special tokens, ENCODE_BATCH documents per encode_batch call."""
    batch: list[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) == ENCODE_BATCH:
            yield from tokenizer.encode_batch(batch, add_special_tokens=False)
            batch = []
    if batch:
        yield from tokenizer.encode_batch(batch, add_special_tokens=False)


def count_tokens(teacher: Teacher, pattern: str, texts: Iterable[str]) -> np.ndarray:
    """Counts of each teacher id over `texts`, the teacher's merges applied inside the pre-tokens of `pattern`."""
    counts = np.zeros(max(teacher.vocab.values()) + 1, dtype=np.int64)
    ids: list[int] = []
    for encoding in _encodings(teacher.bpe(pattern), texts):
        ids.extend(encoding.ids)
        if len(ids) >= 1 << 22:
            counts += np.bincount(np.asarray(ids, dtype=np.int64), minlength=counts.size)
            ids = []
    if ids:
        counts += np.bincount(np.asarray(ids, dtype=np.int64), minlength=counts.size)
    return counts


def reachable(tokenizer: Tokenizer, texts: Iterable[str]) -> int:
    """The number of distinct student ids `texts` produce: the subset's reachable size under its regex."""
    seen: set[int] = set()
    for encoding in _encodings(tokenizer, texts):
        seen.update(encoding.ids)
    return len(seen)


def select(teacher: Teacher, counts: np.ndarray, size: int) -> tuple[set[str], dict[str, int]]:
    """Step 0-2: the subset of `size` teacher tokens, and how it was filled."""
    tokens = {i: t for t, i in teacher.vocab.items()}
    parents = teacher.parents()
    keep = set(pre_tokenizers.ByteLevel.alphabet())
    if size < len(keep):
        raise FreqtrimError(f"a subset of {size} tokens cannot hold the 256 byte tokens")
    stats = {"byte_alphabet": len(keep), "by_frequency": 0, "as_parents": 0, "skipped_for_space": 0}
    observed = np.flatnonzero(counts)
    order = observed[np.lexsort((observed, -counts[observed]))]  # count descending, then id ascending
    for i in order.tolist():
        if len(keep) == size:
            break
        token = tokens.get(i)
        if token is None or token in keep:
            continue
        add, stack = [], [token]
        while stack:
            x = stack.pop()
            if x in keep or x in add:
                continue
            add.append(x)
            if x in parents:
                stack.extend(parents[x])
            elif len(x) > 1:
                raise FreqtrimError(f"teacher token {x!r} is neither a byte nor the result of a merge")
        if len(keep) + len(add) > size:
            stats["skipped_for_space"] += 1
            continue
        keep.update(add)
        stats["by_frequency"] += 1
        stats["as_parents"] += len(add) - 1
    stats["observed_teacher_tokens"] = int(observed.size)
    if len(keep) < size:
        raise FreqtrimError(f"the train split reaches {len(keep)} teacher tokens, not {size}: it is too small")
    return keep, stats


def build_tokenizer(teacher: Teacher, keep: set[str], pattern: str) -> Tokenizer:
    """Steps 3-4: student vocabulary (specials 0-4, then the subset in teacher id order) and the filtered merges."""
    vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
    for token in sorted(keep, key=teacher.vocab.__getitem__):
        vocab[token] = len(vocab)
    merges = [(a, b) for a, b in teacher.merges if a in keep and b in keep and a + b in keep]
    tokenizer = finish_tokenizer(Tokenizer(models.BPE(vocab=vocab, merges=merges)))
    tokenizer.pre_tokenizer = split_pre_tokenizer(pattern)
    tokenizer.add_special_tokens(list(SPECIAL_TOKENS))
    return tokenizer


def build(
    corpus: Corpus,
    out: Path,
    teacher_path: Path,
    ref: TeacherRef,
    *,
    regex: str,
    vocab_size: int = VOCAB_SIZE,
) -> Path:
    """Build a freqtrim tokenizer directory (tokenizer.json, tokenizer_config.json, tokenizer_manifest.json)."""
    out, teacher_path = Path(out), Path(teacher_path)
    if (out / TOKENIZER_FILE).exists():
        raise FileExistsError(f"{out / TOKENIZER_FILE} exists")
    if regex not in REGEXES:
        raise FreqtrimError(f"regex must be one of {sorted(REGEXES)}, got {regex!r}")
    teacher = load_teacher(teacher_path, ref)
    pattern = REGEXES[regex]
    texts = TrainTexts(corpus)
    keep, stats = select(teacher, count_tokens(teacher, pattern, texts), vocab_size - len(SPECIAL_TOKENS))
    sources = texts.sources()
    tokenizer = build_tokenizer(teacher, keep, pattern)
    if tokenizer.get_vocab_size() != vocab_size:
        raise FreqtrimError(f"built {tokenizer.get_vocab_size()} ids, not {vocab_size}")
    reached = reachable(tokenizer, TrainTexts(corpus))
    student_to_teacher = [teacher.vocab[t] for t, _ in sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])[5:]]
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / TOKENIZER_FILE))
    write_config(out)
    manifest = {
        "tlc_tokenizer_version": TOKENIZER_VERSION,
        "kind": KIND,
        "tokenizer_sha256": tokenizer_sha256(out),
        "vocab_size": vocab_size,
        "regex": regex,
        "split_pattern": pattern,
        "special_tokens": list(SPECIAL_TOKENS),
        "normalizer": None,
        "teacher": {
            "repo": ref.repo,
            "revision": ref.revision,
            "licence": ref.licence,
            "file": teacher_path.name,
            "file_sha256": teacher.file_sha256,
            "vocab_identity": teacher.identity,
            "normalizer": teacher.normalizer,
            "ordinary_tokens": len(teacher.vocab),
            "merges": len(teacher.merges),
        },
        "subset": {
            **stats,
            "size": len(keep),
            "merges": len(tokenizer_merges(tokenizer)),
            "reachable_on_train": reached,
            "student_to_teacher_sha256": _canonical_sha256(student_to_teacher),
            "selection": "count descending, then teacher id; closure under the lowest-rank parents; "
            "a token whose closure does not fit is skipped",
        },
        "corpus": texts.record(),
        "sources": sources,
    }
    (out / MANIFEST_FILE).write_text(dump_json(manifest), encoding="utf-8")
    return out


def tokenizer_merges(tokenizer: Tokenizer) -> list[tuple[str, str]]:
    return [_merge_pair(m) for m in json.loads(tokenizer.to_str())["model"]["merges"]]


def _mixes_digit(text: str | None) -> bool:
    """A token text holding a \\p{N} character next to anything else (one digit per token, the notebook's digit rule)."""
    if text is None or len(text) == 1:
        return False
    return any(unicodedata.category(c) in ("Nd", "Nl", "No") for c in text)


def check(directory: Path, teacher: Teacher, texts: Sequence[str] = EDGE_CASES) -> list[str]:
    """The notebook's CPU checks of a freqtrim directory against its teacher; empty when all hold.

    Every ordinary student token is a teacher token; the set is closed under merge parents; each of the 256 byte
    values encodes alone; `texts` round-trip without special ids; no token mixes a digit with anything else; the
    special ids are 0-4; the merges are the teacher's, both parents present, in rank order; and the student BPE
    encodes the bytes of every teacher token alone (the premise of BPM's φ table: φ(w) is the student's first token for
    the bytes of teacher token w).
    """
    path = Path(directory) / TOKENIZER_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        tokenizer = Tokenizer.from_file(str(path))
    except Exception as err:  # tokenizers raises a bare Exception for an inconsistent file
        return [f"tokenizer.json does not load: {err}"]
    problems: list[str] = []
    vocab = data["model"]["vocab"]
    specials = [(t["id"], t["content"]) for t in data["added_tokens"]]
    if specials != list(enumerate(SPECIAL_TOKENS)) or any(vocab.get(t) != i for i, t in specials):
        problems.append(f"special tokens are {specials}, not ids 0-4 {list(SPECIAL_TOKENS)}")
    ordinary = {t: i for t, i in vocab.items() if t not in SPECIAL_TOKENS}
    if data.get("normalizer") is not None:
        problems.append("the student has a normaliser")
    strangers = sorted(t for t in ordinary if t not in teacher.vocab)
    if strangers:
        problems.append(f"{len(strangers)} student tokens are not teacher tokens, e.g. {strangers[:3]}")
    parents = teacher.parents()
    orphans = sorted(t for t in ordinary if t in parents and not all(p in ordinary for p in parents[t]))
    if orphans:
        problems.append(f"{len(orphans)} tokens lack a merge parent, e.g. {orphans[:3]}")
    if sorted(ordinary, key=ordinary.__getitem__) != sorted(ordinary, key=lambda t: teacher.vocab.get(t, -1)):
        problems.append("ordinary tokens are not in teacher id order")
    merges = [_merge_pair(m) for m in data["model"]["merges"]]
    expected = [(a, b) for a, b in teacher.merges if a in ordinary and b in ordinary and a + b in ordinary]
    if merges != expected:
        foreign = set(merges) - set(teacher.merges)
        problems.append(f"merges differ from the teacher's filtered merges ({len(foreign)} are not teacher merges)")
    model = tokenizer.model
    for char in pre_tokenizers.ByteLevel.alphabet():
        if [t.value for t in model.tokenize(char)] != [char] or char not in ordinary:
            problems.append(f"byte {BYTE_OF[char]} ({char!r}) does not encode as one token")
    special_ids = set(range(len(SPECIAL_TOKENS)))
    tokenizer.encode_special_tokens = True  # document text: special-token strings are text
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if tokenizer.decode(ids, skip_special_tokens=False) != text:
            problems.append(f"round trip fails on {text[:40]!r}")
        if special_ids & set(ids):
            problems.append(f"document text {text[:40]!r} produced a special id")
    mixed = [t for t in ordinary if _mixes_digit(token_text(t))]
    if mixed:
        problems.append(f"{len(mixed)} tokens mix a digit with other characters, e.g. {mixed[:3]}")
    unencodable = [w for w in teacher.vocab if "".join(t.value for t in model.tokenize(w)) != w]
    if unencodable:
        problems.append(f"{len(unencodable)} teacher tokens do not encode alone, e.g. {unencodable[:3]}")
    return problems


def token_text(token: str) -> str | None:
    """The text of one byte-level token, or None when its bytes are not whole UTF-8 characters."""
    try:
        return bytes(BYTE_OF[c] for c in token).decode("utf-8")
    except UnicodeDecodeError:
        return None


def check_directory(directory: Path, teacher_path: Path) -> list[str]:
    """check() against the teacher the directory's manifest names; a teacher file of another sha256 is refused."""
    manifest = json.loads((Path(directory) / MANIFEST_FILE).read_text(encoding="utf-8"))
    if manifest.get("kind") != KIND:
        raise FreqtrimError(f"{directory}: not a freqtrim tokenizer (kind {manifest.get('kind')!r})")
    t = manifest["teacher"]
    teacher = load_teacher(teacher_path, TeacherRef(t["repo"], t["revision"], t["licence"], t["file_sha256"]))
    problems = check(directory, teacher)
    if manifest["tokenizer_sha256"] != tokenizer_sha256(directory):
        problems.append("tokenizer.json does not have the manifest's sha256")
    return problems
