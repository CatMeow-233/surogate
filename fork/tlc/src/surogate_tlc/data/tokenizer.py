"""TLC's tokenizer, a joint English + Python byte-level BPE, and its consistency check against the native one.

- Pre-tokenization: one of PRETOKENIZERS, then the byte-level mapping. The choice is open (docs/data.md: Tokenizer):
  "gpt2" and "python" are regex candidates of the experiment notebook (GPT-2's pattern, and an identifier-aware
  pattern after its pyaware2 with Unicode letter classes); "qwen2", the default, is Qwen2's split pattern, GPT-2's
  family with the Unicode classes \\p{L} and \\p{N} and every digit a piece of its own, the regex the teacher-led
  training memo pairs with its Qwen-subset vocabulary; "python-digit" is "python" with every digit a piece of its
  own (the notebook's digit rule). For a model_type it does not know, such as "tlc", the native tokenizer (csrc/src/tokenizer)
  reads the pattern from tokenizer.json, or uses GPT-2's when there is none. It has hand-written splitters for the
  qwen2 and GPT-2 patterns and runs any other one, "python" and "python-digit" included, through its std::regex
  fallback. There is no normaliser, so decode(encode(text)) == text.
- The vocabulary is a BPE trained here (train_tokenizer), or a frequency-chosen subset of a Qwen BPE (freqtrim,
  GAP-18, in surogate_tlc.data.freqtrim; the `freqtrim` command below).
- Special tokens, ids 0-4: <|pad|>, <|bos|>, <|eos|>, <|code_start|>, <|code_end|>; the pad, bos and eos ids of
  hf.default_config() are these.
- The chat template renders <|bos|>, the message contents, then <|eos|> unless a generation prompt is asked
  for: a prompt is the start of a document, as in the shards.

The tokenizer is trained on the train split of a corpus spec (surogate_tlc.data.corpus), so validation documents
stay unseen. compare() runs texts through this tokenizer and a native one; on the GPU host the native one is
surogate._surogate.Tokenizer (docs/runbook.md: G0).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from .. import hf
from ..files import dump_json, file_sha256
from .corpus import RULES, Corpus, decontam_record, documents, load_corpus, provenance, sample, split_of

PAD, BOS, EOS, CODE_START, CODE_END = SPECIAL_TOKENS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|code_start|>",
    "<|code_end|>",
)
VOCAB_SIZE = 16384
# Pre-tokenizer candidates by name: a split pattern, or None for the GPT-2 regex built into the byte-level step.
# "qwen2" is GPT-2's family with single digits; "python" keeps identifiers (with "_") whole and common operators
# together, after the notebook's pyaware2, with Unicode letter and digit classes; "python-digit" (below) is "python"
# with single digits.
PRETOKENIZERS = {
    "qwen2": (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    "gpt2": None,
    "python": (
        r"[ ]?[\p{L}_][\p{L}\p{N}_]*|[ ]?\p{N}+(?:\.\p{N}*)?(?:[eE][+-]?\p{N}+)?"
        r"|[ ]?(?:\*\*=|//=|>>=|<<=|->|:=|==|!=|<=|>=|\*\*|//|<<|>>|[+\-*/%&|^@]=)|[ ]?[^\s\p{L}\p{N}_]|\n[ \t]*|[ \t]+"
    ),
}
# The notebook's per-digit variant of "python": its number branch becomes \p{N}. Its identifier branch also
# drops \p{N}, or "x1" would stay one piece whose merge "x1" breaks the one-digit-per-token rule; so "x1" is "x" + "1",
# as under qwen2.
PRETOKENIZERS["python-digit"] = (
    PRETOKENIZERS["python"]
    .replace(r"[ ]?\p{N}+(?:\.\p{N}*)?(?:[eE][+-]?\p{N}+)?", r"\p{N}")
    .replace(r"[\p{L}_][\p{L}\p{N}_]*", r"[\p{L}_][\p{L}_]*")
)
DEFAULT_PRETOKENIZER = "qwen2"
CHAT_TEMPLATE = (
    "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}"
    "{% if not add_generation_prompt %}{{ eos_token }}{% endif %}"
)
TOKENIZER_FILE = "tokenizer.json"
CONFIG_FILE = "tokenizer_config.json"
MANIFEST_FILE = "tokenizer_manifest.json"
# Bump when the recipe above changes (pattern, special tokens, template).
TOKENIZER_VERSION = 1

# Texts the check always includes: line endings, tabs, non-ASCII identifiers and literals, digits, contractions,
# special-token strings inside text, U+2028 inside a string.
EDGE_CASES = (
    "",
    "def solve(a, b):\n    return a+b\n",
    "x  =  1\t# tab\there\n\n\n",
    "if x:\n\tpass\r\n",
    "a = 1\rb = 2\r",
    "\u53d8\u91cf = 1  # \u4e2d\u6587\u6ce8\u91ca\n",
    "print('h\u00e9llo w\u00f6rld \U0001f642')\n",
    "s = f'{x!r:>10} \U0001f642'\n",
    "n = 1234567890; m = 3.14159e-10; k = 0x1F_FF\n",
    "    \n  \n",
    "trailing spaces   ",
    "a\nb",
    "\n\nclass A:\n    def f(self):  \n        ...\n",
    "<|code_start|>def f(x):\n    return x<|code_end|>",
    "text with <|eos|> and <|pad|> inside",
    "It's a dog's life, isn't it? I'll go. WE'RE HERE.",
    "s = \"\\n\\t\\\\\" + r'\\d+'\n",
    "lambda *args, **kwargs: (args, kwargs)\n",
    "x = '\u2028'  # a line separator inside a string\n",
    "items[i](j):\n",
)


def split_pre_tokenizer(pattern: str | None) -> pre_tokenizers.PreTokenizer:
    """A split pattern isolating its matches, then the byte-level mapping; None is the byte-level step's GPT-2 regex."""
    if pattern is None:
        return pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    split = pre_tokenizers.Split(Regex(pattern), behavior="isolated")
    return pre_tokenizers.Sequence([split, pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])


def _pre_tokenizer(name: str) -> pre_tokenizers.PreTokenizer:
    return split_pre_tokenizer(PRETOKENIZERS[name])


class TrainTexts:
    """The texts every vocabulary builder learns from: the documents of the corpus's train split, in corpus order,
    counted per source (`count`) next to the documents the builder dropped (`dropped`, by source and reason)."""

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.dropped: Counter = Counter()
        self.count: Counter = Counter()

    def __iter__(self):
        for doc in documents(self.corpus, self.dropped):
            if split_of(self.corpus, doc) == "train":
                self.count[doc.source.name] += 1
                yield doc.text

    def sources(self) -> list[dict[str, Any]]:
        """The corpus provenance per source, with the documents read and dropped by the last pass."""
        sources = provenance(self.corpus)
        for s in sources:
            s["documents"] = self.count[s["name"]]
            s["dropped"] = {reason: n for (name, reason), n in sorted(self.dropped.items()) if name == s["name"]}
        return sources

    def record(self) -> dict[str, Any]:
        return {
            "spec_sha256": file_sha256(self.corpus.path),
            "seed": self.corpus.seed,
            "val_fraction": self.corpus.val_fraction,
            "trained_on": "train split",
            "decontam": decontam_record(self.corpus),
            "rules": list(RULES),
        }


def finish_tokenizer(tokenizer: Tokenizer) -> Tokenizer:
    """The decoder and post-processor every TLC tokenizer.json carries."""
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    return tokenizer


def write_config(out: Path) -> None:
    """tokenizer_config.json: the special tokens and the chat template, the same for every vocabulary."""
    config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": BOS,
        "eos_token": EOS,
        "pad_token": PAD,
        "additional_special_tokens": [CODE_START, CODE_END],
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "model_max_length": hf.CONTEXT_LENGTH,
        "chat_template": CHAT_TEMPLATE,
    }
    (Path(out) / CONFIG_FILE).write_text(dump_json(config), encoding="utf-8")


def train_tokenizer(
    corpus: Corpus,
    out: Path,
    *,
    pretokenizer: str = DEFAULT_PRETOKENIZER,
    vocab_size: int = VOCAB_SIZE,
    min_frequency: int = 2,
) -> Path:
    """Train on the corpus's train split and write tokenizer.json, tokenizer_config.json and the manifest."""
    out = Path(out)
    if (out / TOKENIZER_FILE).exists():
        raise FileExistsError(f"{out / TOKENIZER_FILE} exists")
    tokenizer = finish_tokenizer(Tokenizer(models.BPE()))
    tokenizer.pre_tokenizer = _pre_tokenizer(pretokenizer)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    texts = TrainTexts(corpus)
    tokenizer.train_from_iterator(texts, trainer)
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError(f"the corpus yields {tokenizer.get_vocab_size()} tokens, not {vocab_size}: it is too small")
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / TOKENIZER_FILE))
    write_config(out)
    manifest = {
        "tlc_tokenizer_version": TOKENIZER_VERSION,
        "tokenizer_sha256": tokenizer_sha256(out),
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "pretokenizer": pretokenizer,
        "split_pattern": PRETOKENIZERS[pretokenizer],
        "special_tokens": list(SPECIAL_TOKENS),
        "corpus": texts.record(),
        "sources": texts.sources(),
    }
    (out / MANIFEST_FILE).write_text(dump_json(manifest), encoding="utf-8")
    return out


def tokenizer_sha256(path: Path) -> str:
    """The tokenizer's identity: sha256 of tokenizer.json, which shard manifests and results record."""
    return file_sha256(Path(path) / TOKENIZER_FILE)


def load(path: Path, *, ordinary: bool = False) -> Tokenizer:
    """The tokenizer of a tokenizer or model directory. `ordinary`: special-token strings are encoded as text."""
    tokenizer = Tokenizer.from_file(str(Path(path) / TOKENIZER_FILE))
    tokenizer.encode_special_tokens = ordinary
    return tokenizer


def special_ids(tokenizer: Tokenizer) -> dict[str, int]:
    ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    missing = [token for token, i in ids.items() if i is None]
    if missing:
        raise ValueError(f"the tokenizer lacks the special tokens {missing}")
    return ids


def _byte_of() -> dict[str, int]:
    """GPT-2's bytes_to_unicode, inverted: the byte each ByteLevel alphabet character stands for."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1), *range(ord("®"), ord("ÿ") + 1)]
    chars = list(printable)
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            chars.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(printable, chars, strict=True)}


BYTE_OF = _byte_of()


def token_bytes(tokenizer: Tokenizer) -> list[bytes | None]:
    """The bytes every id stands for (its byte-level token with the ByteLevel alphabet inverted), None for the
    special tokens: they frame a document and stand for none of its text, so bits per byte counts no byte for them
    (docs/training.md: 评测). ValueError for a token outside the byte-level alphabet or an id without a token."""
    special = set(special_ids(tokenizer).values())
    out: list[bytes | None] = [None] * tokenizer.get_vocab_size()
    for token, i in tokenizer.get_vocab(with_added_tokens=True).items():
        if i in special:
            continue
        try:
            out[i] = bytes(BYTE_OF[c] for c in token)
        except KeyError:
            raise ValueError(f"token {i} ({token!r}) is not a byte-level token") from None
    missing = [i for i, b in enumerate(out) if b is None and i not in special]
    if missing:
        raise ValueError(f"ids {missing[:5]} have no token")
    return out


def _first_difference(a: Sequence[int], b: Sequence[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))


def compare(path: Path, texts: Sequence[str], native: Any, *, examples: int = 5) -> list[str]:
    """Differences between this tokenizer (directory `path`) and `native` on `texts`; empty means consistent.

    `native` is surogate._surogate.Tokenizer.from_pretrained(path), or anything with its encode(text,
    add_special_tokens), encode_ordinary(text), decode(ids), vocab_size and bos/eos/pad_token_id. Checked: the
    special ids and the vocabulary size, and per text the ordinary encoding (special-token strings are text, as in
    the shards), the encoding with special tokens (as for prompts), and decoding the ordinary ids on both sides.
    """
    ordinary, special = load(path, ordinary=True), load(path)
    ids = special_ids(special)
    problems = [
        f"{attr}: native {getattr(native, attr)}, {token} has id {ids[token]}"
        for token, attr in ((PAD, "pad_token_id"), (BOS, "bos_token_id"), (EOS, "eos_token_id"))
        if getattr(native, attr) != ids[token]
    ]
    if native.vocab_size != special.get_vocab_size():
        problems.append(f"vocab_size: native {native.vocab_size}, here {special.get_vocab_size()}")
    failures: dict[str, list[str]] = {"encode_ordinary": [], "encode": [], "decode (here)": [], "decode (native)": []}
    for i, text in enumerate(texts):
        where = f"text {i} {text[:40]!r}"
        reference = ordinary.encode(text, add_special_tokens=False).ids
        encodings = {
            "encode_ordinary": (reference, list(native.encode_ordinary(text))),
            "encode": (special.encode(text, add_special_tokens=False).ids, list(native.encode(text, False))),
        }
        for check, (here, there) in encodings.items():
            if here != there:
                j = _first_difference(here, there)
                failures[check].append(f"{where}: token {j}: here {here[j : j + 4]}, native {there[j : j + 4]}")
        for check, decoded in (
            ("decode (here)", ordinary.decode(reference, skip_special_tokens=False)),
            ("decode (native)", native.decode(reference)),
        ):
            if decoded != text:
                failures[check].append(f"{where}: decodes to {decoded[:40]!r}")
    for check, lines in failures.items():
        if lines:
            problems += [f"{check}: {len(lines)} of {len(texts)} texts differ", *lines[:examples]]
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tlc.data.tokenizer", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="train on the train split of a corpus spec")
    train.add_argument("corpus", type=Path, help="corpus spec (TOML)")
    train.add_argument("out", type=Path, help="output directory")
    train.add_argument("--pretokenizer", choices=sorted(PRETOKENIZERS), default=DEFAULT_PRETOKENIZER)
    train.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    check = commands.add_parser("check", help="compare with the native tokenizer (needs the native extension)")
    check.add_argument("model_dir", type=Path, help="model (or tokenizer) directory")
    check.add_argument("corpus", type=Path, help="corpus spec (TOML) to draw documents from")
    check.add_argument("--limit", type=int, default=10_000, help="documents, in equal shares per source")
    trim = commands.add_parser("freqtrim", help="a Qwen BPE subset by frequency on the train split (GAP-18)")
    trim.add_argument("corpus", type=Path, help="corpus spec (TOML)")
    trim.add_argument("out", type=Path, help="output directory")
    trim.add_argument("--teacher", type=Path, required=True, help="the Qwen tokenizer.json")
    trim.add_argument("--teacher-repo", required=True, help="its Hugging Face repository, e.g. Qwen/...")
    trim.add_argument("--teacher-revision", required=True, help="the 40-hex commit the file was fetched at")
    trim.add_argument("--teacher-licence", required=True, help="the repository's licence, e.g. apache-2.0")
    trim.add_argument("--teacher-sha256", help="refuse a file with another sha256")
    trim.add_argument("--regex", required=True, help="qwen, fine-a or fine-b (docs/data.md: Tokenizer)")
    trim.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    verify = commands.add_parser("freqtrim-check", help="the CPU checks of a freqtrim directory against its teacher")
    verify.add_argument("tokenizer_dir", type=Path)
    verify.add_argument("--teacher", type=Path, required=True, help="the teacher tokenizer.json the manifest names")
    args = parser.parse_args(argv)
    if args.command == "train":
        out = train_tokenizer(
            load_corpus(args.corpus), args.out, pretokenizer=args.pretokenizer, vocab_size=args.vocab_size
        )
        print(json.dumps({"tokenizer": str(out), "sha256": tokenizer_sha256(out)}))
        return 0
    if args.command in ("freqtrim", "freqtrim-check"):
        from . import freqtrim

        try:
            if args.command == "freqtrim-check":
                problems = freqtrim.check_directory(args.tokenizer_dir, args.teacher)
                print("\n".join(problems) if problems else "freqtrim checks pass")
                return 1 if problems else 0
            ref = freqtrim.TeacherRef(
                args.teacher_repo, args.teacher_revision, args.teacher_licence, args.teacher_sha256
            )
            out = freqtrim.build(
                load_corpus(args.corpus), args.out, args.teacher, ref, regex=args.regex, vocab_size=args.vocab_size
            )
        except (freqtrim.FreqtrimError, FileExistsError) as err:
            print(f"freqtrim: {err}", file=sys.stderr)
            return 1
        print(json.dumps({"tokenizer": str(out), "sha256": tokenizer_sha256(out)}))
        return 0
    from ..native import native_tokenizer

    texts = [*EDGE_CASES, *(doc.text for doc in sample(load_corpus(args.corpus), args.limit))]
    problems = compare(args.model_dir, texts, native_tokenizer(args.model_dir))
    print("\n".join(problems) if problems else f"consistent on {len(texts)} texts")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
