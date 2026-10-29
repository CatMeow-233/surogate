"""TLC's tokenizer, a joint English + Python byte-level BPE, and its consistency check against the native one.

- Pre-tokenization: one of PRETOKENIZERS, then the byte-level mapping. The choice is open (docs/data.md: Tokenizer):
  "gpt2" and "python" are regex candidates of the experiment notebook (GPT-2's pattern, and an identifier-aware
  pattern after its pyaware2 with Unicode letter classes); "qwen2", the default, is Qwen2's split pattern, GPT-2's
  family with the Unicode classes \\p{L} and \\p{N} and every digit a piece of its own, the regex the teacher-led
  training memo pairs with its Qwen-subset vocabulary. For a model_type it does not know, such as "tlc", the native
  tokenizer (csrc/src/tokenizer) reads the pattern from tokenizer.json, or uses GPT-2's when there is none. It has
  hand-written splitters for the qwen2 and GPT-2 patterns and runs any other one, "python" included, through its
  std::regex fallback. There is no normaliser, so decode(encode(text)) == text.
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
# together, after the notebook's pyaware2, with Unicode letter and digit classes.
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


def _pre_tokenizer(name: str) -> pre_tokenizers.PreTokenizer:
    pattern = PRETOKENIZERS[name]
    if pattern is None:
        return pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    split = pre_tokenizers.Split(Regex(pattern), behavior="isolated")
    return pre_tokenizers.Sequence([split, pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])


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
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = _pre_tokenizer(pretokenizer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    dropped: Counter = Counter()
    count = Counter()

    def texts():
        for doc in documents(corpus, dropped):
            if split_of(corpus, doc) == "train":
                count[doc.source.name] += 1
                yield doc.text

    tokenizer.train_from_iterator(texts(), trainer)
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError(f"the corpus yields {tokenizer.get_vocab_size()} tokens, not {vocab_size}: it is too small")
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / TOKENIZER_FILE))
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
    (out / CONFIG_FILE).write_text(dump_json(config), encoding="utf-8")
    sources = provenance(corpus)
    for s in sources:
        s["documents"] = count[s["name"]]
        s["dropped"] = {reason: n for (name, reason), n in sorted(dropped.items()) if name == s["name"]}
    manifest = {
        "tlc_tokenizer_version": TOKENIZER_VERSION,
        "tokenizer_sha256": tokenizer_sha256(out),
        "vocab_size": vocab_size,
        "min_frequency": min_frequency,
        "pretokenizer": pretokenizer,
        "split_pattern": PRETOKENIZERS[pretokenizer],
        "special_tokens": list(SPECIAL_TOKENS),
        "corpus": {
            "spec_sha256": file_sha256(corpus.path),
            "seed": corpus.seed,
            "val_fraction": corpus.val_fraction,
            "trained_on": "train split",
            "decontam": decontam_record(corpus),
            "rules": list(RULES),
        },
        "sources": sources,
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
    args = parser.parse_args(argv)
    if args.command == "train":
        out = train_tokenizer(
            load_corpus(args.corpus), args.out, pretokenizer=args.pretokenizer, vocab_size=args.vocab_size
        )
        print(json.dumps({"tokenizer": str(out), "sha256": tokenizer_sha256(out)}))
        return 0
    from ..native import native_tokenizer

    texts = [*EDGE_CASES, *(doc.text for doc in sample(load_corpus(args.corpus), args.limit))]
    problems = compare(args.model_dir, texts, native_tokenizer(args.model_dir))
    print("\n".join(problems) if problems else f"consistent on {len(texts)} texts")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
