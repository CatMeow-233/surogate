"""The byte-level BPE tokenizer (B-O11; IMPL-08, IMPL-03).

- Byte-level BPE with the 256-byte ByteLevel initial alphabet, so every UTF-8 string encodes without `<unk>`
  (v1.0 trained without it: 176 of the 256 byte symbols were missing and prompts decoded to `<unk>`).
- No normalizer by default (`tokenizer.normalization = "none"`): "，" and "," stay different tokens, because
  punctuation variants are a test condition. `"nfkc"` is the one alternative.
- The pre-tokenizer (`tokenizer.pretokenizer`, the tokenizer study's arms of B-O11): `"bytelevel"`, the default and
  v1.0's recipe, is ByteLevel with its GPT-2 regex; `"qwen2-digits"` splits on Qwen2's regex (every digit a piece
  of its own; Han runs stay together, as words) and then maps bytes without a regex; `"han-isolate"` first makes
  every Han character a piece of its own and then applies the ByteLevel GPT-2 regex to the rest, so no token holds
  more than one Han character.
- The special tokens of v1.0, in v1.0 order, hold ids 0..4: `<unk>`, `<eos>`, `<|zh|>`, `<|en|>`, `<|bridge|>`.
- It trains only on files of the manifest's `seed_train` split: every input file must be listed there with a
  matching sha256, and anything else is refused.
- User text is encoded with `encode_user_text`: special-token strings in it are plain text (tokenizers >= 0.20,
  `encode_special_tokens`), so a literal "<|en|>" never becomes the control tag.

`train_tokenizer` writes `tokenizer.json` and a sidecar `tokenizer.meta.json` (`bilingual-tokenizer/1`: inputs and
their sha256, manifest sha256, settings, tokenizers version). The tokenizer's identity everywhere (seed
fingerprint, shard headers) is `tokenizer_sha256`, the sha256 of the tokenizer.json bytes.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from ..canonical import atomic_write_bytes, atomic_write_json, sha256_bytes, sha256_file
from ..config import DEFAULTS, SPECIAL_TOKENS
from .docs import CONTROL_TAGS, iter_file_docs
from .manifest import Manifest, load_manifest

META_FORMAT = "bilingual-tokenizer/1"
NORMALIZATIONS = ("none", "nfkc")
# Split patterns by pre-tokenizer name; None is ByteLevel's own GPT-2 regex. Qwen2's pattern verbatim, as in
# Qwen2's tokenizer.json (and fork/tlc's "qwen2"); \p{Han} is the Unicode Script property, Extension B included.
PRETOKENIZERS = {
    "bytelevel": None,
    "qwen2-digits": (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    ),
    "han-isolate": r"\p{Han}",
}


class TokenizerError(ValueError):
    pass


def _pre_tokenizer(name: str):
    from tokenizers import Regex, pre_tokenizers

    if name not in PRETOKENIZERS:
        raise TokenizerError(f"pretokenizer must be one of {tuple(PRETOKENIZERS)}: {name!r}")
    if name == "bytelevel":
        return pre_tokenizers.ByteLevel(add_prefix_space=False)
    split = pre_tokenizers.Split(Regex(PRETOKENIZERS[name]), behavior="isolated")
    byte_level = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=name == "han-isolate")
    return pre_tokenizers.Sequence([split, byte_level])


def _new_tokenizer(normalization: str, pretokenizer: str = "bytelevel"):
    from tokenizers import Tokenizer, decoders, models, normalizers

    if normalization not in NORMALIZATIONS:
        raise TokenizerError(f"normalization must be one of {NORMALIZATIONS}: {normalization!r}")
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    if normalization == "nfkc":
        tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = _pre_tokenizer(pretokenizer)
    tok.decoder = decoders.ByteLevel()
    return tok


HELDOUT = ("calib", "dev", "test")


def _training_files(files: Sequence[Path], manifest: Manifest, split: str) -> list[tuple[Path, dict]]:
    if split in HELDOUT:
        raise TokenizerError(f"the tokenizer never trains on a held-out split ({split!r}); B-O11 names seed_train")
    if not files:
        raise TokenizerError(f"no input files; the tokenizer trains on files of the {split!r} split")
    out = []
    for path in files:
        entry = manifest.entry_for(Path(path))
        if entry is None:
            raise TokenizerError(f"{path}: not listed in the manifest; the tokenizer trains on {split!r} files only")
        if entry.get("split") != split:
            raise TokenizerError(f"{path}: in split {entry.get('split')!r} of the manifest, not {split!r}")
        out.append((manifest.verify_file(entry), entry))
    return out


def _texts(inputs: Sequence[tuple[Path, dict]], manifest: Manifest) -> Iterator[str]:
    for path, entry in inputs:
        source = manifest.sources[entry["source"]]
        for doc in iter_file_docs(path, source=source["name"], kind=source["kind"], rel=entry["path"]):
            for segment in doc.segments:
                if segment.text:
                    yield segment.text


def train_tokenizer(
    files: Sequence[Path],
    vocab: int,
    out: Path,
    manifest: Path | Manifest,
    split: str = "seed_train",
    *,
    normalization: str | None = None,
    pretokenizer: str | None = None,
    min_frequency: int = 2,
) -> Path:
    from tokenizers import __version__ as tokenizers_version
    from tokenizers import pre_tokenizers, trainers

    manifest = manifest if isinstance(manifest, Manifest) else load_manifest(manifest)
    normalization = normalization or DEFAULTS["tokenizer"]["normalization"]
    pretokenizer = pretokenizer or DEFAULTS["tokenizer"]["pretokenizer"]
    tok = _new_tokenizer(normalization, pretokenizer)
    inputs = _training_files(files, manifest, split)
    if vocab < len(SPECIAL_TOKENS) + 256:
        raise TokenizerError(f"vocab {vocab} cannot hold the {len(SPECIAL_TOKENS)} specials and 256 byte symbols")
    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab),
        min_frequency=int(min_frequency),
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(_texts(inputs, manifest), trainer=trainer)
    check_special_ids(tok)
    data = tok.to_str().encode("utf-8")
    out = Path(out)
    atomic_write_bytes(out, data)
    atomic_write_json(
        meta_path(out),
        {
            "format": META_FORMAT,
            "tokenizer_sha256": sha256_bytes(data),
            "vocab_size": tok.get_vocab_size(),
            "vocab_requested": int(vocab),
            "min_frequency": int(min_frequency),
            "normalization": normalization,
            "pretokenizer": pretokenizer,
            "pretokenizer_pattern": PRETOKENIZERS[pretokenizer],
            "initial_alphabet": "bytelevel",
            "special_tokens": list(SPECIAL_TOKENS),
            "split": split,
            "manifest_sha256": manifest.sha256,
            "inputs": [{"path": e["path"], "sha256": e["sha256"]} for _, e in inputs],
            "tokenizers_version": tokenizers_version,
        },
    )
    return out


def meta_path(tokenizer_json: Path) -> Path:
    return Path(tokenizer_json).with_name(Path(tokenizer_json).stem + ".meta.json")


def tokenizer_sha256(path: Path) -> str:
    return sha256_file(Path(path))


def check_special_ids(tok) -> None:
    for expected, text in enumerate(SPECIAL_TOKENS):
        got = tok.token_to_id(text)
        if got != expected:
            raise TokenizerError(f"special token {text!r} has id {got}, expected {expected} (v1.0 order, B-O11)")


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    check_special_ids(tok)
    return tok


def special_ids(tok) -> dict[str, int]:
    return {text: int(tok.token_to_id(text)) for text in SPECIAL_TOKENS}


def encode_user_text(tok, text: str) -> list[int]:
    """Token ids of user text, with every special-token string encoded as plain text (IMPL-03)."""
    previous = tok.encode_special_tokens
    tok.encode_special_tokens = True
    try:
        ids = tok.encode(text, add_special_tokens=False).ids
    finally:
        tok.encode_special_tokens = previous
    specials = {tok.token_to_id(s) for s in SPECIAL_TOKENS} - {None}
    if specials & set(ids):
        raise TokenizerError("escaped user text still contains special ids; the tokenizer does not escape them")
    return ids


def decode(tok, ids: Sequence[int]) -> str:
    return tok.decode(list(ids), skip_special_tokens=False)


def round_trips(tok, text: str) -> bool:
    return decode(tok, encode_user_text(tok, text)) == text


# Code points for round-trip probes: ASCII, tabs and newlines, Latin-1, CJK (BMP), CJK Extension B (4-byte UTF-8),
# emoji, full-width punctuation and combining marks. Surrogates are not scalar values and are never drawn.
PROBE_RANGES = (
    (0x20, 0x7E), (0x09, 0x0A), (0xA0, 0xFF), (0x4E00, 0x9FFF), (0x20000, 0x2A6DF), (0x1F300, 0x1F64F),
    (0xFF01, 0xFF5E), (0x3000, 0x303F), (0x0300, 0x036F),
)  # fmt: skip


def random_strings(n: int, seed: int, max_len: int = 24) -> list[str]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        chars = []
        for _ in range(rng.randint(1, max_len)):
            lo, hi = PROBE_RANGES[rng.randrange(len(PROBE_RANGES))]
            chars.append(chr(rng.randint(lo, hi)))
        out.append("".join(chars))
    return out


def check_tokenizer(tok, n: int = 500, seed: int = 0) -> dict:
    """The I4-D1 checks on a trained tokenizer: special ids, `<unk>`-free encoding and exact round trips of random
    UTF-8 strings and every tag string, both as text (escaped) and as the control token."""
    check_special_ids(tok)
    unk = tok.token_to_id("<unk>")
    failures, unks = [], 0
    for text in random_strings(n, seed) + list(SPECIAL_TOKENS):
        unks += encode_user_text(tok, text).count(unk)
        if not round_trips(tok, text):
            failures.append(text)
    for text in CONTROL_TAGS:
        ids = tok.encode(text, add_special_tokens=False).ids
        if ids != [tok.token_to_id(text)] or decode(tok, ids) != text:
            failures.append(text)
    return {"n": n, "unk": unks, "failures": failures, "ok": unks == 0 and not failures}


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual tokenizer", description="train or check")
    sub = ap.add_subparsers(dest="action", required=True)
    t = sub.add_parser("train", help="train on the manifest's seed_train files")
    t.add_argument("--manifest", required=True, type=Path, help="a derived manifest whose files carry their split")
    t.add_argument("--split", default="seed_train")
    t.add_argument("--vocab", type=int, default=DEFAULTS["tokenizer"]["vocab_size"])
    t.add_argument("--pretokenizer", choices=tuple(PRETOKENIZERS), default=DEFAULTS["tokenizer"]["pretokenizer"])
    t.add_argument("--files", nargs="*", type=Path, help="default: every file of the split")
    t.add_argument("--out", type=Path)
    t.add_argument("--dry-run", action="store_true", help="check the inputs against the manifest; train nothing")
    c = sub.add_parser("check", help="special ids, <unk>-free encoding and round trips")
    c.add_argument("tokenizer", type=Path)
    c.add_argument("--samples", type=int, default=500)
    c.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(list(argv))
    try:
        if args.action == "check":
            report = check_tokenizer(load_tokenizer(args.tokenizer), args.samples)
            print(
                f"tokenizer: sha256 {tokenizer_sha256(args.tokenizer)}; {report['n']} random strings, "
                f"{report['unk']} <unk>, {len(report['failures'])} round-trip failures"
            )
            return 0 if report["ok"] else 1
        manifest = load_manifest(args.manifest)
        files = args.files or [manifest.root / e["path"] for e in manifest.files(split=args.split)]
        if args.dry_run:
            inputs = _training_files(files, manifest, args.split)
            print(f"tokenizer: {len(inputs)} input files of split {args.split!r} verified (dry run)")
            return 0
        if args.out is None:
            ap.error("--out is required unless --dry-run")
        out = train_tokenizer(files, args.vocab, args.out, manifest, args.split, pretokenizer=args.pretokenizer)
    except (TokenizerError, ValueError) as err:
        print(f"tokenizer: {err}", file=sys.stderr)
        return 1
    print(f"tokenizer: wrote {out} (sha256 {tokenizer_sha256(out)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
