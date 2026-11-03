"""CPU tests run against this checkout: fork/tokstudy/src goes first on sys.path, then fork/tlc/src and
fork/bilingual/src (the projects whose public pure-data APIs the study reads), and a package imported from anywhere
else errors the session before any test runs. Nothing else is put on sys.path; the suite does not import the native
`surogate` package.

The suite is a CPU suite on every host: `pytest_configure` hides CUDA (CUDA_VISIBLE_DEVICES="") before any test
imports torch, for this process and every subprocess it starts. GPU checks live in tests/gpu and run by hand in
fork/.venv-tokstudy.

TLC model directories (gate TW-1) are written by TLC's own code in a separate isolated interpreter whose sys.path
starts with the checkout and fork/tlc/src (`tlc_model_dir`): TLC's initialisation compiles the row's IR with the
repository's Python `surogate` package (no native extension), which this process never imports.
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[3]
REPO = FORK.parent
SRC = FORK / "tokstudy" / "src"
CHECKOUT = {
    "surogate_tokstudy": SRC,
    "surogate_tlc": FORK / "tlc" / "src",
    "surogate_bilingual": FORK / "bilingual" / "src",
}

for path in reversed(CHECKOUT.values()):  # SRC ends up first
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

for name, root in CHECKOUT.items():
    module = importlib.import_module(name)
    if Path(module.__file__).resolve().parent != root / name:
        raise RuntimeError(f"{name} was imported from {module.__file__}, not from {root}")


def pytest_configure(config):
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise pytest.UsageError("CUDA was initialised before the CPU suite started; run tests/cpu in its own session")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# --- SYNTHETIC tokenizer fixtures ------------------------------------------------------------------------------------
#
# Written by the tests in the byte-level BPE format of tokenizers; never a real project, teacher or reference file.

# Edge cases every codec must reproduce byte for byte: CRLF, lone CR, tab, emoji, CJK Extension B (4-byte UTF-8),
# U+2028 inside a string, an f-string, special-token strings inside text.
EDGE_TEXTS = (
    "",
    "def f(x):\r\n    return x\r\n",
    "a = 1\rb = 2\r",
    "if x:\n\tpass  # tab\n",
    "print('héllo 🙂')\n",
    "变量 = '𠀀𠀁𪚥'  # 扩展B\n",
    "x = '\u2028'\n",
    "s = f'{x!r:>10} {{y}}'\n",
    "text with <|eos|> and <|pad|> and <|zh|> inside",
    "  \n\n   trailing   ",
)


def write_bpe(path, merges=(), *, specials=(), pre="whole", normalizer=None, train_on=None, vocab_size=400):
    """A SYNTHETIC byte-level BPE tokenizer.json at `path`; returns (path, sha256).

    pre "whole": no split (the whole text is one word, so `merges` act anywhere); "gpt2": ByteLevel's own regex.
    With `train_on`, the vocabulary is learned from those texts instead of given by `merges`."""
    from surogate_tokstudy.canonical import sha256_file
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    if train_on is not None:
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=pre == "gpt2")
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=list(specials), show_progress=False,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())  # fmt: skip
        tok.train_from_iterator(list(train_on), trainer)
    else:
        vocab = {s: i for i, s in enumerate(specials)}
        for c in sorted(pre_tokenizers.ByteLevel.alphabet()):
            vocab[c] = len(vocab)
        for a, b in merges:
            vocab.setdefault(a + b, len(vocab))
        tok = Tokenizer(models.BPE(vocab=vocab, merges=list(merges)))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=pre == "gpt2")
        tok.add_special_tokens(list(specials))
    tok.decoder = decoders.ByteLevel()
    if normalizer == "nfc":
        tok.normalizer = normalizers.NFC()
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return path, sha256_file(path)


@pytest.fixture(scope="session")
def bpe_writer():
    return write_bpe


@pytest.fixture(scope="session")
def edge_texts():
    return EDGE_TEXTS


# --- TLC model directories, written by TLC's code in an isolated interpreter ------------------------------------------

# argv: REPO, OUT, overrides (JSON, may name "row"), init seed or "config". A SYNTHETIC zero-merge tokenizer (TLC's
# special tokens, then the 256 byte tokens) stands in for a trained one; "config" prints the row's config.json only.
TLC_MODEL_DIR_SCRIPT = """
import json, sys
from pathlib import Path
sys.path[:0] = [sys.argv[1], sys.argv[1] + "/fork/tlc/src"]
from surogate_tlc import hf
overrides = json.loads(sys.argv[3])
cfg = hf.row_config(overrides.pop("row", "A0"), **overrides)
if sys.argv[4] == "config":
    print(json.dumps(cfg))
    sys.exit(0)
from tokenizers import Tokenizer, models, pre_tokenizers
from surogate_tlc import init, modeldir
from surogate_tlc.data import tokenizer as tok
out = Path(sys.argv[2])
vocab = {s: i for i, s in enumerate(tok.SPECIAL_TOKENS)}
for c in sorted(pre_tokenizers.ByteLevel.alphabet()):
    vocab[c] = len(vocab)
t = tok.finish_tokenizer(Tokenizer(models.BPE(vocab=vocab, merges=[])))
t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
t.add_special_tokens(list(tok.SPECIAL_TOKENS))
(out / "tok").mkdir(parents=True)
t.save(str(out / "tok" / "tokenizer.json"))
tok.write_config(out / "tok")
init.init_model_dir(modeldir.write_model_dir(out / "model", cfg, out / "tok", init_seed=int(sys.argv[4])))
"""


def run_tlc(out: Path, overrides: dict, seed: int | str) -> str:
    command = [sys.executable, "-I", "-B", "-c", TLC_MODEL_DIR_SCRIPT, str(REPO), str(out), json.dumps(overrides),
               str(seed)]  # fmt: skip
    result = subprocess.run(command, cwd=out.parent, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="session")
def tlc_model_dir(tmp_path_factory):
    """write(overrides, seed) -> a TLC model directory (config.json, model.safetensors from TLC's init), written by
    TLC's code; overrides are hf.row_config arguments plus "row" (default "A0")."""

    def write(overrides: dict, seed: int) -> Path:
        out = tmp_path_factory.mktemp("tlc")
        run_tlc(out, dict(overrides), seed)
        return out / "model"

    return write


@pytest.fixture(scope="session")
def tlc_default_config(tmp_path_factory) -> dict:
    """TLC's config.json for row A0 at its default geometry (surogate_tlc.hf.row_config("A0"))."""
    return json.loads(run_tlc(tmp_path_factory.mktemp("tlc-config"), {}, "config"))
