"""CPU tests run against this checkout: the repository's Python `surogate` (no native extension) and
fork/tlc/src. Both go first on sys.path, and an import from anywhere else fails the session.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "fork" / "tlc" / "src"

for path in (str(SRC), str(REPO)):  # REPO ends up first
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import surogate_tlc  # noqa: E402

import surogate  # noqa: E402

for module, expected in ((surogate, REPO / "surogate"), (surogate_tlc, SRC / "surogate_tlc")):
    if Path(module.__file__).resolve().parent != expected:
        raise RuntimeError(f"{module.__name__} was imported from {module.__file__}, not from {expected}")

SPECIAL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|code_start|>", "<|code_end|>")
CORPUS = (
    "def add(a, b):\n    return a + b\n",
    "for i in range(10):\n    total += i\n",
    "class Stack:\n    def push(self, item):\n        self.items.append(item)\n",
    "The little coder reads English and writes Python.\n",
)


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory) -> Path:
    """A small byte-level BPE tokenizer with TLC's special tokens: pad, bos and eos have ids 0, 1 and 2."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=320,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(CORPUS, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    out = tmp_path_factory.mktemp("tokenizer")
    tokenizer.save(str(out / "tokenizer.json"))
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "pad_token": "<|pad|>",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "additional_special_tokens": ["<|code_start|>", "<|code_end|>"],
        "model_max_length": 512,
        "chat_template": "{% for m in messages %}{{ bos_token }}{{ m['content'] }}{{ eos_token }}{% endfor %}",
    }
    (out / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2), encoding="utf-8")
    return out


@pytest.fixture
def model_dir(tmp_path, tokenizer_dir) -> Path:
    """An A0 model directory as write_model_dir() leaves it: everything but the weights."""
    from surogate_tlc import hf, modeldir

    return modeldir.write_model_dir(tmp_path / "model", hf.default_config(), tokenizer_dir, init_seed=0)


@pytest.fixture
def fresh_python(tmp_path):
    """Run `code` in a new isolated interpreter and return its stdout.

    The checkout is first on sys.path (sys.argv[1]); further arguments follow in sys.argv[2:]. Hugging Face
    caches, including the copies of auto_map code, go to an empty directory under tmp_path, offline.
    """
    env = {**os.environ, "HF_HOME": str(tmp_path / "hf-home"), "HF_HUB_OFFLINE": "1"}
    prelude = "import sys\nsys.path.insert(0, sys.argv[1])\n"

    def run(code: str, *args: object) -> str:
        command = [sys.executable, "-I", "-B", "-c", prelude + code, str(REPO), *map(str, args)]
        result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    return run
