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

CORPUS_SPEC = """\
seed = 7
val_fraction = 0.25

[[source]]
name = "code"
kind = "code"
files = ["code/*.py"]
url = "https://example.invalid/code"
license = "MIT"
downloaded = 2026-10-01
filters = ["fixture: generated functions"]

[[source]]
name = "text"
kind = "text"
files = ["text.jsonl"]
url = "https://example.invalid/text"
revision = "r1"
license = "CC-BY-4.0"
downloaded = "2026-10-02"
stage = "M1"
filters = []
"""
NAMES = ("add", "scale", "count", "merge", "split", "total", "clamp", "first", "last", "pairs", "window", "flatten")
SENTENCES = (
    "The little coder reads English and writes Python.",
    "Every function returns the total of the values it was given.",
    "A loop walks over the items, and the counter grows by one each time.",
    "Small models need clean data, careful tokens and honest tests.",
)


def write_corpus(root: Path) -> Path:
    """Local fixture documents and their corpus spec: Python files (one with CRLF, one with lone CR line ends),
    a JSONL of English documents (three in one group, one with a markdown code fence, one with special-token
    strings), and two documents the builder drops (not UTF-8; empty)."""
    (root / "code").mkdir(parents=True)
    for i, name in enumerate(NAMES):
        body = (
            f"def {name}_items(values, limit={i}):\n    total = 0\n    for index, value in enumerate(values):\n"
            f"        if value > limit and index % {i + 2} == 0:\n            total += value * {i + 1}\n"
            f"    return total  # {SENTENCES[i % len(SENTENCES)]}\n"
        )
        body = {1: body.replace("\n", "\r\n"), 2: body.replace("\n", "\r")}.get(i, body)
        (root / "code" / f"mod_{i:02d}.py").write_bytes(body.encode())
    (root / "code" / "bad.py").write_bytes(b"x = '\xff\xfe'\n")
    (root / "code" / "empty.py").write_bytes(b"")
    records = [{"text": " ".join(SENTENCES[i % 4 :] + SENTENCES[: i % 4]) * (1 + i % 3)} for i in range(10)]
    records += [{"text": f"{SENTENCES[i]} Part {i}.", "group": "book"} for i in range(3)]
    records.append({"text": "Intro:\n```python\nx = add_items([1, 2], 0)\n```\nThe value is x.\n"})
    records.append({"text": "A tokenizer marks the end with <|eos|> and pads with <|pad|>."})
    records.append({"text": ""})
    lines = [json.dumps(record) for record in records]
    (root / "text.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "corpus.toml").write_text(CORPUS_SPEC, encoding="utf-8")
    return root / "corpus.toml"


@pytest.fixture(scope="session")
def corpus_spec(tmp_path_factory) -> Path:
    return write_corpus(tmp_path_factory.mktemp("corpus"))


@pytest.fixture(scope="session")
def trained_tokenizer(tmp_path_factory, corpus_spec) -> Path:
    """The TLC tokenizer recipe (default pre-tokenizer) trained on the fixture corpus, with 300 ids."""
    from surogate_tlc.data import corpus, tokenizer

    return tokenizer.train_tokenizer(corpus.load_corpus(corpus_spec), tmp_path_factory.mktemp("tok"), vocab_size=300)


@pytest.fixture(scope="session")
def tokenizer_dir(tmp_path_factory, corpus_spec) -> Path:
    """Another tokenizer of the same recipe (GPT-2 pre-tokenizer): the same special ids, another tokenizer.json."""
    from surogate_tlc.data import corpus, tokenizer

    c = corpus.load_corpus(corpus_spec)
    return tokenizer.train_tokenizer(c, tmp_path_factory.mktemp("tok-gpt2"), pretokenizer="gpt2", vocab_size=300)


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


@pytest.fixture(scope="session")
def small_model(tmp_path_factory, trained_tokenizer) -> Path:
    """A two-layer A0 model directory with initial weights and the fixture corpus's tokenizer."""
    from surogate_tlc import hf, init, modeldir

    cfg = hf.row_config("A0", num_hidden_layers=2, tlc_layer_pattern="MM", chunk_size=16, vocab_size=320)
    path = modeldir.write_model_dir(tmp_path_factory.mktemp("small") / "model", cfg, trained_tokenizer, init_seed=0)
    init.init_model_dir(path)
    return path


@pytest.fixture(scope="session")
def shard_dirs(tmp_path_factory, corpus_spec, trained_tokenizer) -> dict[int, Path]:
    """Shard directories of the fixture corpus with rows of 64 and of 512 tokens, keyed by row length."""
    from surogate_tlc.data import corpus, shards

    root = tmp_path_factory.mktemp("shards")
    for seq_len in (64, 512):
        shards.build_shards(corpus.load_corpus(corpus_spec), trained_tokenizer, root / str(seq_len), seq_len=seq_len)
    return {seq_len: root / str(seq_len) for seq_len in (64, 512)}


@pytest.fixture
def run_yaml(tmp_path, small_model, shard_dirs):
    """write(seq_len=512, tlc=None, **sft) -> a run configuration for small_model; sft keys override the defaults
    (32 x 512 tokens per microbatch, 10 steps), `tlc` entries the tlc table."""
    import yaml

    def write(seq_len: int = 512, tlc: dict | None = None, **sft) -> Path:
        body = {
            "model": str(small_model),
            "output_dir": str(tmp_path / "runs"),
            "sequence_len": seq_len,
            "per_device_train_batch_size": 32,
            "max_steps": 10,
            **sft,
            "tlc": {"data": str(shard_dirs[seq_len]), **(tlc or {})},
        }
        path = tmp_path / "run.yaml"
        path.write_text(yaml.safe_dump(body), encoding="utf-8")
        return path

    return write
