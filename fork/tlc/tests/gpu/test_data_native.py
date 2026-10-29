"""G0 data checks against the native extension: the tokenizer harness with the native tokenizer, and what the native
DataLoader serves from TLC shards. Neither needs the GPU itself, only the native build; at G0 scale the tokenizer
check is the `check` command (README: G0).
"""

import json
from collections import Counter

import numpy as np
import pytest
from surogate_tlc import hf, modeldir
from surogate_tlc.data import corpus, shards
from surogate_tlc.data import tokenizer as tok

SPEC = """\
seed = 3
val_fraction = 0.2
[[source]]
name = "code"
kind = "code"
files = ["code/*.py"]
url = "https://example.invalid/code"
license = "MIT"
downloaded = 2026-10-01
filters = []
[[source]]
name = "text"
kind = "text"
files = ["text.jsonl"]
url = "https://example.invalid/text"
license = "CC-BY-4.0"
downloaded = 2026-10-01
filters = []
"""


@pytest.fixture(scope="module")
def spec(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    (root / "code").mkdir()
    for i, name in enumerate(("add", "scale", "count", "merge", "split", "clamp", "window", "flatten")):
        code = f"def {name}_items(values, limit={i}):\n    total = 0\n    for v in values:\n        total += v * {i}\n"
        (root / "code" / f"m{i}.py").write_text(code + f"    return total  # {name} é中 \U0001f642\n")
    texts = [f"The little coder writes item {i}.\n```\nx = {i}\n```\nIt's done, isn't it?\n" for i in range(12)]
    (root / "text.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    (root / "corpus.toml").write_text(SPEC, encoding="utf-8")
    return root / "corpus.toml"


@pytest.mark.parametrize("pretokenizer", list(tok.PRETOKENIZERS))
def test_native_tokenizer_agrees(tmp_path, spec, pretokenizer):
    from surogate._surogate import Tokenizer

    c = corpus.load_corpus(spec)
    trained = tok.train_tokenizer(c, tmp_path / "tok", pretokenizer=pretokenizer, vocab_size=300)
    model = modeldir.write_model_dir(tmp_path / "model", hf.default_config(), trained, init_seed=0)
    docs = [d.text for d in corpus.documents(c)]
    texts = [*tok.EDGE_CASES, *docs, *(t.replace("\n", "\r\n") for t in docs)]
    assert tok.compare(model, texts, Tokenizer.from_pretrained(str(model))) == []


def test_native_data_loader_serves_the_rows(tmp_path, spec):
    from surogate._surogate import DataLoader

    c = corpus.load_corpus(spec)
    trained = tok.train_tokenizer(c, tmp_path / "tok", vocab_size=300)
    seq_len = 32
    manifest = shards.build_shards(c, trained, tmp_path / "shards", seq_len=seq_len)
    for entry in manifest["shards"]:
        path = tmp_path / "shards" / entry["file"]
        expected = Counter(
            (tuple(i), tuple(t), tuple(p)) for i, t, p in zip(*shards.served_rows(shards.read_shard(path), seq_len))
        )
        loader = DataLoader([str(path)], seq_len, seed=1)
        inputs, targets, positions = (np.zeros((1, seq_len), dtype=np.int32) for _ in range(3))
        served = Counter()
        while loader.has_next():
            loader.load_batch(inputs, targets, positions)
            served[tuple(inputs[0]), tuple(targets[0]), tuple(positions[0])] += 1
        assert served == expected
        assert sum(n * sum(t != -100 for t in row[1]) for row, n in served.items()) == entry["valid_targets"]
