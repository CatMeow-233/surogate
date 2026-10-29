"""Token shards (surogate_tlc.data.shards, docs/data.md: 数据 shard): header fields, positions, mask bits, windows,
padding, max id < vocab_size, the ValidTokenCount accounting and the manifest.

Header fields are read with upstream's parser (surogate.distill.sidecar), and the writer is compared byte for byte
with upstream's TokenizedDataFileWriter, loaded from its source because its module needs the native extension.
What the native DataLoader serves from these files is checked on the GPU host (tests/gpu/test_data_native.py).
"""

import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from surogate_tlc import files
from surogate_tlc.data import corpus, shards
from surogate_tlc.data import tokenizer as tok

from surogate.distill.sidecar import read_token_shard_header

REPO = Path(__file__).resolve().parents[4]
SEQ_LEN = 32
SPECIAL = {token: i for i, token in enumerate(tok.SPECIAL_TOKENS)}
BOS, EOS, START, END = (SPECIAL[t] for t in (tok.BOS, tok.EOS, tok.CODE_START, tok.CODE_END))


@pytest.fixture(scope="module")
def built(tmp_path_factory, corpus_spec, trained_tokenizer):
    out = tmp_path_factory.mktemp("shards") / "out"
    c = corpus.load_corpus(corpus_spec)
    return out, shards.build_shards(c, trained_tokenizer, out, seq_len=SEQ_LEN, train_files=2)


def _framed_by_split(corpus_spec, trained_tokenizer) -> dict[str, list[list[int]]]:
    c = corpus.load_corpus(corpus_spec)
    encoded = shards.encode(tok.load(trained_tokenizer, ordinary=True), corpus.documents(c), SPECIAL)
    out: dict[str, list[list[int]]] = {"train": [], "val": []}
    for doc, ids in encoded:
        out[corpus.split_of(c, doc)].append(ids)
    return out


def test_header_fields(built):
    out, manifest = built
    assert [s["file"] for s in manifest["shards"]] == ["train-00000.bin", "train-00001.bin", "val-00000.bin"]
    for s in manifest["shards"]:
        path = out / s["file"]
        header = read_token_shard_header(str(path))
        assert (header.version, header.bytes_per_token, header.vocab_size) == (3, 4, 300)
        assert header.has_masks and header.non_overlapping
        assert header.n_tokens == s["tokens"] == s["rows"] * SEQ_LEN
        assert path.stat().st_size == 1024 + 8 * header.n_tokens + header.n_tokens // 8


def test_rows_positions_masks_and_padding(built):
    out, manifest = built
    for s in manifest["shards"]:
        shard = shards.read_shard(out / s["file"])
        tokens = np.asarray(shard.tokens).reshape(-1, SEQ_LEN)
        masks = shard.masks.reshape(-1, SEQ_LEN)
        assert (np.asarray(shard.positions).reshape(-1, SEQ_LEN) == np.arange(SEQ_LEN)).all()
        assert 0 <= tokens.min() and tokens.max() < shard.vocab_size == 300
        lengths = masks.sum(axis=1) + 1  # mask bits are set for every position but a row's last real one
        assert (masks == (np.arange(SEQ_LEN) < (lengths - 1)[:, None])).all()
        real = np.arange(SEQ_LEN) < lengths[:, None]
        assert (tokens[~real] == SPECIAL[tok.PAD]).all() and (tokens[real] != SPECIAL[tok.PAD]).all()
        assert int(lengths.sum()) == s["real_tokens"]


def test_every_token_but_a_documents_first_is_a_target_once(built, corpus_spec, trained_tokenizer):
    out, manifest = built
    framed = _framed_by_split(corpus_spec, trained_tokenizer)
    for split in ("train", "val"):
        served = Counter()
        for s in (s for s in manifest["shards"] if s["split"] == split):
            _, targets, _ = shards.served_rows(shards.read_shard(out / s["file"]), SEQ_LEN)
            valid = targets[targets != -100]
            assert len(valid) == s["valid_targets"] == s["real_tokens"] - s["rows"]
            served.update(valid.tolist())
        assert served == Counter(i for ids in framed[split] for i in ids[1:])
        assert manifest["splits"][split]["valid_targets"] == sum(served.values())
        assert BOS not in served


@pytest.mark.parametrize("length", [1, 2, 31, 32, 33, 62, 63, 64, 100])
def test_windows(length):
    ids = list(range(1000, 1000 + length))
    rows = shards.windows(ids, SEQ_LEN)
    assert all(1 <= len(row) <= SEQ_LEN for row in rows)
    assert [row[0] for row in rows[1:]] == [row[-1] for row in rows[:-1]]  # consecutive rows share one token
    assert [t for row in rows for t in row[1:]] == ids[1:]  # every token but the first is a target, once


def test_segments():
    def segments(text, kind="text"):
        return shards.segments(text, kind, SPECIAL)

    assert segments("x = 1\n```\n", kind="code") == [BOS, START, "x = 1\n```\n", END, EOS]
    text = "Intro:\n```python\nx = 1\n```\nMore, with `inline` and ``` mid-line.\n"
    assert segments(text) == [BOS, "Intro:\n", START, "x = 1\n", END, "More, with `inline` and ``` mid-line.\n", EOS]
    # Inside a block only a bare ``` closes it; an open block is closed at the end.
    assert segments("  ```\na\n```js\nb") == [BOS, START, "a\n```js\nb", END, EOS]
    # Lines end at \n only (the text is normalised): U+2028 inside a line does not end it.
    assert segments("s = ' ```'\n") == [BOS, "s = ' ```'\n", EOS]


def test_document_text_never_yields_special_ids(trained_tokenizer):
    doc = corpus.Document(corpus.Source("t", "text", (), {}), "t/x", "t/x", "stop <|eos|> and <|code_end|>")
    [(_, ids)] = shards.encode(tok.load(trained_tokenizer, ordinary=True), [doc], SPECIAL)
    assert ids[0] == BOS and ids[-1] == EOS and not set(ids[1:-1]) & set(SPECIAL.values())


def test_manifest(built, corpus_spec, trained_tokenizer):
    out, manifest = built
    assert manifest == json.loads((out / shards.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["tlc_shard_version"] == shards.SHARD_VERSION
    assert manifest["format"] == {"name": "BIN.TOK", "version": 3, "seq_len": SEQ_LEN, "non_overlapping": True}
    assert manifest["tokenizer"] == {
        "sha256": tok.tokenizer_sha256(trained_tokenizer),
        "vocab_size": 300,
        "special_ids": SPECIAL,
    }
    assert manifest["corpus"] == {
        "spec_sha256": files.file_sha256(corpus_spec),
        "seed": 7,
        "val_fraction": 0.25,
        "train_files": 2,
        "decontam": None,
    }
    for s in manifest["shards"]:
        assert s["sha256"] == files.file_sha256(out / s["file"])
        # Rows are written document by document, source by source: one contiguous range per source.
        assert sum(part["rows"] for part in s["sources"]) == s["rows"]
        assert [part["first_row"] for part in s["sources"]] == [0, *np.cumsum([p["rows"] for p in s["sources"]][:-1])]
    sources = manifest["sources"]
    assert [s["name"] for s in sources] == ["code", "text"]
    assert all(s["url"] and s["license"] and s["downloaded"] and s["files"] for s in sources)
    assert all(f["sha256"] and f["bytes"] >= 0 for s in sources for f in s["files"])
    assert {s["name"]: s["dropped"] for s in sources} == {"code": {"empty": 1, "not_utf8": 1}, "text": {"empty": 1}}
    keys = ("rows", "tokens", "real_tokens", "valid_targets")
    for split in ("train", "val"):
        totals = manifest["splits"][split]
        mine = [s for s in manifest["shards"] if s["split"] == split]
        assert {k: totals[k] for k in keys} == {k: sum(s[k] for s in mine) for k in keys}
        for key in ("rows", "real_tokens"):
            assert totals[key] == sum(s[key][split] for s in sources)
        assert totals["pad_fraction"] == round(1 - totals["real_tokens"] / totals["tokens"], 6)
    assert sum(s["documents"]["train"] for s in sources) == 22 and sum(s["documents"]["val"] for s in sources) == 5
    for source in sources:
        for split in ("train", "val"):
            ranges = [p for s in manifest["shards"] if s["split"] == split for p in s["sources"]]
            assert sum(p["rows"] for p in ranges if p["name"] == source["name"]) == source["rows"][split]
    assert sum(s["train_token_share"] for s in sources) == pytest.approx(1, abs=1e-5)
    assert len(shards.manifest_sha256(out)) == 64


def test_rebuild_is_identical_and_refuses_a_used_directory(tmp_path, built, corpus_spec, trained_tokenizer):
    out, _ = built
    again = tmp_path / "again"
    shards.build_shards(corpus.load_corpus(corpus_spec), trained_tokenizer, again, seq_len=SEQ_LEN, train_files=2)
    assert sorted(p.name for p in again.iterdir()) == sorted(p.name for p in out.iterdir())  # no temporary files
    for path in out.iterdir():
        assert (again / path.name).read_bytes() == path.read_bytes()
    with pytest.raises(FileExistsError):
        shards.build_shards(corpus.load_corpus(corpus_spec), trained_tokenizer, again, seq_len=SEQ_LEN)


def test_writer_matches_upstream_bytes(tmp_path):
    source = (REPO / "surogate" / "train" / "tokenize.py").read_text(encoding="utf-8")
    [cls] = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "TokenizedDataFileWriter"]
    namespace = {"np": np}
    exec(compile(ast.Module([cls], []), "tokenize.py", "exec"), namespace)
    rng = np.random.default_rng(0)
    rows = [rng.integers(1, 300, size=n).tolist() for n in (1, 5, SEQ_LEN, 17, SEQ_LEN - 1)]
    ours = shards.ShardWriter(tmp_path / "ours.bin", seq_len=SEQ_LEN, vocab_size=300, pad_id=0)
    for row in rows:
        ours.add_row(row)
    ours.close()
    with namespace["TokenizedDataFileWriter"](
        str(tmp_path / "theirs.bin"), 300, masking=True, non_overlapping=True
    ) as w:
        for row in rows:
            tokens = np.zeros(SEQ_LEN, dtype=np.int32)
            tokens[: len(row)] = row
            w.add_document(tokens, np.arange(SEQ_LEN), mask=np.arange(SEQ_LEN) < len(row) - 1)
    assert (tmp_path / "ours.bin").read_bytes() == (tmp_path / "theirs.bin").read_bytes()


def test_writer_rejects_bad_rows(tmp_path):
    with pytest.raises(ValueError, match="multiple of 8"):
        shards.ShardWriter(tmp_path / "a.bin", seq_len=30, vocab_size=300, pad_id=0)
    writer = shards.ShardWriter(tmp_path / "b.bin", seq_len=SEQ_LEN, vocab_size=300, pad_id=0)
    with pytest.raises(ValueError, match="token ids"):
        writer.add_row([1, 300])
    with pytest.raises(ValueError, match="a row holds"):
        writer.add_row(list(range(1, SEQ_LEN + 2)))
    with pytest.raises(ValueError, match="a row holds"):
        writer.add_row([])
