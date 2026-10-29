"""The eval command's loss (docs/training.md: 评测), with a stand-in for the native scorer: every served row of the
split is scored once, the loss averages over valid targets (also per source and per kind), and the unigram baseline
counts the train split's targets."""

import math

import numpy as np
import pytest
from surogate_tlc.data import shards
from surogate_tlc.data.shards import manifest, shard_files
from surogate_tlc.eval import loss


def _rows(data, split):
    parts = [shards.served_rows(shards.read_shard(p), 64) for p in shard_files(data, split)]
    return [np.concatenate(arrays) for arrays in zip(*parts, strict=True)]


def test_unigram_loss_counts_the_train_targets(shard_dirs):
    data = shard_dirs[64]
    counts = np.ones(320)
    for target in _rows(data, "train")[1].ravel():
        if target != -100:
            counts[target] += 1
    targets = _rows(data, "val")[1]
    expected = -np.mean([math.log(counts[t] / counts.sum()) for t in targets.ravel() if t != -100])
    assert loss.unigram_loss(data, targets, 320) == pytest.approx(expected)


@pytest.mark.parametrize("batch, max_rows", [(3, None), (4, 2)])
def test_evaluate_scores_each_served_row_once(small_model, shard_dirs, batch, max_rows):
    data = shard_dirs[64]
    seen = []

    def uniform(inputs, targets, positions):
        assert inputs.shape == targets.shape == positions.shape == (batch, 64) and inputs.dtype == np.int32
        seen.append((inputs.copy(), targets.copy()))
        return np.where(targets == -100, 0.0, -math.log(320)).astype(np.float32)

    result = loss.evaluate(small_model, data, batch=batch, max_rows=max_rows, scorer=uniform)
    inputs, targets, _ = (a[:max_rows] for a in _rows(data, "val"))
    scored = np.concatenate([i for i, _ in seen])[: len(inputs)]
    assert np.array_equal(scored, inputs) and result["rows"] == len(inputs)
    assert result["valid_targets"] == int((targets != -100).sum())
    if max_rows is None:
        assert result["valid_targets"] == manifest(data)["splits"]["val"]["valid_targets"]
    # The filled rows of the last batch carry no targets.
    assert (np.concatenate([t for _, t in seen])[len(inputs) :] == -100).all()
    assert result["loss"] == pytest.approx(math.log(320)) and result["unigram_loss"] < result["loss"]
    assert result["below_unigram"] is False
    assert result["data_manifest_sha256"] == shards.manifest_sha256(data)
    # Per source: the row ranges of the manifest; a uniform scorer gives every source the same loss.
    by_source = result["by_source"]
    assert sum(v["valid_targets"] for v in by_source.values()) == result["valid_targets"]
    assert all(v["loss"] == pytest.approx(math.log(320)) for v in by_source.values())
    assert set(result["by_kind"]) <= {"code", "text"}
    if max_rows is None:
        val = {s["name"]: s["real_tokens"]["val"] - s["rows"]["val"] for s in manifest(data)["sources"]}
        assert {name: v["valid_targets"] for name, v in by_source.items()} == {n: c for n, c in val.items() if c}


def test_row_sources_follow_the_documents(shard_dirs, corpus_spec, trained_tokenizer):
    """The manifest's per-shard source ranges give each served row the source of the document it holds."""
    from surogate_tlc.data import corpus, tokenizer

    data, c = shard_dirs[64], corpus.load_corpus(corpus_spec)
    tok = tokenizer.load(trained_tokenizer, ordinary=True)
    special = tokenizer.special_ids(tok)
    expected = [
        doc.source.name
        for doc, ids in shards.encode(tok, corpus.documents(c), special)
        if corpus.split_of(c, doc) == "val"
        for _ in shards.windows(ids, 64)
    ]
    assert loss._row_sources(manifest(data), shard_files(data, "val")) == expected


def test_evaluate_refuses_another_tokenizer(tmp_path, tokenizer_dir, shard_dirs):
    from surogate_tlc import hf, init, modeldir

    cfg = hf.row_config("A0", num_hidden_layers=2, tlc_layer_pattern="MM", chunk_size=16, vocab_size=320)
    other = modeldir.write_model_dir(tmp_path / "other", cfg, tokenizer_dir, init_seed=0)
    init.init_model_dir(other)
    with pytest.raises(ValueError, match="another tokenizer"):
        loss.evaluate(other, shard_dirs[64], scorer=lambda *a: None)


def test_greedy_stops_at_a_stop_id_or_the_row_end():
    from surogate_tlc.eval.samples import greedy

    def next_position(inputs, last):  # the next token is the next position's index
        logits = np.zeros((inputs.shape[0], 16), np.float32)
        logits[np.arange(len(last)), (last + 1) % 16] = 1
        return logits

    kwargs = {"batch": 1, "pad_id": 0, "stop_ids": {3}, "max_new": 5}
    assert greedy([[1, 2], [1]], next_position, seq_len=8, **kwargs) == [[2, 3], [1, 2, 3]]
    assert greedy([[1, 2], [1]], next_position, seq_len=3, **kwargs) == [[2], [1, 2]]
    assert greedy([[1, 2]], next_position, seq_len=8, **{**kwargs, "max_new": 1}) == [[2]]


@pytest.mark.parametrize("stop", ["<|eos|>", "<|code_end|>"])
def test_samples_start_like_shard_documents_and_stop_at_either_end(small_model, shard_dirs, stop):
    """A prompt is the start of a document as the shards frame it (special-token strings are text), and both
    <|eos|> and <|code_end|> end a sample."""
    from surogate_tlc.data import corpus
    from surogate_tlc.data import tokenizer as tok
    from surogate_tlc.eval import samples

    tokenizer = tok.load(small_model, ordinary=True)
    special = tok.special_ids(tokenizer)
    seen = []

    def next_logits(inputs, last):
        seen.append((inputs.copy(), last.copy()))
        logits = np.zeros((inputs.shape[0], 320), np.float32)
        logits[:, special[stop]] = 1.0
        return logits

    prompts = [("code", "x = 1\n"), ("text", "a <|eos|> b <|code_end|> c")]
    result = samples.sample(small_model, shard_dirs[64], prompts=prompts, max_new=4, batch=2, next_logits=next_logits)
    assert [s["tokens"] for s in result["samples"]] == [[special[stop]]] * 2
    (inputs, last), *_ = seen
    for row, (kind, text) in enumerate(prompts):
        doc = corpus.Document(corpus.Source("s", kind, (), {}), "s/x", "s/x", text)
        [(_, framed)] = shards.encode(tokenizer, [doc], special)
        prompt = framed[:-2] if kind == "code" else framed[:-1]  # without <|code_end|> <|eos|>, or <|eos|>
        assert inputs[row, : len(prompt)].tolist() == prompt and last[row] == len(prompt) - 1, kind
