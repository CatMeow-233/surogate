"""The eval command's loss (README: 评测), with a stand-in for the native scorer: every served row of the split is
scored once, the loss averages over valid targets, and the unigram baseline counts the train split's targets."""

import math

import numpy as np
import pytest
from surogate_tlc.data import shards
from surogate_tlc.eval import loss
from surogate_tlc.train import config


def _rows(data, split):
    parts = [shards.served_rows(shards.read_shard(p), 64) for p in config.shard_files(data, split)]
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
        assert result["valid_targets"] == config.manifest(data)["splits"]["val"]["valid_targets"]
    # The filled rows of the last batch carry no targets.
    assert (np.concatenate([t for _, t in seen])[len(inputs) :] == -100).all()
    assert result["loss"] == pytest.approx(math.log(320)) and result["unigram_loss"] < result["loss"]
    assert result["below_unigram"] is False
    assert result["data_manifest_sha256"] == shards.manifest_sha256(data)


def test_evaluate_refuses_another_tokenizer(tmp_path, tokenizer_dir, shard_dirs):
    from surogate_tlc import hf, init, modeldir

    cfg = hf.row_config("A0", num_hidden_layers=2, tlc_layer_pattern="MM", chunk_size=16, vocab_size=320)
    other = modeldir.write_model_dir(tmp_path / "other", cfg, tokenizer_dir, init_seed=0)
    init.init_model_dir(other)
    with pytest.raises(ValueError, match="another tokenizer"):
        loss.evaluate(other, shard_dirs[64], scorer=lambda *a: None)
