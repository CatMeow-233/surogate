"""Real local JSONL, padding and deterministic resume without model downloads."""

import hashlib
import json

import pytest
import torch
import torch.nn.functional as F
from surogate.experimental.train.data import RECORD_KEYS, BatchStream, JSONLDataset, collate_batch


def _file(tmp_path, rows, name="data.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_labels_masks_default_and_causal_padding(tmp_path):
    path = _file(
        tmp_path,
        [
            {"input_ids": [0, 2, 3, 4], "loss_mask": [False, 0, True, 1]},
            {"input_ids": [5, 6], "labels": [-100, 7]},
            {"input_ids": [8, 9]},
        ],
    )
    dataset = JSONLDataset(path, 8, vocab_size=10)
    assert len(dataset) == 3
    assert dataset.fingerprint == hashlib.sha256(path.read_bytes()).hexdigest()
    batch = collate_batch([dataset[index] for index in range(3)], pad_token_id=0)
    assert batch["input_ids"].tolist() == [[0, 2, 3, 4], [5, 6, 0, 0], [8, 9, 0, 0]]
    assert batch["labels"].tolist() == [[-100, -100, 3, 4], [-100, 7, -100, -100], [8, 9, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]]
    assert all(tensor.dtype == torch.long and tensor.device.type == "cpu" for tensor in batch.values())
    # A masked prompt still receives attention, and its last position predicts
    # the first supervised response token. No targets cross padded rows.
    assert batch["attention_mask"][0, 1] == 1 and batch["labels"][0, 2] == 3
    assert (batch["labels"][:, 1:] != -100).sum().item() == 4


@pytest.mark.parametrize(
    "row,message",
    [
        ({"input_ids": [True, 2]}, "integer token"),
        ({"input_ids": [-1, 2]}, "integer token"),
        ({"input_ids": [1, 10]}, "vocabulary"),
        ({"input_ids": [1, 2.0]}, "integer token"),
        ({"input_ids": []}, "nonempty"),
        ({"input_ids": [1]}, "no supervised targets"),
        ({"input_ids": [1, 2], "labels": [1]}, "same length"),
        ({"input_ids": [1, 2], "labels": [1, True]}, "valid integer"),
        ({"input_ids": [1, 2], "labels": [1, -1]}, "valid integer"),
        ({"input_ids": [1, 2], "labels": [1, 10]}, "valid integer"),
        ({"input_ids": [1, 2], "loss_mask": [1, 0.5]}, "bool or 0/1"),
        ({"input_ids": [1, 2], "loss_mask": [1]}, "same length"),
        ({"input_ids": [1, 2], "loss_mask": [True, False]}, "no supervised targets"),
        ({"input_ids": [1, 2], "labels": [1, 2], "loss_mask": [1, 1]}, "mutually exclusive"),
        ({"input_ids": [1, 2], "position_ids": [0, 1]}, "unsupported record keys 'position_ids';"),
        ({"input_ids": [1, 2], "packed": False}, "unsupported record keys 'packed';"),
        ({"input_ids": [1, 2], "segments": [2]}, "unsupported record keys 'segments';"),
        # Common packing/segment metadata that a deny-list would have trained across.
        ({"input_ids": [1, 2], "segment_ids": [0, 0]}, "unsupported record keys 'segment_ids';"),
        ({"input_ids": [1, 2], "seq_lengths": [2]}, "unsupported record keys 'seq_lengths';"),
        ({"input_ids": [1, 2], "cu_seqlens": [0, 2]}, "unsupported record keys 'cu_seqlens';"),
        ({"input_ids": [1, 2], "document_ids": [0, 0]}, "unsupported record keys 'document_ids';"),
        # Misspelled supervision would otherwise fall back to supervising every token.
        ({"input_ids": [1, 2], "label": [-100, 2]}, "'label' \\(did you mean 'labels'\\?\\)"),
        ({"input_ids": [1, 2], "loss-mask": [0, 1]}, "'loss-mask' \\(did you mean 'loss_mask'\\?\\)"),
        ({"input_ids": [1, 2], "text": "a b", "source": "x"}, "unsupported record keys 'source', 'text';"),
        ({"input_ids": [1, 2], "attention_mask": [1, 0]}, "attention_mask must be all ones"),
        ({"input_ids": [1, 2], "attention_mask": [1]}, "attention_mask must be all ones"),
        ({"input_ids": [1, 2], "attention_mask": [1, 1.0]}, "attention_mask must be all ones"),
    ],
)
def test_invalid_records_have_line_context(tmp_path, row, message):
    path = _file(tmp_path, [{"input_ids": [1, 2]}, row])
    with pytest.raises(ValueError, match=message) as failure:
        JSONLDataset(path, 8, vocab_size=10)
    assert ":2:" in str(failure.value)


def test_allow_listed_metadata_and_an_all_ones_attention_mask_do_not_change_the_record(tmp_path):
    plain = {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}
    path = _file(
        tmp_path,
        [
            plain,
            {**plain, "attention_mask": [1, True, 1], "id": "doc-7", "metadata": {"source": "x", "tokens": 3}},
        ],
    )
    dataset = JSONLDataset(path, 8, vocab_size=10)
    assert dataset[0] == dataset[1] == plain
    assert set(RECORD_KEYS) == {"input_ids", "labels", "loss_mask", "attention_mask", "id", "metadata"}


def test_truncation_is_explicit_and_must_keep_supervision(tmp_path):
    path = _file(tmp_path, [{"input_ids": [1, 2, 3, 4], "loss_mask": [0, 1, 1, 1]}])
    with pytest.raises(ValueError, match="exceeds"):
        JSONLDataset(path, 3)
    assert JSONLDataset(path, 3, truncation="right")[0] == {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}
    path = _file(tmp_path, [{"input_ids": [1, 2, 3, 4], "loss_mask": [0, 0, 0, 1]}])
    with pytest.raises(ValueError, match="no supervised targets"):
        JSONLDataset(path, 3, truncation="right")


def test_offsets_skip_blank_lines_and_detect_live_file_changes(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_bytes(b'\n{"input_ids":[1,2]}\n\n{"input_ids":[3,4]}\n')
    dataset = JSONLDataset(path, 8)
    assert dataset[1]["input_ids"] == [3, 4]
    assert dataset._offsets == [(1, 2), (22, 4)]
    assert not any(isinstance(value, torch.Tensor) for value in vars(dataset).values())
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed after indexing"):
        dataset[0]


def test_dataset_rejects_a_file_changed_while_indexing(monkeypatch, tmp_path):
    path = _file(tmp_path, [{"input_ids": [1, 2]}])
    decode = JSONLDataset._decode

    def append_while_decoding(self, raw, line_number):
        result = decode(self, raw, line_number)
        with path.open("ab") as writer:
            writer.write(b"\n")
        return result

    monkeypatch.setattr(JSONLDataset, "_decode", append_while_decoding)
    with pytest.raises(ValueError, match="changed during indexing"):
        JSONLDataset(path, 8)


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("consumed", [0, 1, 3, 4, 7])
def test_stream_resume_preserves_tail_epoch_order_and_global_rng(tmp_path, shuffle, consumed):
    dataset = JSONLDataset(_file(tmp_path, [{"input_ids": [index, index + 1]} for index in range(5)]), 8)
    before = torch.random.get_rng_state().clone()
    stream = BatchStream(dataset, batch_size=2, seed=17, shuffle=shuffle)
    for _ in range(consumed):
        stream.next_batch()
    state = json.loads(json.dumps(stream.state_dict()))
    expected = [stream.next_batch() for _ in range(8)]
    resumed = BatchStream(dataset, batch_size=2, seed=17, shuffle=shuffle)
    resumed.load_state_dict(state)
    actual = [resumed.next_batch() for _ in range(8)]
    for first, second in zip(actual, expected, strict=True):
        for key in first:
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert {batch["input_ids"].shape[0] for batch in actual} == {1, 2}
    assert torch.equal(torch.random.get_rng_state(), before)


def test_stream_rejects_changed_data_or_batch_contract_and_keeps_failed_cursor(tmp_path):
    path = _file(tmp_path, [{"input_ids": [1, 2]}, {"input_ids": [3, 4]}])
    dataset = JSONLDataset(path, 8)
    stream = BatchStream(dataset, 1, 7, True)
    state = stream.state_dict()
    with pytest.raises(ValueError, match="fingerprint"):
        BatchStream(dataset, 2, 7, True).load_state_dict(state)
    with pytest.raises(ValueError, match="fingerprint"):
        BatchStream(JSONLDataset(path, 4), 1, 7, True).load_state_dict(state)
    path.write_text('{"input_ids":[5,6]}\n')
    with pytest.raises(ValueError, match="changed after indexing"):
        stream.next_batch()
    assert stream.state_dict() == state
    with pytest.raises(ValueError, match="fingerprint"):
        BatchStream(JSONLDataset(path, 8), 1, 7, True).load_state_dict(state)


def test_batches_stay_on_cpu_under_a_different_default_device(tmp_path):
    dataset = JSONLDataset(_file(tmp_path, [{"input_ids": [1, 2]}, {"input_ids": [3, 4]}]), 8)
    with torch.device("meta"):
        stream = BatchStream(dataset, 1, 7, True)
        batch = stream.next_batch()
    assert all(tensor.device.type == "cpu" for tensor in batch.values())
    assert batch["attention_mask"].tolist() == [[1, 1]]


def test_token_weighted_accumulation_matches_combined_batch(tmp_path):
    dataset = JSONLDataset(
        _file(
            tmp_path,
            [
                {"input_ids": [1, 2, 3], "loss_mask": [0, 0, 1]},
                {"input_ids": [2, 3, 4, 5], "loss_mask": [0, 1, 1, 1]},
            ],
        ),
        8,
        vocab_size=8,
    )
    generator = torch.Generator().manual_seed(7)
    weights = torch.randn(8, 8, generator=generator, requires_grad=True)
    combined = collate_batch([dataset[0], dataset[1]], 0)
    target_count = (combined["labels"][:, 1:] != -100).sum()
    logits = weights[combined["input_ids"][:, :-1]]
    (
        F.cross_entropy(logits.reshape(-1, 8), combined["labels"][:, 1:].reshape(-1), reduction="sum") / target_count
    ).backward()
    expected = weights.grad.clone()
    weights.grad = None
    for row in dataset:
        batch = collate_batch([row], 0)
        logits = weights[batch["input_ids"][:, :-1]]
        (
            F.cross_entropy(logits.reshape(-1, 8), batch["labels"][:, 1:].reshape(-1), reduction="sum") / target_count
        ).backward()
    torch.testing.assert_close(weights.grad, expected)
