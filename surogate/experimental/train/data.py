"""Local, unpacked token data with deterministic resumable batching."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


class JSONLDataset:
    """Index byte offsets, not token lists; every JSONL row is one document.

    ``loss_mask[t]`` supervises token t, rather than the prediction made at t.
    Labels use the same convention and may contain -100. Causal shifting is
    the runner's responsibility. No packing metadata is accepted.
    """

    def __init__(self, path, max_sequence_length, vocab_size=None, truncation="error"):
        if not _integer(max_sequence_length) or max_sequence_length < 2:
            raise ValueError("max_sequence_length must be an integer >= 2")
        if vocab_size is not None and (not _integer(vocab_size) or vocab_size < 1):
            raise ValueError("vocab_size must be a positive integer")
        if truncation not in ("error", "right"):
            raise ValueError("truncation must be 'error' or 'right'")
        self.path = Path(path).expanduser().resolve()
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.truncation = truncation
        self._offsets = []
        self._signature = self._file_signature()
        digest = hashlib.sha256()
        with self.path.open("rb") as stream:
            line_number = 0
            while True:
                offset = stream.tell()
                raw = stream.readline()
                if not raw:
                    break
                digest.update(raw)
                line_number += 1
                if raw.strip():
                    self._decode(raw, line_number)
                    self._offsets.append((offset, line_number))
        if self._file_signature() != self._signature:
            raise ValueError(f"{self.path}: dataset changed during indexing; rebuild the dataset")
        if not self._offsets:
            raise ValueError(f"{self.path}: dataset contains no records")
        self.fingerprint = digest.hexdigest()

    def _file_signature(self):
        stat = self.path.stat()
        return stat.st_size, stat.st_mtime_ns

    def _decode(self, raw, line_number):
        location = f"{self.path}:{line_number}"
        try:
            row = json.loads(raw)
        except (ValueError, UnicodeError) as error:
            raise ValueError(f"{location}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{location}: a record must be a JSON object")
        if {"packed", "segments", "position_ids"}.intersection(row):
            raise ValueError(f"{location}: packed documents/position_ids are unsupported")
        ids = row.get("input_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{location}: input_ids must be a nonempty list")
        if any(
            not _integer(token) or token < 0 or (self.vocab_size is not None and token >= self.vocab_size)
            for token in ids
        ):
            raise ValueError(f"{location}: input_ids must contain nonnegative integer token IDs within the vocabulary")
        if "labels" in row and "loss_mask" in row:
            raise ValueError(f"{location}: labels and loss_mask are mutually exclusive")
        if "labels" in row:
            labels = row["labels"]
            if not isinstance(labels, list) or len(labels) != len(ids):
                raise ValueError(f"{location}: labels must have the same length as input_ids")
            if any(
                not _integer(label)
                or (label != -100 and (label < 0 or (self.vocab_size is not None and label >= self.vocab_size)))
                for label in labels
            ):
                raise ValueError(f"{location}: labels must contain -100 or valid integer token IDs")
            labels = list(labels)
        elif "loss_mask" in row:
            mask = row["loss_mask"]
            if not isinstance(mask, list) or len(mask) != len(ids):
                raise ValueError(f"{location}: loss_mask must have the same length as input_ids")
            if any(not isinstance(value, bool) and not (_integer(value) and value in (0, 1)) for value in mask):
                raise ValueError(f"{location}: loss_mask values must be bool or 0/1")
            labels = [token if keep else -100 for token, keep in zip(ids, mask, strict=True)]
        else:
            labels = list(ids)
        if len(ids) > self.max_sequence_length:
            if self.truncation == "error":
                raise ValueError(f"{location}: sequence exceeds max_sequence_length={self.max_sequence_length}")
            ids = ids[: self.max_sequence_length]
            labels = labels[: self.max_sequence_length]
        if not any(label != -100 for label in labels[1:]):
            raise ValueError(f"{location}: record has no supervised targets after causal shifting/truncation")
        return {"input_ids": ids, "labels": labels}

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, index):
        if self._file_signature() != self._signature:
            raise ValueError(f"{self.path}: dataset changed after indexing; rebuild the dataset")
        offset, line_number = self._offsets[index]
        # A per-read handle also makes seeks safe across independent workers.
        with self.path.open("rb") as stream:
            stream.seek(offset)
            return self._decode(stream.readline(), line_number)


def collate_batch(records, pad_token_id):
    """Right-pad independent documents; attention follows lengths, not loss masks."""
    import torch

    if not records:
        raise ValueError("cannot collate an empty batch")
    if not _integer(pad_token_id) or pad_token_id < 0:
        raise ValueError("pad_token_id must be a nonnegative integer")
    length = max(len(record["input_ids"]) for record in records)
    shape = (len(records), length)
    inputs = torch.full(shape, pad_token_id, dtype=torch.long, device="cpu")
    labels = torch.full(shape, -100, dtype=torch.long, device="cpu")
    attention = torch.zeros(shape, dtype=torch.long, device="cpu")
    for index, record in enumerate(records):
        count = len(record["input_ids"])
        if len(record["labels"]) != count:
            raise ValueError("labels must have the same length as input_ids")
        inputs[index, :count] = torch.tensor(record["input_ids"], dtype=torch.long, device="cpu")
        labels[index, :count] = torch.tensor(record["labels"], dtype=torch.long, device="cpu")
        attention[index, :count] = 1
    return {"input_ids": inputs, "labels": labels, "attention_mask": attention}


class BatchStream:
    """Deterministic infinite epochs, including the final partial batch.

    The private CPU generator is seeded once per epoch, so a checkpoint needs
    only epoch/cursor rather than the full permutation or global RNG state.
    Cursor advances only after a batch has been read and collated successfully.
    """

    def __init__(self, dataset, batch_size, seed, shuffle, *, pad_token_id=0):
        import torch

        if not _integer(batch_size) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not _integer(seed) or not 0 <= seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")
        if not isinstance(shuffle, bool):
            raise ValueError("shuffle must be bool")
        if (
            not _integer(pad_token_id)
            or pad_token_id < 0
            or (dataset.vocab_size is not None and pad_token_id >= dataset.vocab_size)
        ):
            raise ValueError("pad_token_id must be a valid nonnegative token ID")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pad_token_id = pad_token_id
        self.epoch = 0
        self.cursor = 0
        self._generator = torch.Generator(device="cpu")
        self._order = None

    def next_batch(self):
        import torch

        epoch = self.epoch + int(self.cursor == len(self.dataset))
        cursor = 0 if self.cursor == len(self.dataset) else self.cursor
        order = self._order if epoch == self.epoch else None
        if order is None:
            self._generator.manual_seed((self.seed + epoch) % 2**63)
            order = (
                torch.randperm(len(self.dataset), generator=self._generator, device="cpu").tolist()
                if self.shuffle
                else list(range(len(self.dataset)))
            )
        end = min(cursor + self.batch_size, len(self.dataset))
        batch = collate_batch([self.dataset[index] for index in order[cursor:end]], self.pad_token_id)
        self.epoch, self.cursor, self._order = epoch, end, order
        return batch

    def _contract(self):
        return {
            "dataset_fingerprint": self.dataset.fingerprint,
            "dataset_length": len(self.dataset),
            "max_sequence_length": self.dataset.max_sequence_length,
            "vocab_size": self.dataset.vocab_size,
            "truncation": self.dataset.truncation,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "shuffle": self.shuffle,
            "pad_token_id": self.pad_token_id,
        }

    def state_dict(self):
        return {"version": 1, **self._contract(), "epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state):
        if state.get("version") != 1 or any(state.get(key) != value for key, value in self._contract().items()):
            raise ValueError("batch stream checkpoint does not match the dataset/configuration fingerprint")
        epoch, cursor = state.get("epoch"), state.get("cursor")
        if not _integer(epoch) or epoch < 0 or not _integer(cursor) or not 0 <= cursor <= len(self.dataset):
            raise ValueError("invalid batch stream epoch/cursor")
        self.epoch, self.cursor, self._order = epoch, cursor, None
