"""Greedy samples of a model directory at fixed prompts (docs/training.md: 生成样例), the samples gate G3 records.

Every sample re-runs the whole sequence for each new token through the native next_token_logits, which needs a
trainer with resident LoRA weights: native.SAMPLING_LORA adds a rank-1 adapter whose B factor starts at zero, so the
adapted model computes the base model. With LoRA the native trainer uses the tied embedding as the head, which is
also the head full training trains and saves since fork ledger topic T10 (docs/model.md: tied LM head), so this holds
on trained weights too; G3 checks it on the initial and the pilot weights. There is no decode cache (Mamba decoding
is not native yet), which is slow per token but cheap at G3's few prompts. Prompts are framed as documents start in
the shards: <|bos|> <|code_start|> for code, <|bos|> for English.
"""

from __future__ import annotations

import itertools
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..data.shards import check_tokenizer, manifest_sha256
from ..data.tokenizer import BOS, CODE_END, CODE_START, EOS, PAD, load, special_ids
from ..modeldir import check_model_dir
from ..native import NextLogits, native_next_logits

# (kind, text): the prompts every G3 sample starts from.
PROMPTS = (
    ("code", "def fibonacci(n):\n"),
    ("code", "import os\n\n\ndef list_python_files(root):\n"),
    ("code", "class Stack:\n    def __init__(self):\n"),
    ("code", "def count_words(text):\n    counts = {}\n    for word in text.split():\n"),
    ("text", "The little coder reads"),
    ("text", "To sort a list in Python, "),
)


def greedy(
    prompts: Sequence[Sequence[int]],
    next_logits: NextLogits,
    *,
    batch: int,
    seq_len: int,
    pad_id: int,
    stop_ids: set[int],
    max_new: int,
) -> list[list[int]]:
    """The greedy continuation of every prompt (token ids), up to `max_new` tokens, a stop id or the row's end."""
    sequences = [list(p) for p in prompts]
    done = [not p or len(p) >= seq_len for p in prompts]
    for _ in range(max_new):
        active = [i for i, finished in enumerate(done) if not finished]
        if not active:
            break
        for chunk in itertools.batched(active, batch):
            inputs = np.full((batch, seq_len), pad_id, np.int32)
            last = np.zeros(batch, np.int32)
            for b, i in enumerate(chunk):
                inputs[b, : len(sequences[i])] = sequences[i]
                last[b] = len(sequences[i]) - 1
            logits = np.asarray(next_logits(inputs, last))
            for b, i in enumerate(chunk):
                token = int(np.argmax(logits[b]))
                sequences[i].append(token)
                done[i] = token in stop_ids or len(sequences[i]) >= seq_len
    return [s[len(p) :] for s, p in zip(sequences, prompts, strict=True)]


def sample(
    model_dir: Path,
    data: Path,
    *,
    prompts: Sequence[tuple[str, str]] = PROMPTS,
    max_new: int = 64,
    batch: int = 8,
    next_logits: NextLogits | None = None,
) -> dict[str, Any]:
    """Greedy samples of `model_dir` at `prompts`; rows have the shard directory `data`'s length, and
    `next_logits` defaults to the native trainer."""
    model_dir, data = Path(model_dir), Path(data)
    check_model_dir(model_dir)
    info = check_tokenizer(data, model_dir)
    seq_len = info["format"]["seq_len"]
    tokenizer = load(model_dir, ordinary=True)
    special = special_ids(tokenizer)
    frames = {"code": [special[BOS], special[CODE_START]], "text": [special[BOS]]}
    ids = [frames[kind] + tokenizer.encode(text, add_special_tokens=False).ids for kind, text in prompts]
    with tempfile.TemporaryDirectory(prefix="tlc-sample-") as tmp:
        step = next_logits or native_next_logits(model_dir, data, Path(tmp), batch=batch, seq_len=seq_len)
        stops = {special[EOS], special[CODE_END]}
        completions = greedy(
            ids, step, batch=batch, seq_len=seq_len, pad_id=special[PAD], stop_ids=stops, max_new=max_new
        )
    return {
        "model_dir": str(model_dir),
        "tokenizer_sha256": info["tokenizer"]["sha256"],
        "data_manifest_sha256": manifest_sha256(data),
        "max_new": max_new,
        "samples": [
            {
                "kind": kind,
                "prompt": text,
                "completion": tokenizer.decode(tokens, skip_special_tokens=False),
                "tokens": tokens,
            }
            for (kind, text), tokens in zip(prompts, completions, strict=True)
        ],
    }
