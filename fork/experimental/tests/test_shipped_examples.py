"""Every shipped RTX 5090 example trains with its own settings on a tiny stand-in model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import yaml
from surogate.experimental.train.config import TrainConfig
from surogate.experimental.train.engine import run_training

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

# The examples train Qwen/Qwen3-4B, which ties its embedding and LM head (Qwen3
# Technical Report, arXiv 2505.09388, Table 1: "Tie Embedding: Yes"). The stand-in
# keeps the architecture family and the tie, so a combination of settings that the
# real model rejects at startup (such as optimizer_in_backward with a tied head
# before ties were supported) fails here, offline, without the 8 GB download.
EXAMPLE_MODEL = "Qwen/Qwen3-4B"
TINY_STAND_IN = {
    "model_type": "qwen3",
    "vocab_size": 32,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 8,
    "max_position_embeddings": 32,
    "attention_dropout": 0.0,
    "tie_word_embeddings": True,
}


@pytest.fixture(autouse=True, scope="module")
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("name", sorted(path.name for path in EXAMPLES.glob("5090-*.yaml")))
def test_shipped_example_trains_with_its_settings(tmp_path, name):
    values = yaml.safe_load((EXAMPLES / name).read_text(encoding="utf-8"))
    # A different model needs a matching stand-in (architecture family and tie).
    assert values["model"] == EXAMPLE_MODEL
    data = tmp_path / "tokens.jsonl"
    rows = [{"input_ids": [index, index + 1, index + 2, index + 3]} for index in range(1, 9)]
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    # Only the model, the CPU runtime, the paths and the run length change; the
    # optimizer, LISA, LoRA, OIB, accumulation and checkpointing settings are kept.
    values.update(
        model=None,
        model_config=dict(TINY_STAND_IN),
        device="cpu",
        dtype="float32",
        train_jsonl=str(data),
        output_dir=str(tmp_path / "out"),
        max_sequence_length=16,
        max_steps=2,
        save_every=0,
    )
    result = run_training(TrainConfig(**values))
    assert result["step"] == 2 and result["gradients_cleared"]
    assert math.isfinite(result["last_metrics"]["loss"])
