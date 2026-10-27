"""The opt-in entry point validates its own schema without importing the native trainer."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from surogate.experimental.train.config import TrainConfig, load_config


def _values(**updates):
    return {
        "model_config": {"model_type": "llama", "vocab_size": 32, "hidden_size": 16},
        "train_jsonl": "tokens.jsonl",
        "output_dir": "out",
        "device": "cpu",
        "dtype": "float32",
        **updates,
    }


def test_paths_and_lists_round_trip_relative_to_configuration(tmp_path):
    local = tmp_path / "local-model"
    local.mkdir()
    values = _values(model_config=None, model="local-model", betas=[0.8, 0.95], lora_targets=["q_proj"])
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(values))
    config = load_config(path)
    assert config.model == str(local)
    assert config.train_jsonl == str(tmp_path / "tokens.jsonl")
    assert config.output_dir == str(tmp_path / "out")
    assert config.betas == (0.8, 0.95)
    assert config.lora_targets == ("q_proj",)
    assert TrainConfig(**config.to_dict()) == config


@pytest.mark.parametrize(
    "updates",
    [
        {"schema_version": 1.0},
        {"model": "another/model"},
        {"model_config": None},
        {"optimizer": "typo"},
        {"dtype": "float16"},
        {"device": "cuda:bad"},
        {"batch_size": True},
        {"seed": 2**63},
        {"projection_interval": 0},
        {"max_sequence_length": 1},
        {"learning_rate": float("nan")},
        {"eps": 0},
        {"lora_dropout": 1},
        {"betas": [0.9, 1.0]},
        {"gradient_checkpointing": "false"},
        {"lisa_layers": 2, "lora_rank": 8},
        {"lisa_layers": 2, "compile_model": True},
        {"lisa_layers": 2, "lisa_state_policy": "cpu", "optimizer": "adamw"},
        {"optimizer_in_backward": True},
        {"optimizer_in_backward": True, "gradient_checkpointing": False, "max_grad_norm": 1},
        {"optimizer_in_backward": True, "gradient_checkpointing": False, "gradient_accumulation_steps": 2},
        {"optimizer_in_backward": True, "gradient_checkpointing": False, "compile_model": True},
        {"optimizer_in_backward": True, "gradient_checkpointing": False, "lisa_layers": 1},
    ],
)
def test_unsupported_or_ambiguous_configuration_is_rejected(updates):
    with pytest.raises((ValueError, TypeError)):
        TrainConfig(**_values(**updates))


def test_independent_modes_and_resume_contract():
    ordinary = TrainConfig(**_values(optimizer="adamw", gradient_accumulation_steps=4, max_grad_norm=1))
    lisa = TrainConfig(**_values(lisa_layers=2, lisa_state_policy="cpu"))
    backward = TrainConfig(**_values(optimizer_in_backward=True, gradient_checkpointing=False))
    assert ordinary.optimizer == "adamw" and lisa.lisa_layers == 2 and backward.optimizer_in_backward
    resumed = TrainConfig(**{**lisa.to_dict(), "max_steps": 500, "save_every": 0, "output_dir": "new"})
    assert resumed.resume_contract() == lisa.resume_contract()


def test_maximum_seed_is_accepted():
    assert TrainConfig(**_values(seed=2**63 - 1)).seed == 2**63 - 1


def test_unknown_fields_are_not_silently_ignored(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("unknown_speed_flag: true\n")
    with pytest.raises(ValueError, match="unknown_speed_flag"):
        load_config(path)


@pytest.mark.parametrize("validate_only", [False, True])
def test_help_and_validation_do_not_import_torch_or_native_runtime(tmp_path, validate_only):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_values()))
    arguments = [str(path), "--validate-only"] if validate_only else ["--help"]
    code = """
import runpy, sys
sys.path.insert(0, sys.argv.pop(1))
from _experimental_checkout import use_checkout_portion
use_checkout_portion(sys.argv.pop(1))
sys.argv = ['experimental-train', *sys.argv[1:]]
try:
    runpy.run_module('surogate.experimental.train', run_name='__main__')
except SystemExit as exit_status:
    assert exit_status.code == 0
else:
    raise AssertionError('CLI did not terminate')
assert 'torch' not in sys.modules
assert 'surogate._surogate' not in sys.modules
assert 'surogate.train.trainer' not in sys.modules
"""
    # An inherited PYTHONPATH (for example a regular `surogate`) is kept: the bootstrap must handle it.
    tests = Path(__file__).resolve().parent
    src = tests.parent / "src"
    pythonpath = os.pathsep.join(filter(None, [str(src), os.environ.get("PYTHONPATH")]))
    result = subprocess.run(
        [sys.executable, "-c", code, str(tests), str(src), *arguments],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": pythonpath},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "valid"' in result.stdout if validate_only else "--validate-only" in result.stdout
