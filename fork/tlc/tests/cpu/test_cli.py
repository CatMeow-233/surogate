"""`python -m surogate_tlc` (README: 命令行): the commands on model directories, each from the one command table."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from surogate_tlc import modeldir
from surogate_tlc.__main__ import COMMANDS, main
from surogate_tlc.eval import samples

SRC = Path(__file__).resolve().parents[2] / "src"
REPO = Path(__file__).resolve().parents[4]


def test_module_help_lists_the_commands(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), str(REPO)])}
    command = [sys.executable, "-B", "-m", "surogate_tlc", "--help"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert all(name in result.stdout for name in COMMANDS) and set(COMMANDS) == {
        "init",
        "train",
        "eval",
        "probe",
        "sample",
        "export",
    }


def test_init_writes_an_initialised_model_directory(tmp_path, trained_tokenizer, capsys):
    out = tmp_path / "A0"
    assert main(["init", "A0", str(out), "--tokenizer", str(trained_tokenizer), "--seed", "5"]) == 0
    assert json.loads(capsys.readouterr().out)["model_dir"] == str(out)
    cfg = modeldir.check_model_dir(out)
    assert cfg["tlc_layer_pattern"] == "M" * 12
    assert json.loads((out / modeldir.META_FILE).read_text())["init_seed"] == 5
    with pytest.raises(SystemExit, match="is not empty"):
        main(["init", "A1", str(out), "--tokenizer", str(trained_tokenizer), "--seed", "5"])


def test_export_writes_a_model_directory(tmp_path, small_model, capsys):
    step = tmp_path / "run" / "step_00000007"
    step.mkdir(parents=True)
    for name in ("config.json", "model.safetensors"):
        shutil.copy(small_model / name, step / name)
    (step / "checkpoint.json").write_text("{}")
    assert main(["export", str(step), str(small_model), str(tmp_path / "out")]) == 0
    modeldir.check_model_dir(tmp_path / "out")
    assert json.loads((tmp_path / "out" / modeldir.META_FILE).read_text())["step"] == 7


def test_train_reports_problems_without_training(run_yaml, capsys):
    assert main(["train", str(run_yaml(lora=True))]) == 2
    assert "lora must be False" in capsys.readouterr().err


def test_eval_writes_its_result(small_model, shard_dirs, tmp_path, monkeypatch, capsys):
    from surogate_tlc.eval import loss

    uniform = lambda inputs, targets, positions: np.where(targets == -100, 0.0, -1.0)  # noqa: E731
    monkeypatch.setattr(loss, "native_scorer", lambda *a, **k: uniform)
    out = tmp_path / "eval.json"
    assert main(["eval", str(small_model), str(shard_dirs[64]), "--batch", "4", "--out", str(out)]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(out.read_text())
    assert json.loads(out.read_text())["loss"] == pytest.approx(1.0)


def test_sample_runs_greedy_on_fixed_prompts(small_model, shard_dirs, tmp_path, monkeypatch, capsys):
    calls = []

    def next_logits(inputs, last):
        calls.append(inputs.shape)
        logits = np.zeros((inputs.shape[0], 320), np.float32)
        logits[:, 40] = 1.0  # always the same token
        return logits

    monkeypatch.setattr(samples, "native_next_logits", lambda *a, **k: next_logits)
    out = tmp_path / "samples.json"
    assert main(["sample", str(small_model), str(shard_dirs[64]), "--max-new", "3", "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert [s["prompt"] for s in written["samples"]] == [text for _, text in samples.PROMPTS]
    assert all(s["tokens"] == [40, 40, 40] for s in written["samples"])
    assert set(calls) == {(8, 64)}
