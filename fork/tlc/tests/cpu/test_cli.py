"""`python -m surogate_tlc` (README: 命令行): the TLC commands, and the hand-over of every other command to upstream's
CLI after register()."""

import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from surogate_tlc import modeldir
from surogate_tlc.__main__ import main
from surogate_tlc.data import tokenizer

SRC = Path(__file__).resolve().parents[2] / "src"
REPO = Path(__file__).resolve().parents[4]
COMMANDS = ("init", "train-tokenizer", "build-data", "train", "eval", "export")


def test_module_help_lists_the_commands(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), str(REPO)])}
    command = [sys.executable, "-B", "-m", "surogate_tlc", "--help"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert all(name in result.stdout for name in COMMANDS)


def test_init_writes_an_initialised_model_directory(tmp_path, trained_tokenizer, capsys):
    out = tmp_path / "A0"
    assert main(["init", "A0", str(out), "--tokenizer", str(trained_tokenizer), "--seed", "5"]) == 0
    assert json.loads(capsys.readouterr().out)["model_dir"] == str(out)
    cfg = modeldir.check_model_dir(out)
    assert cfg["tlc_layer_pattern"] == "M" * 12
    assert json.loads((out / modeldir.META_FILE).read_text())["init_seed"] == 5
    with pytest.raises(SystemExit, match="is not empty"):
        main(["init", "A1", str(out), "--tokenizer", str(trained_tokenizer), "--seed", "5"])


def test_train_tokenizer_and_build_data_take_the_module_arguments(tmp_path, corpus_spec, capsys):
    tok = tmp_path / "tok"
    assert main(["train-tokenizer", str(corpus_spec), str(tok), "--vocab-size", "300"]) == 0
    assert json.loads(capsys.readouterr().out)["sha256"] == tokenizer.tokenizer_sha256(tok)
    assert main(["build-data", str(corpus_spec), str(tok), str(tmp_path / "data"), "--seq-len", "64"]) == 0
    assert json.loads((tmp_path / "data" / "manifest.json").read_text())["format"]["seq_len"] == 64


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


def test_other_commands_reach_the_surogate_cli_after_register(monkeypatch):
    seen = {}

    def cli_main():
        from transformers import AutoConfig

        from surogate.dsl.py_compiler import list_registered_models

        seen["argv"] = list(sys.argv)
        seen["tlc"] = "TLCModel" in list_registered_models() and AutoConfig.for_model("tlc").model_type == "tlc"

    monkeypatch.setitem(sys.modules, "surogate.cli.main", types.SimpleNamespace(cli_main=cli_main))
    monkeypatch.setattr(sys, "argv", ["python -m surogate_tlc"])
    assert main(["sft", "run.yaml"]) == 0
    assert seen == {"argv": ["surogate", "sft", "run.yaml"], "tlc": True}
