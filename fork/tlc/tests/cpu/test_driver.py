"""The training driver (docs/training.md: 训练驱动) with a stand-in for the native trainer: every attempt trains the
configured tokens per update and an OOM stops the run instead of shrinking it, failures leave a failure report, a
resume needs the run's weights, data and settings, and finished runs and checkpoints become model directories."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from surogate_tlc import modeldir
from surogate_tlc.data import shards
from surogate_tlc.data.shards import manifest, shard_files
from surogate_tlc.train import config, driver, report

OOM = "Cuda OOM when allocating tensor blocks[0].mamba_scan of shape [32, 512, 768] with dtype bf16 in context fwd."


class FakeTrainer:
    """Stands in for SurogateTrainerWrapper: writes what the native trainer writes, or raises."""

    def __init__(self, sft, behaviour):
        self.sft, self.behaviour = sft, behaviour
        self.out = Path(sft["output_dir"])
        self.config = SimpleNamespace(log_file=str(self.out / "log-fake.json"))

    def train(self):
        model = Path(self.sft["model"])
        save = self.behaviour.get("save", ())
        for step in save(self.sft) if callable(save) else save:
            path = self.out / f"step_{step:08d}"
            path.mkdir(exist_ok=True)
            for name in ("config.json", "model.safetensors"):
                shutil.copy(model / name, path / name)
            (path / "checkpoint.json").write_text("{}")
        error = self.behaviour.get("raise")
        if callable(error):
            error = error(self.sft)
        if error is not None:
            raise error
        for name in ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json"):
            shutil.copy(model / name, self.out / name)  # the final export
        curve = [{"step": float(s), "loss": 9.7 - s / 10, "grad_norm": 1.0, "lr": 1e-3} for s in range(1, 11)]
        regression = {"convergence_curve": curve, "step_time_ms": {"mean": 120.0, "p50": 100.0, "p95": 150.0}}
        regression |= {"cuda_peak_memory_bytes": 123, "block_schema_summary": {"block_schema_records": 0}}
        Path(os.environ[driver.REGRESSION_ENV]).write_text(json.dumps(regression))
        log = [{"log": "step", "step": 4, "loss": 9.3}, {"log": "eval", "step": 5, "loss": 9.1, "eval_tokens": 512}]
        Path(self.config.log_file).write_text(json.dumps(log))


def factory(calls: list, **behaviour):
    def make(sft, train_files, eval_files):
        calls.append({"sft": dict(sft), "train": list(train_files), "eval": list(eval_files)})
        return FakeTrainer(sft, behaviour)

    return make


def oom_above(microbatch: int):
    return lambda sft: RuntimeError(OOM) if sft["per_device_train_batch_size"] > microbatch else None


def test_oom_fails_with_a_failure_report(run_yaml, monkeypatch):
    """The tokens per update never shrink: an OOM stops the run, and the report names the remedy (a new run)."""
    monkeypatch.delenv(driver.REGRESSION_ENV, raising=False)
    run = config.load(run_yaml())
    calls = []
    with pytest.raises(driver.RunFailed, match="Cuda OOM"):
        driver.train(run, factory=factory(calls, **{"raise": oom_above(8)}))
    record = json.loads((run.run_dir / driver.RUN_FILE).read_text())
    assert len(calls) == 1 and record["status"] == "failed" and record["attempts"][0]["status"] == "oom"
    assert [a["tokens_per_update"] for a in record["attempts"]] == [16384]
    failure = record["failure_report"]
    assert failure["oom"] and failure["remedy"] == "a new run with a smaller --microbatch"
    assert (failure["microbatch"], failure["grad_accum"]) == (32, 1) and "Cuda OOM" in failure["traceback"]
    assert record["attempts"][0]["environment"]["git_commit"]
    assert driver.REGRESSION_ENV not in os.environ


def test_other_failures_and_a_resume(run_yaml):
    run = config.load(run_yaml())
    with pytest.raises(driver.RunFailed, match="power cut"):
        driver.train(run, factory=factory([], save=[5], **{"raise": RuntimeError("power cut")}))
    record = json.loads((run.run_dir / driver.RUN_FILE).read_text())
    assert record["failure_report"]["oom"] is False and record["failure_report"]["remedy"] is None
    assert record["finalized_checkpoints"] == ["step_00000005"]  # what was saved is usable
    calls = []
    record = driver.train(run, factory=factory(calls))
    assert [a["status"] for a in record["attempts"]] == ["failed", "completed"]
    assert record["attempts"][-1]["resumed_from_step"] == 5 and record["status"] == "completed"
    assert record["failure_report"] is None


@pytest.mark.parametrize(
    "change, message",
    [
        ({"learning_rate": 1e-4}, "learning_rate"),
        ({"max_steps": 20}, "schedule_total_steps"),
        ({"per_device_train_batch_size": 16}, "per_device_train_batch_size"),
    ],
)
def test_resume_refuses_changed_settings(run_yaml, change, message):
    driver.train(config.load(run_yaml()), factory=factory([], save=[5]))
    with pytest.raises(driver.RunError, match=message):
        driver.train(config.load(run_yaml(**change)), factory=factory([]))
    # Saving and logging may change.
    record = driver.train(config.load(run_yaml(save_steps=3, logging_steps=2)), factory=factory([]))
    assert len(record["attempts"]) == 2


def test_resume_refuses_other_weights(run_yaml, tmp_path):
    run = config.load(run_yaml())
    driver.train(run, factory=factory([], save=[5]))
    other = tmp_path / "other-model"
    shutil.copytree(run.model_dir, other)
    (other / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(driver.RunError, match="model_weights_sha256"):
        driver.train(config.load(run_yaml(model=str(other))), factory=factory([]))


def test_finished_run_and_checkpoints_are_model_directories(run_yaml, small_model, shard_dirs):
    run = config.load(run_yaml())
    calls = []
    record = driver.train(run, factory=factory(calls, save=[5]))
    data = shard_dirs[512]
    assert calls[0]["train"] == [str(p) for p in shard_files(data, "train")]
    assert calls[0]["eval"] == [str(p) for p in shard_files(data, "val")]
    provenance = {
        "tokenizer_sha256": manifest(data)["tokenizer"]["sha256"],
        "data_manifest_sha256": shards.manifest_sha256(data),
        "shard_version": shards.SHARD_VERSION,
    }
    environment = record["attempts"][0]["environment"]
    for path, step, kind in ((run.run_dir, 10, "run"), (run.run_dir / "step_00000005", 5, "checkpoint")):
        modeldir.check_model_dir(path)
        meta = json.loads((path / modeldir.META_FILE).read_text())
        assert meta | provenance == meta and meta["step"] == step and meta["init_seed"] == 0 and meta["kind"] == kind
        assert meta["run"] == {
            "settings_sha256": record["identity"]["settings_sha256"],
            "config_sha256": record["attempts"][0]["config_sha256"],
            "git_commit": environment["git_commit"],
            "git_dirty": environment["git_dirty"],
        }
    assert record["finalized_checkpoints"] == ["step_00000005"]
    summary = record["attempts"][0]["summary"]
    assert summary["steps"] == [1, 10] and summary["first_loss"] == pytest.approx(9.6)
    assert summary["tokens_per_second"] == 163840 and summary["cuda_peak_memory_bytes"] == 123
    assert summary["eval_loss"] == [[5, 9.1]]
    split = manifest(data)["splits"]["train"]
    assert record["data"]["valid_targets_per_update"] == round(16384 * split["valid_targets"] / split["tokens"])
    (row,) = report.runs(run.run_dir.parent)
    assert row["status"] == "completed" and row["tokens_per_second"] == 163840 and row["microbatch"] == 32


def test_run_refuses_problems_before_training(run_yaml):
    calls = []
    with pytest.raises(driver.RunError, match="lora must be False"):
        driver.train(config.load(run_yaml(lora=True)), factory=factory(calls))
    assert calls == []


def test_fixed_batch_holds_the_first_rows(run_yaml, shard_dirs):
    data = shard_dirs[64]
    rows = 4
    tlc = {"tokens_per_update": 64 * rows, "stages": {"fit": {"fixed_batch": True}}}
    run = config.load(run_yaml(64, tlc=tlc, per_device_train_batch_size=rows), stage="fit")
    calls = []
    record = driver.train(run, factory=factory(calls))
    fixed = run.run_dir / driver.FIXED_BATCH_FILE
    assert calls[0]["train"] == [str(fixed)]
    source = shard_files(data, "train")[0]
    expected = [a[:rows] for a in shards.served_rows(shards.read_shard(source), 64)]
    got = shards.served_rows(shards.read_shard(fixed), 64)
    assert all(np.array_equal(a, b) for a, b in zip(expected, got, strict=True))
    valid = int((expected[1] != -100).sum())
    assert record["data"]["fixed_batch"]["valid_targets"] == record["data"]["valid_targets_per_update"] == valid


@pytest.mark.parametrize(
    "error, oom",
    [
        (RuntimeError(OOM), True),
        (RuntimeError("Cuda Error in matmul.cpp:12 (cudaMalloc): cudaErrorMemoryAllocation: out of memory"), True),
        (MemoryError(), True),
        (RuntimeError("no room for the bloom filter"), False),
        (ValueError("out of memory"), False),
    ],
)
def test_is_oom(error, oom):
    assert driver.is_oom(error) is oom


def test_environment_fingerprint():
    env = report.environment()
    assert (report.REPO / ".git").exists()
    head = subprocess.run(["git", "-C", str(report.REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    assert env["git_commit"] == head.stdout.strip() and isinstance(env["git_dirty"], bool)
    assert set(env["packages"]) == set(report.PACKAGES) and env["packages"]["torch"]
    assert env["native"] is None or len(env["native"]["sha256"]) == 64
    assert env["gpu"] is None or "torch_cuda" in env["gpu"]


def test_summarize_tolerates_missing_files(tmp_path):
    assert report.summarize(tmp_path / "absent.json", None, 16384) == {}
