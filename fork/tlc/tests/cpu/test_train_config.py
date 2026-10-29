"""Run configurations of the training driver (README: 训练驱动): the examples, stages, the derived accumulation and
the checks made before a run starts."""

from pathlib import Path

import pytest
import yaml
from surogate_tlc.train import config

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
ROWS = {"a0.yaml": "A0", "a1.yaml": "A1", "hybrid.yaml": "A1h"}


@pytest.fixture
def runs(monkeypatch, tmp_path) -> Path:
    monkeypatch.setenv("TLC_RUNS", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("example", sorted(ROWS))
@pytest.mark.parametrize("stage", ["smoke", "overfit", "pilot"])
@pytest.mark.parametrize("microbatch", [None, 2, 8, 32])
def test_examples_keep_16384_tokens_per_update(runs, example, stage, microbatch):
    run = config.load(EXAMPLES / example, stage=stage, microbatch=microbatch)
    microbatch_, accum = run.batch
    assert microbatch_ * accum * run.sft["sequence_len"] * run.sft["gpus"] == run.tokens_per_update == 16384
    assert run.batch == ((32, 1) if microbatch is None else (microbatch, 16384 // 512 // microbatch))
    suffix = "" if microbatch is None else f"-mb{microbatch}"
    assert run.run_dir == runs / "train" / ROWS[example] / f"{stage}{suffix}"
    assert run.model_dir == runs / "models" / ROWS[example] and run.data == runs / "data"


@pytest.mark.parametrize("example", sorted(ROWS))
def test_example_stages_and_tlc_defaults(runs, example):
    stages = {stage: config.load(EXAMPLES / example, stage=stage) for stage in ("smoke", "overfit", "pilot")}
    assert {stage: run.sft["max_steps"] for stage, run in stages.items()} == {"smoke": 10, "overfit": 100, "pilot": 300}
    assert {stage: run.fixed_batch for stage, run in stages.items()} == {
        "smoke": False,
        "overfit": True,
        "pilot": False,
    }
    assert stages["smoke"].sft["save_steps"] == 5  # the G2 resume check restarts from step 5
    for run in stages.values():
        assert run.on_oom == "fail"
        assert {key: run.sft[key] for key in config.FIXED} == config.FIXED
        assert run.sft["recompute"] is False and run.sft["sequence_len"] == 512
        assert not set(run.sft) & set(config.FORBIDDEN)


def test_grad_accum():
    assert config.grad_accum(16384, 512, 1, 2) == 16
    for microbatch, seq_len in ((3, 512), (64, 512), (0, 512), (1, 3000)):
        with pytest.raises(ValueError):
            config.grad_accum(16384, seq_len, 1, microbatch)


def _write(tmp_path: Path, body: dict) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "change, message",
    [
        ({"tlc": None}, "needs a `tlc` table"),
        ({"tlc": {"data": "d", "seed": 1}}, "unknown: ['seed']"),
        ({"tlc": {"tokens_per_update": 16384}}, "needs `data`"),
        ({"output_dir": "${TLC_UNSET_FOR_TEST}/x"}, "unset environment variable"),
        ({"gradient_accumulation_steps": 4}, "gradient_accumulation_steps must be 1"),
        ({"tlc": {"data": "d", "stages": {"smoke": {"datasets": []}}}}, "sets ['datasets']"),
        ({"per_device_train_batch_size": 5}, "not a whole number of microbatches"),
    ],
)
def test_load_refuses(tmp_path, change, message):
    body = {
        "model": "m",
        "output_dir": "o",
        "sequence_len": 512,
        "per_device_train_batch_size": 32,
        "tlc": {"data": "d"},
    }
    body.update(change)
    stage = "smoke" if "stages" in (body["tlc"] or {}) else None
    with pytest.raises(ValueError) as error:
        config.load(_write(tmp_path, body), stage=stage)
    assert message in str(error.value)


def test_unknown_stage(run_yaml):
    with pytest.raises(ValueError, match="no stage 'smoke'"):
        config.load(run_yaml(), stage="smoke")


def test_problems_of_a_valid_run_and_defaults(run_yaml):
    run = config.load(run_yaml())
    assert config.problems(run) == []
    assert run.batch == (32, 1) and run.run_dir.name == "run"
    assert run.sft["lora"] is False and run.sft["sample_packing"] is False and run.sft["gpus"] == 1


@pytest.mark.parametrize(
    "sft, tlc, message",
    [
        ({"lora": True}, {}, "lora must be False"),
        ({"sample_packing": True}, {}, "sample_packing must be False"),
        ({"datasets": [{"path": "x"}]}, {}, "datasets is not used"),
        ({"checkpoint_dir": "/elsewhere"}, {}, "checkpoint_dir is not used"),
        ({"max_steps": -1}, {}, "max_steps must be a positive integer"),
        ({}, {"on_oom": "retry"}, "tlc.on_oom must be one of"),
        ({}, {"max_downshifts": -1}, "tlc.max_downshifts"),
    ],
)
def test_problems_of_settings(run_yaml, sft, tlc, message):
    found = config.problems(config.load(run_yaml(tlc=tlc, **sft)))
    assert any(message in problem for problem in found), found


def test_problems_of_data_and_model(run_yaml, shard_dirs, tokenizer_dir, tmp_path):
    from surogate_tlc import hf, init, modeldir

    run = config.load(run_yaml(tlc={"data": str(shard_dirs[64])}))
    assert any("have rows of 64" in p for p in config.problems(run))
    # A model whose tokenizer is not the one the shards were built with.
    cfg = hf.row_config("A0", num_hidden_layers=2, tlc_layer_pattern="MM", chunk_size=16, vocab_size=320)
    other = modeldir.write_model_dir(tmp_path / "other", cfg, tokenizer_dir, init_seed=0)
    assert any("missing model.safetensors" in p for p in config.problems(config.load(run_yaml(model=str(other)))))
    init.init_model_dir(other)
    assert any("another tokenizer" in p for p in config.problems(config.load(run_yaml(model=str(other)))))
    assert any("is missing" in p for p in config.problems(config.load(run_yaml(tlc={"data": str(tmp_path)}))))


def test_fixed_batch_needs_enough_rows(run_yaml, shard_dirs):
    rows = config.manifest(shard_dirs[64])["shards"][0]["rows"]
    stages = {"fit": {"fixed_batch": True}}
    tlc = {"tokens_per_update": 64 * rows, "stages": stages}
    fits = config.load(run_yaml(64, tlc=tlc, per_device_train_batch_size=rows), stage="fit")
    assert fits.fixed_batch and config.problems(fits) == []
    too_many = 64 * (rows + 1)
    run = config.load(
        run_yaml(64, tlc={"tokens_per_update": too_many, "stages": stages}, per_device_train_batch_size=rows + 1),
        stage="fit",
    )
    assert any("fixed_batch needs" in p for p in config.problems(run))
