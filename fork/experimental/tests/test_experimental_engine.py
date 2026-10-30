"""Offline real-HF-model tests for the independent experimental training loop."""

from __future__ import annotations

import json
import shutil
import weakref
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from surogate.experimental.train.config import TrainConfig
from surogate.experimental.train.engine import (
    CHECKPOINT_KIND,
    _atomic_save,
    _check_in_backward_aliases,
    _restore_rng,
    _rng_state,
    _seed_rng,
    run_training,
)
from transformers import AutoConfig, AutoModelForCausalLM


@pytest.fixture(autouse=True, scope="module")
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(tmp_path: Path, name: str = "run", **changes) -> TrainConfig:
    data = tmp_path / "tokens.jsonl"
    if not data.exists():
        rows = [
            {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, -100, 5]},
            {"input_ids": [6, 7, 8, 9, 10], "labels": [-100, 7, 8, 9, 10]},
            {"input_ids": [11, 12, 13, 14, 15], "labels": [-100, 12, -100, 14, 15]},
        ]
        data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    values = dict(
        model_config={
            "model_type": "llama",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 32,
            "attention_dropout": 0.0,
        },
        device="cpu",
        dtype="float32",
        train_jsonl=str(data),
        output_dir=str(tmp_path / name),
        gradient_checkpointing=False,
        batch_size=1,
        max_sequence_length=16,
        max_steps=2,
        learning_rate=1e-3,
        seed=23,
        shuffle=False,
        save_every=1,
        projection_rank=2,
        projection_interval=2,
    )
    values.update(changes)
    return TrainConfig(**values)


def checkpoint(result):
    return torch.load(result["checkpoint"], weights_only=True, map_location="cpu")


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    else:
        assert left == right


def test_full_model_updates_and_exports_loadable_hf_artifact(tmp_path):
    config = tiny_config(tmp_path, optimizer="adamw")
    torch.manual_seed(config.seed)
    values = dict(config.model_config)
    initial = AutoModelForCausalLM.from_config(AutoConfig.for_model(values.pop("model_type"), **values))
    result = run_training(config)
    state = checkpoint(result)
    assert state["kind"] == CHECKPOINT_KIND
    assert state["step"] == 2 and result["gradients_cleared"]
    assert any(not torch.equal(initial.state_dict()[name], tensor) for name, tensor in state["model_state"].items())
    assert all(torch.isfinite(tensor).all() for tensor in state["model_state"].values())
    exported = AutoModelForCausalLM.from_pretrained(result["final_export"], local_files_only=True)
    assert_state_equal(exported.state_dict(), state["model_state"])
    metrics = [json.loads(line) for line in Path(result["metrics_path"]).read_text().splitlines()]
    assert [row["step"] for row in metrics] == [1, 2]
    assert [row["valid_tokens"] for row in metrics] == [2, 4]
    assert all(row["step_seconds"] > 0 and row["peak_cuda_bytes"] == 0 for row in metrics)
    manifest = json.loads(Path(result["manifest"]).read_text())
    report = manifest["trainable_parameter_report"]
    assert [row["name"] for row in report] == state["parameter_names"]
    assert manifest["initial_trainable_parameter_count"] == sum(row["numel"] for row in report)
    for row in report:
        tensor = state["model_state"][row["name"]]
        assert row["shape"] == list(tensor.shape)
        assert row["dtype"] == str(tensor.dtype)
        assert row["numel"] == tensor.numel()


def test_ga_uses_valid_token_sum_instead_of_microbatch_mean_of_means(tmp_path):
    micro_config = tiny_config(tmp_path, "micro", optimizer="adamw", gradient_accumulation_steps=2, max_steps=1)
    batch_config = replace(
        micro_config, output_dir=str(tmp_path / "batch"), gradient_accumulation_steps=1, batch_size=2
    )
    micro_result = run_training(micro_config)
    batch_result = run_training(batch_config)
    assert micro_result["last_metrics"]["valid_tokens"] == 6
    assert micro_result["last_metrics"]["loss"] == pytest.approx(batch_result["last_metrics"]["loss"], abs=1e-6)
    micro = checkpoint(micro_result)["model_state"]
    batch = checkpoint(batch_result)["model_state"]
    for name in micro:
        torch.testing.assert_close(micro[name], batch[name], rtol=1e-5, atol=1e-6)


def test_resume_restores_projection_layers_data_and_rng_exactly(tmp_path):
    config = tiny_config(
        tmp_path,
        "uninterrupted",
        max_steps=5,
        shuffle=True,
        lisa_layers=1,
        lisa_interval=1,
        lisa_state_policy="cpu",
        layer_path="model.layers",
        warmup_steps=3,
    )
    config.model_config["attention_dropout"] = 0.1
    uninterrupted = checkpoint(run_training(config))
    partial_config = replace(config, output_dir=str(tmp_path / "partial"), max_steps=2)
    partial = run_training(partial_config)
    # A different path with identical bytes is permitted; processing and data SHA
    # remain in the checkpoint contract.
    moved_data = tmp_path / "moved-tokens.jsonl"
    moved_data.write_bytes(Path(config.train_jsonl).read_bytes())
    resumed_config = replace(config, output_dir=str(tmp_path / "resumed"), train_jsonl=str(moved_data), save_every=0)
    resumed = checkpoint(run_training(resumed_config, resume_from=partial["checkpoint"]))
    for key in ("model_state", "optimizer_state", "layer_state", "data_state", "rng_state", "parameter_names", "step"):
        assert_state_equal(uninterrupted[key], resumed[key])


def test_optimizer_in_backward_matches_regular_step_and_releases_gradients(tmp_path):
    regular_config = tiny_config(tmp_path, "regular", max_steps=3)
    hook_config = replace(regular_config, output_dir=str(tmp_path / "in-backward"), optimizer_in_backward=True)
    regular = run_training(regular_config)
    hooked = run_training(hook_config)
    assert hooked["gradients_cleared"]
    left, right = checkpoint(regular), checkpoint(hooked)
    assert_state_equal(left["model_state"], right["model_state"])
    assert_state_equal(left["optimizer_state"], right["optimizer_state"])


def test_lora_updates_only_adapters_and_exports_reloadable_peft_artifact(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model

    config = tiny_config(
        tmp_path, "lora", lora_rank=2, lora_alpha=4, lora_targets=("q_proj", "v_proj"), gradient_checkpointing=True
    )
    torch.manual_seed(config.seed)
    values = dict(config.model_config)
    base = AutoModelForCausalLM.from_config(AutoConfig.for_model(values.pop("model_type"), **values))
    reference = get_peft_model(
        base, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    )
    initial = {name: tensor.clone() for name, tensor in reference.state_dict().items()}
    result = run_training(config)
    state = checkpoint(result)["model_state"]
    assert result["export_kind"] == "peft_adapter"
    report = json.loads(Path(result["manifest"]).read_text())["trainable_parameter_report"]
    assert report and all("lora_" in row["name"] for row in report)
    assert any(not torch.equal(initial[name], tensor) for name, tensor in state.items() if "lora_" in name)
    for name, tensor in state.items():
        if "lora_" not in name:
            assert torch.equal(initial[name], tensor), name
    torch.manual_seed(config.seed)
    values = dict(config.model_config)
    base = AutoModelForCausalLM.from_config(AutoConfig.for_model(values.pop("model_type"), **values))
    loaded = PeftModel.from_pretrained(base, result["final_export"], local_files_only=True)
    assert_state_equal(loaded.state_dict(), state)


def test_optimizer_in_backward_updates_tied_embeddings_once_like_the_regular_step(tmp_path):
    regular_config = tiny_config(tmp_path, "tied-regular", max_steps=3)
    regular_config.model_config["tie_word_embeddings"] = True
    hook_config = replace(regular_config, output_dir=str(tmp_path / "tied-in-backward"), optimizer_in_backward=True)
    regular = run_training(regular_config)
    hooked = run_training(hook_config)
    assert hooked["gradients_cleared"]
    left, right = checkpoint(regular), checkpoint(hooked)
    assert "lm_head.weight" not in right["parameter_names"]  # one Parameter, listed under its first name
    assert_state_equal(left["model_state"], right["model_state"])
    assert_state_equal(left["optimizer_state"], right["optimizer_state"])
    # The tied weight's hook ran once per backward, with the summed gradient of both uses.
    embedding = right["parameter_names"].index("model.embed_tokens.weight")
    assert right["optimizer_state"]["state"][embedding]["step"] == 3
    exported = AutoModelForCausalLM.from_pretrained(hooked["final_export"], local_files_only=True)
    assert exported.lm_head.weight is exported.model.embed_tokens.weight
    torch.testing.assert_close(
        exported.lm_head.weight, right["model_state"]["model.embed_tokens.weight"], rtol=0, atol=0
    )


def test_in_backward_rejects_distinct_parameters_sharing_storage():
    module = torch.nn.Module()
    backing = torch.randn(8)
    module.first = torch.nn.Parameter(backing[:4])
    module.second = torch.nn.Parameter(backing[4:])
    with pytest.raises(ValueError, match="shared parameter storage"):
        _check_in_backward_aliases(module)


def test_resume_rejects_changed_math_data_and_foreign_checkpoint(tmp_path):
    config = tiny_config(tmp_path)
    result = run_training(config)
    with pytest.raises(ValueError, match="configuration"):
        run_training(replace(config, learning_rate=2e-3), resume_from=result["checkpoint"])
    changed_data = tmp_path / "changed.jsonl"
    changed_data.write_text('{"input_ids":[1,2,3]}\n')
    with pytest.raises(ValueError, match="data content"):
        run_training(replace(config, train_jsonl=str(changed_data)), resume_from=result["checkpoint"])
    foreign = tmp_path / "foreign.pt"
    torch.save({"model": {}}, foreign)
    with pytest.raises(ValueError, match="experimental PyTorch checkpoint"):
        run_training(config, resume_from=str(foreign))


def test_does_not_overwrite_artifacts_or_accept_targetless_window(tmp_path):
    config = tiny_config(tmp_path)
    run_training(config)
    with pytest.raises(ValueError, match="already contains training artifacts"):
        run_training(config)
    targetless = tmp_path / "targetless.jsonl"
    targetless.write_text('{"input_ids":[1,2,3],"labels":[-100,-100,-100]}\n')
    with pytest.raises(ValueError, match="supervised"):
        run_training(replace(config, train_jsonl=str(targetless), output_dir=str(tmp_path / "targetless")))
    assert not list((tmp_path / "targetless").glob("checkpoints/*.pt"))


def test_checkpoint_resumes_without_the_original_pretrained_weights(tmp_path):
    config = tiny_config(tmp_path, "pretrained", max_steps=1)
    torch.manual_seed(config.seed)
    values = dict(config.model_config)
    base = AutoModelForCausalLM.from_config(AutoConfig.for_model(values.pop("model_type"), **values))
    pretrained = tmp_path / "original-model"
    base.save_pretrained(pretrained)
    config = replace(config, model=str(pretrained), model_config=None)
    first = run_training(config)
    shutil.rmtree(pretrained)
    resumed = run_training(
        replace(config, max_steps=2, output_dir=str(tmp_path / "without-original")), resume_from=first["checkpoint"]
    )
    assert resumed["step"] == 2
    assert checkpoint(resumed)["model_config"]["model_type"] == "llama"


def test_failed_checkpoint_write_preserves_previous_file_and_removes_temporary(tmp_path, monkeypatch):
    path = tmp_path / "step.pt"
    _atomic_save({"weights": torch.ones(2)}, path)
    original = path.read_bytes()

    def broken_save(state, handle):
        handle.write(b"incomplete checkpoint")
        raise OSError("simulated storage failure")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="storage failure"):
        _atomic_save({"weights": torch.zeros(2)}, path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_vocabulary_logits_are_released_before_backward(tmp_path, monkeypatch, gradient_checkpointing):
    from surogate.experimental.train import engine

    build = engine._build_model
    backward = torch.Tensor.backward
    observed = []
    backward_calls = []

    def instrumented_build(*args, **kwargs):
        model, architecture = build(*args, **kwargs)

        def before_forward(module, inputs):
            if observed:
                assert observed[-1]() is None, "the previous vocabulary logits survived into the next forward"

        def after_forward(module, inputs, output):
            observed.append(weakref.ref(output.logits))

        model.register_forward_pre_hook(before_forward)
        model.register_forward_hook(after_forward)
        return model, architecture

    def checked_backward(tensor, *args, **kwargs):
        assert observed and observed[-1]() is None, "vocabulary logits survived until backward"
        backward_calls.append(len(observed))
        return backward(tensor, *args, **kwargs)

    monkeypatch.setattr(engine, "_build_model", instrumented_build)
    monkeypatch.setattr(torch.Tensor, "backward", checked_backward)
    run_training(tiny_config(tmp_path, gradient_accumulation_steps=2, gradient_checkpointing=gradient_checkpointing))
    assert len(observed) == 4
    assert backward_calls == [1, 2, 3, 4]
    assert all(reference() is None for reference in observed)


def test_resume_architecture_is_initialized_directly_in_requested_dtype(tmp_path, monkeypatch):
    from surogate.experimental.train.engine import _build_model

    original = AutoModelForCausalLM.from_config
    created_dtypes = []

    def from_config(*args, **kwargs):
        assert kwargs["dtype"] == torch.bfloat16
        model = original(*args, **kwargs)
        # Inspect before _build_model can call .to(): a post-initialization cast
        # would retain the much larger FP32 construction peak on the host.
        created_dtypes.append({parameter.dtype for parameter in model.parameters()})
        return model

    monkeypatch.setattr(AutoModelForCausalLM, "from_config", from_config)
    config = tiny_config(tmp_path)
    model, _ = _build_model(config, torch.device("cpu"), torch.bfloat16, config.model_config)
    assert created_dtypes == [{torch.bfloat16}]
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())


def test_missing_data_is_rejected_before_loading_model_or_creating_output(tmp_path, monkeypatch):
    from surogate.experimental.train import engine

    config = tiny_config(tmp_path)
    config.train_jsonl = str(tmp_path / "missing.jsonl")

    def unexpected_load(*args, **kwargs):
        pytest.fail("missing data must be rejected before any model construction or download")

    monkeypatch.setattr(engine, "_build_model", unexpected_load)
    with pytest.raises(ValueError, match="existing local file"):
        run_training(config)
    assert not Path(config.output_dir).exists()


@pytest.mark.parametrize("already_initialized", [False, True])
def test_multi_process_launch_is_rejected_before_model_or_output(tmp_path, monkeypatch, already_initialized):
    from surogate.experimental.train import engine

    config = tiny_config(tmp_path)
    monkeypatch.setenv("WORLD_SIZE", "1" if already_initialized else "2")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: already_initialized)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def unexpected_load(*args, **kwargs):
        pytest.fail("multi-process mode must be rejected before model loading")

    monkeypatch.setattr(engine, "_build_model", unexpected_load)
    with pytest.raises(ValueError, match="one process only"):
        run_training(config)
    assert not Path(config.output_dir).exists()


def test_cpu_training_and_resume_never_access_cuda_rng_on_gpu_host(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the CPU workflow must not seed, read or restore any CUDA RNG")

    # Model the availability of a GPU without claiming any GPU math was executed.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in (
        "manual_seed",
        "manual_seed_all",
        "get_rng_state",
        "get_rng_state_all",
        "set_rng_state",
        "set_rng_state_all",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    first_config = tiny_config(tmp_path, "cpu-on-gpu-host", max_steps=1)
    first = run_training(first_config)
    assert checkpoint(first)["rng_state"]["cuda"] is None
    resumed = run_training(
        replace(first_config, max_steps=2, output_dir=str(tmp_path / "cpu-resumed")), resume_from=first["checkpoint"]
    )
    assert resumed["step"] == 2
    assert checkpoint(resumed)["rng_state"]["cuda"] is None


def test_selected_cuda_rng_is_restored_to_the_current_workflow_device(monkeypatch):
    saved_device, selected_device = torch.device("cuda:3"), torch.device("cuda:1")
    selected_state = torch.tensor([9, 17, 23], dtype=torch.uint8)
    events = []

    def get_rng_state(device):
        events.append(("get", device))
        return selected_state.clone()

    def set_rng_state(state, device):
        assert torch.equal(state, selected_state)
        events.append(("set", device))

    def forbidden(*args, **kwargs):
        pytest.fail("single-device RNG checkpointing must not enumerate or restore every CUDA generator")

    monkeypatch.setattr(torch.cuda, "get_rng_state", get_rng_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state", set_rng_state)
    for name in ("get_rng_state_all", "set_rng_state_all", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    state = _rng_state(saved_device)
    _restore_rng(state, selected_device)
    assert events == [("get", saved_device), ("set", selected_device)]


def test_cuda_seed_is_scoped_to_the_selected_device(monkeypatch):
    selected_device = torch.device("cuda:2")
    events = []

    @contextmanager
    def selected_context(device):
        assert device == selected_device
        events.append("enter")
        yield
        events.append("exit")

    def seed_selected(seed):
        assert events == ["enter"] and seed == 73
        events.append("seed")

    def forbidden(*args, **kwargs):
        pytest.fail("seeding one workflow must not seed all CUDA devices")

    monkeypatch.setattr(torch.cuda, "device", selected_context)
    monkeypatch.setattr(torch.cuda, "manual_seed", seed_selected)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden)
    _seed_rng(73, selected_device)
    assert events == ["enter", "seed", "exit"]
