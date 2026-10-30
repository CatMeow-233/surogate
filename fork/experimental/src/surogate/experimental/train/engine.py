"""Independent PyTorch experiments; no native training engine is imported here."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from surogate.experimental.train.config import TrainConfig

CHECKPOINT_KIND = "surogate.experimental.train"
# Version 3: rank-scaled optimizer state keeps per-parameter clocks across LISA
# drops; version-2 checkpoints are refused, not migrated.
CHECKPOINT_VERSION = 3


def _check_single_process() -> None:
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as error:
        raise ValueError("WORLD_SIZE must be a positive integer for the single-process experimental trainer") from error
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be a positive integer for the single-process experimental trainer")
    distributed = torch.distributed
    initialized_world_size = (
        distributed.get_world_size() if distributed.is_available() and distributed.is_initialized() else 1
    )
    if world_size > 1 or initialized_world_size > 1:
        raise ValueError(
            "experimental training supports one process only; do not launch it with multi-process torchrun"
        )


def _fingerprint(values: dict[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_save(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _load_checkpoint(path: str, contract: str) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("kind") != CHECKPOINT_KIND:
        raise ValueError("resume requires a surogate experimental PyTorch checkpoint")
    if state.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported experimental checkpoint version {state.get('checkpoint_version')!r}; "
            f"this trainer resumes only version {CHECKPOINT_VERSION}"
        )
    if state.get("config_fingerprint") != contract:
        raise ValueError("resume configuration changes the model, algorithm, or data processing contract")
    if not isinstance(state.get("step"), int) or state["step"] < 0:
        raise ValueError("checkpoint has an invalid completed-step counter")
    return state


def _seed_rng(seed: int, device: torch.device) -> None:
    random.seed(seed)
    # torch.manual_seed also seeds every CUDA device, including for CPU runs.
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(seed)


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def _restore_rng(state: dict[str, Any], device: torch.device) -> None:
    cuda_state = state["cuda"]
    if device.type == "cuda":
        if not isinstance(cuda_state, torch.Tensor) or cuda_state.dtype != torch.uint8 or cuda_state.ndim != 1:
            raise ValueError("CUDA checkpoint is missing its selected-device RNG state")
    elif cuda_state is not None:
        raise ValueError("CPU checkpoint unexpectedly contains CUDA RNG state")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        # The process may select another physical/indexed device for the same
        # single-GPU workflow. Restore its saved stream there, not on the old index.
        torch.cuda.set_rng_state(cuda_state, device)


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    counts: dict[str, int] = {}
    seen: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            if id(value) not in seen:
                seen.add(id(value))
                kind = value.device.type
                counts[kind] = counts.get(kind, 0) + value.numel() * value.element_size()
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(optimizer.state)
    return {
        "optimizer_state_bytes": sum(counts.values()),
        "optimizer_state_cpu_bytes": counts.get("cpu", 0),
        "optimizer_state_cuda_bytes": counts.get("cuda", 0),
    }


def _build_model(config: TrainConfig, device: torch.device, dtype: torch.dtype, saved_config: dict | None):
    from transformers import AutoConfig, AutoModelForCausalLM

    architecture = saved_config if saved_config is not None else config.model_config
    if architecture is not None:
        values = dict(architecture)
        model_type = values.pop("model_type")
        try:
            hf_config = AutoConfig.for_model(model_type, **values)
        except ValueError:
            if not config.trust_remote_code or config.model is None:
                raise
            # Custom Python architectures still need their declared code dependency;
            # checkpoint weights do not require another pretrained-weight download.
            hf_config = AutoConfig.from_pretrained(config.model, revision=config.model_revision, trust_remote_code=True)
            hf_config.update(values)
        model = AutoModelForCausalLM.from_config(
            hf_config,
            trust_remote_code=config.trust_remote_code,
            attn_implementation=config.attention_implementation,
            dtype=dtype,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            revision=config.model_revision,
            trust_remote_code=config.trust_remote_code,
            attn_implementation=config.attention_implementation,
            dtype=dtype,
        )
    model.config.use_cache = False
    architecture = model.config.to_dict()
    model.to(device=device, dtype=dtype)
    if config.lora_rank:
        from peft import LoraConfig, get_peft_model

        adapter_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_targets),
            task_type="CAUSAL_LM",
            revision=config.model_revision or getattr(model.config, "_commit_hash", None),
        )
        model = get_peft_model(model, adapter_config)
    if config.gradient_checkpointing:
        if not getattr(model, "supports_gradient_checkpointing", False):
            raise ValueError("the selected model does not support gradient checkpointing")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, architecture


def _check_in_backward_aliases(model: torch.nn.Module) -> None:
    """Reject distinct trainable parameters over one storage; allow tied names.

    One Parameter object reachable under several names (``tie_word_embeddings``)
    has a single AccumulateGrad node: autograd sums every use's gradient before
    its post-accumulate hook runs, so the hook runs once per backward with the
    full gradient, as a regular step would see it. Distinct Parameter objects
    sharing storage would each run a hook with a partial gradient and update the
    same memory twice.
    """
    seen_storages: dict[tuple[torch.device, int], str] = {}
    # named_parameters() lists each Parameter object once, under its first name.
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not parameter.numel():
            continue
        storage = (parameter.device, parameter.untyped_storage().data_ptr())
        if storage in seen_storages:
            raise ValueError(
                f"optimizer_in_backward does not support shared parameter storage: {seen_storages[storage]}, {name}"
            )
        seen_storages[storage] = name


def _prepare_output(output: Path, resumed: dict[str, Any] | None) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    reserved = [output / name for name in ("metrics.jsonl", "manifest.json", "checkpoints", "final")]
    if resumed is None and any(path.exists() for path in reserved):
        raise ValueError("output_dir already contains training artifacts; use a new directory or resume_from")
    if resumed is not None and any(path.exists() for path in reserved):
        manifest = output / "manifest.json"
        if not manifest.is_file():
            raise ValueError("resume output_dir contains artifacts without an experimental manifest")
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        if any(existing.get(key) != resumed.get(key) for key in ("kind", "config_fingerprint", "data_fingerprint")):
            raise ValueError("resume output_dir belongs to a different training experiment")
    metrics = output / "metrics.jsonl"
    if resumed is not None and metrics.exists():
        with metrics.open(encoding="utf-8") as handle:
            steps = [json.loads(line)["step"] for line in handle if line.strip()]
        if steps and max(steps) > resumed["step"]:
            raise ValueError("output metrics extend beyond this checkpoint; resume into a new output_dir")
    return metrics


def run_training(config: TrainConfig, *, resume_from: str | None = None) -> dict[str, Any]:
    """Train a separate experimental model and save a completed-step resume artifact."""
    from surogate.experimental.train.data import BatchStream, JSONLDataset
    from surogate.experimental.train.layers import LayerSampler
    from surogate.experimental.train.optim import RankScaledAdamW

    config.validate()
    _check_single_process()
    if not Path(config.train_jsonl).is_file():
        raise ValueError(f"train_jsonl must name an existing local file: {config.train_jsonl}")
    device_name = config.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; CPU experiments require device=cpu and dtype=float32")
    dtype = torch.float32 if config.dtype == "float32" else torch.bfloat16
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU experiments currently require dtype=float32")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError("the selected CUDA device does not support bfloat16")
    contract = _fingerprint(config.resume_contract())
    resumed = _load_checkpoint(resume_from, contract) if resume_from is not None else None
    runtime = {"device_type": device.type, "dtype": str(dtype)}
    if resumed is not None and resumed.get("runtime") != runtime:
        raise ValueError("resume requires the same device type and dtype")
    output = Path(config.output_dir)
    metrics_path = _prepare_output(output, resumed)
    _seed_rng(config.seed, device)
    model, architecture = _build_model(config, device, dtype, resumed["model_config"] if resumed else None)
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named_parameters:
        raise ValueError("the model has no trainable parameters")
    parameter_names = [name for name, _ in named_parameters]
    parameters = [parameter for _, parameter in named_parameters]
    parameter_report = [
        {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype), "numel": parameter.numel()}
        for name, parameter in named_parameters
    ]
    if config.optimizer_in_backward:
        _check_in_backward_aliases(model)
    if config.optimizer == "rank_scaled":
        optimizer = RankScaledAdamW(
            parameters,
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
            rank=config.projection_rank,
            update_interval=config.projection_interval,
            scale=config.projection_scale,
            seed=config.seed,
            stochastic_rounding=config.stochastic_rounding,
        )
    else:
        optimizer = torch.optim.AdamW(
            parameters, lr=config.learning_rate, betas=config.betas, eps=config.eps, weight_decay=config.weight_decay
        )
    dataset = JSONLDataset(
        config.train_jsonl,
        config.max_sequence_length,
        vocab_size=model.config.vocab_size,
        truncation=config.truncation,
    )
    if config.pad_token_id >= model.config.vocab_size:
        raise ValueError("pad_token_id is outside the model vocabulary")
    stream = BatchStream(dataset, config.batch_size, config.seed, config.shuffle, pad_token_id=config.pad_token_id)
    selector = (
        LayerSampler(
            model,
            config.lisa_layers,
            config.lisa_interval,
            state_policy=config.lisa_state_policy,
            layer_path=config.layer_path,
            seed=config.seed,
        )
        if config.lisa_layers
        else None
    )
    step = 0
    if resumed is not None:
        if resumed.get("data_fingerprint") != dataset.fingerprint:
            raise ValueError("resume training data content differs from the checkpoint")
        if resumed.get("parameter_names") != parameter_names:
            raise ValueError("resume trainable parameter names or ordering differ from the checkpoint")
        model.load_state_dict(resumed["model_state"], strict=True)
        if selector is not None:
            selector.load_state_dict(resumed["layer_state"])
        optimizer.load_state_dict(resumed["optimizer_state"])
        stream.load_state_dict(resumed["data_state"])
        step = resumed["step"]
    manifest = {
        "kind": CHECKPOINT_KIND,
        "schema_version": config.schema_version,
        "config": config.to_dict(),
        "config_fingerprint": contract,
        "data_fingerprint": dataset.fingerprint,
        "runtime": runtime,
        "base_model": config.model,
        "base_revision": config.model_revision or getattr(model.config, "_commit_hash", None),
        "remote_code_dependency": config.trust_remote_code,
        "parameter_report_scope": "initial trainable candidates before optional layer sampling",
        "initial_trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_report": parameter_report,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    forward_model = torch.compile(model) if config.compile_model else model
    hooks = []
    if config.optimizer_in_backward:

        def update_parameter(parameter: torch.Tensor) -> None:
            optimizer.step_parameter(parameter)
            parameter.grad = None

        hooks = [parameter.register_post_accumulate_grad_hook(update_parameter) for parameter in parameters]
    if resumed is not None:
        _restore_rng(resumed["rng_state"], device)
        # The live model/optimizer now own their restored state. Do not retain a
        # second full checkpoint dictionary throughout a large-model run.
        resumed = None

    def save_checkpoint() -> str:
        path = output / "checkpoints" / f"step-{step:08d}.pt"
        _atomic_save(
            {
                "kind": CHECKPOINT_KIND,
                "checkpoint_version": CHECKPOINT_VERSION,
                "config": config.to_dict(),
                "config_fingerprint": contract,
                "data_fingerprint": dataset.fingerprint,
                "parameter_names": parameter_names,
                "model_config": architecture,
                "runtime": runtime,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "layer_state": selector.state_dict() if selector is not None else None,
                "data_state": stream.state_dict(),
                "rng_state": _rng_state(device),
                "step": step,
            },
            path,
        )
        return str(path)

    checkpoint = None
    last_saved_step = -1
    last_metrics: dict[str, Any] = {}
    try:
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            while step < config.max_steps:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                if selector is not None:
                    selector.on_step(step, optimizer)
                lr = config.learning_rate * min(1.0, (step + 1) / max(1, config.warmup_steps))
                for group in optimizer.param_groups:
                    group["lr"] = lr
                microbatches = [stream.next_batch() for _ in range(config.gradient_accumulation_steps)]
                valid_tokens = sum(int((batch["labels"][:, 1:] != -100).sum()) for batch in microbatches)
                if valid_tokens == 0:
                    raise ValueError("the accumulation window has no supervised causal target tokens")
                summed_loss = torch.zeros((), dtype=torch.float32, device=device)
                for batch in microbatches:
                    batch = {name: tensor.to(device) for name, tensor in batch.items()}
                    context = (
                        torch.autocast("cuda", dtype=dtype)
                        if device.type == "cuda" and dtype == torch.bfloat16
                        else nullcontext()
                    )
                    with context:
                        outputs = forward_model(
                            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
                        )
                        logits = outputs.logits[:, :-1, :].float()
                        loss_sum = F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            batch["labels"][:, 1:].reshape(-1),
                            ignore_index=-100,
                            reduction="sum",
                        )
                        loss = loss_sum / valid_tokens
                    summed_loss += loss_sum.detach()
                    # CE retains the tensors needed for its gradient. Release
                    # model logits and their FP32 cast before backward allocates
                    # gradients, rather than keeping these large outputs alive.
                    del outputs, logits
                    loss.backward()
                    del loss_sum, loss, batch
                mean_loss = float((summed_loss / valid_tokens).item())
                if not math.isfinite(mean_loss):
                    raise FloatingPointError("non-finite experimental loss; no new checkpoint was saved")
                grad_norm = None
                if not config.optimizer_in_backward:
                    if config.max_grad_norm:
                        grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm, error_if_nonfinite=True)
                        )
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step += 1
                elapsed = time.perf_counter() - started
                last_metrics = {
                    "step": step,
                    "loss": mean_loss,
                    "valid_tokens": valid_tokens,
                    "step_seconds": elapsed,
                    "valid_tokens_per_second": valid_tokens / elapsed,
                    "learning_rate": lr,
                    "grad_norm": grad_norm,
                    "trainable_parameters": sum(
                        parameter.numel() for parameter in parameters if parameter.requires_grad
                    ),
                    "selected_layers": list(selector.selected_layers) if selector is not None else [],
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                    **_optimizer_state_bytes(optimizer),
                }
                metrics_file.write(json.dumps(last_metrics, allow_nan=False) + "\n")
                metrics_file.flush()
                if config.save_every and step % config.save_every == 0:
                    checkpoint = save_checkpoint()
                    last_saved_step = step
            if last_saved_step != step:
                checkpoint = save_checkpoint()
    finally:
        for hook in hooks:
            hook.remove()
    final = output / "final"
    model.save_pretrained(final, safe_serialization=True)
    export_kind = "peft_adapter" if config.lora_rank else "huggingface_model"
    (final / "experimental_export.json").write_text(
        json.dumps(
            {
                "kind": export_kind,
                "checkpoint": checkpoint,
                "base_model": config.model,
                "base_revision": manifest["base_revision"],
                "step": step,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "step": step,
        "checkpoint": checkpoint,
        "final_export": str(final),
        "export_kind": export_kind,
        "metrics_path": str(metrics_path),
        "manifest": str(output / "manifest.json"),
        "gradients_cleared": all(parameter.grad is None for parameter in parameters),
        "last_metrics": last_metrics,
    }
