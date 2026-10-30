"""Configuration for the independent experimental PyTorch trainer."""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    schema_version: int = 1
    model: str | None = None
    model_config: dict[str, Any] | None = None
    model_revision: str | None = None
    trust_remote_code: bool = False
    attention_implementation: str = "sdpa"
    device: str = "cuda"
    dtype: str = "bfloat16"
    train_jsonl: str = ""
    output_dir: str = ""
    max_sequence_length: int = 2048
    pad_token_id: int = 0
    truncation: str = "error"
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_steps: int = 100
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 0.0
    optimizer: str = "rank_scaled"
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    projection_rank: int = 1
    projection_interval: int = 200
    projection_scale: float = 1.0
    projection_scale_rule: str = "rms"
    stochastic_rounding: bool = True
    seed: int = 42
    lisa_layers: int = 0
    lisa_interval: int = 20
    lisa_state_policy: str = "drop"
    layer_path: str | None = None
    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    lora_targets: tuple[str, ...] = field(
        default_factory=lambda: ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    )
    gradient_checkpointing: bool = True
    optimizer_in_backward: bool = False
    compile_model: bool = False
    save_every: int = 50
    shuffle: bool = True
    warmup_steps: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("experimental training requires schema_version: 1")
        if (self.model is None) == (self.model_config is None):
            raise ValueError("provide exactly one of model or model_config")
        if self.model is not None and (not isinstance(self.model, str) or not self.model.strip()):
            raise ValueError("model must be a nonempty local path or Hugging Face model ID")
        if self.model_config is not None:
            if not isinstance(self.model_config, dict) or not isinstance(self.model_config.get("model_type"), str):
                raise ValueError("model_config must contain a model_type string")
            json.dumps(self.model_config, allow_nan=False)
        if self.model_revision is not None and not isinstance(self.model_revision, str):
            raise ValueError("model_revision must be a string or null")
        for name in ("train_jsonl", "output_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty path")
        if self.attention_implementation not in {"sdpa", "eager"}:
            raise ValueError("attention_implementation must be sdpa or eager")
        if not isinstance(self.device, str) or re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", self.device) is None:
            raise ValueError("device must be auto, cpu, cuda, or cuda:N")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("dtype must be float32 or bfloat16; FP16 GradScaler is not implemented")
        if self.optimizer not in {"adamw", "rank_scaled"}:
            raise ValueError("optimizer must be adamw or rank_scaled")
        if self.truncation not in {"error", "right"}:
            raise ValueError("truncation must be error or right")
        if self.lisa_state_policy not in {"drop", "cpu"}:
            raise ValueError("lisa_state_policy must be drop or cpu")
        if self.projection_scale_rule not in {"rms", "constant"}:
            raise ValueError("projection_scale_rule must be rms or constant")
        for name in (
            "max_sequence_length",
            "batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "projection_rank",
            "projection_interval",
            "lisa_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least two for causal training")
        for name in ("pad_token_id", "lisa_layers", "lora_rank", "save_every", "warmup_steps", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.seed >= 2**63:
            raise ValueError("seed must be in [0, 2**63 - 1]")
        for name in ("learning_rate", "eps", "projection_scale", "lora_alpha"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("weight_decay", "max_grad_norm", "lora_dropout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.lora_dropout >= 1:
            raise ValueError("lora_dropout must be less than one")
        if not isinstance(self.betas, (list, tuple)) or len(self.betas) != 2:
            raise ValueError("betas must contain two numbers")
        if any(
            isinstance(b, bool) or not isinstance(b, (int, float)) or not math.isfinite(b) or not 0 <= b < 1
            for b in self.betas
        ):
            raise ValueError("betas must be finite values in [0, 1)")
        self.betas = tuple(self.betas)
        if not isinstance(self.lora_targets, (list, tuple)) or not self.lora_targets:
            raise ValueError("lora_targets must be a nonempty list of module names")
        if any(not isinstance(name, str) or not name for name in self.lora_targets):
            raise ValueError("lora_targets must contain nonempty strings")
        self.lora_targets = tuple(self.lora_targets)
        if self.layer_path is not None and (not isinstance(self.layer_path, str) or not self.layer_path):
            raise ValueError("layer_path must be a nonempty module path or null")
        for name in (
            "trust_remote_code",
            "gradient_checkpointing",
            "optimizer_in_backward",
            "compile_model",
            "shuffle",
            "stochastic_rounding",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")
        if self.lora_rank and self.lisa_layers:
            raise ValueError("choose LoRA or LISA-style layer sampling; their combination is not implemented")
        if self.lisa_layers and self.lisa_state_policy == "cpu" and self.optimizer != "rank_scaled":
            raise ValueError("CPU-offloaded LISA states require rank_scaled for lazy state migration")
        if self.compile_model and self.lisa_layers:
            raise ValueError("compile_model cannot be combined with changing LISA parameter masks")
        if self.optimizer_in_backward:
            if self.optimizer != "rank_scaled":
                raise ValueError("optimizer_in_backward currently requires rank_scaled")
            if self.gradient_accumulation_steps != 1 or self.max_grad_norm != 0:
                raise ValueError("optimizer_in_backward requires accumulation=1 and max_grad_norm=0")
            if self.gradient_checkpointing or self.compile_model or self.lisa_layers:
                raise ValueError("optimizer_in_backward cannot combine with checkpointing, compile, or LISA")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resume_contract(self) -> dict[str, Any]:
        """Numerical/model contract; dataset contents are fingerprinted separately."""
        ignored = {"max_steps", "output_dir", "save_every", "train_jsonl"}
        return {key: value for key, value in self.to_dict().items() if key not in ignored}


def load_config(path: str | Path) -> TrainConfig:
    """Read JSON/YAML; local paths are relative to the configuration file."""
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as handle:
        if source.suffix.lower() == ".json":
            values = json.load(handle)
        else:
            import yaml

            values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("the experimental configuration must be a mapping")
    unknown = set(values) - {item.name for item in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"unknown experimental configuration keys: {', '.join(sorted(map(str, unknown)))}")
    values = dict(values)
    for name in ("train_jsonl", "output_dir"):
        if name in values and isinstance(values[name], str) and values[name].strip():
            candidate = Path(values[name]).expanduser()
            values[name] = str((source.parent / candidate).resolve())
    if isinstance(values.get("model"), str):
        candidate = Path(values["model"]).expanduser()
        local = source.parent / candidate
        if local.exists():
            values["model"] = str(local.resolve())
    return TrainConfig(**values)
