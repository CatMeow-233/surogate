"""Typed, frozen configurations, the one table of defaults, size profiles and the config hash (docs/spec.md, "开关").

Every default of the package lives in `DEFAULTS`; the dataclass fields below read their defaults from it, so a value
is stated once. A row config (`load_toml`) overlays DEFAULTS and refuses keys DEFAULTS does not know. This module
imports only the stdlib (it never imports torch), so reading a config costs nothing.
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonical import sha256_json

# The special tokens of the v1.0 tokenizer, in their v1.0 order (B-O11): ids 0..4. Probe batches and the
# non-special vocabulary start after them.
SPECIAL_TOKENS = ("<unk>", "<eos>", "<|zh|>", "<|en|>", "<|bridge|>")
N_SPECIAL = len(SPECIAL_TOKENS)

FAMILIES = ("swiglu", "slot")
# The fusion a model applies to rows with two nonzero gates; one-hot rows always take the Top-1 path.
FUSIONS = {"swiglu": ("output", "concat"), "slot": ("pre", "post")}
DEFAULT_FUSION = {"swiglu": "output", "slot": "post"}
GQA_MODES = ("sdpa_gqa", "repeat")
NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")

DEFAULTS: dict[str, dict[str, Any]] = {
    "model": {
        "family": None,  # per row (MS §7): "swiglu" (family I) or "slot" (family II)
        "vocab_size": 32768,
        "d_model": 512,
        "n_layers": 24,
        "n_heads": 8,
        "n_kv_heads": 2,
        "ffn_width": 1728,  # f (swiglu) / k (slot), B-O4
        "slot_width": 789,  # n (slot only), B-O4; B10-II uses 395
        "expert_layers": None,  # None = every layer; LP-k = the first k and the last k layers
        "max_seq": 1024,
        "rope_theta": 10000.0,
        "norm_eps": 1e-6,
        "tie_embeddings": True,  # P0 requires True
        "gqa": "sdpa_gqa",  # execution only (not part of ModelConfig or its hash); "repeat" is the fallback
    },
    "routing": {
        "signal": "oob",  # B-O3: per-token route ids travel out of band in the shards
        "gate_policy": None,  # per row: "one_hot" for Top-1 rows, "dual_const" for dual rows (B-O5)
        "g": (0.5, 0.5),  # B-O5
        "window": 32,  # B-O5, dual_window
    },
    "fusion": {"name": None},  # per row (MS §7); None = the family's default (DEFAULT_FUSION)
    "train": {
        "global_batch_tokens": 65536,  # B-O10: must equal microbatch * seq * accum
        "microbatch": 16,
        "seq": 1024,
        "accum": 4,
        "lr": None,  # per recipe (MS §8 grid)
        "warmup_frac": 0.02,
        "min_lr_frac": 0.10,
        "weight_decay": 0.1,  # B-O9: the seed; norms and the embedding are never decayed
        "expert_decay": "none",  # B-O9: "none" | "to_zero" | "to_seed"
        "expert_decay_lambda": 0.0,
        "anchor_kl": 0.0,
        "grad_clip": 1.0,
        "dtype": "auto",  # bf16 autocast on CUDA, fp32 on CPU; parameters stay fp32
        "activation_checkpointing": False,
        "bridge_variant": "c",  # B-O13
    },
    "data": {
        "window": "packed",  # documents separated by <eos>; "doc_aligned" is deferred
        "seed_mix": {"natural": 0.80, "track_s": 0.12, "codeswitch_parallel": 0.05, "tagged": 0.03},  # B-O7
    },
    "tokenizer": {
        "vocab_size": 32768,
        "initial_alphabet": "bytelevel",  # the 256-byte ByteLevel alphabet
        "normalization": "none",  # B-O11
        "special_tokens": list(SPECIAL_TOKENS),
    },
    "eval": {"length_norm": "auto"},  # sum when every candidate has the same token length, else mean
    "registry": {"allow_dirty": False},
}

EXPERT_DECAYS = ("none", "to_zero", "to_seed")
TRAIN_DTYPES = ("auto", "bf16", "fp32")
BRIDGE_VARIANTS = ("a", "b", "c")


def _default(section: str, key: str):
    value = DEFAULTS[section][key]
    if isinstance(value, (dict, list)):
        return field(default_factory=lambda v=value: type(v)(v))
    return field(default=value)


def _expert_layers(value, n_layers: int) -> tuple[int, ...] | None:
    if value is None:
        return None
    layers = tuple(int(x) for x in value)
    if not layers or list(layers) != sorted(set(layers)) or layers[0] < 0 or layers[-1] >= n_layers:
        raise ValueError(f"expert_layers must be a sorted, duplicate-free subset of 0..{n_layers - 1}: {value!r}")
    return layers


@dataclass(frozen=True)
class ModelConfig:
    """Geometry of one model. `config_hash` covers exactly these fields."""

    family: str
    vocab_size: int = _default("model", "vocab_size")
    d_model: int = _default("model", "d_model")
    n_layers: int = _default("model", "n_layers")
    n_heads: int = _default("model", "n_heads")
    n_kv_heads: int = _default("model", "n_kv_heads")
    ffn_width: int = _default("model", "ffn_width")
    slot_width: int = _default("model", "slot_width")
    expert_layers: tuple[int, ...] | None = _default("model", "expert_layers")
    max_seq: int = _default("model", "max_seq")
    rope_theta: float = _default("model", "rope_theta")
    norm_eps: float = _default("model", "norm_eps")
    tie_embeddings: bool = _default("model", "tie_embeddings")

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}: {self.family!r}")
        for name in (
            "vocab_size",
            "d_model",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "ffn_width",
            "slot_width",
            "max_seq",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive int: {getattr(self, name)!r}")
        if self.d_model % self.n_heads or self.n_heads % self.n_kv_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model must split into n_heads even-width heads, and n_heads into n_kv_heads groups")
        if self.vocab_size <= N_SPECIAL:
            raise ValueError(f"vocab_size must exceed the {N_SPECIAL} special tokens")
        if self.tie_embeddings is not True:
            raise ValueError("P0 requires tie_embeddings = True (the embedding is the LM head)")
        object.__setattr__(self, "expert_layers", _expert_layers(self.expert_layers, self.n_layers))
        object.__setattr__(self, "rope_theta", float(self.rope_theta))
        object.__setattr__(self, "norm_eps", float(self.norm_eps))

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def layers_with_experts(self) -> tuple[int, ...]:
        return tuple(range(self.n_layers)) if self.expert_layers is None else self.expert_layers

    @property
    def shared_ffn_layers(self) -> tuple[int, ...]:
        """Layers whose FFN is the frozen seed FFN inside the backbone (LP-k only; empty otherwise)."""
        experts = set(self.layers_with_experts)
        return tuple(layer for layer in range(self.n_layers) if layer not in experts)

    def replace(self, **changes) -> ModelConfig:
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["expert_layers"] = None if self.expert_layers is None else list(self.expert_layers)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelConfig:
        unknown = set(data) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown ModelConfig keys: {sorted(unknown)}")
        return cls(**dict(data))


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings of one run (the loop lands in step I5). Validated on construction, so a run whose
    batch arithmetic does not add up is refused before anything is trained (I5-T16)."""

    global_batch_tokens: int = _default("train", "global_batch_tokens")
    microbatch: int = _default("train", "microbatch")
    seq: int = _default("train", "seq")
    accum: int = _default("train", "accum")
    lr: float | None = _default("train", "lr")
    warmup_frac: float = _default("train", "warmup_frac")
    min_lr_frac: float = _default("train", "min_lr_frac")
    weight_decay: float = _default("train", "weight_decay")
    expert_decay: str = _default("train", "expert_decay")
    expert_decay_lambda: float = _default("train", "expert_decay_lambda")
    anchor_kl: float = _default("train", "anchor_kl")
    grad_clip: float = _default("train", "grad_clip")
    dtype: str = _default("train", "dtype")
    activation_checkpointing: bool = _default("train", "activation_checkpointing")
    bridge_variant: str = _default("train", "bridge_variant")

    def __post_init__(self):
        if self.microbatch * self.seq * self.accum != self.global_batch_tokens:
            raise ValueError(
                f"microbatch x seq x accum = {self.microbatch} x {self.seq} x {self.accum} = "
                f"{self.microbatch * self.seq * self.accum} != global_batch_tokens = {self.global_batch_tokens}"
            )
        if self.expert_decay not in EXPERT_DECAYS:
            raise ValueError(f"expert_decay must be one of {EXPERT_DECAYS}: {self.expert_decay!r}")
        if self.dtype not in TRAIN_DTYPES:
            raise ValueError(f"dtype must be one of {TRAIN_DTYPES}: {self.dtype!r}")
        if self.bridge_variant not in BRIDGE_VARIANTS:
            raise ValueError(f"bridge_variant must be one of {BRIDGE_VARIANTS}: {self.bridge_variant!r}")
        if not 0.0 <= self.warmup_frac < 1.0 or not 0.0 <= self.min_lr_frac <= 1.0:
            raise ValueError("warmup_frac must lie in [0, 1) and min_lr_frac in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# Size profiles: overrides of DEFAULTS["model"]. slot_width follows one rule everywhere, the largest n whose family-II
# layer (Q, b, R, W) has no more parameters than the SwiGLU layer of the same d and f: n = floor(f (2d - 1) / (d + f)).
PROFILES: dict[str, dict[str, Any]] = {
    "tiny": {  # fp64 unit tests
        "vocab_size": 64, "d_model": 32, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
        "ffn_width": 48, "slot_width": 37, "max_seq": 16,
    },
    "smoke": {  # the v1.0 smoke geometry (moe_pilot.cfg_for_profile("smoke", 256))
        "vocab_size": 256, "d_model": 64, "n_layers": 2, "n_heads": 4, "n_kv_heads": 2,
        "ffn_width": 128, "slot_width": 84, "max_seq": 128,
    },
    "mech8": {"n_layers": 8},  # the 8-layer mechanism pilot of G1' (MS §13)
    "pilot": {},  # the pilot geometry of MS §5.1
}  # fmt: skip


def matched_slot_width(d_model: int, ffn_width: int) -> int:
    return ffn_width * (2 * d_model - 1) // (d_model + ffn_width)


def profile(name: str, family: str, **overrides) -> ModelConfig:
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; known: {sorted(PROFILES)}")
    values = {k: v for k, v in DEFAULTS["model"].items() if k not in ("family", "gqa")}
    values.update(PROFILES[name])
    values.update(overrides)
    return ModelConfig(family=family, **values)


def config_hash(cfg: ModelConfig | TrainConfig | Mapping) -> str:
    """sha256 (hex) of the canonical JSON of the config's fields, tagged with its kind."""
    if isinstance(cfg, (ModelConfig, TrainConfig)):
        return sha256_json({"kind": type(cfg).__name__, "fields": cfg.to_dict()})
    if isinstance(cfg, Mapping):
        return sha256_json({"kind": "mapping", "fields": dict(cfg)})
    raise TypeError(f"cannot hash {type(cfg).__name__}")


@dataclass(frozen=True)
class RunConfig:
    """One row config read from TOML: DEFAULTS overlaid with the file. The stage tables of row files and the run
    identity fields (seeds, matrix row) land with the training loop in step I5."""

    model: ModelConfig
    train: TrainConfig
    routing: dict[str, Any]
    fusion: str
    gqa: str
    data: dict[str, Any]
    tokenizer: dict[str, Any]
    eval: dict[str, Any]
    registry: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)  # top-level tables DEFAULTS does not own (I5 stages)


def _overlay(section: str, values: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(values) - set(DEFAULTS[section])
    if unknown:
        raise ValueError(f"unknown keys in [{section}]: {sorted(unknown)}")
    merged = {k: (type(v)(v) if isinstance(v, (dict, list)) else v) for k, v in DEFAULTS[section].items()}
    merged.update(values)
    return merged


def run_config(data: Mapping[str, Any], *, extra_tables: tuple[str, ...] = ()) -> RunConfig:
    unknown = set(data) - set(DEFAULTS) - set(extra_tables)
    if unknown:
        raise ValueError(f"unknown top-level tables: {sorted(unknown)}")
    model = _overlay("model", data.get("model", {}))
    gqa = model.pop("gqa")
    if gqa not in GQA_MODES:
        raise ValueError(f"model.gqa must be one of {GQA_MODES}: {gqa!r}")
    if model["family"] is None:
        raise ValueError("model.family must be set by the row config")
    cfg = ModelConfig(**model)
    fusion = _overlay("fusion", data.get("fusion", {}))["name"] or DEFAULT_FUSION[cfg.family]
    if fusion not in FUSIONS[cfg.family]:
        raise ValueError(f"fusion {fusion!r} does not belong to family {cfg.family!r}: {FUSIONS[cfg.family]}")
    routing = _overlay("routing", data.get("routing", {}))
    routing["g"] = tuple(float(x) for x in routing["g"])
    return RunConfig(
        model=cfg,
        train=TrainConfig(**_overlay("train", data.get("train", {}))),
        routing=routing,
        fusion=fusion,
        gqa=gqa,
        data=_overlay("data", data.get("data", {})),
        tokenizer=_overlay("tokenizer", data.get("tokenizer", {})),
        eval=_overlay("eval", data.get("eval", {})),
        registry=_overlay("registry", data.get("registry", {})),
        extra={k: data[k] for k in extra_tables if k in data},
    )


def load_toml(path: Path, *, extra_tables: tuple[str, ...] = ()) -> RunConfig:
    with open(path, "rb") as f:
        return run_config(tomllib.load(f), extra_tables=extra_tables)
