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

from .canonical import TEMP_MARK, sha256_json

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
        "profile": None,  # row files only: a size profile (PROFILES) applied under the file's own model keys
    },
    "routing": {
        # B-O3: per-token route ids travel out of band in the shards. "tags" (routes from in-band control tags,
        # routing.tags) is deferred: a row file that asks for it is refused. Recorded in every run's config.
        "signal": "oob",
        # The gate policy and its parameters of the dual evaluation arms (B-O5). Training plans are one-hot from the
        # shard routes, so these are evaluation settings: eval.items.Routing and routing.policies take their
        # defaults from here, eval-tasks overrides them per arm (--gate, --g, --window), and a row file may not set
        # them. None = the arm preset decides ("one_hot" for Top-1 rows, "dual_const" for dual rows).
        "gate_policy": None,
        "g": (0.5, 0.5),
        "window": 32,
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
        # [建议] AdamW moments: torch's defaults, which v1.0 used too (MS §8 names no betas)
        "adam_betas": (0.9, 0.999),
        "adam_eps": 1e-8,
    },
    "data": {
        "window": "packed",  # documents separated by <eos>; "doc_aligned" is deferred
        "seed_mix": {"natural": 0.80, "track_s": 0.12, "codeswitch_parallel": 0.05, "tagged": 0.03},  # B-O7
        # [建议] the share of the codeswitch_parallel tokens taken by concatenated parallel pairs; the rest is
        # code-switched text (MS §6.1 names both and no ratio)
        "seed_mix_parallel_fraction": 0.5,
        # B-O3: a tagged D_seed document opens with the tag of its language and carries one at every language
        # switch; tags are ordinary input and target tokens. <|bridge|> stays out of D_seed unless set (MS §6.4)
        "seed_tags": {"start": True, "switch": True, "bridge": False},
        # [建议] the declared floor of the D_seed shard's mixed-window fraction at train.seq (v1.0: 2.6e-7)
        "min_mixed_window_fraction": 0.01,
        # [建议] document-level split fractions per kind (SCI-10); the seven sets are disjoint by construction
        "split_fractions": {
            "zh": {"seed_train": 0.70, "zh_train": 0.20, "calib": 0.02, "dev": 0.04, "test": 0.04},
            "en": {"seed_train": 0.70, "en_train": 0.20, "calib": 0.02, "dev": 0.04, "test": 0.04},
            "parallel": {"seed_train": 0.30, "parallel_train": 0.60, "calib": 0.02, "dev": 0.04, "test": 0.04},
            "codeswitch": {"seed_train": 0.90, "calib": 0.02, "dev": 0.04, "test": 0.04},
            "synthetic": {"seed_train": 1.0},  # Track S assigns its own sets (data.synth); this covers strays
        },
        "decontam_rules": ["en_word13", "zh_han25", "token13"],  # MS §6.1, SCI-10
    },
    "tokenizer": {
        "vocab_size": 32768,
        "initial_alphabet": "bytelevel",  # the 256-byte ByteLevel alphabet
        "normalization": "none",  # B-O11
        "special_tokens": list(SPECIAL_TOKENS),
    },
    "eval": {
        "length_norm": "auto",  # MS §9.1: sum when every candidate has the same token length, else mean
        "autocast": "none",  # [建议] evaluation in fp32 (bf16 on request); training uses bf16 autocast on CUDA
        "max_new": 160,  # [建议] greedy tokens per E4 stage (the longest test reference is about 100 BPE tokens)
        "lang_share": 0.9,  # [建议] lang_ok: the answer language's share of Han + Latin letters, names removed
        "n_boot": 10_000,  # MS §9.2: clustered bootstrap resamples, fixed seed
        "switch_k": 16,  # MS §9.1: the after-switch NLL spike covers the first 16 target tokens
        # B-O8 / MS §9.3 [建议], locked before the pilot is unblinded (MS §9.7)
        "alpha_branch": 0.0125,
        "sesoi": {"H_A": 0.05, "H_B": 0.03, "H_C": 0.05},
        "noninferiority": {"E1": 0.02, "bpb": 0.02},  # E1: absolute accuracy drop; BPB: bit/byte increase
        "g1p": {"chance": 0.125, "margin": 0.20},  # G1': best of B0, B2, B12a >= chance + 20 pp on dev
        "power": 0.8,  # MS §9.5: target power at the SESOI
        "replicates": (3, 6),  # MS §9.5: R in {3, ..., 6}
        # [建议] MS §9.6: one 5-point coefficient grid for every merge arm (task arithmetic and TIES lambda, chosen on
        # dev); TIES keeps the top 20 % of each delta (the TIES paper's k). The confirmatory dual arms keep B-O5's g.
        "merge": {"lam_grid": (0.2, 0.4, 0.6, 0.8, 1.0), "density": 0.2},
    },
    "registry": {"allow_dirty": False},
    # One run of a row file (step I5). Paths are relative to the run root (`train --root`, default the working
    # directory). Only the fields that enter the run identity (recipe and its arguments, seeds, steps, inputs by
    # hash) change a run's id; device, save/keep/log cadence do not.
    "run": {
        "name": None,  # the run name; a stage's name by default (registry: one name, one run_id)
        "recipe": None,  # seed | expert | joint_wide | bridge | target_copy | router
        "matrix_row": None,  # the MS §7 row id, e.g. "B1"
        "replicate": 0,
        "init_seed": None,  # torch.Generator seed of the model initialisation (IMPL-01)
        "data_seed": None,  # EpochChunkSampler seed
        "steps": None,  # exactly one of steps and tokens; steps = ceil(tokens / train.global_batch_tokens)
        "tokens": None,
        "shard": None,  # the training shard directory (bilingual-shard/1)
        "tokenizer": None,  # seed recipe: the tokenizer.json the shard was built with; others use the seed pack's
        "seed_pack": None,  # every recipe but seed
        "packs": {},  # name -> expert pack: frozen experts the recipe runs (Bridge (c), T1', expert context)
        "routes": {},  # shard route name -> expert name, for an expert recipe that runs frozen context experts
        "expert": None,  # expert / joint_wide: the name of the trained expert
        "target": None,  # target_copy (T1'): the expert whose copy is trained
        "source": None,  # target_copy (T1'): the frozen source-language expert
        # a tuning-ledger trial (MS §9.6), e.g. "lr-3e-4": recorded in run.json and tabulated as tuning compute by
        # `registry budget`; not part of the run identity (the trial's own settings, such as train.lr, are)
        "trial": None,
        "device": "auto",  # auto = cuda when available, else cpu
        "save_every": 200,  # [建议] state every N steps; the last step always saves
        "keep_every": 0,  # retention: the last two states, plus every state whose step is a multiple of this
        "log_every": 1,
    },
}

EXPERT_DECAYS = ("none", "to_zero", "to_seed")
TRAIN_DTYPES = ("auto", "bf16", "fp32")
BRIDGE_VARIANTS = ("a", "b", "c")
RECIPES = ("seed", "expert", "joint_wide", "bridge", "target_copy", "router")
ROW_ONLY_MODEL_KEYS = ("gqa", "profile")  # model keys of a row file that are not ModelConfig fields
DEVICES = ("auto", "cpu", "cuda")
RUN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def run_name_problem(name) -> str | None:
    """Why `name` cannot be a run name, or None. Besides the pattern: no "..", and never the temporary-file mark
    of canonical (".tmp-"): the registry deletes every entry of the runs directory that holds it as a crash
    leftover, so a run of that name would lose its directory at the next start."""
    if not isinstance(name, str) or not RUN_NAME_PATTERN.fullmatch(name):
        return f"run names must match {RUN_NAME_PATTERN.pattern}: {name!r}"
    if ".." in name:
        return f"run names may not contain '..': {name!r}"
    if TEMP_MARK in name:
        return f"run names may not contain {TEMP_MARK!r}, the mark of temporary files: {name!r}"
    return None


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
    adam_betas: tuple[float, float] = _default("train", "adam_betas")
    adam_eps: float = _default("train", "adam_eps")

    def __post_init__(self):
        object.__setattr__(self, "adam_betas", tuple(float(b) for b in self.adam_betas))
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
        if self.lr is not None and not self.lr > 0:
            raise ValueError(f"lr must be positive: {self.lr!r}")
        for name in ("weight_decay", "expert_decay_lambda", "anchor_kl", "grad_clip"):
            if not getattr(self, name) >= 0:
                raise ValueError(f"{name} must be >= 0: {getattr(self, name)!r}")
        if len(self.adam_betas) != 2 or not all(0.0 <= b < 1.0 for b in self.adam_betas) or not self.adam_eps > 0:
            raise ValueError(f"adam_betas must be two values in [0, 1) and adam_eps positive: {self.adam_betas}")

    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        out["adam_betas"] = list(self.adam_betas)
        return out


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
    values = {k: v for k, v in DEFAULTS["model"].items() if k != "family" and k not in ROW_ONLY_MODEL_KEYS}
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


def _optional_int(name: str, value) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"run.{name} must be an integer: {value!r}")
    return value


@dataclass(frozen=True)
class RunSpec:
    """The [run] table of a row config: what one training run is, where its inputs are, how often it saves.

    Paths stay strings relative to the run root; `train.loop` resolves them. `steps` and `tokens` are exclusive."""

    name: str
    recipe: str
    init_seed: int
    data_seed: int
    shard: str
    matrix_row: str | None = None
    replicate: int = 0
    steps: int | None = None
    tokens: int | None = None
    tokenizer: str | None = None
    seed_pack: str | None = None
    packs: dict[str, str] = field(default_factory=dict)
    routes: dict[str, str] = field(default_factory=dict)
    expert: str | None = None
    target: str | None = None
    source: str | None = None
    trial: str | None = None
    device: str = "auto"
    save_every: int = 200
    keep_every: int = 0
    log_every: int = 1

    def __post_init__(self):
        problem = run_name_problem(self.name)
        if problem:
            raise ValueError(f"run.name: {problem}")
        if self.recipe not in RECIPES:
            raise ValueError(f"run.recipe must be one of {RECIPES}: {self.recipe!r}")
        for name in ("init_seed", "data_seed"):
            if _optional_int(name, getattr(self, name)) is None:
                raise ValueError(f"run.{name} must be set (IMPL-01: every run records both seeds)")
        if not self.shard:
            raise ValueError("run.shard must name the training shard")
        if self.trial is not None and (not isinstance(self.trial, str) or not self.trial):
            raise ValueError(f"run.trial must be a non-empty label or unset: {self.trial!r}")
        steps, tokens = _optional_int("steps", self.steps), _optional_int("tokens", self.tokens)
        if (steps is None) == (tokens is None):
            raise ValueError("set exactly one of run.steps and run.tokens")
        if (steps is not None and steps <= 0) or (tokens is not None and tokens <= 0):
            raise ValueError("run.steps / run.tokens must be positive")
        for name in ("replicate", "save_every", "keep_every", "log_every"):
            value = _optional_int(name, getattr(self, name))
            if value is None or value < 0 or (name == "log_every" and value == 0):
                raise ValueError(f"run.{name} must be a non-negative integer (log_every positive): {value!r}")
        if self.device not in DEVICES:
            raise ValueError(f"run.device must be one of {DEVICES}: {self.device!r}")
        for name in ("expert", "target", "source"):
            value = getattr(self, name)
            if value is not None and not NAME_PATTERN.fullmatch(str(value)):
                raise ValueError(f"run.{name} must match {NAME_PATTERN.pattern}: {value!r}")
        for table in ("packs", "routes"):
            value = getattr(self, table)
            if not isinstance(value, Mapping) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in value.items()
            ):
                raise ValueError(f"run.{table} must be a table of strings: {value!r}")
            object.__setattr__(self, table, dict(value))
        bad = [n for n in [*self.packs, *self.routes.values()] if not NAME_PATTERN.fullmatch(n)]
        if bad:
            raise ValueError(f"expert names in run.packs / run.routes must match {NAME_PATTERN.pattern}: {bad}")

    def total_steps(self, train: TrainConfig) -> int:
        if self.steps is not None:
            return self.steps
        return -(-self.tokens // train.global_batch_tokens)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RunConfig:
    """One row config: DEFAULTS overlaid with the file (and, for a row file with stages, with one stage)."""

    model: ModelConfig
    train: TrainConfig
    routing: dict[str, Any]
    fusion: str
    gqa: str
    data: dict[str, Any]
    tokenizer: dict[str, Any]
    eval: dict[str, Any]
    registry: dict[str, Any]
    run: RunSpec | None = None  # the [run] table; None for a config that describes no training run
    extra: dict[str, Any] = field(default_factory=dict)  # top-level tables DEFAULTS does not own


def _overlay(section: str, values: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(values, Mapping):
        raise ValueError(f"[{section}] must be a table")
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
    raw_model = dict(data.get("model", {}))
    model = _overlay("model", raw_model)
    if model["profile"] is not None:  # DEFAULTS, then the profile, then the file's own keys
        if model["profile"] not in PROFILES:
            raise ValueError(f"unknown profile {model['profile']!r}; known: {sorted(PROFILES)}")
        model.update(PROFILES[model["profile"]])
        model.update(raw_model)
    model.pop("profile")
    gqa = model.pop("gqa")
    if gqa not in GQA_MODES:
        raise ValueError(f"model.gqa must be one of {GQA_MODES}: {gqa!r}")
    if model["family"] is None:
        raise ValueError("model.family must be set by the row config")
    cfg = ModelConfig(**model)
    fusion = _overlay("fusion", data.get("fusion", {}))["name"] or DEFAULT_FUSION[cfg.family]
    if fusion not in FUSIONS[cfg.family]:
        raise ValueError(f"fusion {fusion!r} does not belong to family {cfg.family!r}: {FUSIONS[cfg.family]}")
    raw_routing = dict(data.get("routing", {}))
    routing = _overlay("routing", raw_routing)
    routing["g"] = tuple(float(x) for x in routing["g"])
    if routing["signal"] == "tags":
        raise ValueError(
            'routing.signal = "tags" (routes from in-band control tags) is deferred; P0 trains on out-of-band '
            'shard routes, routing.signal = "oob" (B-O3)'
        )
    if routing["signal"] != "oob":
        raise ValueError(f'routing.signal must be "oob" (B-O3; "tags" is deferred): {routing["signal"]!r}')
    eval_only = sorted(k for k in ("gate_policy", "g", "window") if k in raw_routing)
    if eval_only:
        raise ValueError(
            f"routing.{', routing.'.join(eval_only)}: evaluation settings, not row settings (training plans are"
            " one-hot from the shard routes); eval-tasks sets them per arm (--gate, --g, --window), with the"
            ' defaults of DEFAULTS["routing"]'
        )
    train = TrainConfig(**_overlay("train", data.get("train", {})))
    if train.seq > cfg.max_seq:
        raise ValueError(f"train.seq = {train.seq} exceeds model.max_seq = {cfg.max_seq}")
    spec = RunSpec(**_overlay("run", data["run"])) if "run" in data else None
    return RunConfig(
        model=cfg,
        train=train,
        routing=routing,
        fusion=fusion,
        gqa=gqa,
        data=_overlay("data", data.get("data", {})),
        tokenizer=_overlay("tokenizer", data.get("tokenizer", {})),
        eval=_overlay("eval", data.get("eval", {})),
        registry=_overlay("registry", data.get("registry", {})),
        run=spec,
        extra={k: data[k] for k in extra_tables if k in data},
    )


# --- row files with stages ------------------------------------------------------------------------------------------
#
# A row file may hold top-level tables (shared by every stage) and [stage.NAME] tables. A stage table holds section
# tables ([stage.NAME.run], [stage.NAME.model], ...) that update the file's sections key by key, and optionally
# `replicates = [0, 1, 2]`: the stage then expands into NAME-r0, NAME-r1, ..., with "{r}" replaced by the replicate
# in every string, run.replicate set, and run.init_seed / run.data_seed taken from same-length lists by position.

STAGE_KEYS = ("replicates",)
PER_REPLICATE = ("init_seed", "data_seed")


def _substitute(value, r: int):
    if isinstance(value, str):
        return value.replace("{r}", str(r))
    if isinstance(value, dict):
        return {k: _substitute(v, r) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, r) for v in value]
    return value


def expand_stages(stages: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """{stage name: its section tables}, replicates expanded, in file order."""
    out: dict[str, dict[str, Any]] = {}
    for name, table in stages.items():
        if not isinstance(table, Mapping):
            raise ValueError(f"[stage.{name}] must be a table")
        sections = {k: dict(v) for k, v in table.items() if k not in STAGE_KEYS}
        bad = sorted(k for k in sections if k not in DEFAULTS)
        if bad:
            raise ValueError(f"[stage.{name}] holds unknown tables {bad}")
        replicates = table.get("replicates")
        if replicates is None:
            sections.setdefault("run", {}).setdefault("name", name)
            expanded = {name: sections}
        else:
            if not isinstance(replicates, list) or not replicates or len(set(replicates)) != len(replicates):
                raise ValueError(f"[stage.{name}] replicates must be a non-empty list of distinct integers")
            expanded = {}
            for i, r in enumerate(replicates):
                one = _substitute(sections, r)
                run = one.setdefault("run", {})
                for key in PER_REPLICATE:
                    if isinstance(run.get(key), list):
                        if len(run[key]) != len(replicates):
                            raise ValueError(f"[stage.{name}] run.{key} needs one value per replicate")
                        run[key] = run[key][i]
                run["replicate"] = r
                run.setdefault("name", f"{name}-r{r}")
                expanded[f"{name}-r{r}"] = one
        for key in expanded:
            if key in out:
                raise ValueError(f"stage {key!r} is defined twice")
        out.update(expanded)
    return out


def parse_override(text: str) -> tuple[tuple[str, ...], Any]:
    """`section.key=VALUE` (VALUE in TOML syntax, else a bare string) -> (path, value)."""
    key, sep, raw = text.partition("=")
    path = tuple(part for part in key.strip().split("."))
    if not sep or len(path) < 2 or not all(path):
        raise ValueError(f"an override is SECTION.KEY=VALUE, got {text!r}")
    try:
        value = tomllib.loads(f"v = {raw.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    return path, value


def _apply_override(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node = data
    for part in path[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"override {'.'.join(path)}: {part} is not a table")
    node[path[-1]] = value


def read_row_file(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """(the file's shared tables, its expanded stages)."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data, expand_stages(data.pop("stage", {}))


def stage_names(path: Path) -> list[str]:
    return list(read_row_file(path)[1])


def load_toml(
    path: Path,
    *,
    stage: str | None = None,
    overrides: tuple[str, ...] = (),
    extra_tables: tuple[str, ...] = (),
) -> RunConfig:
    data, stages = read_row_file(path)
    if stages and stage is None:
        raise ValueError(f"{path} defines stages {list(stages)}; choose one (train --stage NAME)")
    if stage is not None:
        if stage not in stages:
            raise ValueError(f"{path} has no stage {stage!r}; stages: {list(stages)}")
        for section, values in stages[stage].items():
            data[section] = {**data.get(section, {}), **values}
    for text in overrides:
        _apply_override(data, *parse_override(text))
    return run_config(data, extra_tables=extra_tables)
