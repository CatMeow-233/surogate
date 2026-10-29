"""The config.json contract of P0 rows: the TLC schema, its transformers class and the checks.

The native trainer reads config.json itself and accepts several wrong configs without an error (a missing
head count, a list-valued layer pattern), and the DSL fails late on others (a missing norm epsilon), so every fork
entry point checks a config with check_config() before it builds or loads a model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from importlib import resources
from typing import Any

from .configuration_tlc import TLCConfig

__all__ = [
    "TLCConfig",
    "check_config",
    "config_problems",
    "configuration_source",
    "default_config",
    "positive_int",
    "row_config",
]

ARCHITECTURE = "TLCForCausalLM"
MODEL_TYPE = TLCConfig.model_type
CONFIGURATION_FILE = "configuration_tlc.py"
AUTO_MAP = {"AutoConfig": "configuration_tlc.TLCConfig"}
# The one context length P0 trains and checks (rows of the shards, max_position_embeddings, the tokenizer's
# model_max_length). Formal training at 2048 is an open owner decision; it changes this constant.
CONTEXT_LENGTH = 512

# tlc_layer_pattern alphabet. A GDN layer kind (a later phase) must require all four linear_* geometry keys: without
# them the GDN kernel falls back to the attention geometry and compiles anyway.
LAYER_KINDS = {"M": "mamba", "A": "attention"}
MAMBA_KEYS = ("mamba_num_heads", "mamba_head_dim", "ssm_state_size", "n_groups", "conv_kernel", "chunk_size")
ATTENTION_KEYS = ("num_attention_heads", "num_key_value_heads", "head_dim")
TOKEN_ID_KEYS = ("bos_token_id", "eos_token_id", "pad_token_id")
# true: layer 1 is traced as model-level ops ahead of the layer stack (the A0-split row). Absent means false.
SPLIT_KEY = "tlc_split_first_layer"

# The P0 Mamba-2 geometry, fixed: H=12 heads of P=64 (inner width 768 = 2 x hidden), N=64 state, G=1 group.
# N is 64 rather than 128 because the scan kernel loops over the state dimension serially.
MAMBA_GEOMETRY = {
    "mamba_num_heads": 12,
    "mamba_head_dim": 64,
    "ssm_state_size": 64,
    "n_groups": 1,
    "conv_kernel": 4,
    "chunk_size": 128,
}

# Differences of each P0 row from default_config(). A1's FFN width 1504 puts it within 0.1 % of A0's parameter
# count, tied or untied. A1h replaces layers 6 and 12 of A0 with attention layers.
ROWS: dict[str, dict[str, Any]] = {
    "A0": {},
    "A0-split": {SPLIT_KEY: True},
    "A1": {"tlc_layer_pattern": "A" * 12, "intermediate_size": 1504},
    "A1h": {"tlc_layer_pattern": "MMMMMAMMMMMA"},
}


def configuration_source() -> str:
    """Text of configuration_tlc.py, written next to config.json so that `auto_map` resolves."""
    return resources.files(__package__).joinpath(CONFIGURATION_FILE).read_text(encoding="utf-8")


def default_config(**overrides: Any) -> dict[str, Any]:
    """A0: 12 layers of width 384, SwiGLU 1024, vocabulary 16384, context CONTEXT_LENGTH, MAMBA_GEOMETRY.

    The special token ids must match the tokenizer; check_model_dir() compares them.
    """
    cfg: dict[str, Any] = {
        "architectures": [ARCHITECTURE],
        "model_type": MODEL_TYPE,
        "auto_map": dict(AUTO_MAP),
        "vocab_size": 16384,
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "intermediate_size": 1024,
        "max_position_embeddings": CONTEXT_LENGTH,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        **MAMBA_GEOMETRY,
        # Used by attention layers. A0 keeps them too: the native runtime divides by the head count when
        # head_dim is 0.
        "num_attention_heads": 6,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "rope_theta": 10000.0,
        "torch_dtype": "bfloat16",
    }
    cfg.update(overrides)
    cfg.setdefault("tlc_layer_pattern", "M" * cfg["num_hidden_layers"])
    return cfg


def row_config(row: str, **overrides: Any) -> dict[str, Any]:
    """config.json of P0 row `row` (a ROWS key); a pattern without Mamba layers drops the Mamba keys."""
    cfg = default_config(**{**ROWS[row], **overrides})
    if "M" not in cfg["tlc_layer_pattern"]:
        for key in MAMBA_KEYS:
            if key not in overrides:
                del cfg[key]
    return cfg


def positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def config_problems(cfg: Mapping[str, Any]) -> list[str]:
    """Every rule `cfg` breaks; [] for a config.json every fork entry point accepts."""
    problems = []
    if cfg.get("architectures") != [ARCHITECTURE]:
        problems.append(f"architectures must be [{ARCHITECTURE!r}], got {cfg.get('architectures')!r}")
    if cfg.get("model_type") != MODEL_TYPE:
        problems.append(f"model_type must be {MODEL_TYPE!r}, got {cfg.get('model_type')!r}")
    if cfg.get("auto_map") != AUTO_MAP:
        problems.append(f"auto_map must be {AUTO_MAP!r}, so that a process without the fork can read the config")
    for key in ("vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size"):
        if not positive_int(cfg.get(key)):
            problems.append(f"{key} must be a positive integer, got {cfg.get(key)!r}")
    context = cfg.get("max_position_embeddings")
    if not (positive_int(context) and context == CONTEXT_LENGTH):
        problems.append(f"max_position_embeddings must be the integer {CONTEXT_LENGTH}, got {context!r}")
    eps = cfg.get("rms_norm_eps")
    if not (isinstance(eps, int | float) and not isinstance(eps, bool) and 0 < eps < math.inf):
        problems.append(f"rms_norm_eps must be a positive number, got {eps!r}")
    if not isinstance(cfg.get("tie_word_embeddings"), bool):
        problems.append("tie_word_embeddings must be given as true or false")
    vocab = cfg.get("vocab_size")
    for key in TOKEN_ID_KEYS:
        value = cfg.get(key)
        if not (isinstance(value, int) and not isinstance(value, bool) and value >= 0):
            problems.append(f"{key} must be a non-negative integer, got {value!r}")
        elif positive_int(vocab) and value >= vocab:
            problems.append(f"{key}={value} is outside the vocabulary (vocab_size={vocab})")
    # The native head_size() divides by num_attention_heads when head_dim is 0, attention layers or not.
    if not (positive_int(cfg.get("num_attention_heads")) or positive_int(cfg.get("head_dim"))):
        problems.append("num_attention_heads or head_dim must be positive, even without attention layers")
    pattern = cfg.get("tlc_layer_pattern")
    if not isinstance(pattern, str):
        problems.append(f"tlc_layer_pattern must be a string (the DSL drops list attributes), got {pattern!r}")
        return problems
    if len(pattern) != cfg.get("num_hidden_layers"):
        problems.append(
            f"tlc_layer_pattern has {len(pattern)} layers, num_hidden_layers is {cfg.get('num_hidden_layers')!r}"
        )
    unknown = sorted(set(pattern) - set(LAYER_KINDS))
    if unknown:
        problems.append(f"tlc_layer_pattern has unknown layer kinds {unknown}; known: {sorted(LAYER_KINDS)}")
    split = cfg.get(SPLIT_KEY, False)
    if not isinstance(split, bool):
        problems.append(f"{SPLIT_KEY} must be true or false, got {split!r}")
    elif split and (len(pattern) < 2 or pattern[0] != "M"):
        problems.append(f"{SPLIT_KEY} needs a Mamba first layer and at least one more layer, got {pattern!r}")
    if "M" in pattern:
        problems += [
            f"{key} must be given as a positive integer" for key in MAMBA_KEYS if not positive_int(cfg.get(key))
        ]
        heads, groups = cfg.get("mamba_num_heads"), cfg.get("n_groups")
        if positive_int(heads) and positive_int(groups) and heads % groups:
            problems.append(f"mamba_num_heads={heads} is not a multiple of n_groups={groups}")
    if "A" in pattern:
        problems += [f"{key} must be a positive integer" for key in ATTENTION_KEYS if not positive_int(cfg.get(key))]
        q, kv = cfg.get("num_attention_heads"), cfg.get("num_key_value_heads")
        if positive_int(q) and positive_int(kv) and q % kv:
            problems.append(f"num_attention_heads={q} is not a multiple of num_key_value_heads={kv}")
    return problems


def check_config(cfg: Mapping[str, Any]) -> None:
    problems = config_problems(cfg)
    if problems:
        raise ValueError("config.json breaks the P0 contract:\n" + "\n".join(f"  - {p}" for p in problems))
