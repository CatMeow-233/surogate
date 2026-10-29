"""From-scratch weights for a TLC model directory: model.safetensors under the names the native loader reads.

The native trainer can initialise a model itself, but by parameter name only (bias -> 0, norm -> 1, everything
else normal). Mamba-2 would start with dt_bias = 0 (dt = softplus(0) = 0.69 in every head), A_log near 0 (A = -1
in every head) and D near 0: all heads decay alike and the skip path is off. This module writes the reference
initialisation instead.

Scheme, for every P0 row, so that A0, A1 and A1h start alike. Sources: state-spaces/mamba (main at e9594ce1),
mamba_ssm/modules/mamba2.py Mamba2.__init__ and mamba_ssm/models/mixer_seq_simple.py _init_weights; PyTorch's
Linear and Conv1d reset_parameters (kaiming_uniform_ with a=sqrt(5), i.e. U(+-1/sqrt(fan_in))).

- dt_bias = softplus^-1(dt), dt = exp(U(ln 1e-3, ln 1e-1)) floored at 1e-4; A_log = ln U(1, 16); D = 1.
- Mamba conv weight and bias: U(+-1/sqrt(K)), the Conv1d default for a depthwise conv (fan_in = K).
- Linear weights: U(+-1/sqrt(fan_in)). The projections that write into the residual stream (Mamba out_proj,
  attention output, MLP down) are divided by sqrt(2 * num_hidden_layers): one mixer and one MLP write per layer,
  as MixerModel passes n_residuals_per_layer=2 when layers have an MLP. MambaLMHeadModel applies _init_weights a
  second time with the default of 1, which leaves sqrt(num_hidden_layers) in that model; TLC keeps 2 per layer.
- Embedding N(0, 0.02); an untied LM head is a Linear; norm weights 1.

Each tensor draws from its own generator, seeded from (seed, layer, field), so the same seed gives A0 and A0-split
the same weights under their two naming schemes (gate G1(d)).

The native AdamW decays every parameter by name alike (weight_decay applies to A_log, dt_bias and D too, which the
reference excludes), so the P0 run configuration trains without weight decay.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from . import hf, modeldir
from .dsl.models import layer_param
from .ir import compile_ir, param_shapes

DT_MIN, DT_MAX, DT_INIT_FLOOR = 1e-3, 1e-1, 1e-4
A_INIT_RANGE = (1.0, 16.0)
EMBEDDING_STD = 0.02
RESIDUAL_WRITES_PER_LAYER = 2
# Written into the safetensors metadata; bump it when the scheme changes.
INIT_VERSION = 1

# Rule per field (the per-layer parameter name, dsl.models.layer_param). A field without a rule is an error, so a
# parameter added later gets a deliberate initialisation.
RULES = {
    "embedding": "embedding",
    "lm_head": "linear",
    "in_proj_weight": "linear",
    "qkv_weight": "linear",
    "mlp_up_weight": "linear",
    "out_proj_weight": "residual",
    "out_weight": "residual",
    "mlp_down_weight": "residual",
    "conv_weight": "conv",
    "conv_bias": "conv",
    "A_log": "a_log",
    "dt_bias": "dt_bias",
    "D_param": "one",
    "final_norm": "one",
    "ln1_weight": "one",
    "ln2_weight": "one",
    "gated_norm_weight": "one",
    "q_norm_weight": "one",
    "k_norm_weight": "one",
}
_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def _generator(seed: int, layer: int | None, field: str) -> torch.Generator:
    digest = hashlib.sha256(f"{seed}/{layer}/{field}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") >> 1)


def _value(rule: str, shape: tuple[int, ...], cfg: Mapping[str, Any], generator: torch.Generator) -> torch.Tensor:
    t = torch.empty(shape, dtype=torch.float32)
    if rule == "one":
        return t.fill_(1.0)
    if rule == "embedding":
        return t.normal_(0.0, EMBEDDING_STD, generator=generator)
    if rule == "a_log":
        return t.uniform_(*A_INIT_RANGE, generator=generator).log_()
    if rule == "dt_bias":
        dt = t.uniform_(math.log(DT_MIN), math.log(DT_MAX), generator=generator).exp_().clamp_(min=DT_INIT_FLOOR)
        return dt + torch.log(-torch.expm1(-dt))  # softplus^-1
    if rule == "conv":
        bound = cfg["conv_kernel"] ** -0.5
    elif rule == "linear":
        bound = shape[1] ** -0.5
    elif rule == "residual":
        bound = shape[1] ** -0.5 / math.sqrt(RESIDUAL_WRITES_PER_LAYER * cfg["num_hidden_layers"])
    else:
        raise ValueError(f"unknown initialisation rule {rule!r}")
    return t.uniform_(-bound, bound, generator=generator)


def init_state_dict(
    cfg: Mapping[str, Any], module: Mapping[str, Any] | None = None, *, seed: int, rules: Mapping[str, str] = RULES
) -> dict[str, torch.Tensor]:
    """Initial weights of config.json `cfg` (its compiled IR `module`), keyed by the name the native loader reads.

    That name is the IR's hf_mapping entry, or the IR parameter name itself. Frozen tensors (RoPE tables) are left
    out because the runtime computes them; so is a tied LM head, which the loader fills from the embedding. Each
    tensor has the dtype of its IR parameter. `rules` maps each field to its rule (a probe model adds its own).
    """
    module = module if module is not None else compile_ir(cfg)
    params = module["forward"]["params"]
    names = module.get("hf_mapping") or {}
    split = bool(cfg.get(hf.SPLIT_KEY, False))
    out = {}
    for name, shape in param_shapes(module).items():
        if name == "lm_head" and cfg["tie_word_embeddings"]:
            continue
        layer, field = layer_param(name, split)
        if field not in rules:
            raise ValueError(f"{name}: no initialisation rule for {field!r}; add one to RULES")
        key = names.get(name, name)
        if not isinstance(key, str):
            raise ValueError(f"{name}: hf_mapping {key!r} is not a plain tensor name")
        value = _value(rules[field], shape, cfg, _generator(seed, layer, field))
        out[key] = value.to(_DTYPES[params[name]["dtype"]])
    return out


def init_model_dir(path: Path) -> Path:
    """Write model.safetensors into `path`, a directory from modeldir.write_model_dir(..., init_seed=...).

    The seed is tlc_meta.json's init_seed. An existing weights file is never replaced.
    """
    path = Path(path)
    cfg = modeldir.check_model_dir(path, weights=False)
    weights = path / modeldir.WEIGHTS_FILE
    if weights.exists():
        raise FileExistsError(f"{weights} exists; initialise only a model directory without weights")
    seed = json.loads((path / modeldir.META_FILE).read_text(encoding="utf-8")).get("init_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{path / modeldir.META_FILE}: init_seed must be an integer, got {seed!r}")
    metadata = {"format": "pt", "tlc_init_version": str(INIT_VERSION), "tlc_init_seed": str(seed)}
    tmp = weights.with_name(f".{weights.name}.tmp")
    save_file(init_state_dict(cfg, seed=seed), tmp, metadata=metadata)
    os.replace(tmp, weights)
    modeldir.check_model_dir(path)
    return weights
