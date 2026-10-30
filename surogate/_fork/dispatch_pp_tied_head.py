"""Refuse dispatch-PP full training that would lose half of a tied gradient (ledger topic T10).

Dispatch-PP reads each stage's gradients from the GPU that ran the stage
(`dispatch_pp_train_step_multigpu` in csrc/src/binding/py_train.cpp: stage s
runs on GPU s % gpus, `embedding` is read with stage 0 and `lm_head` with the
loss stage). `dispatch_pp_write_grads` (dsl_model_execution.cpp) then copies
them into GPU 0's gradient store one name at a time. Under a tied head both
names are one buffer (DslGradStore), so the loss stage's copy, written last,
is the one the optimizer sees. A GPU's buffer holds only the gradient parts
computed on that GPU: when stage 0 and the loss stage share a GPU, the copy
holds the sum of both parts and the update is right. Otherwise it holds only
the head's part, and the embedding lookup's part is lost without a message.
Summing the copies across GPUs would fix it inside the native step; until
then the trainers refuse the plans that lose it. LoRA copies only adapter
gradients and is unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path

STAGE_BLOCKS_ENV = "SUROGATE_DISPATCH_STAGE_BLOCKS"


def _json_bool(value) -> bool | None:
    """as_bool in pretrained_config.cpp: a JSON boolean, an integer, or the strings true/false/1/0."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        if value.lower() == "true" or value == "1":
            return True
        if value.lower() == "false" or value == "0":
            return False
    return None


def native_tie_word_embeddings(config_json: dict, ir: dict) -> bool:
    """PretrainedConfig.TiedWordEmbeddings as the native trainer resolves it for this config.json and IR.

    load_pretrained_config (csrc/src/config/pretrained_config.cpp) starts from false, sets it for LFM2, then takes a
    top-level tie_word_embeddings (which marks it specified) and a top-level tie_embeddings. DslModel's
    apply_declared_config_defaults (dsl_model.cpp) gives a value that is not specified the model module's declared
    tie_word_embeddings, when that is a boolean.
    """
    architectures = config_json.get("architectures")
    architecture = architectures[0] if isinstance(architectures, list) and architectures else None
    tied = config_json.get("model_type") == "lfm2" or architecture == "Lfm2ForCausalLM"
    specified = False
    if (value := _json_bool(config_json.get("tie_word_embeddings"))) is not None:
        tied, specified = value, True
    if (value := _json_bool(config_json.get("tie_embeddings"))) is not None:
        tied = value
    if not specified:
        models = [module for module in ir.get("modules", []) if module.get("kind") == "model"]
        declared = models[0].get("config", {}).get("tie_word_embeddings") if len(models) == 1 else None
        if isinstance(declared, bool):
            tied = declared
    return tied


def loses_tied_gradient(*, tied: bool, lora: bool, num_stages: int, gpus: int) -> bool:
    """Full training of a tied model with stage 0 (GPU 0) and the loss stage on different GPUs."""
    return tied and not lora and (num_stages - 1) % gpus != 0


def _alternatives(num_layers: int, num_stages: int, stage_blocks: int, gpus: int) -> list[str]:
    """Plans that put stage 0 and the loss stage on GPU 0: up to three stage counts, each at the stage size nearest
    the current one, then the most GPUs that work at the current stage size, then LoRA.

    A stage size that gives at least `gpus` stages is used as is by plan_stages (surogate/train/dispatch_pp.py), so
    these need no other setting. Fewer GPUs at the current stage size keep the same stages.
    """
    nearest: dict[int, int] = {}  # stage count -> stage size
    for size in range(1, num_layers + 1):
        stages = -(-num_layers // size)
        if stages >= gpus and (stages - 1) % gpus == 0:
            if stages not in nearest or abs(size - stage_blocks) < abs(nearest[stages] - stage_blocks):
                nearest[stages] = size
    sizes = sorted(sorted(nearest.values(), key=lambda size: abs(size - stage_blocks))[:3])
    options = [f"{STAGE_BLOCKS_ENV}={size} ({-(-num_layers // size)} stages)" for size in sizes]
    fewer = [count for count in range(gpus - 1, 1, -1) if (num_stages - 1) % count == 0]
    if fewer:
        options.append(f"{fewer[0]} GPUs with {STAGE_BLOCKS_ENV}={stage_blocks}")
    options.append("lora: true")
    return options


def refuse_lost_tied_gradient(config, ir_json: str) -> None:
    """Raise ValueError when dispatch-PP full training would drop the embedding lookup's part of a tied gradient.

    `config` is the trainer config (parallelism, lora, gpus, model_dir, model_info); `ir_json` the model's DSL IR.
    """
    if getattr(config, "parallelism", None) != "dispatch_pp" or config.lora:
        return
    from surogate.train.dispatch_pp import plan_stages

    _, _, num_layers, num_stages, _, stage_blocks = plan_stages(config.model_info.config, config.gpus)
    if not loses_tied_gradient(tied=True, lora=False, num_stages=num_stages, gpus=config.gpus):
        return
    config_json = json.loads((Path(config.model_dir) / "config.json").read_text(encoding="utf-8"))
    if not native_tie_word_embeddings(config_json, json.loads(ir_json)):
        return
    options = "; ".join(_alternatives(num_layers, num_stages, stage_blocks, config.gpus))
    raise ValueError(
        f"parallelism=dispatch_pp with lora: false would lose part of the tied lm_head/embedding gradient: "
        f"{num_stages} stages of {stage_blocks} layers on {config.gpus} GPUs run stage 0 on GPU 0 and the loss stage "
        f"on GPU {(num_stages - 1) % config.gpus}, and the step keeps only the loss stage's copy of the shared "
        f"gradient, so the embedding lookup's part never reaches the update (fork ledger T10, "
        f"fork/docs/upstream-errata.md). Use one of: {options}."
    )
