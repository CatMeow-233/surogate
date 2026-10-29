"""Compiled TLC IR on CPU: compile a config, count its parameters, and check the graph before any GPU time.

The native graph compiler accepts several broken graphs without an error: an op input that nothing defines (a
parameter dropped by a `when=` condition), a tensor written twice (a module called twice, two layer stacks),
`moe_softmax` on a 3-D tensor (it reads the first two sizes as tokens x experts), `concat`/`mul` inputs of
different dtypes (there is no cast op). ir_problems() lists these.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

GRAPH_COMPILER = Path("csrc/src/runtime/dsl/graph_compiler.cpp")
_OP_ENTRY = re.compile(r'\{"([a-z0-9_]+)",\s*CompiledOpType::')
RANK2_OPS = ("moe_softmax",)
SAME_DTYPE_OPS = ("concat", "mul")


def compile_ir(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The IR module the trainer would build from config.json `cfg`; raises ValueError if compilation fails."""
    from surogate.dsl.py_compiler import compile_model_for_hf

    from . import register

    register()
    result = json.loads(compile_model_for_hf(cfg["architectures"][0], dict(cfg)))
    if not result["success"]:
        raise ValueError(f"DSL compilation failed: {result.get('errors')}")
    return result["modules"][0]


def native_op_names(repo: Path) -> frozenset[str]:
    """Kernel types the native graph compiler accepts: the op_type_from_string table, read from the source."""
    source = (Path(repo) / GRAPH_COMPILER).read_text(encoding="utf-8")
    start = source.index("CompiledOpType op_type_from_string(")
    return frozenset(_OP_ENTRY.findall(source, start, source.index("};", start)))


def symbol_values(config: Mapping[str, Any]) -> dict[str, int]:
    """Values of the symbolic dimensions in IR parameter shapes, as the native parameter store defines them."""
    hq, hkv, head = config.get("num_query_heads", 0), config.get("num_kv_heads", 0), config.get("head_size", 0)
    ff = config.get("d_ff", 0)
    return {
        "C": config["d_model"],
        "M": ff,
        "MUp": 2 * ff,
        "AttnDim": hq * head,
        "QKV": (hq + 2 * hkv) * head,
    }


def param_shapes(module: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Shapes of the trainable parameters, symbols resolved. Frozen tensors (RoPE tables) are left out."""
    symbols = symbol_values(module["config"])
    shapes = {}
    for name, ref in module["forward"]["params"].items():
        if ref.get("frozen"):
            continue
        try:
            shapes[name] = tuple(d if isinstance(d, int) else symbols[d] for d in ref["shape"])
        except KeyError as error:
            raise ValueError(f"{name}: no value for dimension {error} in {ref['shape']}") from None
    return shapes


EMBEDDINGS = ("embedding", "lm_head")


def param_count(module: Mapping[str, Any], *, tied: bool, embedding: bool = True) -> int:
    """Trainable parameters; with `tied` the LM head shares the embedding and does not count. embedding=False
    leaves out the embedding and the LM head (the non-embedding count that scaling comparisons use)."""
    return sum(
        math.prod(shape)
        for name, shape in param_shapes(module).items()
        if not (tied and name == "lm_head") and (embedding or name not in EMBEDDINGS)
    )


def flops_per_token(cfg: Mapping[str, Any], seq_len: int) -> int:
    """Estimated forward FLOPs per token of config.json `cfg` at context `seq_len`: 2 x the multiply-adds of every
    matmul, plus the sequence mixers (the chunked SSD scan and the depthwise conv of a Mamba layer; QK^T and AV of a
    causal attention layer, about 2 x seq_len x query width). Training costs about 3 x this. An estimate for budgets
    and MFU, not a kernel count: norms, activations and the loss are left out."""
    d, ffn, vocab = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    total = 2 * d * vocab  # LM head
    for kind in cfg["tlc_layer_pattern"]:
        total += 2 * (d * 2 * ffn + ffn * d)  # SwiGLU: fused up|gate, then down
        if kind == "M":
            heads, state, groups = cfg["mamba_num_heads"], cfg["ssm_state_size"], cfg["n_groups"]
            inner, chunk, kernel = heads * cfg["mamba_head_dim"], cfg["chunk_size"], cfg["conv_kernel"]
            total += 2 * (d * (2 * inner + 2 * groups * state + heads) + inner * d)  # in_proj, out_proj
            total += 4 * inner * state + 2 * chunk * (groups * state + inner)  # SSD states and intra-chunk terms
            total += 2 * kernel * (inner + 2 * groups * state)  # depthwise conv
        else:
            query = cfg["num_attention_heads"] * cfg["head_dim"]
            kv = cfg["num_key_value_heads"] * cfg["head_dim"]
            total += 2 * (d * query + d * 2 * kv + query * d) + 2 * seq_len * query
    return total


def ir_problems(module: Mapping[str, Any], native_ops: Collection[str]) -> list[str]:
    """Everything wrong with the forward graph of `module` that the checks above can see; [] when it is clean."""
    graph = module["forward"]
    known = {**graph["params"], **graph["inputs"]}
    ops = graph["operations"]
    produced: dict[str, int] = {}
    every_output = {name for op in ops for name in op["outputs"]}
    problems = []
    for index, op in enumerate(ops):
        kind = op["kernel_type"]
        where = f"op {index} ({kind})"
        if kind not in native_ops:
            problems.append(f"{where}: kernel type not in the native op table")
        for name in filter(None, op["inputs"]):
            if name in known or name in produced:
                continue
            if name in every_output:
                problems.append(f"{where} reads {name!r} before any op writes it")
            else:
                problems.append(f"{where} reads {name!r}, which is no parameter, graph input or op output")
        for name in filter(None, op["outputs"]):
            if name in known:
                problems.append(f"{where} overwrites {name!r}, a parameter or graph input")
            elif name in produced:
                problems.append(f"{where} writes {name!r} again (first written by op {produced[name]})")
            else:
                produced[name] = index
        if kind in RANK2_OPS:
            shape = _shape(op["inputs"][0], graph, ops, produced)
            if shape is None or len(shape) != 2:
                problems.append(f"{where} needs a rank-2 input, {op['inputs'][0]!r} has shape {shape}; view it first")
        if kind in SAME_DTYPE_OPS:
            dtypes = {d for d in (_dtype(name, module) for name in op["inputs"]) if d is not None}
            if len(dtypes) > 1:
                problems.append(f"{where} mixes input dtypes {sorted(dtypes)}; the native op needs one")
    for name in graph["outputs"]:
        if name not in produced:
            problems.append(f"graph output {name!r} is never written")
    return problems


def _shape(name: str, graph: Mapping[str, Any], ops: list, produced: Mapping[str, int]) -> list | None:
    for table in ("params", "inputs", "intermediates"):
        ref = graph[table].get(name)
        if ref and ref.get("shape"):
            return ref["shape"]
    if name in produced and ops[produced[name]]["kernel_type"] == "view":
        return ops[produced[name]]["attrs"].get("shape")
    return None


def _dtype(name: str, module: Mapping[str, Any]) -> str | None:
    graph = module["forward"]
    for table in ("params", "inputs", "intermediates"):
        ref = graph[table].get(name)
        if ref and ref.get("dtype"):
            return ref["dtype"]
    for slot in (module.get("activation_layout") or {}).get("slots", ()):
        if slot["name"] == name and slot.get("dtype"):
            return slot["dtype"]
    return None
