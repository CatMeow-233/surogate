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
    return {
        "C": config["d_model"],
        "M": config["d_ff"],
        "MUp": 2 * config["d_ff"],
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


def param_count(module: Mapping[str, Any], *, tied: bool) -> int:
    """Trainable parameters; with `tied` the LM head shares the embedding and does not count."""
    return sum(math.prod(shape) for name, shape in param_shapes(module).items() if not (tied and name == "lm_head"))


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
