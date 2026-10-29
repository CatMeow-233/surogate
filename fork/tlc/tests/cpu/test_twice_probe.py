"""The "W used twice" probe of the GPU runbook (tests/gpu/twice_probe.py) compiles on CPU: one parameter feeds both
matmuls of its layer, the IR is otherwise clean, initial weights exist for it and its PyTorch gradient is finite.
It runs in a fresh interpreter because importing the probe adds a model to the DSL registry."""

import json
from pathlib import Path

PROBE = Path(__file__).resolve().parents[1] / "gpu" / "twice_probe.py"
CHECK = """
import importlib.util, json, torch
spec = importlib.util.spec_from_file_location("twice_probe", sys.argv[2])
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
from surogate_tlc import init, ir
from pathlib import Path
cfg = probe.probe_config(300, bos_token_id=1, eos_token_id=2, pad_token_id=0)
module = ir.compile_ir(cfg)
weight = f"blocks[0].{probe.FIELD}"
readers = [op["kernel_type"] for op in module["forward"]["operations"] if weight in op["inputs"]]
weights = init.init_state_dict(cfg, module, seed=0, rules={**init.RULES, probe.FIELD: "linear"})
grad = probe.reference_gradient(weights, cfg, torch.randint(0, 300, (2, 16), generator=torch.Generator().manual_seed(0)))
print(json.dumps({
    "problems": ir.ir_problems(module, ir.native_op_names(Path(sys.argv[1]))),
    "readers": readers,
    "weights": sorted(weights),
    "finite": bool(torch.isfinite(grad).all()) and float(grad.abs().max()) > 0,
}))
"""


def test_probe_compiles_and_uses_one_weight_twice(fresh_python):
    code = "sys.path.insert(0, sys.argv[1] + '/fork/tlc/src')\n" + CHECK
    report = json.loads(fresh_python(code, PROBE).splitlines()[-1])
    assert report["problems"] == [] and report["readers"] == ["matmul", "matmul"] and report["finite"]
    assert report["weights"] == [
        "blocks[0].ln1_weight",
        "blocks[0].twice_weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
    ]
