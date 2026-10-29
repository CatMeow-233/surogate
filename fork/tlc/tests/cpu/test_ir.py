"""The compiled IR of every P0 row passes the graph checks of surogate_tlc.ir (docs/model.md: DSL 模型与 IR 检查).

Covers the op table, single assignment, defined-before-use, rank-2 moe_softmax and same-dtype concat/mul, the
DSL hazards these checks exist for (two layer stacks, one module called twice, a `when=` parameter without its
model-level alias), A0-split against A0, and A1 against upstream Qwen3.
"""

import functools
from collections import Counter
from pathlib import Path

import pytest
from surogate_tlc import hf, ir

REPO = Path(__file__).resolve().parents[4]
MODEL_LEVEL = {"embedding": "model.embed_tokens.weight", "final_norm": "model.norm.weight", "lm_head": "lm_head.weight"}


@pytest.fixture(scope="module")
def native_ops() -> frozenset[str]:
    return ir.native_op_names(REPO)


@pytest.fixture(scope="module")
def rows() -> dict[str, dict]:
    return {row: ir.compile_ir(hf.row_config(row)) for row in hf.ROWS}


def _kinds(module: dict) -> Counter:
    return Counter(op["kernel_type"] for op in module["forward"]["operations"])


def _correspondence(a: dict, b: dict) -> dict[str, str]:
    """Tensor names of `a` -> `b`, for two graphs with the same ops, attributes and order; fails if inconsistent."""
    names: dict[str, str] = {}
    for index, (op, other) in enumerate(zip(a["forward"]["operations"], b["forward"]["operations"], strict=True)):
        assert (op["kernel_type"], op["attrs"]) == (other["kernel_type"], other["attrs"]), index
        for name, other_name in zip(op["inputs"] + op["outputs"], other["inputs"] + other["outputs"], strict=True):
            assert names.setdefault(name, other_name) == other_name, (index, name)
    assert len(set(names.values())) == len(names)
    return names


@pytest.mark.parametrize("row", list(hf.ROWS))
def test_every_row_passes_the_ir_checks(rows, native_ops, row):
    assert ir.ir_problems(rows[row], native_ops) == []


@pytest.mark.parametrize("row", list(hf.ROWS))
def test_model_level_weights_have_explicit_names(rows, row):
    # A module default left in hf_mapping is a literal "{prefix}..." path that no checkpoint has.
    assert rows[row]["hf_mapping"] == MODEL_LEVEL


def test_rows_have_the_layers_they_claim(rows):
    layers = {row: [_kinds(m)[k] for k in ("mamba_ssm_scan", "flash_attention", "swiglu")] for row, m in rows.items()}
    assert layers == {"A0": [12, 0, 12], "A0-split": [12, 0, 12], "A1": [0, 12, 12], "A1h": [10, 2, 12]}
    patterns = {row: m["config"].get("hybrid_pattern") for row, m in rows.items()}
    # A0-split: the stack only. A1: none, so every layer is a dense block as for upstream Qwen3.
    assert patterns == {"A0": "M" * 12, "A0-split": "M" * 11, "A1": None, "A1h": "MMMMMAMMMMMA"}
    assert {row: m["config"]["n_layers"] for row, m in rows.items()} == {"A0": 12, "A0-split": 11, "A1": 12, "A1h": 12}


def test_a0_split_is_a0_with_layer_1_outside_the_stack(rows):
    """Same ops, attributes and order as A0; A0's parameters reappear under split_param_name() with their shapes."""
    from surogate_tlc.dsl.models import FIRST_LAYER, split_param_name

    a0, split = rows["A0"], rows["A0-split"]
    names = _correspondence(a0, split)
    params = a0["forward"]["params"]
    assert {name: names[name] for name in params} == {name: split_param_name(name) for name in params}
    assert {split_param_name(name): shape for name, shape in ir.param_shapes(a0).items()} == ir.param_shapes(split)
    model_level = {name for name in split["forward"]["params"] if not name.startswith("blocks[")} - set(MODEL_LEVEL)
    assert model_level == {split_param_name(name) for name in params if name.startswith("blocks[0].")}
    assert all(name.startswith(f"{FIRST_LAYER}_") for name in model_level)
    assert split_param_name("blocks[11].A_log") == "blocks[10].A_log" and split_param_name("lm_head") == "lm_head"


def test_a1_compiles_to_the_upstream_qwen3_graph(rows):
    """A1 is Qwen3ForCausalLM at the same size: same ops and parameters, only the checkpoint names differ."""
    import surogate.dsl.models  # noqa: F401 - registers upstream Qwen3

    qwen3 = ir.compile_ir({**hf.row_config("A1"), "architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"})
    a1 = rows["A1"]
    names = _correspondence(a1, qwen3)
    assert all(names[name] == name for name in a1["forward"]["params"])
    assert a1["forward"]["params"] == qwen3["forward"]["params"]
    keys = ("d_model", "n_layers", "d_ff", "num_query_heads", "num_kv_heads", "head_size", "use_qk_norm", "rotary_dim")
    assert {k: a1["config"][k] for k in keys} == {k: qwen3["config"][k] for k in keys}
    assert "hybrid_pattern" not in qwen3["config"]


# ----------------------------------------------------------------------------- the checks on hand-made graphs


def _problems(*ops: tuple, intermediates: dict | None = None) -> list[str]:
    module = {
        "forward": {
            "inputs": {"x": {"shape": ["B", "T", "C"], "dtype": "bf16"}},
            "params": {"w": {"shape": [8, 4], "dtype": "bf16"}, "s": {"shape": [8], "dtype": "fp32"}},
            "intermediates": intermediates or {},
            "outputs": {"loss": {}},
            "operations": [
                {"kernel_type": kind, "inputs": inputs, "outputs": outputs, "attrs": attrs}
                for kind, inputs, outputs, attrs in ops
            ],
        },
    }
    return ir.ir_problems(module, {"view", "matmul", "moe_softmax", "mul", "concat"})


TO_2D = ("view", ["x"], ["x2"], {"shape": ["B * T", "C"]})


def test_a_clean_graph_has_no_problems():
    assert _problems(TO_2D, ("matmul", ["x2", "w"], ["loss"], {})) == []


@pytest.mark.parametrize(
    "ops, problem",
    [
        ([("softmax", ["x"], ["loss"], {})], "op 0 (softmax): kernel type not in the native op table"),
        ([("matmul", ["x", "conv_bias"], ["loss"], {})], "reads 'conv_bias', which is no parameter, graph input"),
        ([("matmul", ["y", "w"], ["loss"], {}), ("view", ["x"], ["y"], {})], "reads 'y' before any op writes it"),
        ([("view", ["x"], ["y"], {}), ("view", ["x"], ["y"], {})], "writes 'y' again (first written by op 0)"),
        ([("view", ["x"], ["w"], {})], "overwrites 'w', a parameter or graph input"),
        ([("view", ["x"], ["y"], {})], "graph output 'loss' is never written"),
        ([("moe_softmax", ["x"], ["loss"], {})], "needs a rank-2 input, 'x' has shape ['B', 'T', 'C']"),
        ([("matmul", ["x", "w"], ["y"], {}), ("moe_softmax", ["y"], ["loss"], {})], "'y' has shape None"),
        ([("mul", ["x", "s"], ["loss"], {})], "mixes input dtypes ['bf16', 'fp32']"),
    ],
)
def test_each_check_reports_its_problem(ops, problem):
    problems = _problems(*ops)
    assert any(problem in p for p in problems), problems


def test_rank_comes_from_views_and_dtypes_only_from_annotations():
    assert _problems(TO_2D, ("moe_softmax", ["x2"], ["loss"], {})) == []
    copy = ("view", ["x"], ["y"], {})
    assert _problems(copy, ("mul", ["x", "y"], ["loss"], {})) == []  # y has no dtype annotation: not checked
    fp32 = {"y": {"shape": [], "dtype": "fp32"}}
    assert _problems(copy, ("concat", ["x", "y"], ["loss"], {}), intermediates=fp32) == [
        "op 1 (concat) mixes input dtypes ['bf16', 'fp32']; the native op needs one"
    ]


# ----------------------------------------------------------------------------- the DSL hazards


def _compile_probe(name: str, init, forward) -> dict:
    """Register a throwaway DSL model under a TLCProbe name and compile it."""
    from surogate.dsl import nn

    cls = type(f"TLCProbe{name}", (nn.Model,), {"__init__": init, "forward": forward})
    nn.hf_config(architecture=f"TLCProbe{name}ForCausalLM", model_type=f"tlc_probe_{name.lower()}")(cls)
    return ir.compile_ir({"architectures": [f"TLCProbe{name}ForCausalLM"]})


def _probe_init(self, **modules):
    from surogate.dsl import nn
    from surogate.dsl.modules import Embedding, LMHead

    nn.Model.__init__(self)
    self.vocab_size, self.d_model, self.use_conv_bias, self.use_bias = 64, 64, True, False
    self.embedding, self.lm_head = Embedding(64, 64), LMHead(64, 64)
    for name, module in modules.items():
        setattr(self, name, module)


def _probe_io(model):
    from surogate.dsl.specs import ActivationScope

    for slot in ("token_ids", "targets"):
        model._register_activation(slot, ("B", "T"), dtype="int32", scope=ActivationScope.GLOBAL)


def test_two_layer_stacks_collide(native_ops):
    """Each stack names its layers blocks[0..], so the second overwrites the first. TLC models keep one stack."""
    from surogate_tlc.dsl.blocks import TLCMambaFFNBlock

    from surogate.dsl import nn

    mamba = dict(mamba_num_heads=4, mamba_head_dim=16, ssm_state_size=16, n_groups=1, conv_kernel=4, chunk_size=64)
    block = dict(d_model=64, d_ff=128, eps=1e-5, **mamba)

    def stack():
        return nn.HybridBlockStack([("mamba_blocks", TLCMambaFFNBlock, 2, block)], ["mamba"] * 2, 2)

    def forward(self, token_ids, position_ids, targets):
        _probe_io(self)
        x, residual = self.embedding(token_ids), self._zeros(["B", "T", "d_model"])
        x, residual = self.stack_a(x, residual, position_ids)
        x, residual = self.stack_b(x, residual, position_ids)
        return self.lm_head(x, targets)

    def init(self):
        _probe_init(self, stack_a=stack(), stack_b=stack())

    problems = ir.ir_problems(_compile_probe("TwoStacks", init, forward), native_ops)
    assert any("writes 'blocks[0].ssm_out' again" in p for p in problems), problems


def test_a_module_called_twice_reuses_its_tensor_names(native_ops):
    """What a naive shared-weight loop (D1) emits: the second call overwrites the first call's activations."""
    from surogate.dsl.modules import Mamba2Mixer

    def init(self):
        _probe_init(self, mixer=Mamba2Mixer(64, mamba_num_heads=4, mamba_head_dim=16, ssm_state_size=16, n_groups=1))

    def forward(self, token_ids, position_ids, targets):
        _probe_io(self)
        return self.lm_head(self.mixer(self.mixer(self.embedding(token_ids))), targets)

    problems = ir.ir_problems(_compile_probe("Reuse", init, forward), native_ops)
    assert any("writes 'mixer_ssm_out' again" in p for p in problems), problems


def test_a_conditional_parameter_needs_its_model_level_alias(native_ops, monkeypatch):
    """Without use_conv_bias on the model, conv_bias leaves the parameters while the graph still reads it."""
    from surogate_tlc.dsl.models import TLCModel

    init = TLCModel.__init__

    @functools.wraps(init)  # keeps the signature the compiler reads
    def without_alias(self, *args, **kwargs):
        init(self, *args, **kwargs)
        del self.use_conv_bias, self.use_bias

    monkeypatch.setattr(TLCModel, "__init__", without_alias)
    problems = ir.ir_problems(ir.compile_ir(hf.row_config("A0", num_hidden_layers=1)), native_ops)
    assert any("reads 'blocks[0].conv_bias', which is no parameter" in p for p in problems), problems
