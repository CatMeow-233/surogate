"""Parameter counts of the P0 rows, read from IR shapes (README: P0 的行).

A0's count pins the fixed Mamba geometry; A1 must stay within 2 % of A0, tied and untied. A closed-form count per
layer kind cross-checks the IR shapes of every row.
"""

import pytest
from surogate_tlc import hf, ir

A0_TIED, A0_UNTIED = 31_782_192, 38_073_648


@pytest.fixture(scope="module")
def counts() -> dict[str, tuple[int, int]]:
    modules = {row: ir.compile_ir(hf.row_config(row)) for row in hf.ROWS}
    return {row: (ir.param_count(m, tied=True), ir.param_count(m, tied=False)) for row, m in modules.items()}


def _closed_form(cfg: dict) -> int:
    """Untied parameter count of `cfg` from the layer formulas, independent of the DSL."""
    d, ff, vocab = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    swiglu_and_norms = 3 * d * ff + 2 * d
    layer = {}
    if "M" in cfg["tlc_layer_pattern"]:
        heads, inner = cfg["mamba_num_heads"], cfg["mamba_num_heads"] * cfg["mamba_head_dim"]
        conv = inner + 2 * cfg["n_groups"] * cfg["ssm_state_size"]
        in_proj = (inner + conv + heads) * d
        # in_proj, conv weight + bias, A_log + D + dt_bias, gated norm, out_proj
        layer["M"] = in_proj + conv * cfg["conv_kernel"] + conv + 3 * heads + inner + d * inner + swiglu_and_norms
    if "A" in cfg["tlc_layer_pattern"]:
        q, kv, head = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        # fused qkv, out, q/k norms
        layer["A"] = (q + 2 * kv) * head * d + d * q * head + 2 * head + swiglu_and_norms
    return 2 * vocab * d + d + sum(layer[kind] for kind in cfg["tlc_layer_pattern"])


def test_a0_count_pins_the_fixed_geometry(counts):
    assert counts["A0"] == (A0_TIED, A0_UNTIED)


def test_a0_split_has_exactly_a0s_parameters(counts):
    assert counts["A0-split"] == counts["A0"]


@pytest.mark.parametrize("tied", [True, False], ids=["tied", "untied"])
def test_a1_matches_a0_within_two_percent(counts, tied):
    index = 0 if tied else 1
    a0, a1 = counts["A0"][index], counts["A1"][index]
    assert abs(a1 / a0 - 1) <= 0.02, (a0, a1)


@pytest.mark.parametrize("row", list(hf.ROWS))
def test_ir_count_matches_the_layer_formulas(counts, row):
    cfg = hf.row_config(row)
    tied, untied = counts[row]
    assert untied == _closed_form(cfg)
    assert untied - tied == cfg["vocab_size"] * cfg["hidden_size"]  # tying drops exactly the LM head


def test_non_embedding_count_and_flops():
    """The counts a scaling comparison uses: A0 without its embedding, and the estimated forward FLOPs per token
    (2 x matmul multiply-adds plus the sequence mixers), which grow with the context only for attention layers."""
    module = ir.compile_ir(hf.row_config("A0"))
    assert ir.param_count(module, tied=True, embedding=False) == A0_TIED - 16384 * 384 == 25_490_736
    assert ir.param_count(module, tied=False, embedding=False) == 25_490_736
    flops = {
        row: (ir.flops_per_token(hf.row_config(row), 512), ir.flops_per_token(hf.row_config(row), 2048))
        for row in ("A0", "A1", "A1h")
    }
    assert flops["A0"] == (68_419_584, 68_419_584)
    assert flops["A1"] == (68_321_280, 82_477_056)
    assert flops["A1h"][0] < flops["A0"][0] < flops["A1h"][1]
