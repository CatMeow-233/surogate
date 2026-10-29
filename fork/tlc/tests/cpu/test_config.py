"""The config.json invariants (README: "config.json 的不变量") and the DSL reading the layer pattern as a string."""

import ast
import json

import pytest
from surogate_tlc import hf

A0 = hf.default_config()
A1H = hf.default_config(tlc_layer_pattern="MMMMMAMMMMMA")


def _without(cfg: dict, *keys: str) -> dict:
    return {k: v for k, v in cfg.items() if k not in keys}


@pytest.mark.parametrize("cfg", [A0, A1H], ids=["A0", "A1h"])
def test_valid_configs_have_no_problems(cfg):
    assert hf.config_problems(cfg) == []
    hf.check_config(cfg)


@pytest.mark.parametrize(
    "cfg, problem",
    [
        ({**A0, "architectures": ["Qwen3ForCausalLM"]}, "architectures must be"),
        ({**A0, "model_type": "qwen3"}, "model_type must be 'tlc'"),
        (_without(A0, "auto_map"), "auto_map must be"),
        ({**A0, "tlc_layer_pattern": ["M"] * 12}, "must be a string"),
        ({**A0, "tlc_layer_pattern": "M" * 11}, "tlc_layer_pattern has 11 layers"),
        ({**A0, "tlc_layer_pattern": "M" * 11 + "G"}, "unknown layer kinds ['G']"),
        (_without(A0, "ssm_state_size"), "ssm_state_size must be given"),
        ({**A0, "n_groups": 5}, "not a multiple of n_groups"),
        (_without(A1H, "num_key_value_heads"), "num_key_value_heads must be"),
        ({**A1H, "num_key_value_heads": 4}, "not a multiple of num_key_value_heads"),
        ({**A0, "max_position_embeddings": 1024}, "max_position_embeddings must be 512"),
        (_without(A0, "vocab_size"), "vocab_size must be"),
        ({**A0, "pad_token_id": 16384}, "pad_token_id=16384 is outside the vocabulary"),
        (_without(A0, "eos_token_id"), "eos_token_id must be"),
        (_without(A0, "tie_word_embeddings"), "tie_word_embeddings must be given"),
        ({**A0, "num_attention_heads": 0, "head_dim": 0}, "num_attention_heads or head_dim must be positive"),
        ({**A0, "linear_num_key_heads": 4}, "a GDN geometry needs all of"),
    ],
)
def test_each_invariant_is_enforced(cfg, problem):
    problems = hf.config_problems(cfg)
    assert any(problem in p for p in problems), problems
    with pytest.raises(ValueError, match="breaks the P0 contract"):
        hf.check_config(cfg)


def test_attention_free_a0_still_states_head_geometry():
    # A0 has no attention layer, yet the native runtime divides by the head count when head_dim is 0.
    assert "A" not in A0["tlc_layer_pattern"]
    assert hf.config_problems({**A0, "num_attention_heads": 0}) == []
    assert hf.config_problems({**A0, "head_dim": 0}) == []


def test_non_tlc_rows_follow_the_common_rules_only():
    a1 = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        **{k: A0[k] for k in ("vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size")},
        **{k: A0[k] for k in ("max_position_embeddings", "tie_word_embeddings", *hf.TOKEN_ID_KEYS)},
        "num_attention_heads": 6,
        "head_dim": 64,
    }
    assert hf.config_problems(a1) == []
    assert hf.config_problems({**a1, "max_position_embeddings": 40960}) == [
        "max_position_embeddings must be 512, got 40960"
    ]


def test_auto_map_file_is_self_contained():
    source = hf.configuration_source()
    imports = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module.split(".")[0] if node.level == 0 else ".")
    assert imports == {"transformers"}
    assert hf.AUTO_MAP == {"AutoConfig": f"{hf.CONFIGURATION_FILE.removesuffix('.py')}.{hf.TLCConfig.__name__}"}
    assert hf.TLCConfig.model_type == hf.MODEL_TYPE == "tlc"


def test_transformers_keeps_every_key():
    config = hf.TLCConfig(**A0).to_dict()
    for key, value in A0.items():
        if key != "torch_dtype":
            assert config[key] == value, key


@pytest.mark.parametrize("cfg", [A0, A1H], ids=["A0", "A1h"])
def test_ir_carries_the_layer_pattern_as_a_string(cfg):
    import surogate_tlc

    from surogate.dsl.py_compiler import compile_model_for_hf

    surogate_tlc.register()
    result = json.loads(compile_model_for_hf(hf.ARCHITECTURE, cfg))
    assert result["success"], result.get("errors")
    ir_config = result["modules"][0]["config"]
    assert ir_config["hybrid_pattern"] == cfg["tlc_layer_pattern"]
    assert ir_config["n_mamba_blocks"] == cfg["tlc_layer_pattern"].count("M")


def test_dsl_rejects_a_pattern_of_the_wrong_length():
    import surogate_tlc

    from surogate.dsl.py_compiler import compile_model_for_hf

    surogate_tlc.register()
    result = json.loads(compile_model_for_hf(hf.ARCHITECTURE, {**A0, "tlc_layer_pattern": "M" * 11}))
    assert not result["success"]
    assert "tlc_layer_pattern" in json.dumps(result["errors"])
