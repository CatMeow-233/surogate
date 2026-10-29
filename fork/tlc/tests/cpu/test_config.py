"""The config.json invariants (docs/model.md: config.json 的不变量) and the DSL reading the layer pattern as a string."""

import ast
import json

import pytest
from surogate_tlc import hf

A0 = hf.row_config("A0")
A1H = hf.row_config("A1h")


def _without(cfg: dict, *keys: str) -> dict:
    return {k: v for k, v in cfg.items() if k not in keys}


@pytest.mark.parametrize("row", list(hf.ROWS))
def test_valid_configs_have_no_problems(row):
    cfg = hf.row_config(row)
    assert hf.config_problems(cfg) == []
    hf.check_config(cfg)


def test_rows_state_their_geometry_explicitly():
    assert A0 == hf.default_config() and A0["tlc_layer_pattern"] == "M" * 12
    for row in ("A0", "A0-split", "A1h"):
        assert hf.MAMBA_GEOMETRY.items() <= hf.row_config(row).items(), row
    # Without Mamba layers the Mamba keys would only be noise in config.json, unless asked for.
    assert not set(hf.MAMBA_KEYS) & set(hf.row_config("A1"))
    assert hf.row_config("A1", n_groups=1)["n_groups"] == 1


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
        ({**A0, "max_position_embeddings": 1024}, "max_position_embeddings must be the integer 512"),
        ({**A0, "max_position_embeddings": 512.0}, "max_position_embeddings must be the integer 512"),
        ({**A0, "max_position_embeddings": True}, "max_position_embeddings must be the integer 512"),
        (_without(A0, "rms_norm_eps"), "rms_norm_eps must be a positive number"),
        ({**A0, "rms_norm_eps": "x"}, "rms_norm_eps must be a positive number"),
        ({**A0, "rms_norm_eps": 0.0}, "rms_norm_eps must be a positive number"),
        ({**A0, "rms_norm_eps": float("nan")}, "rms_norm_eps must be a positive number"),
        ({**A0, "rms_norm_eps": float("inf")}, "rms_norm_eps must be a positive number"),
        ({**A0, "rms_norm_eps": True}, "rms_norm_eps must be a positive number"),
        (_without(A0, "vocab_size"), "vocab_size must be"),
        *[
            ({**A0, key: 0}, f"{key} must be a positive integer")
            for key in ("vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size")
        ],
        *[
            ({**A0, key: True}, f"{key} must be a non-negative integer")
            for key in ("bos_token_id", "eos_token_id", "pad_token_id")
        ],
        ({**A0, "eos_token_id": -1}, "eos_token_id must be a non-negative integer"),
        ({**A0, "tie_word_embeddings": 1}, "tie_word_embeddings must be given"),
        ({**A0, "pad_token_id": 16384}, "pad_token_id=16384 is outside the vocabulary"),
        (_without(A0, "eos_token_id"), "eos_token_id must be"),
        (_without(A0, "tie_word_embeddings"), "tie_word_embeddings must be given"),
        ({**A0, "num_attention_heads": 0, "head_dim": 0}, "num_attention_heads or head_dim must be positive"),
        ({**A0, hf.SPLIT_KEY: 1}, "tlc_split_first_layer must be true or false"),
        ({**A1H, "tlc_layer_pattern": "A" * 12, hf.SPLIT_KEY: True}, "needs a Mamba first layer"),
        (hf.default_config(num_hidden_layers=1, **{hf.SPLIT_KEY: True}), "and at least one more layer"),
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


@pytest.mark.parametrize(
    "cfg, error",
    [
        ({**A0, "tlc_layer_pattern": "M" * 11}, "tlc_layer_pattern"),
        ({**A1H, "tlc_layer_pattern": "A" + "M" * 11, hf.SPLIT_KEY: True}, hf.SPLIT_KEY),
    ],
)
def test_dsl_rejects_what_check_config_rejects(cfg, error):
    import surogate_tlc

    from surogate.dsl.py_compiler import compile_model_for_hf

    surogate_tlc.register()
    result = json.loads(compile_model_for_hf(hf.ARCHITECTURE, cfg))
    assert not result["success"]
    assert error in json.dumps(result["errors"])
