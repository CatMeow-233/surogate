"""Initial weights (surogate_tlc.init, docs/model.md: 初始化): what model.safetensors holds and how it is distributed.

The expected tensor names are re-derived here from the per-layer parameter names of one-layer models and the
model-level mappings, not from init's own naming, so a renamed or added parameter shows up as a mismatch.
"""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file
from surogate_tlc import hf, init, ir, modeldir
from surogate_tlc.dsl import TLCModel
from surogate_tlc.dsl.models import FIRST_LAYER, split_param_name

SEED = 3
FP32_FIELDS = {"A_log", "D_param", "dt_bias"}


@pytest.fixture(scope="module")
def layer_fields() -> dict[str, set[str]]:
    """Trainable per-layer parameter names of each layer kind, from one-layer models."""
    fields = {}
    for kind, row in (("M", "A0"), ("A", "A1")):
        module = ir.compile_ir(hf.row_config(row, num_hidden_layers=1, tlc_layer_pattern=kind))
        params = module["forward"]["params"]
        fields[kind] = {n.removeprefix("blocks[0].") for n, p in params.items() if "[" in n and not p.get("frozen")}
    return fields


@pytest.fixture(scope="module")
def weights() -> dict[str, dict[str, torch.Tensor]]:
    """init_state_dict of every row (tied) and of A0 untied, all with SEED."""
    out = {row: init.init_state_dict(hf.row_config(row), seed=SEED) for row in hf.ROWS}
    out["A0-untied"] = init.init_state_dict(hf.row_config("A0", tie_word_embeddings=False), seed=SEED)
    return out


def _config(name: str) -> dict:
    return hf.row_config("A0", tie_word_embeddings=False) if name == "A0-untied" else hf.row_config(name)


def _expected_names(cfg: dict, fields: dict[str, set[str]]) -> set[str]:
    model = TLCModel._hf_block_mappings_
    names = {model["embedding"], model["final_norm"]} | (set() if cfg["tie_word_embeddings"] else {model["lm_head"]})
    split = int(cfg.get(hf.SPLIT_KEY, False))
    for layer, kind in enumerate(cfg["tlc_layer_pattern"]):
        prefix = f"{FIRST_LAYER}_" if split and layer == 0 else f"blocks[{layer - split}]."
        names |= {prefix + field for field in fields[kind]}
    return names


@pytest.mark.parametrize("name", [*hf.ROWS, "A0-untied"])
def test_names_are_what_the_native_loader_reads(weights, layer_fields, name):
    assert set(weights[name]) == _expected_names(_config(name), layer_fields)


@pytest.mark.parametrize("name", [*hf.ROWS, "A0-untied"])
def test_shapes_and_dtypes_follow_the_ir(weights, name):
    module = ir.compile_ir(_config(name))
    to_key = {n: (module["hf_mapping"] or {}).get(n, n) for n in module["forward"]["params"]}
    shapes = {to_key[n]: shape for n, shape in ir.param_shapes(module).items()}
    for key, tensor in weights[name].items():
        assert tuple(tensor.shape) == shapes[key], key
        fp32 = key.rsplit(".", 1)[-1].removeprefix(f"{FIRST_LAYER}_") in FP32_FIELDS
        assert tensor.dtype == (torch.float32 if fp32 else torch.bfloat16), key


def _gather(state: dict[str, torch.Tensor], field: str) -> torch.Tensor:
    tensors = [t.float().flatten() for k, t in state.items() if k.endswith(f".{field}")]
    assert tensors, field
    return torch.cat(tensors)


def _check_uniform(values: torch.Tensor, bound: float, what: str) -> None:
    # bf16 rounding may land one ulp beyond the bound
    assert values.abs().max() <= bound * (1 + 2**-7), what
    assert abs(values.std().item() / (bound / math.sqrt(3)) - 1) < 0.05, what
    assert abs(values.mean().item()) < 0.05 * bound, what


@pytest.mark.parametrize("row", ["A0", "A1"])
def test_statistics_follow_the_reference_scheme(weights, row):
    state, cfg = weights[row], hf.row_config(row)
    d, ff, layers = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    residual = math.sqrt(2 * layers)
    uniform = {"mlp_up_weight": d**-0.5, "mlp_down_weight": ff**-0.5 / residual}
    ones = ["ln1_weight", "ln2_weight"]
    if row == "A0":
        inner = cfg["mamba_num_heads"] * cfg["mamba_head_dim"]
        kernel = cfg["conv_kernel"] ** -0.5
        uniform |= {"in_proj_weight": d**-0.5, "out_proj_weight": inner**-0.5 / residual}
        uniform |= {"conv_weight": kernel, "conv_bias": kernel}
        ones += ["gated_norm_weight", "D_param"]
    else:
        attention = cfg["num_attention_heads"] * cfg["head_dim"]
        uniform |= {"qkv_weight": d**-0.5, "out_weight": attention**-0.5 / residual}
        ones += ["q_norm_weight", "k_norm_weight"]
    for field, bound in uniform.items():
        _check_uniform(_gather(state, field), bound, field)
    for field in ones:
        assert torch.all(_gather(state, field) == 1), field
    assert torch.all(state["model.norm.weight"].float() == 1)
    embedding = state["model.embed_tokens.weight"].float()
    assert abs(embedding.std().item() / init.EMBEDDING_STD - 1) < 0.02
    assert abs(embedding.mean().item()) < 1e-3

    if row == "A0":
        a = _gather(state, "A_log").exp()
        assert a.min() >= 1 and a.max() <= 16 * (1 + 1e-6) and abs(a.mean().item() - 8.5) < 1.5
        dt = F.softplus(_gather(state, "dt_bias").double())
        assert dt.min() >= init.DT_MIN * (1 - 1e-5) and dt.max() <= init.DT_MAX * (1 + 1e-5)
        assert abs(dt.log().mean().item() - math.log(1e-2)) < 0.5  # log-uniform over [1e-3, 1e-1]
        # every layer draws its own heads
        assert not torch.equal(state["blocks[0].A_log"], state["blocks[1].A_log"])
        assert not torch.equal(state["blocks[0].dt_bias"], state["blocks[1].dt_bias"])


def test_untied_head_is_a_linear(weights):
    head = weights["A0-untied"]["lm_head.weight"].float()
    _check_uniform(head.flatten(), hf.default_config()["hidden_size"] ** -0.5, "lm_head")


def test_seed_decides_and_a0_split_gets_a0s_weights(weights):
    a0, split = weights["A0"], weights["A0-split"]
    again = init.init_state_dict(hf.row_config("A0"), seed=SEED)
    assert all(torch.equal(again[k], a0[k]) for k in a0)
    other = init.init_state_dict(hf.row_config("A0"), seed=SEED + 1)
    constant = {k for k, t in a0.items() if torch.all(t == 1)}
    assert all(not torch.equal(other[k], a0[k]) for k in set(a0) - constant)
    # one seed, one weight set: A0-split holds A0's tensors under its own names (gate G1(d))
    assert set(split) == {split_param_name(k) for k in a0}
    assert all(torch.equal(split[split_param_name(k)], a0[k]) for k in a0)


def test_unknown_parameter_is_refused():
    cfg = hf.row_config("A0", num_hidden_layers=1, tlc_layer_pattern="M")
    module = ir.compile_ir(cfg)
    module["forward"]["params"]["blocks[0].mystery_weight"] = {"shape": [4, 4], "dtype": "bf16"}
    with pytest.raises(ValueError, match="mystery_weight"):
        init.init_state_dict(cfg, module, seed=0)


def test_init_model_dir_completes_the_directory(tmp_path, tokenizer_dir):
    """The weights come from the recorded seed (SEED, not 0: a seed dropped on the way would give seed 0's)."""
    model_dir = modeldir.write_model_dir(tmp_path / "model", hf.default_config(), tokenizer_dir, init_seed=SEED)
    weights_file = init.init_model_dir(model_dir)

    assert weights_file == model_dir / modeldir.WEIGHTS_FILE
    cfg = modeldir.check_model_dir(model_dir)
    with safe_open(weights_file, framework="pt") as f:
        assert f.metadata() == {"format": "pt", "tlc_init_version": str(init.INIT_VERSION), "tlc_init_seed": str(SEED)}
    expected = init.init_state_dict(cfg, seed=SEED)
    stored = load_file(weights_file)
    assert set(stored) == set(expected)
    assert all(torch.equal(stored[k], expected[k]) for k in expected)
    assert not list(model_dir.glob(".*.tmp"))

    with pytest.raises(FileExistsError):
        init.init_model_dir(model_dir)
    unseeded = modeldir.write_model_dir(tmp_path / "unseeded", hf.default_config(), tokenizer_dir)
    with pytest.raises(ValueError, match="init_seed"):
        init.init_model_dir(unseeded)
    assert not (unseeded / modeldir.WEIGHTS_FILE).exists()


def test_init_writes_its_weights_atomically(tmp_path, tokenizer_dir, monkeypatch):
    """A crash or a full disk while writing leaves no model.safetensors, and the leftover temporary file does not
    stop the next attempt."""
    cfg = hf.row_config("A0", num_hidden_layers=1, tlc_layer_pattern="M")
    path = modeldir.write_model_dir(tmp_path / "m", cfg, tokenizer_dir, init_seed=SEED)
    real = init.save_file

    def partial(tensors, filename, metadata=None):
        Path(filename).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(init, "save_file", partial)
    with pytest.raises(OSError, match="disk full"):
        init.init_model_dir(path)
    assert not (path / modeldir.WEIGHTS_FILE).exists()
    monkeypatch.setattr(init, "save_file", real)
    assert load_file(init.init_model_dir(path)).keys() == init.init_state_dict(cfg, seed=SEED).keys()
