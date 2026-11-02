"""Configurations (docs/spec.md, "开关"): one table of defaults, profiles, the config hash, row configs."""

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest

from surogate_bilingual.config import (
    DEFAULTS,
    FUSIONS,
    PROFILES,
    ModelConfig,
    TrainConfig,
    config_hash,
    load_toml,
    matched_slot_width,
    profile,
    run_config,
)


def test_dataclass_defaults_come_from_defaults():
    for cls, section in ((ModelConfig, "model"), (TrainConfig, "train")):
        for f in dataclasses.fields(cls):
            if f.name == "family":
                continue
            default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
            assert default == DEFAULTS[section][f.name], (section, f.name)
    assert set(f.name for f in dataclasses.fields(ModelConfig)) == set(DEFAULTS["model"]) - {"gqa"}
    assert set(f.name for f in dataclasses.fields(TrainConfig)) == set(DEFAULTS["train"])


def test_owner_decisions_are_the_defaults():
    pilot = profile("pilot", "slot")
    assert (pilot.d_model, pilot.n_layers, pilot.n_heads, pilot.n_kv_heads, pilot.vocab_size) == (512, 24, 8, 2, 32768)
    assert (pilot.slot_width, pilot.ffn_width) == (789, 1728)  # B-O4
    assert DEFAULTS["routing"]["signal"] == "oob" and DEFAULTS["routing"]["g"] == (0.5, 0.5)  # B-O3, B-O5
    assert DEFAULTS["routing"]["window"] == 32
    train = TrainConfig()
    assert (train.global_batch_tokens, train.microbatch, train.seq, train.accum) == (65536, 16, 1024, 4)  # B-O10
    assert (train.warmup_frac, train.min_lr_frac) == (0.02, 0.10)
    assert (train.expert_decay, train.weight_decay, train.bridge_variant) == ("none", 0.1, "c")  # B-O9, B-O13
    assert DEFAULTS["tokenizer"]["normalization"] == "none"  # B-O11
    assert DEFAULTS["tokenizer"]["special_tokens"] == ["<unk>", "<eos>", "<|zh|>", "<|en|>", "<|bridge|>"]


def test_profiles_follow_the_slot_width_rule():
    for name in PROFILES:
        cfg = profile(name, "slot")
        assert cfg.slot_width == matched_slot_width(cfg.d_model, cfg.ffn_width), name
        slot_layer = cfg.d_model * cfg.ffn_width + cfg.ffn_width + cfg.slot_width * (cfg.d_model + cfg.ffn_width)
        assert slot_layer <= 3 * cfg.d_model * cfg.ffn_width, name
    smoke = profile("smoke", "swiglu")
    assert (smoke.vocab_size, smoke.d_model, smoke.n_layers, smoke.ffn_width, smoke.max_seq) == (256, 64, 2, 128, 128)
    assert profile("mech8", "swiglu").n_layers == 8


def test_config_hash_is_stable_and_sensitive():
    a = profile("tiny", "swiglu")
    assert config_hash(a) == config_hash(profile("tiny", "swiglu"))
    assert config_hash(a) != config_hash(a.replace(expert_layers=(0,)))
    assert config_hash(a) != config_hash(profile("tiny", "slot"))
    assert config_hash({"x": 1}) != config_hash({"x": 2})
    assert ModelConfig.from_dict(a.to_dict()) == a
    with pytest.raises(ValueError, match="unknown ModelConfig keys"):
        ModelConfig.from_dict({**a.to_dict(), "gqa": "repeat"})


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"family": "moe"}, "family"),
        ({"n_heads": 3}, "heads"),
        ({"expert_layers": (1, 0)}, "sorted"),
        ({"expert_layers": (5,)}, "subset"),
        ({"tie_embeddings": False}, "tie_embeddings"),
        ({"vocab_size": 5}, "special tokens"),
    ],
)
def test_model_config_validation(changes, message):
    with pytest.raises(ValueError, match=message):
        profile("tiny", "swiglu").replace(**changes)


def test_train_config_refuses_inconsistent_batch_arithmetic():
    with pytest.raises(ValueError, match="global_batch_tokens"):
        TrainConfig(microbatch=8)
    assert TrainConfig(microbatch=8, accum=8).global_batch_tokens == 65536
    with pytest.raises(ValueError, match="expert_decay"):
        TrainConfig(expert_decay="sometimes")


def test_row_config_from_toml(tmp_path):
    path = tmp_path / "row.toml"
    path.write_text('[model]\nfamily = "slot"\nn_layers = 8\ngqa = "repeat"\n[routing]\ngate_policy = "dual_const"\n')
    run = load_toml(path)
    assert run.model == profile("mech8", "slot") and run.gqa == "repeat"
    assert run.fusion == "post" and run.fusion in FUSIONS["slot"]
    assert run.routing["gate_policy"] == "dual_const" and run.routing["g"] == (0.5, 0.5)
    assert run.train == TrainConfig()
    for bad, message in (
        ({"model": {"family": "slot", "colour": 1}}, "unknown keys in \\[model\\]"),
        ({"model": {"family": "slot"}, "stage": {}}, "unknown top-level"),
        ({"model": {}}, "family must be set"),
        ({"model": {"family": "slot"}, "fusion": {"name": "concat"}}, "does not belong"),
    ):
        with pytest.raises(ValueError, match=message):
            run_config(bad)
    assert run_config({"model": {"family": "slot"}, "stage": {"a": 1}}, extra_tables=("stage",)).extra == {
        "stage": {"a": 1}
    }


def test_reading_a_config_imports_no_torch(tmp_path):
    src = Path(__file__).resolve().parents[2] / "src"
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import surogate_bilingual.config, surogate_bilingual.canonical;"
        " sys.exit('torch' in sys.modules)"
    )
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(src)], cwd=tmp_path, timeout=120)
    assert result.returncode == 0
