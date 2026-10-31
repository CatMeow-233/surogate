"""The versioned resume contract: classification, fingerprints of recorded checkpoints, refusals."""

import copy
import dataclasses
import hashlib
import json

import pytest
from surogate.experimental.train import contract
from surogate.experimental.train.config import TrainConfig

# TrainConfig().to_dict() of one configuration as 488e5f00 recorded it in format-3
# checkpoints and manifests (contract version 1: no contract_version entry), and
# the fingerprint 488e5f00's engine computed for it.
RECORDED_488E5F00 = json.loads(
    '{"attention_implementation": "sdpa", "batch_size": 1, "betas": [0.9, 0.999], "compile_model": false, '
    '"device": "cpu", "dtype": "float32", "eps": 1e-08, "gradient_accumulation_steps": 1, '
    '"gradient_checkpointing": true, "layer_path": null, "learning_rate": 2e-05, "lisa_interval": 20, '
    '"lisa_layers": 0, "lisa_state_policy": "drop", "lora_alpha": 32.0, "lora_dropout": 0.0, "lora_rank": 4, '
    '"lora_targets": ["q_proj", "v_proj"], "max_grad_norm": 0.0, "max_sequence_length": 2048, "max_steps": 100, '
    '"model": null, "model_config": {"hidden_size": 16, "model_type": "llama", "vocab_size": 32}, '
    '"model_revision": null, "optimizer": "rank_scaled", "optimizer_in_backward": false, "output_dir": "out", '
    '"pad_token_id": 0, "projection_interval": 200, "projection_rank": 1, "projection_scale": 1.0, '
    '"projection_scale_rule": "rms", "save_every": 50, "schema_version": 1, "seed": 42, "shuffle": true, '
    '"stochastic_rounding": true, "train_jsonl": "tokens.jsonl", "truncation": "error", '
    '"trust_remote_code": false, "warmup_steps": 0, "weight_decay": 0.0}'
)
V1_FINGERPRINT_488E5F00 = "f2b491725e82a5e9614429eccd49dedc236cc7f28c3ed4deca2287877afe275e"
# The same configuration under 856bd86c, which introduced format 3 before
# projection_scale_rule existed (its rule was always constant).
RECORDED_856BD86C = {key: value for key, value in RECORDED_488E5F00.items() if key != "projection_scale_rule"}
V1_FINGERPRINT_856BD86C = "25fa1c57f92f91e82998d1d6768ec243757f0e070f70a7fdfe941f752edffdf8"
# Pinned: changing the version-2 rule changes every version-2 fingerprint and
# needs a new CONTRACT_VERSION (see contract.py).
V2_FINGERPRINT_488E5F00 = "bd6f15caae103a49f3146645d671044745d246e5424a77d1ca24e01c95816c40"

# The TrainConfig fields of 856bd86c, the checkout that introduced checkpoint format 3.
FORMAT_3_FIELDS = {
    "schema_version", "model", "model_config", "model_revision", "trust_remote_code", "attention_implementation",
    "device", "dtype", "train_jsonl", "output_dir", "max_sequence_length", "pad_token_id", "truncation",
    "batch_size", "gradient_accumulation_steps", "max_steps", "learning_rate", "weight_decay", "max_grad_norm",
    "optimizer", "betas", "eps", "projection_rank", "projection_interval", "projection_scale",
    "stochastic_rounding", "seed", "lisa_layers", "lisa_interval", "lisa_state_policy", "layer_path", "lora_rank",
    "lora_alpha", "lora_dropout", "lora_targets", "gradient_checkpointing", "optimizer_in_backward",
    "compile_model", "save_every", "shuffle", "warmup_steps",
}  # fmt: skip


def _requested(values, **changes):
    return TrainConfig(**{**copy.deepcopy(values), **changes}).to_dict()


def _independent_v1_fingerprint(values):
    """488e5f00's engine._fingerprint(config.resume_contract()), restated."""
    kept = {
        key: value
        for key, value in values.items()
        if key not in {"max_steps", "output_dir", "save_every", "train_jsonl"}
    }
    return hashlib.sha256(json.dumps(kept, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _future_config(default="off"):
    """A TrainConfig with one field added after this checkout, as a later change would add it."""
    return dataclasses.make_dataclass(
        "FutureTrainConfig", [("future_switch", str, dataclasses.field(default=default))], bases=(TrainConfig,)
    )


@pytest.fixture
def future_legacy(monkeypatch):
    monkeypatch.setattr(contract, "LEGACY_VALUES", {**contract.LEGACY_VALUES, "future_switch": "off"})


def test_every_field_has_exactly_one_contract_class():
    classes = [
        contract.RUN_CONTROL_KEYS,
        contract.PERFORMANCE_KEYS,
        contract.BASELINE_KEYS,
        frozenset(contract.LEGACY_VALUES),
    ]
    assert sum(len(keys) for keys in classes) == len(frozenset().union(*classes)), "a key is in two classes"
    assert frozenset().union(*classes) == {item.name for item in dataclasses.fields(TrainConfig)}, (
        "classify each TrainConfig field in contract.py: PERFORMANCE_KEYS only when results provably do not "
        "depend on it, otherwise LEGACY_VALUES with the value that reproduces the behavior from before it existed"
    )


def test_baseline_keys_are_the_format_3_fields_and_legacy_values_are_frozen():
    assert contract.BASELINE_KEYS | contract.RUN_CONTROL_KEYS | contract.PERFORMANCE_KEYS == FORMAT_3_FIELDS
    # A released legacy value never changes: checkpoints without the key were
    # trained with it. Add new fields; do not edit these entries.
    assert dict(contract.LEGACY_VALUES) == {"projection_scale_rule": "constant"}
    assert contract.PERFORMANCE_KEYS == {"device", "gradient_checkpointing"}
    assert contract.CONTRACT_VERSION == 2


def test_recorded_version_1_fingerprints_still_verify():
    for recorded, expected in (
        (RECORDED_488E5F00, V1_FINGERPRINT_488E5F00),
        (RECORDED_856BD86C, V1_FINGERPRINT_856BD86C),
    ):
        assert _independent_v1_fingerprint(recorded) == expected
        assert contract._FINGERPRINT_RULES[1](recorded) == expected
    # Tuples (as asdict leaves betas and lora_targets) encode like the lists a manifest holds.
    as_saved = {**RECORDED_488E5F00, "betas": (0.9, 0.999), "lora_targets": ("q_proj", "v_proj")}
    assert contract._FINGERPRINT_RULES[1](as_saved) == V1_FINGERPRINT_488E5F00


def test_version_2_fingerprint_is_pinned_and_ignores_performance_keys():
    assert contract.fingerprint(_requested(RECORDED_488E5F00)) == V2_FINGERPRINT_488E5F00
    for key, value in (
        ("device", "auto"),
        ("device", "cuda:1"),
        ("gradient_checkpointing", False),
        ("max_steps", 7),
        ("save_every", 0),
        ("output_dir", "elsewhere"),
        ("train_jsonl", "moved.jsonl"),
    ):
        assert contract.fingerprint(_requested(RECORDED_488E5F00, **{key: value})) == V2_FINGERPRINT_488E5F00, key
    for key, value in (
        ("compile_model", True),
        ("attention_implementation", "eager"),
        ("lora_dropout", 0.1),
        ("projection_scale_rule", "constant"),
        ("seed", 43),
    ):
        assert contract.fingerprint(_requested(RECORDED_488E5F00, **{key: value})) != V2_FINGERPRINT_488E5F00, key


def test_a_key_at_its_legacy_value_counts_as_unrecorded():
    constant = contract.fingerprint(_requested(RECORDED_488E5F00, projection_scale_rule="constant"))
    assert contract._FINGERPRINT_RULES[2](RECORDED_856BD86C) == constant
    assert contract.compared_values(RECORDED_856BD86C)["projection_scale_rule"] == "constant"


def test_version_1_checkpoints_resume_under_version_2_rules():
    current = {"config": RECORDED_488E5F00, "config_fingerprint": V1_FINGERPRINT_488E5F00}
    contract.check_resume(current, _requested(RECORDED_488E5F00))
    # Performance keys were hashed by version 1 but are free to change now.
    contract.check_resume(current, _requested(RECORDED_488E5F00, gradient_checkpointing=False, device="auto"))
    with pytest.raises(ValueError, match=r"seed \(checkpoint 42, requested 7\)"):
        contract.check_resume(current, _requested(RECORDED_488E5F00, seed=7))
    # Before 8e048387 every update used the constant rule; the record says nothing.
    older = {"config": RECORDED_856BD86C, "config_fingerprint": V1_FINGERPRINT_856BD86C}
    contract.check_resume(older, _requested(RECORDED_856BD86C, projection_scale_rule="constant"))
    with pytest.raises(ValueError, match=r'projection_scale_rule \(checkpoint "constant", requested "rms"\)'):
        contract.check_resume(older, _requested(RECORDED_856BD86C, projection_scale_rule="rms"))


def test_every_differing_key_is_named():
    current = {"config": RECORDED_488E5F00, "config_fingerprint": V1_FINGERPRINT_488E5F00}
    with pytest.raises(ValueError, match="contract: ") as failure:
        contract.check_resume(current, _requested(RECORDED_488E5F00, learning_rate=1e-4, compile_model=True))
    assert (
        "compile_model (checkpoint false, requested true); learning_rate (checkpoint 2e-05, requested 0.0001)"
        in str(failure.value)
    )


def test_a_field_added_at_its_legacy_value_keeps_fingerprints_and_resume(future_legacy):
    future = _future_config()
    base = _requested(RECORDED_488E5F00)
    assert contract.fingerprint(future(**base).to_dict()) == contract.fingerprint(base)
    assert contract.fingerprint(future(**base, future_switch="on").to_dict()) != contract.fingerprint(base)
    written = {**contract.record(base), "config": base}
    contract.check_resume(written, future(**base).to_dict())
    with pytest.raises(ValueError, match=r'future_switch \(checkpoint "off", requested "on"\)'):
        contract.check_resume(written, future(**base, future_switch="on").to_dict())


def test_unclassified_field_fails_and_is_named():
    future = _future_config()(**_requested(RECORDED_488E5F00))
    with pytest.raises(ValueError, match="future_switch; add each to PERFORMANCE_KEYS or LEGACY_VALUES"):
        future.resume_contract()
    with pytest.raises(ValueError, match="future_switch"):
        contract.record(future.to_dict())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"contract_version": 3}, "contract version 3 is unknown"),
        ({"contract_version": "2"}, "contract version '2' is unknown"),
        ({"contract_version": True}, "contract version True is unknown"),
        ({"config": None}, "lacks its recorded configuration"),
        ({"config_fingerprint": "0" * 64}, "does not match its fingerprint"),
    ],
)
def test_records_this_trainer_cannot_verify_are_refused(change, message):
    base = _requested(RECORDED_488E5F00)
    saved = {**contract.record(base), "config": base, **change}
    with pytest.raises(ValueError, match=message):
        contract.check_resume(saved, base)


def test_newer_or_damaged_records_are_refused():
    base = _requested(RECORDED_488E5F00)
    newer = {**base, "future_switch": "off"}
    with pytest.raises(ValueError, match="does not know: future_switch; resume with the trainer that wrote it"):
        contract.check_resume({"config": newer, "config_fingerprint": _independent_v1_fingerprint(newer)}, base)
    damaged = {key: value for key, value in base.items() if key != "seed"}
    with pytest.raises(ValueError, match="lacks contract keys: seed"):
        contract.check_resume({**contract.record(base), "config": damaged}, base)
    with pytest.raises(ValueError, match=r"seed \(checkpoint not recorded, requested 42\)"):
        contract.check_resume({"config": damaged, "config_fingerprint": _independent_v1_fingerprint(damaged)}, base)
    with pytest.raises(ValueError, match="does not match its fingerprint"):
        contract.check_resume({"config": {**base, "seed": object()}, "config_fingerprint": "x"}, base)


def test_values_compare_by_canonical_encoding():
    found = contract.differences({"a": (1, 2), "b": 1, "c": True}, {"a": [1, 2], "b": 1.0, "c": 1, "d": 0}, {"d": 0})
    assert found == {"b": (1, 1.0), "c": (True, 1)}
    assert contract.describe({"x": (contract._MISSING, "y" * 80)}, "manifest") == (
        f'x (manifest not recorded, requested "{"y" * 56}...)'
    )
