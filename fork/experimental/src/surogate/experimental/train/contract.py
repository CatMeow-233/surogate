"""Resume contract: the configuration a resumed experimental run must keep.

A checkpoint (and the run's ``manifest.json``) records the configuration that
wrote it (``config``), the contract version it was written under
(``contract_version``; absent means version 1) and a fingerprint of the
contract (``config_fingerprint``). Resume first checks that the fingerprint
can be recomputed from the recorded configuration under that version's rule,
then compares the recorded and the requested configuration key by key under
this trainer's rules, and names every key that differs.

Every ``TrainConfig`` field is in exactly one class:

``RUN_CONTROL_KEYS``
    Where and how long the run goes. Never compared.
``PERFORMANCE_KEYS``
    Change speed or memory. Results do not depend on them except in the
    configurations ``COMPARED_WHEN`` names, where they are compared and
    fingerprinted like baseline keys; otherwise they are not compared
    (contract version 2 and later). ``device`` is here because resume separately requires the same device type
    and dtype (``runtime`` in the checkpoint); only the device index and the
    ``auto`` spelling are free to change.
``BASELINE_KEYS``
    Recorded by every configuration this trainer can resume (the fields of
    checkpoint format 3 when it was introduced, 856bd86c). Compared.
``LEGACY_VALUES``
    Fields added after that, each mapped to its legacy value: the value that
    reproduces the trainer's behavior from before the field existed (not
    necessarily the field's default). A configuration recorded before the field
    existed is compared as if it had recorded this value. Compared.

Adding a ``TrainConfig`` field therefore means adding it to
``PERFORMANCE_KEYS`` (only when results provably do not depend on it, with a
``COMPARED_WHEN`` entry for each configuration in which they do) or to
``LEGACY_VALUES``. An unclassified field makes every fingerprint and resume
check fail, and ``tests/test_resume_contract.py`` names it. A legacy value never
changes once released. Moving an existing key between classes changes
fingerprints and needs a new ``CONTRACT_VERSION`` with its rule kept in
``_FINGERPRINT_RULES``, so that older records still verify.

Version-2 fingerprints leave out each ``LEGACY_VALUES`` key whose value equals
its legacy value. A field added at its legacy value thus keeps every
fingerprint, and a newer trainer can still recompute the fingerprint of a
configuration recorded before the field existed.

The contract covers configuration only. It does not detect a code change
between two checkouts that run the same configuration differently.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

CONTRACT_VERSION = 2

RUN_CONTROL_KEYS = frozenset({"max_steps", "output_dir", "save_every", "train_jsonl"})

# gradient_checkpointing: in eager mode, non-reentrant recomputation restores
# the RNG and autocast state and reruns the same operations, so a resume that
# toggles it continues bitwise on CPU (tests/test_experimental_engine.py). On a
# GPU this holds only up to the run-to-run variation of its kernels; see the
# guide. Under torch.compile it changes the compiled graph (what is saved or
# recomputed, and how operations are fused), so it changes the arithmetic and
# is compared there (COMPARED_WHEN).
PERFORMANCE_KEYS = frozenset({"device", "gradient_checkpointing"})

# Performance keys that change results in some configurations: key -> (baseline
# key, value). When the configuration has that value, the key is compared and,
# from contract version 2 on, fingerprinted. Part of the version-2 rule: once
# released, an entry is never edited or added without a new CONTRACT_VERSION.
COMPARED_WHEN: Mapping[str, tuple[str, Any]] = MappingProxyType(
    {
        # CPU fp32, tiny Llama, 3 steps: compiled runs with and without it
        # differ in 4 of 21 weight tensors; two compiled runs differ in none.
        "gradient_checkpointing": ("compile_model", True),
    }
)

BASELINE_KEYS = frozenset(
    {
        "attention_implementation",
        "batch_size",
        "betas",
        "compile_model",
        "dtype",
        "eps",
        "gradient_accumulation_steps",
        "layer_path",
        "learning_rate",
        "lisa_interval",
        "lisa_layers",
        "lisa_state_policy",
        "lora_alpha",
        "lora_dropout",
        "lora_rank",
        "lora_targets",
        "max_grad_norm",
        "max_sequence_length",
        "model",
        "model_config",
        "model_revision",
        "optimizer",
        "optimizer_in_backward",
        "pad_token_id",
        "projection_interval",
        "projection_rank",
        "projection_scale",
        "schema_version",
        "seed",
        "shuffle",
        "stochastic_rounding",
        "truncation",
        "trust_remote_code",
        "warmup_steps",
        "weight_decay",
    }
)

LEGACY_VALUES: Mapping[str, Any] = MappingProxyType(
    {
        # 8e048387: until then every rank-scaled update used the constant rule.
        "projection_scale_rule": "constant",
    }
)

# The keys contract version 1 (no contract_version recorded) left out of its
# fingerprint; it hashed every other recorded key, so it is computed from the
# recorded configuration, whatever keys that has.
_V1_IGNORED_KEYS = frozenset({"max_steps", "output_dir", "save_every", "train_jsonl"})

_MISSING = object()


def _encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_encode(value).encode()).hexdigest()


def _same(left: Any, right: Any) -> bool:
    try:
        return _encode(left) == _encode(right)
    except (TypeError, ValueError):
        return False


def _short(value: Any) -> str:
    if value is _MISSING:
        return "not recorded"
    try:
        text = _encode(value)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def differences(
    recorded: Mapping[str, Any], requested: Mapping[str, Any], legacy: Mapping[str, Any] | None = None
) -> dict[str, tuple[Any, Any]]:
    """Map each key of ``requested`` whose recorded value differs to (recorded, requested).

    A key ``recorded`` lacks is compared with its ``legacy`` value; without one
    it differs. Values are compared by their canonical JSON encoding, so a tuple
    equals the list it was saved as, while 1 and 1.0 or 1 and True differ.
    Keys only ``recorded`` has are not reported.
    """
    legacy = legacy or {}
    found = {}
    for key, value in requested.items():
        saved = recorded[key] if key in recorded else legacy.get(key, _MISSING)
        if saved is _MISSING or not _same(saved, value):
            found[key] = (saved, value)
    return found


def describe(found: Mapping[str, tuple[Any, Any]], source: str = "checkpoint") -> str:
    return "; ".join(
        f"{key} ({source} {_short(saved)}, requested {_short(value)})" for key, (saved, value) in sorted(found.items())
    )


def _known_keys() -> frozenset[str]:
    return RUN_CONTROL_KEYS | PERFORMANCE_KEYS | BASELINE_KEYS | frozenset(LEGACY_VALUES)


def _check_classified(values: Mapping[str, Any]) -> None:
    unclassified = sorted(set(values) - _known_keys())
    if unclassified:
        raise ValueError(
            "TrainConfig fields without a resume-contract class: "
            f"{', '.join(unclassified)}; add each to PERFORMANCE_KEYS or LEGACY_VALUES in "
            "surogate/experimental/train/contract.py"
        )


def compared_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """The contract keys of a configuration, legacy-filled; it must be one this trainer knows.

    ``values`` is ``TrainConfig.to_dict()`` or a recorded configuration.
    """
    _check_classified(values)
    missing = sorted(BASELINE_KEYS - set(values))
    if missing:
        raise ValueError(f"configuration record lacks contract keys: {', '.join(missing)}")
    compared = {key: values[key] for key in BASELINE_KEYS}
    for key, legacy in LEGACY_VALUES.items():
        compared[key] = values.get(key, legacy)
    conditional = [key for key, (when, value) in COMPARED_WHEN.items() if _same(compared[when], value)]
    missing = sorted(set(conditional) - set(values))
    if missing:
        raise ValueError(f"configuration record lacks contract keys: {', '.join(missing)}")
    for key in conditional:
        compared[key] = values[key]
    return compared


def _fingerprint_v1(values: Mapping[str, Any]) -> str:
    return _sha256({key: value for key, value in values.items() if key not in _V1_IGNORED_KEYS})


def _fingerprint_v2(values: Mapping[str, Any]) -> str:
    compared = compared_values(values)
    kept = {
        key: value for key, value in compared.items() if not (key in LEGACY_VALUES and _same(value, LEGACY_VALUES[key]))
    }
    return _sha256({"contract_version": 2, "values": kept})


_FINGERPRINT_RULES = MappingProxyType({1: _fingerprint_v1, 2: _fingerprint_v2})


def fingerprint(values: Mapping[str, Any]) -> str:
    """This trainer's (version ``CONTRACT_VERSION``) fingerprint of a configuration."""
    return _FINGERPRINT_RULES[CONTRACT_VERSION](values)


def record(values: Mapping[str, Any]) -> dict[str, Any]:
    """The contract entries a checkpoint and the manifest store next to ``config``."""
    return {"contract_version": CONTRACT_VERSION, "config_fingerprint": fingerprint(values)}


def check_resume(saved: Mapping[str, Any], requested: Mapping[str, Any], source: str = "checkpoint") -> None:
    """Raise ValueError unless ``saved`` (a checkpoint or manifest, named by ``source``) may continue as ``requested``.

    ``requested`` is the ``TrainConfig.to_dict()`` of the resumed run.
    """
    version = saved.get("contract_version", 1)
    if type(version) is not int or version not in _FINGERPRINT_RULES:
        raise ValueError(
            f"resume contract version {version!r} is unknown to this trainer, which reads versions "
            f"{', '.join(map(str, _FINGERPRINT_RULES))}; resume with the trainer that wrote the {source}"
        )
    values = saved.get("config")
    if not isinstance(values, Mapping):
        raise ValueError(f"{source} lacks its recorded configuration")
    unknown = sorted(map(str, set(values) - _known_keys()))
    if unknown:
        raise ValueError(
            f"{source} records configuration keys this trainer does not know: {', '.join(unknown)}; "
            "resume with the trainer that wrote it"
        )
    current = compared_values(requested)
    try:
        expected = _FINGERPRINT_RULES[version](values)
    except TypeError:  # a recorded value JSON cannot encode
        expected = None
    if expected is None or saved.get("config_fingerprint") != expected:
        raise ValueError(f"{source} configuration record does not match its fingerprint")
    found = differences(values, current, LEGACY_VALUES)
    if found:
        raise ValueError(
            "resume configuration changes the model, algorithm, or data processing contract: " + describe(found, source)
        )
