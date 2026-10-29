"""The decision logic of the GPU gate tests (tests/gpu), replayed on CPU with stand-ins for the native trainer: a gate
must fail on the defect it targets, keep at most one native trainer alive, and record what the w-twice probe
measured. The GPU host runs the gates themselves; a mistake in their logic would let a defect through unseen."""

import importlib.util
import json
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

GPU = Path(__file__).resolve().parents[1] / "gpu"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"gpu_{name}", GPU / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def gates():
    return _load("test_gates")


class Trainer:
    def __init__(self, model: str) -> None:
        self.model = model


class Trainers:
    """Stand-in for native_trainer(): trainers that remember their model and count how many are alive."""

    def __init__(self):
        self.alive, self.alive_at_start = 0, []

    def __call__(self, model, data, out, **kwargs):
        self.alive_at_start.append(self.alive)
        self.alive += 1
        trainer = Trainer(Path(model).name)
        weakref.finalize(trainer, self._gone)
        return SimpleNamespace(trainer=trainer)

    def _gone(self):
        self.alive -= 1


def _runs(tmp_path, rows, vocab=50):
    runs = tmp_path / "runs"
    for row in rows:
        (runs / "models" / row).mkdir(parents=True)
        (runs / "models" / row / "config.json").write_text(json.dumps({"vocab_size": vocab}))
        (runs / "models" / row / "tlc_meta.json").write_text(json.dumps({"init_seed": 1}))
    return runs


@pytest.mark.parametrize("reach, passes", [(0, True), (1, False)], ids=["causal", "reads one token ahead"])
def test_g1n3_sees_the_token_next_to_the_last_target(gates, monkeypatch, tmp_path, reach, passes):
    """A backward that reads one position past the last target must move the gradients; one trainer at a time."""
    trainers = Trainers()
    monkeypatch.setattr(gates, "native_trainer", trainers)

    def gradients(trainer, ids, targets):
        last = int(np.flatnonzero(targets[0] != gates.IGNORE).max())
        return {"w": ids[:, : last + 1 + reach].astype(np.float64).sum(1)}

    monkeypatch.setattr(gates, "gradients", gradients)
    runs = _runs(tmp_path, ["A0"])
    run = lambda: gates.test_g1n3_gradients_ignore_tokens_after_the_last_target(  # noqa: E731
        runs, "A0", gates.CHUNK, tmp_path / "t"
    )
    if passes:
        run()
    else:
        with pytest.raises(AssertionError):
            run()
    assert trainers.alive_at_start == [0, 0, 0]  # each with-block drops its trainer before the next one is built


@pytest.mark.parametrize(
    "broken, report",
    [
        (None, None),
        ("zero", r"^\['first_layer_conv_bias'\]$"),  # the dead gradients, by name, before any cosine
        ("nan", r"^\['first_layer_conv_bias'\]$"),
        ("flipped", r"^\{'first_layer_conv_bias': -"),  # alive but pointing the other way: the cosine check
    ],
)
def test_g1d_fails_on_a_lost_or_nan_gradient(gates, monkeypatch, tmp_path, broken, report):
    """A0-split is no G0 row: a model-level gradient that is all zero or NaN must fail G1(d), wherever it sorts, and
    is reported as dead rather than as a NaN cosine."""
    runs = _runs(tmp_path, ["A0", "A0-split"])
    (runs / "goldens").mkdir()
    ids = np.zeros((1, 4), np.int32)
    np.savez(runs / "goldens" / "A0.npz", input_ids=ids, targets=ids, tolerance=np.float64(1e-3))
    rng = np.random.default_rng(0)
    a0 = {name: rng.normal(size=8) for name in ("blocks[0].A_log", "blocks[0].conv_bias", "blocks[1].dt_bias")}
    split = {"first_layer_A_log": a0["blocks[0].A_log"], "blocks[0].dt_bias": a0["blocks[1].dt_bias"]}
    conv_bias = {"zero": np.zeros(8), "nan": np.full(8, np.nan), "flipped": -a0["blocks[0].conv_bias"]}
    split["first_layer_conv_bias"] = conv_bias.get(broken, a0["blocks[0].conv_bias"])
    trainers = Trainers()
    monkeypatch.setattr(gates, "native_trainer", trainers)
    monkeypatch.setattr(gates, "logprobs", lambda trainer, ids, targets: np.zeros(ids.shape))
    monkeypatch.setattr(gates, "gradients", lambda t, ids, targets: dict(a0 if t.model == "A0" else split))
    if broken is None:
        gates.test_g1d_split_first_layer_is_equivalent(runs, tmp_path / "t")
    else:
        with pytest.raises(AssertionError, match=report):
            gates.test_g1d_split_first_layer_is_equivalent(runs, tmp_path / "t")
    assert trainers.alive_at_start == [0, 0]


def test_below_counts_a_nan_cosine_as_low(gates):
    """G1(d) and G2's gradient check: a NaN cosine (zero or non-finite gradient) fails like a low one."""
    assert gates.below({"nan": float("nan"), "low": 0.5, "at bound": 0.99, "high": 1.0}, 0.99) == pytest.approx(
        {"nan": float("nan"), "low": 0.5}, nan_ok=True
    )


def test_a_trainer_is_unusable_after_its_block(gates, monkeypatch, tmp_path):
    monkeypatch.setattr(gates, "native_trainer", Trainers())
    with gates.native(tmp_path / "A0", tmp_path, tmp_path) as trainer:
        assert trainer.model == "A0"
    with pytest.raises(RuntimeError, match="after its with-block"):
        trainer.model  # noqa: B018


@pytest.mark.parametrize(
    "child, timeout, expected",
    [
        ("os.abort()", 60, {"outcome": "garbage", "exit": -6}),  # dies in teardown after measuring
        ("os._exit(1)", 60, {"outcome": "garbage", "exit": 1}),
        ("time.sleep(60)", 3, {"outcome": "garbage", "exit": None}),  # hangs in teardown
        ("", 60, {"outcome": "garbage"}),
    ],
    ids=["abort", "exit 1", "hang", "clean"],
)
def test_w_twice_keeps_a_measurement_made_before_the_child_died(monkeypatch, tmp_path, child, timeout, expected):
    module = _load("test_w_twice")
    line = json.dumps({"outcome": "garbage", "cosine": 0.71, "norm_ratio": 0.5})
    script = f"import json, os, time\nprint({line!r}, flush=True)\n{child}\n"
    monkeypatch.setattr(module, "CHILD", script)
    monkeypatch.setattr(module, "TIMEOUT", timeout)
    module.test_w_used_twice_is_recorded(tmp_path, tmp_path / "t")
    record = json.loads((tmp_path / "env" / "w-twice.json").read_text())
    assert record["outcome"] == expected["outcome"] and record["cosine"] == 0.71
    assert ("exit" in record) == ("exit" in expected)
    if "exit" in expected:
        assert record["exit"]["returncode"] == expected["exit"]


@pytest.mark.parametrize("child, outcome", [("os.abort()", "crash"), ("sys.exit(3)", "error")], ids=["crash", "error"])
def test_w_twice_without_a_measurement(monkeypatch, tmp_path, child, outcome):
    module = _load("test_w_twice")
    monkeypatch.setattr(module, "CHILD", f"import os, sys\n{child}\n")
    module.test_w_used_twice_is_recorded(tmp_path, tmp_path / "t")
    assert json.loads((tmp_path / "env" / "w-twice.json").read_text())["outcome"] == outcome
