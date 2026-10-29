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
import torch
from safetensors.torch import load_file, save_file
from surogate_tlc import hf, init

GPU = Path(__file__).resolve().parents[1] / "gpu"
REF_FILE = Path(__file__).resolve().parents[2] / "prototypes" / "ref_mamba2.py"


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


@pytest.fixture(scope="module")
def ref():
    spec = importlib.util.spec_from_file_location("ref_mamba2_gate_replay", REF_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def small_a0(tmp_path_factory):
    """A two-layer A0 with its initial weights, as the runbook's model directory would hold it."""
    model = tmp_path_factory.mktemp("runs") / "models" / "A0"
    model.mkdir(parents=True)
    cfg = hf.row_config("A0", num_hidden_layers=2, tlc_layer_pattern="MM", chunk_size=16)
    (model / "config.json").write_text(json.dumps(cfg))
    save_file(init.init_state_dict(cfg, seed=0), model / "model.safetensors")
    return model


class TiedTrainer:
    """Stand-in for the trainer of test_g2_tied_head_trains_as_the_embedding, scored by the reference in bf16: an
    update moves the embedding. `defect` leaves the head at the initial embedding ("stale head", the forward of the
    unfixed runtime), saves it as well ("head saved", the unfixed checkpoint) or trains nothing ("untrained")."""

    def __init__(self, model: Path, defect: str | None, ref) -> None:
        self.cfg = json.loads((model / "config.json").read_text())
        self.w, self.ref, self.defect = load_file(model / "model.safetensors"), ref, defect
        self.initial = self.w[ref.EMBEDDING].clone()

    def step(self, ids, targets, positions) -> None:
        pass

    def update_with_config(self, optimizer, step: int) -> None:
        if self.defect != "untrained":
            noise = torch.randn(self.initial.shape, generator=torch.Generator().manual_seed(step))
            self.w[self.ref.EMBEDDING] = (self.w[self.ref.EMBEDDING].float() + 0.01 * noise).to(self.initial.dtype)

    def compute_logprobs(self, ids, targets, use_lora, position_ids=None) -> np.ndarray:
        cfg, weights = self.cfg, self.w
        if self.defect == "stale head":
            cfg, weights = {**cfg, "tie_word_embeddings": False}, {**weights, self.ref.LM_HEAD: self.initial}
        with torch.no_grad():
            reference = self.ref.Reference.from_weights(cfg, weights, bf16=True)
            return reference.logprobs(torch.from_numpy(ids).long(), torch.from_numpy(targets).long()).numpy()

    def save_checkpoint(self, path: str, step: int) -> None:
        out = Path(path) / f"step_{step:08d}"
        out.mkdir(parents=True)
        head = self.defect == "head saved"
        save_file({**self.w, **({self.ref.LM_HEAD: self.initial} if head else {})}, out / "model.safetensors")
        names = ["embedding", "final_norm", *(["lm_head"] if head else [])]
        save_file({name: torch.zeros(1) for name in names}, out / "weights.shard_000_of_001.safetensors")


@pytest.mark.parametrize(
    "defect, report",
    [
        (None, None),
        ("stale head", "error <= tolerance"),
        ("head saved", '"lm_head" not in'),
        ("untrained", "torch.equal"),
    ],
    ids=["tied", "stale head", "head saved", "untrained"],
)
def test_g2_tied_head_fails_on_a_head_left_behind(gates, monkeypatch, ref, small_a0, tmp_path, defect, report):
    """The tied-head gate passes a runtime that trains one matrix, and fails one whose forward keeps the initial
    head, one whose checkpoint saves a head, and one that trains nothing; one trainer, dropped at its block's end."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # the reference recurrence is many small ops
    trainers = []

    def native_trainer(model, data, out, **kwargs):
        trainers.append(TiedTrainer(Path(model), defect, ref))
        return SimpleNamespace(trainer=trainers[-1])

    monkeypatch.setattr(gates, "native_trainer", native_trainer)
    monkeypatch.setattr(gates, "adamw", lambda: None)
    runs = small_a0.parents[1]
    try:
        if defect is None:
            gates.test_g2_tied_head_trains_as_the_embedding(runs, tmp_path)
        else:
            with pytest.raises(AssertionError) as failure:
                gates.test_g2_tied_head_trains_as_the_embedding(runs, tmp_path)
            assert report in str(failure.traceback[-1].statement)  # the assertion that failed
    finally:
        torch.set_num_threads(threads)
    assert len(trainers) == 1
