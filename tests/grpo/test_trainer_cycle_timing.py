"""Measure acquisition/publication separately without changing compute throughput semantics."""

import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from surogate.train.metrics_writer import MetricsWriter


@pytest.fixture
def trainer_module(monkeypatch):
    # Exercise the real Python training loop on CPU. Only the native engine
    # dependency is replaced, scoped to this fixture rather than left globally
    # installed as a fake extension for later GPU tests.
    import surogate

    modules_before = set(sys.modules)
    stubbed = False
    with monkeypatch.context() as native_patch:
        try:
            importlib.import_module("surogate._surogate")
        except ImportError:
            stubbed = True
            native = ModuleType("surogate._surogate")
            native_patch.setitem(sys.modules, "surogate._surogate", native)
            native_patch.setattr(surogate, "_surogate", native, raising=False)
        try:
            path = Path(__file__).resolve().parents[2] / "surogate/grpo/trainer.py"
            spec = importlib.util.spec_from_file_location("surogate.grpo._cycle_timing_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            if stubbed:
                # Config modules also capture the extension in module globals.
                # Evict modules loaded under the stub, including package
                # attributes, so later tests import against the real runtime.
                added = set(sys.modules) - modules_before
                for name in sorted(added, key=lambda value: value.count("."), reverse=True):
                    if name.startswith("surogate.") and name != "surogate._surogate":
                        loaded = sys.modules.pop(name)
                        parent_name, _, attribute = name.rpartition(".")
                        parent = sys.modules.get(parent_name)
                        if parent is not None and getattr(parent, attribute, None) is loaded:
                            delattr(parent, attribute)
    # The isolated module uses only an optimizer configuration factory from
    # the engine. Keep even a real installed extension untouched on GPU hosts.
    module._surogate = SimpleNamespace(OptimizerConfig=lambda **kwargs: SimpleNamespace(**kwargs))
    return module


class _Clock:
    def __init__(self):
        self.seconds = 0.0
        self.wall_offset = 0.0

    def advance(self, seconds, result=None):
        self.seconds += seconds
        return result

    def perf_counter(self):
        return 100.0 + self.seconds

    def time(self):
        return 1000.0 + self.seconds + self.wall_offset


@pytest.mark.parametrize("dispatch_pp", [False, True])
def test_cycle_records_stage_boundaries_and_excludes_checkpointing(
    trainer_module,
    monkeypatch,
    tmp_path,
    dispatch_pp,
):
    clock = _Clock()
    monkeypatch.setattr(trainer_module, "time", clock)
    trainer = trainer_module.GRPOTrainer.__new__(trainer_module.GRPOTrainer)
    trainer.config = SimpleNamespace(
        max_steps=2,
        model="test",
        gpus=1,
        sequence_len=4,
        learning_rate=0.01,
        lora=True,
        lora_rank=2,
        lora_alpha=4,
        recipe="bf16",
        optimizer="adamw",
        doc_masking=True,
        noise_scheduler=None,
        turn_diagnostics=False,
        logging_steps=1,
        weight_decay=0.0,
        max_grad_norm=1.0,
        adamw_beta1=0.9,
        adamw_beta2=0.99,
        adamw_epsilon=1e-8,
        save_steps=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        output_dir=str(tmp_path),
        merge_adapter=False,
        loss=SimpleNamespace(
            kl_tau=0.1,
            adv_tau=1.0,
            teacher_tau=0.0,
            ipo_mask_low=0.2,
            ipo_mask_high=0.2,
            ratio_clip=7.0,
        ),
    )
    micro_batch = {
        "input_ids": np.array([[1, 2]], dtype=np.int32),
        "position_ids": np.array([[0, 1]], dtype=np.int32),
        "targets": np.array([[2, 0]], dtype=np.int32),
        "advantages": np.array([[0.0, 1.0]], dtype=np.float32),
        "inference_logprobs": np.array([[0.0, -0.5]], dtype=np.float32),
        "loss_mask": np.array([[False, True]]),
        "temperatures": np.ones((1, 2), dtype=np.float32),
        "teacher_logprobs": None,
    }
    metrics = {"total_tokens": 2, "policy_loss": 0.25}
    trainer.start_step = 1  # A resumed step publishes weights before acquisition.
    trainer.phase_controller = None
    trainer._setup_data = Mock()
    trainer.broadcast = SimpleNamespace(
        broadcast=lambda *_: clock.advance(2),
        cleanup=lambda *_: None,
    )

    def pack():
        clock.advance(3)
        clock.wall_offset -= 100  # A wall-clock correction must not affect cycle timings.

    trainer.packer = SimpleNamespace(pack=pack)
    trainer.data_loader = SimpleNamespace(
        wait_for_batch=lambda: clock.advance(5),
        get_batch=lambda: clock.advance(7, [micro_batch]),
    )
    trainer.lr_schedule = SimpleNamespace(get_lr=lambda _: 0.01)
    trainer._dispatch_pp = dispatch_pp
    trainer._r3_warned_pp = False
    trainer._r3_active = False
    trainer._r3_stats = {}
    trainer._r3_step_summary = {}
    trainer.trainer = SimpleNamespace(
        set_grad_accumulation=Mock(),
        step_grpo_native=lambda *args, **kwargs: clock.advance(11),
        update_with_config=lambda *_: clock.advance(13, {"norm": 1.0}),
        get_grpo_native_metrics=lambda: metrics,
        save_checkpoint=lambda *_: clock.advance(19),
        export_adapter=Mock(),
    )

    def dispatch_step(*_):
        trainer._diag_metrics = metrics
        return clock.advance(24, {"norm": 1.0})

    trainer._dispatch_pp_step = dispatch_step
    metrics_path = tmp_path / "metrics.jsonl"
    with MetricsWriter(str(metrics_path)) as writer:
        trainer.metrics_writer = writer
        trainer.train()

    (row,) = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    assert row["step"] == 1 and row["train/orch_step"] == 1
    assert row["train/publish_duration_ms"] == 2000
    assert row["train/pack_duration_ms"] == 3000
    assert row["train/batch_wait_duration_ms"] == 5000
    assert row["train/batch_receive_duration_ms"] == 7000
    assert row["train/cycle_duration_ms"] == 41000
    # Existing metrics still describe only the 11+13 seconds of training.
    assert row["train/duration_ms"] == 24000
    assert row["train/tokens_per_second"] == pytest.approx(2 / 24)
    # The 19-second checkpoint and terminal publications happen after the
    # measured cycle; omitting them from cycle duration is intentional.
    assert clock.seconds > row["train/cycle_duration_ms"] / 1000 + 19
    trainer.trainer.set_grad_accumulation.assert_called_once_with(1)
    trainer.trainer.export_adapter.assert_called_once()
