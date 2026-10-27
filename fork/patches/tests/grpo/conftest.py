"""Shared fixtures for the CPU tests of the fork's GRPO patches (ledger topics T6-T8)."""

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

# This directory is not a package, so pytest puts only the directory itself on
# sys.path. Put the checkout first so `surogate` resolves to it under a bare
# `pytest` too, not only under `python -m pytest` from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))


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
            path = REPO_ROOT / "surogate/grpo/trainer.py"
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
