"""CPU tests run against this checkout: fork/tokstudy/src goes first on sys.path, then fork/tlc/src and
fork/bilingual/src (the projects whose public pure-data APIs the study reads), and a package imported from anywhere
else errors the session before any test runs. Nothing else is put on sys.path; the suite does not import the native
`surogate` package.

The suite is a CPU suite on every host: `pytest_configure` hides CUDA (CUDA_VISIBLE_DEVICES="") before any test
imports torch, for this process and every subprocess it starts. GPU checks live in tests/gpu and run by hand in
fork/.venv-tokstudy.
"""

import importlib
import os
import sys
from pathlib import Path

import pytest

FORK = Path(__file__).resolve().parents[3]
SRC = FORK / "tokstudy" / "src"
CHECKOUT = {
    "surogate_tokstudy": SRC,
    "surogate_tlc": FORK / "tlc" / "src",
    "surogate_bilingual": FORK / "bilingual" / "src",
}

for path in reversed(CHECKOUT.values()):  # SRC ends up first
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

for name, root in CHECKOUT.items():
    module = importlib.import_module(name)
    if Path(module.__file__).resolve().parent != root / name:
        raise RuntimeError(f"{name} was imported from {module.__file__}, not from {root}")


def pytest_configure(config):
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise pytest.UsageError("CUDA was initialised before the CPU suite started; run tests/cpu in its own session")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
