"""GPU tests (runbook G0, step 3): run in fork/.venv-bilingual on the GPU host; every test is skipped without CUDA.

Like tests/cpu, the suite imports this checkout: fork/bilingual/src goes first on sys.path. The tiny-root builders
of tests/cpu/conftest.py are loaded by path and reused (`build_tiny_root`, `tiny_run`).

`BILINGUAL_GPU_TESTS_ON_CPU=1` runs the tests on the CPU instead. That only rehearses the test code on a host
without a GPU (CUDA-only checks are skipped); it says nothing about CUDA and is not a G0 result.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

import surogate_bilingual  # noqa: E402

if Path(surogate_bilingual.__file__).resolve().parent != SRC / "surogate_bilingual":
    raise RuntimeError(f"surogate_bilingual was imported from {surogate_bilingual.__file__}, not from {SRC}")

_spec = importlib.util.spec_from_file_location(
    "bilingual_cpu_conftest", Path(__file__).parents[1] / "cpu" / "conftest.py"
)
cpu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpu)


def gpu_device() -> str | None:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu" if os.environ.get("BILINGUAL_GPU_TESTS_ON_CPU") == "1" else None


@pytest.fixture(scope="session")
def device() -> str:
    dev = gpu_device()
    if dev is None:
        pytest.skip("no CUDA device (GPU tests run in fork/.venv-bilingual on the GPU host, runbook G0)")
    return dev


@pytest.fixture(scope="session")
def tiny_root(tmp_path_factory):
    return cpu.build_tiny_root(tmp_path_factory.mktemp("gpu_tiny_root"))


@pytest.fixture(scope="session")
def tiny_run():
    return cpu.tiny_run


@pytest.fixture(scope="session")
def cpu_helpers():
    """tests/cpu/conftest.py, loaded by path (build_tiny_root, tiny_run, ...)."""
    return cpu
