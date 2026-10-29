"""GPU tests run in fork/.venv-tlc on the GPU host (runbook in fork/tlc/README.md).

They use the installed native `surogate`, so only fork/tlc/src goes on sys.path; the repository
root is left off because its `surogate/` may lack the built extension. Without CUDA every test
here is skipped.
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "src"

if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def pytest_collection_modifyitems(config, items):
    skip = None if _cuda_available() else pytest.mark.skip(reason="needs CUDA (GPU runbook in fork/tlc/README.md)")
    for item in items:
        if item.path.is_relative_to(HERE):
            item.add_marker(pytest.mark.gpu)
            if skip:
                item.add_marker(skip)
