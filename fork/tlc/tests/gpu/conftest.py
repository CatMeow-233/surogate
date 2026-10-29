"""GPU tests run in fork/.venv-tlc on the GPU host (runbook in fork/tlc/README.md).

That environment imports `surogate` from this checkout, whose `surogate/` holds the extension
`fork/dev build train` copies there, through a .pth file the runbook writes; only fork/tlc/src is
put on sys.path here. Without CUDA every test here is skipped. The gate tests also need TLC_RUNS,
the directory the runbook prepares, and skip without it.
"""

import os
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


@pytest.fixture(scope="session")
def runs() -> Path:
    """$TLC_RUNS as the runbook prepares it: tokenizer/, data/, models/<row>/ and goldens/A0.npz."""
    path = os.environ.get("TLC_RUNS")
    if not path:
        pytest.skip("set TLC_RUNS to the runbook's directory (fork/tlc/README.md, G0)")
    return Path(path)
