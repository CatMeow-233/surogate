"""CPU tests run against this checkout: the repository's Python `surogate` (no native extension) and
fork/tlc/src. Both go first on sys.path, and an import from anywhere else fails the session.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "fork" / "tlc" / "src"

for path in (str(SRC), str(REPO)):  # REPO ends up first
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

import surogate_tlc  # noqa: E402

import surogate  # noqa: E402

for module, expected in ((surogate, REPO / "surogate"), (surogate_tlc, SRC / "surogate_tlc")):
    if Path(module.__file__).resolve().parent != expected:
        raise RuntimeError(f"{module.__name__} was imported from {module.__file__}, not from {expected}")
