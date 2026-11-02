"""CPU tests run against this checkout: fork/bilingual/src goes first on sys.path, and a `surogate_bilingual`
imported from anywhere else (an installed copy, another checkout) errors the session before any test runs.
Nothing else is put on sys.path; the suite does not import the native `surogate` package.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

import surogate_bilingual  # noqa: E402

if Path(surogate_bilingual.__file__).resolve().parent != SRC / "surogate_bilingual":
    raise RuntimeError(f"surogate_bilingual was imported from {surogate_bilingual.__file__}, not from {SRC}")
