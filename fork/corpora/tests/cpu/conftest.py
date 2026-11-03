"""CPU tests run against this checkout: fork/corpora/src goes first on sys.path, and a `surogate_corpora` imported
from anywhere else (an installed copy, another checkout) errors the session before any test runs.

The export tests also read the exports back with the projects' own loaders, `surogate_tlc.data.corpus` and
`surogate_bilingual.data.manifest`, imported from fork/tlc/src and fork/bilingual/src of the same checkout (tests
only: the package itself never imports a project). Nothing here touches the network: the Hub is a local
http.server (hubserver.py), and every parquet file is a fixture the tests write in a registry schema with content
marked SYNTHETIC FIXTURE.
"""

import sys
from pathlib import Path

FORK = Path(__file__).resolve().parents[3]
SRC = FORK / "corpora" / "src"
PROJECT_SRCS = (FORK / "tlc" / "src", FORK / "bilingual" / "src")

for path in (*PROJECT_SRCS, SRC):  # SRC ends up first
    if str(path) in sys.path:
        sys.path.remove(str(path))
    sys.path.insert(0, str(path))

import surogate_corpora  # noqa: E402

if Path(surogate_corpora.__file__).resolve().parent != SRC / "surogate_corpora":
    raise RuntimeError(f"surogate_corpora was imported from {surogate_corpora.__file__}, not from {SRC}")
