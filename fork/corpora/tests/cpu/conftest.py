"""CPU tests run against this checkout: fork/corpora/src goes first on sys.path, and a `surogate_corpora` imported
from anywhere else (an installed copy, another checkout) errors the session before any test runs.

The export tests also read the exports back with the projects' own loaders, `surogate_tlc.data.corpus` and
`surogate_bilingual.data.manifest`, imported from fork/tlc/src and fork/bilingual/src of the same checkout (tests
only: the package itself never imports a project). Nothing here touches the network: the Hub is a local
http.server (corpora_fixtures.FakeHub), and every parquet file is a fixture the tests write in a registry schema with content
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

import pytest  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.append(str(HERE))  # corpora_fixtures, the shared test helpers


@pytest.fixture
def fake_hub(monkeypatch):
    """A started FakeHub (corpora_fixtures) with no files; tests fill `hub.files` before the client lists them.
    Requests to 127.0.0.1 bypass any proxy of the environment."""
    from corpora_fixtures import FakeHub

    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    for name in ("http_proxy", "HTTP_PROXY"):
        monkeypatch.delenv(name, raising=False)
    hub = FakeHub(files={}).start()
    yield hub
    hub.stop()


@pytest.fixture
def client_for():
    """client_for(hub, token=None) -> a HubClient on the fake Hub that never sleeps between retries."""
    from surogate_corpora.hub import HubClient

    return lambda hub, token=None: HubClient(
        token=token, endpoint=hub.endpoint, timeout=10, retries=2, sleep=lambda s: None
    )
