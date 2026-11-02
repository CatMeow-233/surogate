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

import importlib.util  # noqa: E402

import pytest  # noqa: E402

REFERENCE = Path(__file__).resolve().parents[2] / "reference"


@pytest.fixture(autouse=True)
def _one_cpu_thread():
    """Bitwise checks on CPU hold with one thread only (docs/spec.md, "实现步骤"); every test runs that way."""
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def load_reference(stem: str):
    """One of the owner's v1.0 scripts in `reference/`, loaded by path (they are not package modules)."""
    spec = importlib.util.spec_from_file_location(f"bilingual_reference_{stem}", REFERENCE / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def moe_pilot():
    return load_reference("moe_pilot")


@pytest.fixture(scope="session")
def verify_merge_math():
    return load_reference("verify_merge_math")


@pytest.fixture(scope="session")
def compare_v10():
    return load_reference("compare")


@pytest.fixture(scope="session")
def pilot():
    """The pilot budget report in its JSON form (what the golden file holds) and its rows by id. Built once per
    session: it constructs every row's model on the meta device and runs its forward (about 5 s)."""
    import json

    from surogate_bilingual.audit.budget import report

    doc = json.loads(json.dumps(report("pilot")))
    return doc, {row["id"]: row for row in doc["rows"]}
