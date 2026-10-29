"""Gates G1 and G2 on the Mamba-2 reference (fork/tlc/prototypes/g1_suite.py and g2_suite.py): the reference is
causal and its paths agree in fp64 and fp32, each of five planted bugs fails a check, and the G2 checks hold. The
native gates compare the trainer with this reference, so it is checked first."""

import importlib.util
import sys
import time
from pathlib import Path

import pytest
import torch

PROTOTYPES = Path(__file__).resolve().parents[2] / "prototypes"


def _load(name: str):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PROTOTYPES / f"{name}.py")
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    """The recurrence is thousands of small ops; on a shared CPU intra-op threads make it much slower."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["fp64", "fp32"])
def test_g1_checks_pass(dtype):
    start = time.process_time()
    results = _load("g1_suite").run(dtype)
    assert len(results) == 7 and not {name: failure for name, failure in results.items() if failure}
    assert time.process_time() - start < 10  # the notebook's budget for the whole suite, single-threaded


@pytest.mark.parametrize("mutation", list(_load("g1_suite").mutations(_load("ref_mamba2"))))
def test_g1_catches_every_mutation(mutation):
    suite, ref = _load("g1_suite"), _load("ref_mamba2")
    patched = suite.mutations(ref)[mutation]
    before = {name: getattr(ref, name) for name in patched}
    assert suite.caught(mutation), f"no G1 check fails with: {mutation}"
    assert {name: getattr(ref, name) for name in patched} == before  # the patch is undone


def test_g2_checks_pass():
    results = _load("g2_suite").run()
    assert set(results) == {"G2.1 gradients", "G2.2 initial loss", "G2.3 overfit"}
    assert not {name: failure for name, failure in results.items() if failure}


def test_g2_bound_is_the_best_current_token_predictor():
    g2 = _load("g2_suite")
    ids = torch.tensor([[0, 1, 0, 2, 0, 1]])
    # after 0: 1, 2, 1 (H = -(2/3 ln 2/3 + 1/3 ln 1/3)); after 1 and 2: always 0
    expected = 3 / 5 * -(2 / 3 * torch.log(torch.tensor(2 / 3)) + 1 / 3 * torch.log(torch.tensor(1 / 3)))
    assert g2.conditional_entropy(ids) == pytest.approx(float(expected))
    assert g2.conditional_entropy(g2.small_alphabet_batch()) >= 1.5
