"""Seed of G0: the environment has the native extension and the TLC package from this checkout."""

from pathlib import Path


def test_native_extension_and_checkout_package():
    import surogate_tlc
    import torch

    from surogate import _surogate  # noqa: F401  the training extension must load

    src = Path(__file__).resolve().parents[2] / "src"
    assert Path(surogate_tlc.__file__).resolve().is_relative_to(src), surogate_tlc.__file__
    # The fork's native builds and TLC's budgets target the RTX 5090 (sm_120).
    assert torch.cuda.get_device_capability(0) == (12, 0), torch.cuda.get_device_name(0)
