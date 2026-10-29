"""G0 environment: the native extension and the TLC package come from this checkout, the GPU is an sm_120 part, and
the environment fingerprint every run records is complete (docs/runbook.md: G0)."""

import json
from pathlib import Path


def test_g0_native_extension_and_checkout_package():
    import surogate_tlc
    import torch

    from surogate import _surogate

    src = Path(__file__).resolve().parents[2] / "src"
    assert Path(surogate_tlc.__file__).resolve().is_relative_to(src), surogate_tlc.__file__
    repo = src.parents[2]
    assert Path(_surogate.__file__).resolve().is_relative_to(repo / "surogate"), _surogate.__file__
    # The fork's native builds and TLC's budgets target the RTX 5090 (sm_120).
    assert torch.cuda.get_device_capability(0) == (12, 0), torch.cuda.get_device_name(0)


def test_g0_environment_fingerprint():
    from surogate_tlc.train import report

    env = report.environment()
    print(json.dumps(env, indent=2))
    assert env["git_commit"] and env["native"] and env["gpu"]["available"]
    assert env["gpu"]["capability"] == [12, 0] and env["gpu"]["driver"]
    assert all(env["packages"][name] for name in ("torch", "transformers", "tokenizers", "safetensors", "surogate-tlc"))
