"""G0 environment: the native extension and the TLC package come from this checkout, the GPU is an sm_120 part, and
the environment fingerprint every run records is complete (docs/runbook.md: G0)."""

import json
import subprocess
import sys
from pathlib import Path

# What `python -m surogate_tlc ...` and the w-twice child import: a fresh interpreter outside the checkout, so neither
# the conftest's sys.path entry nor a repository-root working directory stands in for the editable install and the
# runbook's .pth file.
IMPORTS = "import surogate_tlc, surogate._surogate as s; print(surogate_tlc.__file__); print(s.__file__)"


def test_g0_native_extension_and_checkout_package(tmp_path):
    import torch

    src = Path(__file__).resolve().parents[2] / "src"
    repo = src.parents[2]
    result = subprocess.run([sys.executable, "-c", IMPORTS], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    tlc, native = (Path(line).resolve() for line in result.stdout.split())
    # fork/tlc/src exactly: a non-editable install sits in fork/.venv-tlc/.../site-packages, inside the checkout too.
    assert tlc.is_relative_to(src), tlc
    assert native.is_relative_to(repo / "surogate"), native
    # The fork's native builds and TLC's budgets target the RTX 5090 (sm_120).
    assert torch.cuda.get_device_capability(0) == (12, 0), torch.cuda.get_device_name(0)


def test_g0_environment_fingerprint():
    from surogate_tlc.train import report

    env = report.environment()
    print(json.dumps(env, indent=2))
    assert env["git_commit"] and env["native"] and env["gpu"]["available"]
    assert env["native"]["libsurogate-common.so"]["sha256"]  # the runtime and kernels, apart from the binding
    assert env["gpu"]["capability"] == [12, 0] and env["gpu"]["driver"]
    assert all(env["packages"][name] for name in ("torch", "transformers", "tokenizers", "safetensors", "surogate-tlc"))
