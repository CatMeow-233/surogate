"""The environment fingerprint (`environment.json`, `bilingual-env/1`) and the code identity of this checkout.

`environment()` records what a measurement or a run depends on: Python, torch and its CUDA/cuDNN builds, the
driver and the GPUs as `nvidia-smi` reports them, the CPU, the thread settings, the library versions and the git
commit of the checkout with its dirty flag. A field that cannot be read on this host is null, never guessed: on a
CPU-only host the CUDA, cuDNN, driver and GPU fields are null.

`code_identity()` is the one implementation of (commit, dirty) used by runs, packs and assembly manifests.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..canonical import atomic_write_json

ENV_FORMAT = "bilingual-env/1"
SRC_DIR = Path(__file__).resolve().parents[2]  # fork/bilingual/src
REQUIRED_KEYS = (
    "format", "python", "platform", "torch", "cuda", "cudnn", "driver", "gpu", "cpu", "threads", "numpy",
    "tokenizers", "safetensors", "git_commit", "git_dirty", "created",
)  # fmt: skip


def code_identity() -> tuple[str | None, bool | None]:
    """(commit, dirty) of the checkout this package runs from; dirty = uncommitted or untracked changes under
    fork/bilingual/src. (None, None) outside a git checkout."""
    try:
        commit = subprocess.run(
            ["git", "-C", str(SRC_DIR), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(SRC_DIR), "status", "--porcelain", "--", "."],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    return commit, bool(status.strip())


def _version(module: str) -> str | None:
    try:
        return getattr(__import__(module), "__version__", None)
    except ImportError:
        return None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text("utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _nvidia_smi() -> list[dict[str, str]] | None:
    query = "index,name,driver_version,memory.total,compute_cap"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    keys = query.split(",")
    return [dict(zip(keys, (v.strip() for v in line.split(",")), strict=False)) for line in out.splitlines() if line]


def environment() -> dict[str, Any]:
    import datetime

    import torch

    commit, dirty = code_identity()
    smi = _nvidia_smi()
    cuda = torch.cuda.is_available()
    gpus = None
    if cuda:
        gpus = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({"index": i, "name": p.name, "capability": f"{p.major}.{p.minor}",
                         "memory_bytes": int(p.total_memory)})  # fmt: skip
    return {
        "format": ENV_FORMAT,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if cuda else None,
        "cudnn": torch.backends.cudnn.version() if cuda and torch.backends.cudnn.is_available() else None,
        "driver": smi[0].get("driver_version") if smi else None,
        "gpu": gpus,
        "nvidia_smi": smi,
        "cpu": {"model": _cpu_model(), "machine": platform.machine(), "logical_cpus": os.cpu_count()},
        "threads": {
            "torch": torch.get_num_threads(),
            "torch_interop": torch.get_num_interop_threads(),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },  # fmt: skip
        "numpy": _version("numpy"),
        "tokenizers": _version("tokenizers"),
        "safetensors": _version("safetensors"),
        "git_commit": commit,
        "git_dirty": dirty,
        "created": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_environment(path: Path) -> Path:
    return atomic_write_json(Path(path), environment())


def main(argv: Sequence[str]) -> int:
    """`env --out PATH [--dry-run]`"""
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual env", description="write environment.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the fingerprint; write it into a temp dir only")
    args = ap.parse_args(list(argv))
    if args.dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_environment(Path(tmp) / "environment.json")
            print(path.read_text("utf-8"), end="")
        return 0
    print(write_environment(args.out))
    return 0
