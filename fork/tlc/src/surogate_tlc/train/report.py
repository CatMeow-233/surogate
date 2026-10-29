"""What a TLC run records besides its weights: the environment fingerprint and the summary of an attempt.

Both are plain data in the run record (RUN_FILE). The summary reads two files upstream writes: the training
log (a JSON array, csrc/src/runtime/training/logging.cpp) and the regression artifact the trainer writes when
SUROGATE_REGRESSION_ARTIFACT names a path (loss curve, step time percentiles, peak device memory, block-schema
summary). The CPU pin tests fix the fields read here.

`python -m surogate_tlc.train.report DIR [--peak-tflops X]` prints one JSON line per run record under DIR (gates
G0.5 and G3): the model FLOPs utilisation needs the GPU's dense bf16 peak, which the runbook takes from the GPU's
data sheet. Loss and eval points join every attempt of a run, so a run that failed and was resumed reads from step 1.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..files import read_json
from ..ir import flops_per_token
from ..native import extension_fingerprint


def _checkout() -> Path:
    """The git checkout this package runs from (editable install or the source tree)."""
    here = Path(__file__).resolve()
    return next((p for p in here.parents if (p / ".git").exists()), here.parent)


REPO = _checkout()
RUN_FILE = "tlc_run.json"  # the run record, in the run directory
PACKAGES = ("torch", "transformers", "tokenizers", "safetensors", "numpy", "surogate-tlc")


def _run(*command: str) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return {"torch_cuda": torch.version.cuda, "available": False}
    properties = torch.cuda.get_device_properties(0)
    driver = _run("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
    return {
        "torch_cuda": torch.version.cuda,
        "available": True,
        "name": properties.name,
        "capability": list(torch.cuda.get_device_capability(0)),
        "memory_bytes": properties.total_memory,
        "driver": driver.splitlines()[0] if driver else None,
    }


def environment() -> dict[str, Any]:
    """Environment fingerprint: commit, interpreter, package versions, native build, GPU and driver, so that a result
    can be traced to the code and machine that produced it."""
    status = _run("git", "-C", str(REPO), "status", "--porcelain")
    return {
        "git_commit": _run("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _version(name) for name in PACKAGES},
        "native": extension_fingerprint(),
        "gpu": _gpu(),
    }


def summarize(regression: Path, log: Path | None, tokens_per_update: int) -> dict[str, Any]:
    """Loss, throughput and memory of one attempt, from the files the trainer wrote; missing files give less."""
    out: dict[str, Any] = {}
    if regression.is_file():
        data = json.loads(regression.read_text(encoding="utf-8"))
        curve = data.get("convergence_curve") or []
        if curve:
            out["steps"] = [int(curve[0]["step"]), int(curve[-1]["step"])]
            out["first_loss"], out["last_loss"] = curve[0]["loss"], curve[-1]["loss"]
            out["loss"] = [[int(point["step"]), point["loss"]] for point in curve]
        p50 = (data.get("step_time_ms") or {}).get("p50")
        if p50:
            out["step_time_ms_p50"] = p50
            out["tokens_per_second"] = round(tokens_per_update / (p50 / 1000))
        out["cuda_peak_memory_bytes"] = data.get("cuda_peak_memory_bytes")
        out["block_schema_summary"] = data.get("block_schema_summary")
    if log is not None and log.is_file():
        entries = json.loads(log.read_text(encoding="utf-8"))
        out["eval_loss"] = [[entry["step"], entry["loss"]] for entry in entries if entry.get("log") == "eval"]
    return out


def _log_entries(log: str | None) -> list[dict[str, Any]]:
    """Entries of an attempt's training log; none when it is missing or unreadable."""
    try:
        entries = json.loads(Path(log).read_text(encoding="utf-8")) if log else []
    except (OSError, ValueError):
        return []
    return entries if isinstance(entries, list) else []


def joined(record: dict[str, Any]) -> dict[str, list[list[Any]]]:
    """Loss and eval points of every attempt of a run as one run: an attempt's points replace those after the step it
    resumed from. A completed attempt's loss comes from its summary (the regression artifact, steps from 1); an attempt
    that failed wrote no artifact, so its loss comes from the step lines of its training log, which count steps from 0
    (pinned). Eval points are the log's eval lines."""
    loss: dict[int, Any] = {}
    evals: dict[int, Any] = {}
    for attempt in record["attempts"]:
        start = attempt.get("resumed_from_step") or 0
        loss = {step: value for step, value in loss.items() if step <= start}
        evals = {step: value for step, value in evals.items() if step <= start}
        summary = attempt.get("summary") or {}
        entries = _log_entries(attempt.get("log_file"))
        if "loss" in summary:
            loss.update((int(step), value) for step, value in summary["loss"])
        else:
            loss.update((int(e["step"]) + 1, e["loss"]) for e in entries if e.get("log") == "step")
        evals.update((int(e["step"]), e["loss"]) for e in entries if e.get("log") == "eval")
    return {"loss": [[s, loss[s]] for s in sorted(loss)], "eval_loss": [[s, evals[s]] for s in sorted(evals)]}


def runs(root: Path, *, peak_tflops: float | None = None) -> list[dict[str, Any]]:
    """Every run record under `root`, flattened for a results table: loss and eval points of all attempts (joined()),
    throughput and memory of the latest. With the GPU's dense bf16 peak, the model FLOPs utilisation: tokens/s x
    training FLOPs per token (3 x forward, ir.flops_per_token) / peak."""
    out = []
    for path in sorted(Path(root).rglob(RUN_FILE)):
        record = read_json(path)
        attempt = record["attempts"][-1] if record["attempts"] else {}
        summary = attempt.get("summary") or {}
        sft = record.get("sft") or {}
        points = joined(record)
        curve = points["loss"]
        row = {
            "run_dir": str(path.parent),
            "stage": record["stage"],
            "status": record["status"],
            "attempts": len(record["attempts"]),
            "resumed_from_step": attempt.get("resumed_from_step"),
            "microbatch": attempt.get("microbatch"),
            "grad_accum": attempt.get("grad_accum"),
            "use_cuda_graphs": sft.get("use_cuda_graphs"),
            "tokens_per_update": record["tokens_per_update"],
            "valid_targets_per_update": (record.get("data") or {}).get("valid_targets_per_update"),
            "steps": [curve[0][0], curve[-1][0]] if curve else None,
            "first_loss": curve[0][1] if curve else None,
            "last_loss": curve[-1][1] if curve else None,
            "eval_loss": points["eval_loss"],
            "tokens_per_second": summary.get("tokens_per_second"),
            "cuda_peak_memory_bytes": summary.get("cuda_peak_memory_bytes"),
        }
        config = Path(record["model_dir"]) / "config.json"
        if config.is_file() and sft.get("sequence_len"):
            row["forward_flops_per_token"] = flops_per_token(read_json(config), sft["sequence_len"])
            if peak_tflops and row["tokens_per_second"]:
                training = 3 * row["forward_flops_per_token"] * row["tokens_per_second"]
                row["mfu"] = round(training / (peak_tflops * 1e12), 4)
        out.append(row)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tlc.train.report", description=runs.__doc__)
    parser.add_argument("root", type=Path, help=f"directory searched for {RUN_FILE}")
    parser.add_argument("--peak-tflops", type=float, help="the GPU's dense bf16 peak (data sheet), for the MFU")
    args = parser.parse_args(argv)
    for row in runs(args.root, peak_tflops=args.peak_tflops):
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
