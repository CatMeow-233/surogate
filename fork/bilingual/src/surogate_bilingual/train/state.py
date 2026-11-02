"""Training state (`bilingual-state/1`; docs/spec.md, "训练状态"; IMPL-02): atomic save, retention, verified load.

    runs/<run>/state/step_00003052/
        trainable.safetensors   the trainable tensors only, by parameter name
        optimizer.pt            optimizer.state_dict(), read back with torch.load(weights_only=True)
        rng_torch.pt            the torch CPU (and CUDA) RNG states
        extra.json              run_id, step, tokens, frozen_hashes, scheduler, sampler, python/numpy RNG states,
                                the trainable names, the sha256 of the three files above

A state directory is written by `canonical.atomic_directory`: the files go into `step_N.tmp-<pid>-<rand>`, are
fsynced, and one `os.replace` makes the state visible. A crash before the rename leaves only the temporary
directory, which `remove_stale_temps` deletes on the next start; the previous state is untouched. Retention keeps
the last two states plus every state whose step is a multiple of `keep_every`; a dropped state is first renamed to
a temporary name, so a crash while deleting it never leaves a partial `step_N` behind.

`load_state` checks the format, the run_id and the file hashes. The loop then checks the frozen hashes against the
live model, and restores the sampler cursor and the RNG states before it draws the first batch.
"""

from __future__ import annotations

import json
import random
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from ..canonical import (
    TEMP_MARK,
    atomic_directory,
    check_format,
    remove_stale_temps,
    sha256_file,
)

STATE_FORMAT = "bilingual-state/1"
STATE_DIR = "state"
STEP_DIR = re.compile(r"step_(\d{8})")
TRAINABLE = "trainable.safetensors"
OPTIMIZER = "optimizer.pt"
RNG_TORCH = "rng_torch.pt"
EXTRA = "extra.json"


class StateError(ValueError):
    pass


def step_name(step: int) -> str:
    return f"step_{int(step):08d}"


def list_states(run_dir: Path) -> list[tuple[int, Path]]:
    root = Path(run_dir) / STATE_DIR
    if not root.is_dir():
        return []
    out = []
    for path in root.iterdir():
        match = STEP_DIR.fullmatch(path.name)
        if match and path.is_dir():
            out.append((int(match.group(1)), path))
    return sorted(out)


def latest_state(run_dir: Path) -> Path | None:
    states = list_states(run_dir)
    return states[-1][1] if states else None


def clean_stale(run_dir: Path) -> list[Path]:
    """Remove the temporary directories crashed saves (or prunes) left behind."""
    return remove_stale_temps(Path(run_dir) / STATE_DIR)


# --- RNG ------------------------------------------------------------------------------------------------------------


def capture_rng() -> dict[str, Any]:
    version, internal, gauss = random.getstate()
    np_state = np.random.get_state(legacy=False)
    np_state = {**np_state, "state": {**np_state["state"], "key": np_state["state"]["key"].tolist()}}
    return {"python": [version, list(internal), gauss], "numpy": np_state}


def restore_rng(state: Mapping[str, Any]) -> None:
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(internal), gauss))
    np_state = dict(state["numpy"])
    np_state["state"] = {**np_state["state"], "key": np.asarray(np_state["state"]["key"], dtype=np.uint32)}
    np.random.set_state(np_state)


def torch_rng_states() -> dict[str, Any]:
    cuda = [torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    return {"cpu": torch.get_rng_state(), "cuda": cuda}


def restore_torch_rng(states: Mapping[str, Any]) -> None:
    torch.set_rng_state(states["cpu"])
    if states["cuda"]:
        if not torch.cuda.is_available() or torch.cuda.device_count() < len(states["cuda"]):
            raise StateError("the state holds CUDA RNG states for more devices than this host has")
        for i, s in enumerate(states["cuda"]):
            torch.cuda.set_rng_state(s, i)


# --- save -----------------------------------------------------------------------------------------------------------


def save_state(
    run_dir: Path,
    step: int,
    *,
    trainable: Mapping[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    extra: Mapping[str, Any],
    keep_every: int = 0,
) -> Path:
    """Write state `step` atomically, then apply the retention rule. `extra` carries run_id, tokens,
    frozen_hashes, scheduler and sampler; the format, step, RNG states, names and file hashes are added here."""
    root = Path(run_dir) / STATE_DIR
    target = root / step_name(step)
    with atomic_directory(target) as tmp:
        save_file({k: v.detach().to("cpu").contiguous() for k, v in trainable.items()}, str(tmp / TRAINABLE))
        torch.save(optimizer.state_dict(), tmp / OPTIMIZER)
        torch.save(torch_rng_states(), tmp / RNG_TORCH)
        doc = {
            **dict(extra),
            "format": STATE_FORMAT,
            "step": int(step),
            "trainable_names": sorted(trainable),
            "rng": capture_rng(),
            "files": {name: sha256_file(tmp / name) for name in (TRAINABLE, OPTIMIZER, RNG_TORCH)},
        }
        (tmp / EXTRA).write_text(json.dumps(doc, sort_keys=True, indent=1, allow_nan=False) + "\n", "utf-8")
    prune(run_dir, keep_every)
    return target


def retained(steps: list[int], keep_every: int) -> set[int]:
    keep = set(steps[-2:])
    if keep_every > 0:
        keep |= {s for s in steps if s % keep_every == 0}
    return keep


def prune(run_dir: Path, keep_every: int) -> list[int]:
    states = list_states(run_dir)
    keep = retained([s for s, _ in states], keep_every)
    dropped = []
    for step, path in states:
        if step not in keep:
            doomed = path.with_name(f"{path.name}{TEMP_MARK}prune")
            path.rename(doomed)
            shutil.rmtree(doomed)
            dropped.append(step)
    return dropped


# --- load -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedState:
    path: Path
    step: int
    trainable: dict[str, torch.Tensor]
    optimizer: dict[str, Any]
    torch_rng: dict[str, Any]
    extra: dict[str, Any]


def load_state(path: Path, *, run_id: str, map_location: str | torch.device = "cpu") -> LoadedState:
    path = Path(path)
    try:
        extra = json.loads((path / EXTRA).read_text("utf-8"))
        check_format(extra, "state")
    except (OSError, ValueError) as exc:
        raise StateError(f"{path}: {exc}") from exc
    if extra.get("run_id") != run_id:
        raise StateError(
            f"{path} belongs to run_id {extra.get('run_id')}, this run is {run_id}; refusing to resume another run"
        )
    for name, digest in extra["files"].items():
        if sha256_file(path / name) != digest:
            raise StateError(f"{path / name} does not match the sha256 recorded in {EXTRA}")
    trainable = load_file(str(path / TRAINABLE))
    if sorted(trainable) != extra["trainable_names"]:
        raise StateError(f"{path / TRAINABLE} holds other tensors than {EXTRA} lists")
    optimizer = torch.load(path / OPTIMIZER, map_location=map_location, weights_only=True)
    torch_rng = torch.load(path / RNG_TORCH, map_location="cpu", weights_only=True)
    return LoadedState(path, int(extra["step"]), trainable, optimizer, torch_rng, extra)
