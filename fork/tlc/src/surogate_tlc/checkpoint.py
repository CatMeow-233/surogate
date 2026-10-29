"""Turn native checkpoint and export directories into model directories (docs/model.md: checkpoint 与导出).

The native trainer saves `<checkpoint_dir>/step_<8 digits>/` with a config.json of generic keys only (no
auto_map, no TLC keys) and no tokenizer, and writes checkpoint.json there last. Its final export copies a fixed
file list that lacks configuration_tlc.py and tlc_meta.json. finalize_dir() copies the source model directory's
companion files into such a directory; a directory that has not been finalized must not be used as model_dir.

The native trainer owns step directories, so finalize_dir() can only add files to them: each file is replaced
atomically and tlc_meta.json, which check_model_dir() requires, comes last. export() builds its directory under a
temporary name and renames it into place, so `out` either is a complete model directory or does not change.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import hf
from .files import copy_atomic, dump_json, write_text_atomic
from .modeldir import CONFIG_FILE, META_FILE, WEIGHTS_FILE, check_model_dir, companion_files

STEP_DIR = re.compile(r"step_(\d{8})")
NATIVE_DONE_FILE = "checkpoint.json"
# Keys the native trainer writes back as it read them; a difference means the wrong source model directory.
STRUCTURAL_KEYS = ("architectures", "model_type", "vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size")


def finalize_dir(target: Path, source: Path, *, meta: Mapping[str, Any] | None = None) -> Path:
    """Copy the companion files of model directory `source` into `target`; tlc_meta.json gets `meta` merged in.

    Idempotent. tlc_meta.json is written last, so an interrupted first run leaves a directory that fails
    check_model_dir() until finalize_dir() runs again.
    """
    target, source = Path(target), Path(source)
    source_cfg = check_model_dir(source, weights=False)
    if not target.is_dir():
        raise FileNotFoundError(f"{target} is not a directory")
    if (target / CONFIG_FILE).is_file():
        written = json.loads((target / CONFIG_FILE).read_text(encoding="utf-8"))
        differ = [k for k in STRUCTURAL_KEYS if k in written and written[k] not in _native_values(source_cfg, k)]
        if differ:
            raise ValueError(
                f"{target / CONFIG_FILE} disagrees with {source} on {differ}: wrong source model directory?"
            )
    for name in companion_files(source):
        if name != META_FILE:
            copy_atomic(source / name, target / name)
    tlc_meta = json.loads((source / META_FILE).read_text(encoding="utf-8")) | dict(meta or {})
    write_text_atomic(target / META_FILE, dump_json(tlc_meta))
    return target


def _native_values(source_cfg: Mapping[str, Any], key: str) -> list[Any]:
    """Values of `key` the native trainer may write for a model built from `source_cfg`."""
    values = [source_cfg.get(key)]
    if key == "num_hidden_layers" and source_cfg.get(hf.SPLIT_KEY):
        # The native config takes the DSL's layer count, which leaves out the model-level layer 1.
        values.append(source_cfg[key] - 1)
    return values


def checkpoint_step(path: Path) -> int | None:
    """Step of a complete native checkpoint directory, None for anything else."""
    path = Path(path)
    match = STEP_DIR.fullmatch(path.name)
    if not match or not (path / NATIVE_DONE_FILE).is_file():
        return None
    return int(match.group(1))


def finalize_checkpoints(checkpoint_dir: Path, source: Path, *, meta: Mapping[str, Any] | None = None) -> list[Path]:
    """finalize_dir() every complete step directory under `checkpoint_dir`; incomplete ones are left alone."""
    done = []
    for path in sorted(Path(checkpoint_dir).iterdir()):
        step = checkpoint_step(path)
        if step is not None:
            done.append(finalize_dir(path, source, meta={**(meta or {}), "step": step}))
    return done


def export(checkpoint: Path, source: Path, out: Path) -> Path:
    """Write a model directory `out` from one native checkpoint: its weights plus `source`'s companion files.

    Optimizer state and shards stay behind. `out` must be absent or empty, and is left unchanged on a failure.
    """
    checkpoint, out = Path(checkpoint), Path(out)
    step = checkpoint_step(checkpoint)
    if step is None:
        raise ValueError(f"{checkpoint} is not a complete native checkpoint (step_<8 digits> with {NATIVE_DONE_FILE})")
    if not (checkpoint / WEIGHTS_FILE).is_file():
        raise FileNotFoundError(f"{checkpoint / WEIGHTS_FILE} is missing")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        tmp.chmod(0o755)  # mkdtemp's 0700 would carry over to `out`
        # The native config.json comes along so that finalize_dir() can compare it with the source.
        for name in (CONFIG_FILE, WEIGHTS_FILE):
            if (checkpoint / name).is_file():
                shutil.copyfile(checkpoint / name, tmp / name)
        # A finalized checkpoint's tlc_meta.json carries the run's provenance (tokenizer and data hashes); keep it.
        meta_file = checkpoint / META_FILE
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.is_file() else {}
        finalize_dir(tmp, source, meta={**meta, "step": step})
        check_model_dir(tmp)
        if out.exists():
            out.rmdir()
        os.replace(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out
