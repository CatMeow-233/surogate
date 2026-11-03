"""The run registry: run identities, run records and the prereg lock of each table, under one runs root.

    <runs>/<table>/prereg-lock.json             tokstudy-prereg-lock/1: the table's accepted prereg sha256s, in order
    <runs>/<table>/<arm>-s<seed>-<run id>/      one run (train.py): run.json (tokstudy-run/1), metrics.jsonl,
                                                events.jsonl, ckpt/, stable/, final.pt, branches/E<e>/

**Run id.** The first 16 hex of the sha256 of the canonical JSON of {arm_sha256, stream_sha256, config_hash,
code_commit, seed, f_target, schedule, budget}: the arm (its tokenizer by sha256), the stream, the training
configuration, the code, the seed, the table's F_target and learning-rate schedule, and the prereg's other fields
that change what a run computes (`budget`: the FLOP convention every step's F is counted in, and the budget points
that place the stable checkpoints). The prereg's thresholds, seeds, stages and signature are not in it: they do not
change a run. Any change gives another run; resuming or branching a run directory under another id is refused. The
code commit is `git rev-parse HEAD` of the checkout this package runs from (null outside a git checkout); the dirty
flag is recorded in the run record, not in the id, and a verdict may refuse dirty runs.

**Prereg lock.** The first run of a table records its prereg's sha256. A later run with another prereg is refused
unless it names an amendment reason, which is appended with the new sha256 (spec §3.4: changing the thresholds
after the first run needs `--amend "<reason>"`); a prereg that was accepted before is accepted again for a new run.
An existing run (resume, decay branch, evaluation) accepts only the prereg it started with or one accepted after it
(`since`, the run record's prereg sha256): an earlier or unrecorded prereg needs `--amend` as well, and the lock is
the one of the registry layout, `<runs>/<table>/prereg-lock.json` next to the run directory.
"""

from __future__ import annotations

import datetime
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .canonical import TokstudyError, atomic_write_json, read_json, sha256_json

RUN_FILE = "run.json"
LOCK_FILE = "prereg-lock.json"
ID_KEYS = ("arm_sha256", "stream_sha256", "config_hash", "code_commit", "seed", "f_target", "schedule", "budget")


def run_id(
    arm_sha256: str,
    stream_sha256: str,
    config_hash: str,
    code_commit: str | None,
    seed: int,
    f_target: int,
    schedule: Mapping[str, Any],
    budget: Mapping[str, Any],
) -> str:
    values = (arm_sha256, stream_sha256, config_hash, code_commit, int(seed), int(f_target), dict(schedule))
    inputs = dict(zip(ID_KEYS, (*values, dict(budget)), strict=True))
    return sha256_json(inputs)[:16]


def id_inputs(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record["id_inputs"][key] for key in ID_KEYS}


def code_identity(start: Path | None = None) -> dict[str, Any]:
    """{commit, dirty} of the git checkout holding `start` (default: this package); nulls outside one."""
    where = Path(start or __file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "-C", str(where), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(where), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def run_dir_for(root: Path, table: str, arm_id: str, seed: int, rid: str) -> Path:
    return Path(root) / table / f"{arm_id}-s{seed}-{rid}"


def now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def lock_path(root: Path, table: str) -> Path:
    return Path(root) / table / LOCK_FILE


def lock_prereg(root: Path, prereg, *, amend: str | None = None, since: str | None = None) -> dict[str, Any]:
    """Accept `prereg` for its table under `root` (module docstring); returns the lock document. With `since` (the
    prereg sha256 an existing run started with) only that prereg and the ones accepted after it are accepted."""
    path = lock_path(root, prereg.table)
    if not path.exists():
        if since is not None:
            raise TokstudyError(f"{path}: no prereg lock, but a run of table {prereg.table} exists (under {since})")
        lock = {
            "format": "tokstudy-prereg-lock/1",
            "table": prereg.table,
            "history": [{"sha256": prereg.sha256, "file": prereg.path.name, "reason": "first run", "at": now()}],
        }
        atomic_write_json(path, lock)
        return lock
    lock = read_json(path, "prereg-lock")
    if lock.get("table") != prereg.table:
        raise TokstudyError(f"{path}: the lock of table {lock.get('table')!r}, not {prereg.table!r}")
    history = [entry["sha256"] for entry in lock["history"]]
    if since is not None:
        if since not in history:
            raise TokstudyError(f"{path}: the lock does not hold the prereg {since} the run started with")
        history = history[history.index(since) :]
    if prereg.sha256 in history:
        return lock
    if not amend:
        what = "the run started with or accepted after it" if since is not None else "accepted for the table"
        raise TokstudyError(
            f"{prereg.path}: sha256 {prereg.sha256} is not a prereg {what} (latest {history[-1]}); "
            "a change needs --amend REASON"
        )
    lock["history"].append({"sha256": prereg.sha256, "file": prereg.path.name, "reason": amend, "at": now()})
    atomic_write_json(path, lock)
    return lock


def read_run(run_dir: Path) -> dict[str, Any]:
    return read_json(Path(run_dir) / RUN_FILE, "run")


def write_run(run_dir: Path, record: Mapping[str, Any]) -> Path:
    if record.get("format") != "tokstudy-run/1":
        raise TokstudyError("a run record is a tokstudy-run/1 document")
    if run_id(**id_inputs(record)) != record["run_id"]:
        raise TokstudyError("the run record's run_id does not match its id inputs")
    return atomic_write_json(Path(run_dir) / RUN_FILE, record)


def list_runs(root: Path, table: str | None = None) -> list[dict[str, Any]]:
    """Every run record under `root` (one table, or all), each with its directory, sorted by directory name; a
    record whose id does not match its inputs is refused."""
    root = Path(root)
    tables = [root / table] if table else sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    out = []
    for directory in tables:
        if not directory.is_dir():
            continue
        for run_dir in sorted(p for p in directory.iterdir() if (p / RUN_FILE).is_file()):
            record = read_run(run_dir)
            if run_id(**id_inputs(record)) != record.get("run_id"):
                raise TokstudyError(f"{run_dir}: run.json's run_id does not match its id inputs")
            out.append({**record, "dir": str(run_dir)})
    return out


def read_events(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "events.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
