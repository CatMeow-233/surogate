"""Run identity and run directories (`bilingual-run/1`; docs/spec.md, "运行登记"; IMPL-21).

    data/runs/index.json            {"run_name": "run_id"}: a name is bound to exactly one id
    data/runs/<run_name>/run.json   identity, config, status
                         environment.json (+ environment.step_<N>.json for each resume)
                         metrics.jsonl   state/   packs/   eval/

The runs directory is `data/runs` under the run root (the repository root on the GPU host): the root .gitignore
already ignores data/, so run outputs never show up in `git status`, never make the checkout dirty and never block
`fork/dev sync` (whose clean-tree check counts untracked files). `--runs` puts it elsewhere.

`run_id` is the first 16 hex characters of the sha256 of the canonical JSON of the identity fields (config_hash,
data_manifest_sha256, tokenizer_sha256, seed_fingerprint, code_commit, recipe, init_seed, data_seed). Creating a
run whose name is bound to another id is refused; the same name with the same id resumes. A dirty checkout
(`code_dirty: true`) is refused unless `allow_dirty` is set, and the flag is then recorded in run.json.

`registry list` / `registry show NAME` / `registry budget` read run directories; they never write. `budget` tabulates
tokens, P_update and device time per matrix row and replicate, checks the token matching of the pilot rows
(TOKEN_MATCHES) and totals the tuning-ledger trials (`run.trial`).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..canonical import atomic_write_json, check_format, remove_stale_temps, sha256_json
from ..config import run_name_problem
from . import env

RUN_FORMAT = "bilingual-run/1"
RUNS_DIR = Path("data") / "runs"  # under the run root
INDEX = "index.json"
RUN_JSON = "run.json"
IDENTITY_KEYS = (
    "config_hash",
    "data_manifest_sha256",
    "tokenizer_sha256",
    "seed_fingerprint",
    "code_commit",
    "recipe",
    "init_seed",
    "data_seed",
)
STATUSES = ("created", "running", "done", "failed")


class RegistryError(ValueError):
    pass


def now() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_id(identity: Mapping[str, Any]) -> str:
    missing = [k for k in IDENTITY_KEYS if k not in identity]
    if missing:
        raise RegistryError(f"run identity lacks {missing}")
    return sha256_json({k: identity[k] for k in IDENTITY_KEYS})[:16]


def check_name(name: str) -> str:
    problem = run_name_problem(name)
    if problem:
        raise RegistryError(problem)
    return name


def read_index(runs: Path) -> dict[str, str]:
    path = Path(runs) / INDEX
    if not path.is_file():
        return {}
    index = json.loads(path.read_text("utf-8"))
    if not isinstance(index, dict) or not all(isinstance(v, str) for v in index.values()):
        raise RegistryError(f"{path} is not a {{run_name: run_id}} table")
    return index


def read_run(run_dir: Path) -> dict[str, Any]:
    doc = json.loads((Path(run_dir) / RUN_JSON).read_text("utf-8"))
    check_format(doc, "run")
    return doc


def open_run(
    runs: Path,
    name: str,
    identity: Mapping[str, Any],
    fields: Mapping[str, Any],
    *,
    allow_dirty: bool,
    code: tuple[str | None, bool | None] | None = None,
) -> tuple[Path, dict[str, Any], bool]:
    """Create the run `name`, or reopen it when it exists with the same id. Returns (run dir, run.json, resumed).

    `identity` holds every IDENTITY_KEYS field except code_commit, which comes from `code` (default: this
    checkout's `env.code_identity()`); `fields` are the other run.json fields (matrix_row, replicate, config, ...).
    """
    check_name(name)
    runs = Path(runs)
    runs.mkdir(parents=True, exist_ok=True)
    remove_stale_temps(runs)
    commit, dirty = env.code_identity() if code is None else code
    if dirty and not allow_dirty:
        raise RegistryError(
            "the checkout has uncommitted changes under fork/bilingual/src (code_dirty); commit them, or set"
            " registry.allow_dirty = true to run anyway (the flag is recorded in run.json)"
        )
    ident = {**dict(identity), "code_commit": commit}
    rid = run_id(ident)
    index = read_index(runs)
    if name in index and index[name] != rid:
        raise RegistryError(
            f"run name {name!r} is bound to run_id {index[name]}, this config gives {rid}: the config, inputs, seeds"
            " or code differ. Use another run name (or delete the old run deliberately)."
        )
    run_dir = runs / name
    resumed = False
    if (run_dir / RUN_JSON).is_file():
        doc = read_run(run_dir)
        if doc["run_id"] != rid:
            raise RegistryError(f"{run_dir / RUN_JSON} holds run_id {doc['run_id']}, this config gives {rid}")
        resumed = True
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RegistryError(f"{run_dir} exists without {RUN_JSON}; refusing to reuse it")
        doc = {
            "format": RUN_FORMAT,
            "run_name": name,
            "run_id": rid,
            **{k: ident[k] for k in IDENTITY_KEYS},
            **dict(fields),
            "code_dirty": dirty,
            "allow_dirty": bool(allow_dirty),
            "status": "created",
            "failure": None,
            "created": now(),
        }
        atomic_write_json(run_dir / RUN_JSON, doc)
    if index.get(name) != rid:
        atomic_write_json(runs / INDEX, {**index, name: rid})
    return run_dir, doc, resumed


def update_run(run_dir: Path, **changes) -> dict[str, Any]:
    doc = read_run(run_dir)
    if "status" in changes and changes["status"] not in STATUSES:
        raise RegistryError(f"status must be one of {STATUSES}: {changes['status']!r}")
    doc.update(changes)
    atomic_write_json(Path(run_dir) / RUN_JSON, doc)
    return doc


def list_runs(runs: Path) -> list[dict[str, Any]]:
    out = []
    for name, rid in sorted(read_index(runs).items()):
        path = Path(runs) / name / RUN_JSON
        doc = read_run(path.parent) if path.is_file() else {"run_name": name, "run_id": rid, "status": "missing"}
        out.append(doc)
    return out


# Token matching of the pilot (MS §8.4, §13): per replicate, the finished tokens of the left row equal the summed
# finished tokens of the right row(s). B2 = ZH + EN (both B1 runs); B12a = B2; T1' (both directions) = T2 = T3' = T3''.
TOKEN_MATCHES = (("B2", "B1"), ("B12a", "B2"), ("T2", "T1'"), ("T3'", "T2"), ("T3''", "T2"))


def budget_table(runs: Path) -> dict[str, Any]:
    """Tokens, P_update and device time per (matrix row, replicate) of the confirmatory runs, the token-matching
    checks, and the tuning-ledger trials (run.trial) with their compute (MS §9.6: tuning counts toward the FLOPs)."""
    groups: dict[tuple, dict[str, Any]] = {}
    tuning = []
    for doc in list_runs(runs):
        if doc.get("status") == "missing":
            continue
        tokens = doc.get("finished_tokens") if doc.get("status") == "done" else None
        seconds = (doc.get("compute") or {}).get("device_seconds")
        gpu_hours = (doc.get("compute") or {}).get("gpu_hours")
        if doc.get("trial") is not None:
            tuning.append(
                {
                    "run": doc["run_name"],
                    "row": doc.get("matrix_row"),
                    "replicate": doc.get("replicate"),
                    "trial": doc["trial"],
                    "status": doc.get("status"),
                    "tokens": tokens,
                    "gpu_hours": gpu_hours,
                }
            )
            continue
        key = (doc.get("matrix_row"), doc.get("replicate"))
        g = groups.setdefault(
            key, {"row": key[0], "replicate": key[1], "runs": [], "tokens": 0, "p_update": [], "gpu_hours": 0.0}
        )
        g["runs"].append(doc["run_name"])
        g["p_update"].append(doc.get("p_update"))
        g["tokens"] = None if tokens is None or g["tokens"] is None else g["tokens"] + tokens
        g["gpu_hours"] = None if gpu_hours is None or g["gpu_hours"] is None else g["gpu_hours"] + gpu_hours
        g.setdefault("device_seconds", 0.0)
        g["device_seconds"] = None if seconds is None or g["device_seconds"] is None else g["device_seconds"] + seconds
    checks = []
    for left, right in TOKEN_MATCHES:
        for rep in sorted({r for (row, r) in groups if row in (left, right)}, key=str):
            a, b = groups.get((left, rep)), groups.get((right, rep))
            if a is None or b is None or a["tokens"] is None or b["tokens"] is None:
                status = "incomplete"
            else:
                status = "matched" if a["tokens"] == b["tokens"] else "mismatch"
            checks.append(
                {
                    "rows": [left, right],
                    "replicate": rep,
                    "tokens": [a and a["tokens"], b and b["tokens"]],
                    "status": status,
                }
            )
    hours = [t["gpu_hours"] for t in tuning]
    return {
        "rows": [groups[k] for k in sorted(groups, key=str)],
        "token_matches": checks,
        "tuning": {
            "trials": tuning,
            "gpu_hours": sum(hours) if hours and all(h is not None for h in hours) else None,
        },
    }


def main(argv: Sequence[str]) -> int:
    """`registry list|budget [--runs DIR]` | `registry show NAME [--runs DIR]` (read-only; --dry-run is accepted)"""
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual registry", description="list or show runs")
    ap.add_argument("action", choices=["list", "show", "budget"])
    ap.add_argument("name", nargs="?")
    ap.add_argument("--runs", type=Path, default=RUNS_DIR, help="default: data/runs")
    ap.add_argument("--dry-run", action="store_true", help="no effect: the command only reads")
    args = ap.parse_args(list(argv))
    try:
        if args.action == "list":
            for doc in list_runs(args.runs):
                print(
                    f"{doc['run_name']}\t{doc['run_id']}\t{doc.get('status')}\t{doc.get('recipe')}\t"
                    f"{doc.get('matrix_row')}\tr{doc.get('replicate')}\t{doc.get('trial') or '-'}"
                )
            return 0
        if args.action == "budget":
            print(json.dumps(budget_table(args.runs), indent=1, sort_keys=True, ensure_ascii=False))
            return 0
        if not args.name:
            ap.error("show needs a run name")
        index = read_index(args.runs)
        if args.name not in index:
            print(f"registry: no run {args.name!r} in {args.runs / INDEX}", file=sys.stderr)
            return 1
        print(json.dumps(read_run(args.runs / args.name), indent=1, sort_keys=True, ensure_ascii=False))
        return 0
    except (OSError, ValueError) as exc:
        print(f"registry: {exc}", file=sys.stderr)
        return 1
