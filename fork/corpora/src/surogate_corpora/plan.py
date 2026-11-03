"""`corpora-plan/1`: which row groups a profile reads, chosen deterministically up to a byte budget.

- Every row group of every file of the profile's sources is a unit. Its rank key is
  sha256(`seed \\0 repo \\0 revision \\0 path \\0 rg`); units are taken in ascending rank key. Row order inside a shard
  is not random (Ultra-FineWeb zh comes in runs of one upstream source, about 270-400 documents each, and the first
  rows of a file are all Tele), so a plan never takes "the first rows": the unit is a whole row group, drawn from all
  files.
- A lazy metadata pass walks the units in rank order. For each it reads only the cheap columns its profile's
  metadata filters need (Stack-v3: licence type and vendor flag, plus the per-file size; Ultra-FineWeb: score and
  source) and estimates the bytes kept: the summed `size_bytes` of the rows that pass, or the content column's
  uncompressed size times the share of rows that pass. A profile without metadata filters estimates the content
  column's uncompressed size. Content filters (dedup, UTF-8, non-Latin share) are not estimated; `margin` covers them.
- Units are taken until the summed estimate reaches `budget × (1 + margin)`, so the plan overshoots by less than one
  unit. A plan whose units run out first records `budget_reached: false`.
- `plan_sha256` is the sha256 of the canonical JSON of the plan without that field. The same inputs give a
  byte-identical plan file.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import CorporaError, atomic_write_json, check_sha256, read_json, with_sha256
from .hub import HubClient, RemoteFile
from .parquet import FooterInfo, RemoteFiles, RemoteParquet, iter_rows
from .profiles import Pipeline, Profile, parse_params, resolve
from .sources import SourceDef
from .sources import get as get_source

FORMAT = "corpora-plan/1"


def rank_key(seed: int, repo: str, revision: str, path: str, rg: int) -> str:
    return hashlib.sha256(f"{seed}\0{repo}\0{revision}\0{path}\0{rg}".encode()).hexdigest()


@dataclass(frozen=True)
class Plan:
    doc: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        return self.doc["plan_sha256"]

    @property
    def units(self) -> list[Mapping[str, Any]]:
        return list(self.doc["units"])

    def file(self, path: str) -> Mapping[str, Any]:
        return next(f for f in self.doc["files"] if f["path"] == path)

    def remote(self, path: str) -> RemoteFile:
        entry = self.file(path)
        source = get_source(entry["source"])
        return RemoteFile(source.repo, source.revision, path, entry["size"], entry["lfs_sha256"], entry["xet_hash"])

    def profile(self) -> Profile:
        """The profile the plan was made with; refuses a plan whose profile the code now defines differently."""
        profile = resolve(self.doc["profile"], self.doc["params"], allow_unlicensed=self.doc["allow_unlicensed"])
        if profile.describe() != self.doc["profile_def"]:
            raise CorporaError(
                f"profile {profile.name} is defined differently now than when the plan was made; re-plan"
            )
        return profile


def load_plan(path: Path) -> Plan:
    doc = read_json(path, "plan")
    check_sha256(doc, "plan_sha256")
    return Plan(doc)


def _estimate(pipeline: Pipeline, source: SourceDef, remote: RemoteParquet, rg: int) -> int:
    """Estimated kept bytes of one row group under the profile's metadata filters (reads only their columns)."""
    info = remote.footer.row_groups[rg]
    _, content_uncompressed = info.size_of(source.content)
    columns = list(pipeline.profile.metadata_columns())
    if source.size_column and source.size_column not in columns:
        columns.append(source.size_column)
    table = remote.read(rg, columns)
    metadata = pipeline.filters[: len(pipeline.profile.metadata_filters)]
    kept = kept_size = 0
    for row in iter_rows(table, ()):
        if all(f(row)[0] for f in metadata):
            kept += 1
            if source.size_column:
                kept_size += row.get(source.size_column) or 0
    return kept_size if source.size_column else content_uncompressed * kept // max(info.num_rows, 1)


def build_plan(
    profile: Profile,
    *,
    budget_bytes: int,
    seed: int,
    client: HubClient,
    margin: float = 0.05,
    dry_run_footers: Mapping[str, FooterInfo] | None = None,
    allow_unlicensed: bool = False,
) -> Plan:
    """The plan of `profile` for `budget_bytes` kept bytes. `dry_run_footers` (path -> FooterInfo) supplies footers
    already read; any other footer is read from the Hub."""
    if budget_bytes <= 0 or margin < 0:
        raise CorporaError("the budget must be positive and the margin non-negative")
    target = budget_bytes * (1 + margin)
    files: list[dict[str, Any]] = []
    units: list[tuple[str, str, int, SourceDef]] = []
    footers: dict[str, FooterInfo] = {}
    metadata_bytes = 0
    remotes: dict[str, RemoteFile] = {}
    opened_files = RemoteFiles(client)
    for source_id in profile.sources:
        source = get_source(source_id)
        for remote in client.list_files(source.repo, source.revision, source.files):
            if dry_run_footers is not None and remote.path in dry_run_footers:
                footer = dry_run_footers[remote.path]
            else:
                opened, fetched = opened_files.get(remote)
                footer, metadata_bytes = opened.footer, metadata_bytes + fetched
            if dict(footer.schema) != dict(source.schema):
                diff = sorted(set(footer.schema.items()) ^ set(source.schema.items()))
                raise CorporaError(f"{remote.path}: the parquet schema differs from the registry's: {diff[:4]}")
            footers[remote.path], remotes[remote.path] = footer, remote
            for rg in footer.row_groups:
                units.append(
                    (rank_key(seed, source.repo, source.revision, remote.path, rg.index), remote.path, rg.index, source)
                )
            files.append(
                {
                    "source": source_id,
                    "path": remote.path,
                    "size": remote.size,
                    "lfs_sha256": remote.lfs_sha256,
                    "xet_hash": remote.xet_hash,
                    "num_rows": footer.num_rows,
                    "created_by": footer.created_by,
                    "row_groups": [],
                }
            )
    units.sort()
    pipelines = {s: Pipeline(profile, get_source(s)) for s in profile.sources}
    estimates: dict[tuple[str, int], int] = {}
    chosen, total = [], 0
    for key, path, rg, source in units:
        if total >= target:
            break
        if not profile.metadata_columns():
            estimate = footers[path].row_groups[rg].size_of(source.content)[1]
        else:
            opened, fetched = opened_files.get(remotes[path])
            before = opened.bytes_fetched
            estimate = _estimate(pipelines[source.id], source, opened, rg)
            metadata_bytes += fetched + opened.bytes_fetched - before  # a re-opened footer, then the columns
        estimates[path, rg] = estimate
        total += estimate
        chosen.append({"path": path, "rg": rg, "rank_key": key, "est_kept_bytes": estimate})
    transfer = 0
    for entry in files:
        source = get_source(entry["source"])
        for info in footers[entry["path"]].row_groups:
            compressed, uncompressed = info.size_of(source.content)
            entry["row_groups"].append(
                {
                    "index": info.index,
                    "num_rows": info.num_rows,
                    "content_compressed": compressed,
                    "content_uncompressed": uncompressed,
                    "pass_compressed": info.size_of(source.columns)[0],
                    "est_kept_bytes": estimates.get((entry["path"], info.index)),
                }
            )
    by_path = {f["path"]: f for f in files}
    for unit in chosen:
        transfer += by_path[unit["path"]]["row_groups"][unit["rg"]]["pass_compressed"]
    doc = {
        "format": FORMAT,
        "profile": profile.name,
        "params": dict(sorted(profile.params.items())),
        "allow_unlicensed": allow_unlicensed,
        "profile_def": profile.describe(),
        "seed": seed,
        "budget": {"unit": profile.budget_unit, "value": budget_bytes, "margin": margin},
        "sources": [{"id": s, "repo": get_source(s).repo, "revision": get_source(s).revision} for s in profile.sources],
        "files": files,
        "units": chosen,
        "estimate": {
            "transfer_bytes": transfer,
            "kept_bytes": total,
            "metadata_bytes_fetched": metadata_bytes,
            "units": len(chosen),
            "units_total": len(units),
            "budget_reached": total >= target,
        },
    }
    return Plan(with_sha256(doc, "plan_sha256"))


def summary(plan: Plan) -> str:
    e = plan.doc["estimate"]
    budget = plan.doc["budget"]
    lines = [
        f"profile {plan.doc['profile']}  seed {plan.doc['seed']}  budget {budget['value']:,} {budget['unit']} "
        f"(+{budget['margin']:.0%})",
        f"units {e['units']:,} of {e['units_total']:,} row groups; estimated kept bytes {e['kept_bytes']:,}; "
        f"estimated transfer {e['transfer_bytes']:,} bytes; metadata fetched {e['metadata_bytes_fetched']:,} bytes",
    ]
    if not e["budget_reached"]:
        lines.append("warning: every row group is in the plan and the budget is still not reached")
    lines.append(f"plan_sha256 {plan.sha256}")
    return "\n".join(lines)


def client_from(args) -> HubClient:
    return HubClient(token=os.environ.get("HF_TOKEN") or None, endpoint=args.endpoint)


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_corpora plan", description="write a corpora-plan/1")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--budget-bytes", type=int, required=True, help="kept bytes to plan for")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--param", action="append", help="name=value: a profile parameter (repeatable)")
    parser.add_argument("--allow-unlicensed", action="store_true", help="owner flag for the Code-L2 ablation")
    parser.add_argument("--out", type=Path, help="the plan file to write (required unless --dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="build and print the plan's estimates; write nothing")
    parser.add_argument("--endpoint", default="https://huggingface.co")
    args = parser.parse_args(argv)
    if not args.dry_run and args.out is None:
        parser.error("--out is required unless --dry-run")
    try:
        profile = resolve(args.profile, parse_params(args.param), allow_unlicensed=args.allow_unlicensed)
        plan = build_plan(
            profile,
            budget_bytes=args.budget_bytes,
            seed=args.seed,
            client=client_from(args),
            margin=args.margin,
            allow_unlicensed=args.allow_unlicensed,
        )
    except CorporaError as err:
        print(f"surogate_corpora plan: refused: {err}", file=sys.stderr)
        return 1
    print(summary(plan))
    if not args.dry_run:
        atomic_write_json(args.out, dict(plan.doc))
        print(f"wrote {args.out}")
    return 0
