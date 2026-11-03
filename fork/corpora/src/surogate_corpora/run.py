"""`fetch`: read a plan's row groups in rank order, filter them with the plan's profile and export them, resumably.

- Units are processed in the plan's order and rows in file order within a unit, so the same plan, profile and code
  give byte-identical parts, whether or not the run was interrupted.
- After each unit the parts are fsynced and one line goes to `done.jsonl`: the unit, its provenance record and the
  part state (paths, sizes, document counts). A rerun skips the units listed there (none is read twice), cuts the
  parts back to the last recorded state (dropping what an interrupted unit wrote), and rebuilds the dedup set and
  the attribution from the parts already written.
- When every unit is done, the parts are hashed and `spec_fragment.toml`, `attribution.jsonl` (code) and
  `provenance.json` (`corpora-export/1`) are written atomically. Rerunning a finished export rewrites only these.
- A plan whose profile the code now defines differently is refused, and so is a directory holding another plan's
  export or files that no done list explains.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .canonical import CorporaError, atomic_write_bytes, atomic_write_json, canonical_json, sha256_file, with_sha256
from .export import PART_BYTES, Writer, make_writer, write_attribution, write_fragment
from .filters import TEXTS
from .hub import HubClient
from .parquet import RemoteFiles, iter_rows
from .plan import Plan, load_plan, summary
from .profiles import Pipeline
from .sources import get as get_source

DONE = "done.jsonl"
DONE_FORMAT = "corpora-done/1"
EXPORT_FORMAT = "corpora-export/1"
FINAL_FILES = ("spec_fragment.toml", "attribution.jsonl", "provenance.json")


def code_state() -> dict[str, Any]:
    """The checkout's commit and whether tracked files differ from it (None when git is unavailable)."""
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True, timeout=30).stdout.strip()  # fmt: skip
        status = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, check=True, timeout=30).stdout  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def tool_versions() -> dict[str, str]:
    import pyarrow

    return {"python": platform.python_version(), "pyarrow": pyarrow.__version__, "surogate_corpora": __version__}


class DoneLog:
    """`done.jsonl`: a header line, then one line per finished unit."""

    def __init__(self, path: Path, plan: Plan):
        self.path = path
        self.records: list[dict[str, Any]] = []
        header = {"format": DONE_FORMAT, "plan_sha256": plan.sha256, "profile": plan.doc["profile"]}
        if path.exists():
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines or lines[0] != header:
                raise CorporaError(f"{path}: the done list belongs to another plan or profile; use a new directory")
            self.records = lines[1:]
        else:
            atomic_write_bytes(path, canonical_json(header) + b"\n")
        units = plan.units
        for i, record in enumerate(self.records):
            if i >= len(units) or (record["unit"]["path"], record["unit"]["rg"]) != (units[i]["path"], units[i]["rg"]):
                raise CorporaError(f"{path}: line {i + 2} is not the plan's unit {i}; use a new directory")

    def append(self, record: Mapping[str, Any]) -> None:
        with open(self.path, "ab") as f:
            f.write(canonical_json(record) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        self.records.append(dict(record))


def _check_directory(out_dir: Path, resume: bool) -> None:
    if not out_dir.exists():
        return
    entries = {p.name for p in out_dir.iterdir()}
    if DONE in entries and not resume:
        raise CorporaError(f"{out_dir} holds an export (done list); rerun without --no-resume, or use a new directory")
    if DONE not in entries and entries:
        raise CorporaError(f"{out_dir} is not empty and holds no done list; use a new directory")


def _unit(plan: Plan, unit: Mapping[str, Any], files: RemoteFiles, pipeline: Pipeline, writer: Writer) -> dict:
    """Read one row group, filter it, write its kept rows; the unit's provenance record. The footer bytes are those
    of opening the file, when this unit opened it."""
    entry = plan.file(unit["path"])
    source = get_source(entry["source"])
    remote, footer_bytes = files.get(plan.remote(unit["path"]))
    groups = remote.footer.row_groups
    planned = [(g["index"], g["num_rows"]) for g in entry["row_groups"]]
    if [(g.index, g.num_rows) for g in groups] != planned:
        raise CorporaError(f"{unit['path']}: the row groups differ from the plan's footer")
    before = remote.bytes_fetched
    table = remote.read(unit["rg"], source.columns)
    drops: Counter = Counter()
    read = kept = text_bytes = 0
    for index, row in enumerate(iter_rows(table, source.content)):
        read += 1
        keep, reason = pipeline(row)
        if not keep:
            drops[reason] += 1
            continue
        writer.write(row, unit["path"], unit["rg"], index, pipeline.annotate(row))
        kept += 1
        text_bytes += sum(len(t.encode("utf-8")) for t in row[TEXTS].values())
    return {
        "path": unit["path"],
        "rg": unit["rg"],
        "rank_key": unit["rank_key"],
        "file_size": remote.remote.size,
        "etag": remote.resolved.etag,
        "xet_hash": remote.resolved.xet_hash,
        "footer_num_rows": groups[unit["rg"]].num_rows,
        "rows_read": read,
        "rows_kept": kept,
        "text_bytes": text_bytes,
        "drops": dict(sorted(drops.items())),
        "footer_bytes_fetched": footer_bytes,
        "bytes_fetched": remote.bytes_fetched - before,
    }


def fetch(
    plan: Plan,
    out_dir: Path,
    *,
    client: HubClient,
    resume: bool = True,
    part_bytes: int = PART_BYTES,
    downloaded: str | None = None,
    after_unit: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run (or finish) the export of `plan` into `out_dir`; returns the provenance. `after_unit(k)` is called after
    the k-th unit is recorded as done (tests use it to interrupt a run)."""
    out_dir = Path(out_dir)
    profile = plan.profile()
    _check_directory(out_dir, resume)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in FINAL_FILES:
        (out_dir / name).unlink(missing_ok=True)
    done = DoneLog(out_dir / DONE, plan)
    sources = {s: get_source(s) for s in profile.sources}
    writers = {s: make_writer(out_dir, source, profile, part_bytes=part_bytes) for s, source in sources.items()}
    pipelines = {s: Pipeline(profile, source) for s, source in sources.items()}
    last = done.records[-1]["writers"] if done.records else {}
    for source_id, writer in writers.items():
        writer.restore(last.get(source_id, {}))
        texts = (writer.texts(obj) for obj in writer.restored_lines())
        if pipelines[source_id].dedup is not None:
            pipelines[source_id].dedup.seed(texts)
        else:
            for _ in texts:  # still rebuilds the attribution
                pass
    files = RemoteFiles(client)
    for k, unit in enumerate(plan.units):
        if k < len(done.records):
            continue
        source_id = plan.file(unit["path"])["source"]
        record = _unit(plan, unit, files, pipelines[source_id], writers[source_id])
        for writer in writers.values():
            writer.sync()
        state = {s: w.state() for s, w in writers.items()}
        done.append({"unit": {"path": unit["path"], "rg": unit["rg"]}, "record": record, "writers": state})
        if after_unit is not None:
            after_unit(k)
    return finish(plan, out_dir, profile, writers, done, downloaded)


def finish(plan, out_dir, profile, writers, done, downloaded) -> dict[str, Any]:
    downloaded = downloaded or datetime.datetime.now(datetime.UTC).date().isoformat()
    filters = [*profile.filter_names(), f"corpora export {plan.sha256}"]
    entries, parts, attribution = [], [], None
    for writer in writers.values():
        for name, infos in writer.close().items():
            entries.append(writer.fragment_entry(name, infos, filters, downloaded))
            parts += [p.to_json() for p in infos]
        if profile.writer == "tlc" and profile.kind == "code":
            attribution = write_attribution(out_dir, writer.attribution_lines())
    fragment = write_fragment(out_dir, entries, plan.sha256)
    units = [r["record"] for r in done.records]
    drops: Counter = Counter()
    for u in units:
        drops.update(u["drops"])
    doc = {
        "format": EXPORT_FORMAT,
        "plan_sha256": plan.sha256,
        "profile": profile.describe(),
        "allow_unlicensed": plan.doc["allow_unlicensed"],
        "sources": plan.doc["sources"],
        "code": code_state(),
        "tools": tool_versions(),
        "downloaded": downloaded,
        "units": units,
        "parts": sorted(parts, key=lambda p: p["path"]),
        "fragment": {"path": fragment.name, "sha256": sha256_file(fragment)},
        "attribution": None if attribution is None else {"path": attribution.name, "sha256": sha256_file(attribution)},
        "totals": {
            "units": len(units),
            "rows_read": sum(u["rows_read"] for u in units),
            "rows_kept": sum(u["rows_kept"] for u in units),
            "text_bytes": sum(u["text_bytes"] for u in units),
            "drops": dict(sorted(drops.items())),
            "bytes_fetched": sum(u["bytes_fetched"] + u["footer_bytes_fetched"] for u in units),
            "parts": len(parts),
            "part_bytes": sum(p["bytes"] for p in parts),
            "docs": sum(p["docs"] for p in parts),
        },
    }
    doc = with_sha256(doc, "provenance_sha256")
    atomic_write_json(out_dir / "provenance.json", doc)
    return doc


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_corpora fetch", description="run a plan's export")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="the export directory (created; resumed if partial)")
    parser.add_argument("--no-resume", action="store_true", help="refuse a directory that already holds a done list")
    parser.add_argument("--dry-run", action="store_true", help="print the plan's estimates; read and write nothing")
    parser.add_argument("--part-bytes", type=int, default=PART_BYTES)
    parser.add_argument("--endpoint", default="https://huggingface.co")
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan)
        if args.dry_run:
            plan.profile()
            print(summary(plan))
            return 0
        client = HubClient(token=os.environ.get("HF_TOKEN") or None, endpoint=args.endpoint)
        doc = fetch(plan, args.out, client=client, resume=not args.no_resume, part_bytes=args.part_bytes)
    except CorporaError as err:
        print(f"surogate_corpora fetch: refused: {err}", file=sys.stderr)
        return 1
    totals = doc["totals"]
    print(f"{totals['units']} units, {totals['rows_kept']:,} of {totals['rows_read']:,} rows kept, "
          f"{totals['text_bytes']:,} text bytes in {totals['parts']} parts; fetched {totals['bytes_fetched']:,} bytes")  # fmt: skip
    print(f"wrote {args.out / 'provenance.json'} and {args.out / 'spec_fragment.toml'}")
    return 0
