"""`compose`: one project corpus spec from several export directories (the per-project wiring of the exports).

Every export writes a `spec_fragment.toml` whose file paths are relative to the export directory. A project spec
needs all its sources in one file with paths relative to one base: for TLC the spec's own directory (the TLC corpus
reader resolves globs there, and `..` is allowed), for bilingual the corpus root (`manifest --root`; globs may not
leave it). `compose`

- re-checks every export with verify_export (any problem refuses) and that its profile writes this project's format;
- rewrites each fragment entry's `files` relative to the base and keeps every other key as the export wrote it;
- appends owner-written `[[source]]` tables (`--include`: the synthetic sources, TLC's generated binding programs
  and the bilingual Track S D_seed component) verbatim, their paths already relative to the base;
- refuses a source name that occurs twice and an output that exists;
- writes the spec (TLC: `seed`, `val_fraction` and optionally `decontam` first) with a comment block naming each
  export's plan and provenance sha256, and reports the kept text bytes per kind and their shares: the byte mixture
  the plan budgets set (TLC's 70/30 is chosen in bytes when the inputs are prepared; the tokenizer study fixes its
  mixture by bytes too).

No project is imported; the tests read the composed specs with the projects' own loaders.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .canonical import CorporaError, atomic_write_bytes, read_json
from .export import toml_value
from .verify import verify_export

PROJECT_WRITERS = {"tlc": ("tlc",), "bilingual": ("bilingual", "parallel")}
FRAGMENT = "spec_fragment.toml"


def _read_toml(path: Path, what: str) -> dict[str, Any]:
    try:
        return tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError) as err:
        raise CorporaError(f"{path}: cannot read {what}: {err}") from None


def _relative(path: Path, base: Path, project: str) -> str:
    rel = Path(os.path.relpath(path.resolve(), base.resolve())).as_posix()
    if project == "bilingual" and (rel == ".." or rel.startswith("../")):
        raise CorporaError(f"{path} is outside the corpus root {base}: bilingual file globs may not leave the root")
    return rel


def export_sources(export: Path, project: str, base: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The fragment entries of one checked export, paths relative to `base`, and what the spec records of it."""
    export = Path(export)
    problems = verify_export(export)
    if problems:
        raise CorporaError(f"{export}: the export does not verify: " + "; ".join(problems[:5]))
    provenance = read_json(export / "provenance.json", "export")
    writer = provenance["profile"]["writer"]
    if writer not in PROJECT_WRITERS[project]:
        raise CorporaError(f"{export}: profile {provenance['profile']['name']} writes the {writer} format, "
                           f"not a {project} corpus")  # fmt: skip
    entries = _read_toml(export / FRAGMENT, "a spec fragment").get("source", [])
    if not entries:
        raise CorporaError(f"{export}: {FRAGMENT} has no [[source]] tables")
    out = []
    for entry in entries:
        files = [_relative(export / f, base, project) for f in entry["files"]]
        out.append({**entry, "files": files})
    record = {
        "export": export.name,
        "profile": provenance["profile"]["name"],
        "kind": provenance["profile"]["kind"],
        "plan_sha256": provenance["plan_sha256"],
        "provenance_sha256": provenance["provenance_sha256"],
        "docs": provenance["totals"]["docs"],
        "text_bytes": provenance["totals"]["text_bytes"],
        "sources": [e["name"] for e in out],
    }
    return out, record


def compose(project: str, exports: Sequence[Path], out: Path, *, root: Path | None = None,
            includes: Sequence[Path] = (), top: Mapping[str, Any] | None = None) -> dict[str, Any]:  # fmt: skip
    """Write the project spec `out` from `exports` (and `includes`); returns the summary the command prints."""
    if project not in PROJECT_WRITERS:
        raise CorporaError(f"project must be one of {sorted(PROJECT_WRITERS)}, got {project!r}")
    out = Path(out)
    if out.exists():
        raise CorporaError(f"{out} exists; compose never overwrites a spec")
    if not exports:
        raise CorporaError("compose needs at least one --export")
    if project == "tlc" and root is not None:
        raise CorporaError("--root is bilingual's corpus root; TLC globs are relative to the spec's directory")
    base = (root if root is not None else out.parent).resolve()
    top = dict(top or {})
    if project == "bilingual" and top:
        raise CorporaError(f"a bilingual spec holds only [[source]] tables, not {sorted(top)}")
    entries, records = [], []
    for export in exports:
        found, record = export_sources(Path(export), project, base)
        entries += found
        records.append(record)
    included = []
    for path in includes:
        data = _read_toml(path, "an --include file")
        extra = sorted(set(data) - {"source"})
        if extra or not data.get("source"):
            raise CorporaError(f"{path}: an --include file holds [[source]] tables only (found {extra or 'none'})")
        included.append((Path(path), data["source"]))
    names = Counter([e["name"] for e in entries] + [s.get("name") for _, tables in included for s in tables])
    duplicates = sorted(str(n) for n, k in names.items() if k > 1)
    if duplicates:
        raise CorporaError(f"source names occur twice: {duplicates}")
    kinds: Counter = Counter()
    for record in records:
        kinds[record["kind"]] += record["text_bytes"]
    total = sum(kinds.values())
    mixture = {k: {"text_bytes": v, "share": round(v / total, 6) if total else None} for k, v in sorted(kinds.items())}
    lines = [f"# {project} corpus spec composed by `python -m surogate_corpora compose` from these exports",
             f"# (file paths relative to {'the corpus root' if project == 'bilingual' else 'this file'}):"]  # fmt: skip
    for r in records:
        lines.append(f"#   {r['export']}: profile {r['profile']}, plan {r['plan_sha256']}, "
                     f"provenance {r['provenance_sha256']}, {r['docs']} docs, {r['text_bytes']} text bytes")  # fmt: skip
    lines.append("# kept text bytes by kind (exports only): " + ", ".join(
        f"{k} {v['text_bytes']} ({v['share']})" for k, v in mixture.items()))  # fmt: skip
    if top:
        lines.append("")
        lines += [f"{k} = {toml_value(v)}" for k, v in top.items()]
    for entry in entries:
        lines += ["", "[[source]]"] + [f"{k} = {toml_value(v)}" for k, v in entry.items()]
    for path, _ in included:
        lines += ["", f"# --- included verbatim from {path.name} ---", path.read_text(encoding="utf-8").rstrip("\n")]
    atomic_write_bytes(out, ("\n".join(lines) + "\n").encode("utf-8"))
    return {
        "spec": str(out),
        "project": project,
        "base": str(base),
        "exports": records,
        "included": [{"file": p.name, "sources": [s.get("name") for s in t]} for p, t in included],
        "mixture_by_bytes": mixture,
    }


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_corpora compose",
                                     description="one project corpus spec from several export directories")  # fmt: skip
    parser.add_argument("--project", choices=sorted(PROJECT_WRITERS), required=True)
    parser.add_argument("--export", type=Path, action="append", default=[], help="an export directory (repeat)")
    parser.add_argument("--out", type=Path, required=True, help="the spec to write (must not exist)")
    parser.add_argument("--root", type=Path, help="bilingual: the corpus root (default the spec's directory)")
    parser.add_argument("--include", type=Path, action="append", default=[],
                        help="owner-written [[source]] tables appended verbatim (paths relative to the base)")  # fmt: skip
    parser.add_argument("--seed", type=int, help="TLC: the split seed (default 0)")
    parser.add_argument("--val-fraction", type=float, help="TLC: the share of documents in val (default 0.01)")
    parser.add_argument("--decontam", help="TLC: the decontamination report, relative to the spec")
    args = parser.parse_args(argv)
    top: dict[str, Any] = {}
    if args.project == "tlc":
        top = {"seed": 0 if args.seed is None else args.seed,
               "val_fraction": 0.01 if args.val_fraction is None else args.val_fraction}  # fmt: skip
        if args.decontam:
            top["decontam"] = args.decontam
    elif args.seed is not None or args.val_fraction is not None or args.decontam:
        print("compose: --seed, --val-fraction and --decontam are TLC spec keys; bilingual splits with "
              "`split assign`", file=sys.stderr)  # fmt: skip
        return 1
    try:
        summary = compose(args.project, args.export, args.out, root=args.root, includes=args.include, top=top)
    except CorporaError as err:
        print(f"compose: {err}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0
