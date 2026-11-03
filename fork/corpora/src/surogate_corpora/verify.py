"""Checks of what the package wrote and of what the Hub serves.

- `verify_export(out_dir)` re-reads an export: every part's sha256, size and line count against the provenance, no
  part on disk that the provenance does not list, the fragment and attribution hashes, the totals, and the done list.
  It returns the problems, one precise message each; [] means clean.
- `verify_samples(rows_json, source, profile, out_dir)` checks real rows from datasets-server
  (`/rows` or `/first-rows`: `{"features": [...], "rows": [{"row_idx", "row", "truncated_cells"}]}`) against the
  registry's server schema, then runs the profile's filters and writer on them exactly as `fetch` would and checks
  that the exported text bytes equal the input bytes. Rows are local evidence only: the report keeps counts, never
  text, and the export it writes is for local inspection (never commit it).
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .canonical import CorporaError, atomic_write_json, check_sha256, read_json, sha256_file
from .export import PART_BYTES, PARTS, make_writer
from .filters import TEXT, TEXTS, script_stats
from .profiles import Pipeline, check_source, parse_params, resolve

SAMPLES_FORMAT = "corpora-samples/1"


# --- exports --------------------------------------------------------------------------------------------------------


def verify_export(out_dir: Path) -> list[str]:
    out_dir = Path(out_dir)
    try:
        doc = read_json(out_dir / "provenance.json", "export")
        check_sha256(doc, "provenance_sha256")
    except CorporaError as err:
        return [f"provenance.json: {err}"]
    problems: list[str] = []
    listed = {p["path"]: p for p in doc["parts"]}
    for path, part in sorted(listed.items()):
        target = out_dir / path
        if not target.is_file():
            problems.append(f"{path}: listed in provenance.json but missing")
            continue
        size = target.stat().st_size
        if size != part["bytes"]:
            problems.append(f"{path}: {size} bytes, provenance.json records {part['bytes']}")
        digest = sha256_file(target)
        if digest != part["sha256"]:
            problems.append(f"{path}: sha256 {digest}, provenance.json records {part['sha256']}")
        lines = 0
        with open(target, "rb") as f:
            for number, raw in enumerate(f, 1):
                lines += 1
                try:
                    obj = json.loads(raw)
                except ValueError:
                    problems.append(f"{path}:{number}: not a JSON line")
                    continue
                if not isinstance(obj, dict) or not ("text" in obj or "segments" in obj):
                    problems.append(f"{path}:{number}: neither a text nor a segments document")
        if lines != part["docs"]:
            problems.append(f"{path}: {lines} documents, provenance.json records {part['docs']}")
    on_disk = {f"{PARTS}/{p.name}" for p in (out_dir / PARTS).glob("*")} if (out_dir / PARTS).is_dir() else set()
    for path in sorted(on_disk - set(listed)):
        problems.append(f"{path}: on disk but not listed in provenance.json")
    for key in ("fragment", "attribution"):
        record = doc.get(key)
        if record is None:
            continue
        target = out_dir / record["path"]
        if not target.is_file():
            problems.append(f"{record['path']}: missing")
        elif sha256_file(target) != record["sha256"]:
            problems.append(f"{record['path']}: sha256 differs from provenance.json")
    if (out_dir / "spec_fragment.toml").is_file():
        try:
            fragment = tomllib.loads((out_dir / "spec_fragment.toml").read_text(encoding="utf-8"))
            named = [f for entry in fragment.get("source", []) for f in entry["files"]]
            if sorted(named) != sorted(listed):
                problems.append("spec_fragment.toml: its files are not exactly the parts provenance.json lists")
        except (ValueError, KeyError) as err:
            problems.append(f"spec_fragment.toml: {err}")
    totals = doc["totals"]
    if sum(p["docs"] for p in listed.values()) != totals["docs"]:
        problems.append(
            f"totals.docs {totals['docs']} differs from the parts' {sum(p['docs'] for p in listed.values())}"
        )
    if sum(u["rows_kept"] for u in doc["units"]) != totals["docs"]:
        problems.append("totals.docs differs from the units' rows_kept")
    done = out_dir / "done.jsonl"
    if done.is_file():
        n = sum(1 for line in done.read_text(encoding="utf-8").splitlines() if line.strip()) - 1
        if n != len(doc["units"]):
            problems.append(f"done.jsonl lists {n} units, provenance.json {len(doc['units'])}")
    return problems


# --- datasets-server samples ----------------------------------------------------------------------------------------


def feature_types(features: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """`{dotted leaf: dtype}` of datasets-server `features` (Value, nested dicts, lists, Translation)."""
    out: dict[str, str] = {}

    def walk(kind: Any, name: str) -> None:
        if isinstance(kind, list):
            inner = kind[0] if kind else {}
            out[name] = f"list<{inner.get('dtype', '?')}>" if isinstance(inner, dict) else "list<?>"
        elif isinstance(kind, dict) and kind.get("_type") == "Value":
            out[name] = kind["dtype"]
        elif isinstance(kind, dict) and kind.get("_type") == "Translation":
            for lang in kind["languages"]:
                out[f"{name}.{lang}"] = "string"
        elif isinstance(kind, dict) and kind.get("_type") in ("Sequence", "List"):
            inner = kind.get("feature", {})
            out[name] = f"list<{inner.get('dtype', '?')}>"
        elif isinstance(kind, dict) and "_type" not in kind:
            for key, sub in kind.items():
                walk(sub, f"{name}.{key}")
        else:
            out[name] = f"unknown:{kind}"

    for feature in features:
        walk(feature["type"], feature["name"])
    return out


def flatten(row: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out


PY_TYPES = {
    "string": (str,),
    "float32": (float, int),
    "float64": (float, int),
    "int64": (int,),
    "bool": (bool,),
}


def _value_ok(value: Any, dtype: str) -> bool:
    if value is None:
        return True
    if dtype.startswith("list<"):
        inner = dtype[5:-1]
        return isinstance(value, list) and all(_value_ok(v, inner) for v in value)
    if dtype in ("int64", "float32", "float64") and isinstance(value, bool):
        return False
    return isinstance(value, PY_TYPES.get(dtype, (object,)))


def verify_samples(
    rows_json: Path,
    source_id: str,
    profile_name: str,
    out_dir: Path,
    *,
    allow_unlicensed: bool = False,
    params: Mapping[str, Any] | None = None,
    part_bytes: int = PART_BYTES,
) -> dict[str, Any]:
    """The sample report; refuses (CorporaError) a profile or source the registry refuses, and a schema mismatch."""
    profile = resolve(profile_name, params, allow_unlicensed=allow_unlicensed)
    source = check_source(profile, source_id)
    rows_json, out_dir = Path(rows_json), Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise CorporaError(f"{out_dir} is not empty; verify-samples writes a fresh export")
    data = json.loads(rows_json.read_text(encoding="utf-8"))
    expected = source.server_schema()
    problems = []
    if "features" in data:
        got = feature_types(data["features"])
        if got != expected:
            diff = sorted(set(got.items()) ^ set(expected.items()))
            problems.append(f"features differ from the registry's server schema: {diff[:6]}")
    rows = data["rows"]
    truncated = [r["row_idx"] for r in rows if r.get("truncated_cells")]
    for r in rows:
        flat = flatten(r["row"])
        if set(flat) != set(expected):
            problems.append(f"row {r['row_idx']}: columns {sorted(set(flat) ^ set(expected))} differ")
            continue
        bad = [c for c, v in flat.items() if not _value_ok(v, expected[c])]
        if bad:
            problems.append(f"row {r['row_idx']}: values of {bad} do not have the registry's types")
    if problems:
        raise CorporaError(f"{rows_json.name} against source {source_id}: " + "; ".join(problems[:5]))

    pipeline = Pipeline(profile, source)
    writer = make_writer(out_dir, source, profile, part_bytes=part_bytes)
    unit_path = f"samples/{rows_json.stem}"
    drops: Counter = Counter()
    kept_texts: list[dict[str, str]] = []
    kept_by_series: dict[str, list[dict[str, str]]] = {}
    text_in = text_kept = nfkc = hant_dominant = with_hant = 0
    for r in rows:
        if r["row_idx"] in truncated:
            drops["truncated_cell"] += 1
            continue
        flat = flatten(r["row"])
        for column in source.content:  # as parquet.iter_rows hands them over: bytes
            if isinstance(flat.get(column), str):
                flat[column] = flat[column].encode("utf-8", "surrogatepass")
        text_in += sum(len(flat[c]) for c in source.content if isinstance(flat.get(c), bytes))
        keep, reason = pipeline(flat)
        if not keep:
            drops[reason] += 1
            continue
        writer.write(flat, unit_path, 0, r["row_idx"], pipeline.annotate(flat))
        texts = flat[TEXTS]
        kept_texts.append(texts)
        kept_by_series.setdefault(writer.series_name(flat), []).append(texts)
        text_kept += sum(len(t.encode("utf-8")) for t in texts.values())
        nfkc += any(unicodedata.normalize("NFKC", t) != t for t in texts.values())
        if TEXT in flat:
            stats = script_stats(flat[TEXT])
            hant_dominant += stats["hant"] > stats["hans"]
            with_hant += stats["hant"] > 0
    parts = writer.close()
    written: list[dict[str, Any]] = []
    out_by_series: dict[str, list[dict[str, str]]] = {}
    for name, infos in parts.items():
        for info in infos:
            with open(out_dir / info.path, "rb") as f:
                for line in f:
                    obj = json.loads(line)
                    written.append(obj)
                    out_by_series.setdefault(name, []).append(writer.texts(obj))
    groups = [obj.get("group") for obj in written]
    report = {
        "format": SAMPLES_FORMAT,
        "rows_json": rows_json.name,
        "rows_sha256": sha256_file(rows_json),
        "source": source_id,
        "profile": profile.describe(),
        "schema": "ok",
        "rows": len(rows),
        "truncated_rows": len(truncated),
        "kept": len(kept_texts),
        "drops": dict(sorted(drops.items())),
        "text_bytes_in": text_in,
        "text_bytes_kept": text_kept,
        "text_bytes_out": sum(len(t.encode("utf-8")) for d in sum(out_by_series.values(), []) for t in d.values()),
        "bytes_equal": out_by_series == kept_by_series,
        "ids_unique": len({obj["id"] for obj in written}) == len(written),
        "parts": {name: sum(i.docs for i in infos) for name, infos in parts.items()},
        "groups": len({g for g in groups if g is not None}),
        "every_doc_has_a_group": bool(written) and all(isinstance(g, str) for g in groups),
        "nfkc_would_change": nfkc,
        "hant_dominant_docs": hant_dominant if source.upstream_column else None,
        "docs_with_hant_markers": with_hant if source.upstream_column else None,
    }
    atomic_write_json(out_dir / "sample_report.json", report)
    return report


# --- command line ---------------------------------------------------------------------------------------------------


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_corpora verify", description="re-check an export")
    parser.add_argument("export_dir", type=Path)
    args = parser.parse_args(argv)
    problems = verify_export(args.export_dir)
    for problem in problems:
        print(problem)
    print(f"{args.export_dir}: {'clean' if not problems else f'{len(problems)} problems'}")
    return 1 if problems else 0


def samples_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m surogate_corpora verify-samples", description="check datasets-server rows against the registry"
    )
    parser.add_argument("rows_json", type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", type=Path, required=True, help="a new directory for the sample export and report")
    parser.add_argument("--param", action="append")
    parser.add_argument("--allow-unlicensed", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = verify_samples(
            args.rows_json,
            args.source,
            args.profile,
            args.out,
            allow_unlicensed=args.allow_unlicensed,
            params=parse_params(args.param),
        )
    except CorporaError as err:
        print(f"surogate_corpora verify-samples: refused: {err}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in report.items() if k != "profile"}, ensure_ascii=False, indent=1, sort_keys=True))
    return 0
