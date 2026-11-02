"""Result reports (`bilingual-result/1`) and their markdown table (the resource report is audit.resources).

- `results_from_predictions`: one result row per (row, arm, metric, replicate) from `bilingual-pred/1` lines:
  accuracy per family/variant and per direction, and for E4 the slot accuracy, the all-slots rate and the language
  rate. Each value carries a clustered bootstrap CI (clusters = templates, MS §9.2), which is descriptive and
  within one replicate. Skipped items are counted, never scored.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .stats import clustered_bootstrap

RESULT_FORMAT = "bilingual-result/1"
NOT_RUN = "未运行"


def result_row(
    row: str,
    arm: str,
    metric: str,
    replicate: int,
    value: float | None = None,
    n_items: int = 0,
    ci: Mapping | None = None,
    status: str = NOT_RUN,
    **extra,
) -> dict[str, Any]:
    return {
        "row": row,
        "arm": arm,
        "metric": metric,
        "replicate": replicate,
        "value": value,
        "n_items": n_items,
        "ci": dict(ci) if ci else None,
        "status": status,
        **extra,
    }


def _metric(preds: Sequence[Mapping[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray, int]:
    scored = [p for p in preds if p.get("skip_reason") is None and p.get(key) is not None]
    values = np.array([float(p[key]) for p in scored], dtype=np.float64)
    clusters = np.array([p.get("cluster") or p["item_id"] for p in scored], dtype=object)
    return values, clusters, len(preds) - len(scored)


def results_from_predictions(
    preds: Iterable[Mapping[str, Any]], *, row: str | None = None, n_boot: int | None = None, seed: int = 0
) -> list[dict[str, Any]]:
    from ..config import DEFAULTS

    n_boot = DEFAULTS["eval"]["n_boot"] if n_boot is None else n_boot
    groups: dict[tuple, list] = defaultdict(list)
    for p in preds:
        base = (row or p["arm"], p["arm"], int(p["replicate"]), p["family"], p["variant"])
        groups[(*base, None)].append(p)
        groups[(*base, p["direction"])].append(p)
    out = []
    for (row_id, arm, rep, family, variant, direction), group in sorted(groups.items(), key=lambda kv: str(kv[0])):
        keys = [("acc", "correct"), ("margin", "margin")]  # margin: gold score minus the best other (continuous)
        if family == "E4":
            keys = [("slot_acc", "slot_acc"), ("all_slots", "correct"), ("lang_ok", "lang_ok")]
        for name, key in keys:
            values, clusters, skipped = _metric(group, key)
            metric = "/".join(x for x in (family, variant, direction, name) if x)
            if values.size == 0:
                out.append(result_row(row_id, arm, metric, rep, None, 0, None, NOT_RUN, skipped=skipped))
                continue
            ci = clustered_bootstrap(values, clusters, n=n_boot, seed=seed)
            out.append(
                result_row(
                    row_id,
                    arm,
                    metric,
                    rep,
                    float(values.mean()),
                    int(values.size),
                    {
                        "lo": ci.lo,
                        "hi": ci.hi,
                        "level": ci.level,
                        "n_boot": ci.n_boot,
                        "n_clusters": ci.n_clusters,
                        "seed": ci.seed,
                    },
                    "done",
                    skipped=skipped,
                )
            )
    return out


def results_document(rows: Sequence[Mapping[str, Any]], **meta) -> dict[str, Any]:
    return {"format": RESULT_FORMAT, **meta, "results": list(rows)}


def results_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| row | arm | metric | r | value | 95% CI (clustered) | n | status |",
        "|---|---|---|---|---:|---|---:|---|",
    ]
    for r in rows:
        value = "—" if r["value"] is None else f"{r['value']:.4f}"
        ci = "—" if not r.get("ci") else f"[{r['ci']['lo']:.4f}, {r['ci']['hi']:.4f}]"
        lines.append(
            f"| {r['row']} | {r['arm']} | {r['metric']} | {r['replicate']} | {value} | {ci} | "
            f"{r['n_items']} | {r['status']} |"
        )
    return "\n".join(lines)
