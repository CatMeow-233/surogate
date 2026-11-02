"""Independence and leak tests of Track S (MS §6.2 anti-triviality rules 1–4; I4-D7).

- `chi2_sf`: the chi-square survival function from the regularized upper incomplete gamma function (series below
  a + 1, Lentz continued fraction above), so the tests need no scipy.
- `independence`: Pearson's chi-square test of a contingency table, with the Bayes accuracy of predicting the
  column from the row alone (sum of row maxima / n).
- `single_fact_leak`: for E3-par items, the table (binned value of one person's fact) x (label). A leak-free
  design leaves the label independent of either single fact; the "born earlier?" control (`born_earlier_items`)
  is the leaky design MS §6.2 rejects (Bayes accuracy 0.75 from one birth year) and must be flagged.
- `attribute_independence`: on fresh draws, every attribute against the partition and the name parts.
- `cooccurrence`, `fresh_codes`, `template_disjointness`: no test pair in one training document, no item code in
  any training document or other item, and split-disjoint template ids per family.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from .entities import CODE_PATTERN, PARTITIONS, Person, draw_people, name_parts
from .values import ATTRIBUTES

_CODE = re.compile(CODE_PATTERN)


def _gamma_q(a: float, x: float) -> float:
    """Q(a, x) = Gamma(a, x) / Gamma(a), the regularized upper incomplete gamma function (Numerical Recipes 6.2)."""
    if x < 0 or a <= 0:
        raise ValueError("Q(a, x) needs a > 0 and x >= 0")
    if x == 0:
        return 1.0
    log_prefix = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(10_000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return max(0.0, 1.0 - total * math.exp(log_prefix))
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10_000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-16:
            break
    return math.exp(log_prefix) * h


def chi2_sf(x: float, df: int) -> float:
    """P(X >= x) for X ~ chi-square(df)."""
    return _gamma_q(df / 2.0, x / 2.0)


def independence(table: np.ndarray) -> dict[str, float]:
    """Pearson chi-square test of independence of rows and columns (empty rows and columns dropped)."""
    t = np.asarray(table, dtype=np.float64)
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    n = t.sum()
    if t.shape[0] < 2 or t.shape[1] < 2:
        return {"chi2": 0.0, "df": 0, "p": 1.0, "n": int(n), "bayes_accuracy": float(t.max(axis=1).sum() / n)}
    expected = np.outer(t.sum(axis=1), t.sum(axis=0)) / n
    chi2 = float(((t - expected) ** 2 / expected).sum())
    df = (t.shape[0] - 1) * (t.shape[1] - 1)
    return {"chi2": chi2, "df": df, "p": chi2_sf(chi2, df), "n": int(n),
            "bayes_accuracy": float(t.max(axis=1).sum() / n)}  # fmt: skip


def contingency(rows: Sequence[int], cols: Sequence[int], n_rows: int, n_cols: int) -> np.ndarray:
    table = np.zeros((n_rows, n_cols), dtype=np.int64)
    np.add.at(table, (np.asarray(rows), np.asarray(cols)), 1)
    return table


def bin_of(value: int, n_values: int, n_bins: int) -> int:
    return value * n_bins // n_values


def single_fact_leak(
    items: Sequence[Mapping[str, Any]], people: Mapping[str, Person], side: str, attr: str, n_values: int,
    n_labels: int, n_bins: int = 10,
) -> dict[str, float]:  # fmt: skip
    """Chi-square test of (binned value of one person's `attr`) x (label) over E3-par items."""
    bins = min(n_bins, n_values)
    rows = [bin_of(people[it["structured"][side]].attrs[attr], n_values, bins) for it in items]
    cols = [it["structured"]["label"] for it in items]
    return independence(contingency(rows, cols, bins, n_labels))


def born_earlier_items(a_side: Sequence[Person], b_side: Sequence[Person], n: int, rng) -> list[dict]:
    """The leaky control of MS §6.2: "was A born before B?" over random pairs (ties redrawn), labels balanced by
    rejection. One birth year alone predicts the label with Bayes accuracy near 0.75."""
    out, want = [], [0, 1] * (n // 2) + [0] * (n % 2)
    for label in want:
        while True:
            a = a_side[int(rng.integers(len(a_side)))]
            b = b_side[int(rng.integers(len(b_side)))]
            if a.attrs["year"] == b.attrs["year"]:
                continue
            if int(a.attrs["year"] < b.attrs["year"]) == (1 - label):  # label 0 = "yes, earlier"
                out.append({"structured": {"a": a.id, "b": b.id, "label": label}})
                break
    return out


def leak_flagged(stats: Mapping[str, float], *, alpha: float = 0.001) -> bool:
    return stats["p"] < alpha


def attribute_independence(
    n: int, n_values: Mapping[str, int], seed: int, n_bins: int = 10
) -> dict[str, dict[str, float]]:
    """Draw n people with the generator's own functions and test every attribute against the partition and the
    two name parts (given-name head, surname head, binned)."""
    rng = np.random.default_rng([seed, 0x1EA4])
    share = n // 4
    sizes = {"heldout_fraction": 0.5, "eval_dev_fraction": 0.2, "par_train_fraction": 0.5}
    people = draw_people({p: share for p in PARTITIONS}, n_values, sizes, rng)
    out = {}
    for attr in ATTRIBUTES:
        nv = n_values[attr]
        b = min(n_bins, nv)
        values = [bin_of(p.attrs[attr], nv, b) for p in people]
        features = {
            "partition": ([PARTITIONS.index(p.partition) for p in people], len(PARTITIONS)),
            "given_name": ([name_parts(p.name_index)[0] % n_bins for p in people], n_bins),
            "surname": ([name_parts(p.name_index)[2] % n_bins for p in people], n_bins),
        }
        for feature, (codes, k) in features.items():
            out[f"{attr}~{feature}"] = independence(contingency(values, codes, b, k))
    return out


def cooccurrence(test_pairs: Iterable[tuple[str, str]], doc_people: Iterable[Iterable[str]]) -> list[tuple[str, str]]:
    """Test pairs whose two people are named together in one training document (by text-derived names)."""
    pairs = {tuple(sorted(p)) for p in test_pairs}
    hits = set()
    for names in doc_people:
        names = sorted(set(names))
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                if (a, b) in pairs:
                    hits.add((a, b))
    return sorted(hits)


def codes_in(text: str) -> set[str]:
    return set(_CODE.findall(text))


def fresh_codes(item_codes: Mapping[str, Iterable[str]], training_texts: Iterable[str]) -> dict[str, Any]:
    """Codes shared between items, and item codes found in any training text (both must be empty)."""
    owner: dict[str, str] = {}
    shared = []
    for item_id, codes in item_codes.items():
        for code in codes:
            if code in owner:
                shared.append(code)
            owner[code] = item_id
    in_training = set()
    for text in training_texts:
        in_training |= codes_in(text) & owner.keys()
    return {"shared_between_items": sorted(set(shared)), "in_training": sorted(in_training), "n_codes": len(owner)}


def template_sets(items: Iterable[Mapping[str, Any]], train_ids: Mapping[str, set[str]] | None = None):
    """{family: {split: set of template ids}}; training ids (from the training documents) join as split 'train'."""
    out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for item in items:
        for tid in item["template_ids"].values():
            out[item["family"]][item["split"]].add(tid)
    for family, ids in (train_ids or {}).items():
        out[family]["train"] |= set(ids)
    return out


def template_disjointness(sets: Mapping[str, Mapping[str, set[str]]], parts: Callable[[str], set[str]]) -> dict:
    """Per family: overlaps between splits of composed template ids and of their parts (both must be empty)."""
    report = {}
    for family, by_split in sets.items():
        splits = sorted(by_split)
        overlaps, part_overlaps = {}, {}
        for i, a in enumerate(splits):
            for b in splits[i + 1 :]:
                common = by_split[a] & by_split[b]
                pa = set().union(*(parts(t) for t in by_split[a])) if by_split[a] else set()
                pb = set().union(*(parts(t) for t in by_split[b])) if by_split[b] else set()
                overlaps[f"{a}&{b}"] = sorted(common)
                part_overlaps[f"{a}&{b}"] = sorted(pa & pb)
        report[family] = {"counts": {s: len(v) for s, v in by_split.items()}, "overlap": overlaps,
                          "part_overlap": part_overlaps}  # fmt: skip
    return report


def split_parts(template_id: str) -> set[str]:
    return set(template_id.split("+"))
