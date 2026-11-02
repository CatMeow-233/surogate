"""Confirmatory statistics of MS §9.2–9.6 (SCI-07, SCI-09): the run-level paired t test, the clustered bootstrap,
TOST, the graphical gatekeeper of MS §9.3, the three-way decision, power, MDE and the replicate rule, and the G1'
and futility rules. numpy and the stdlib only: no scipy, so it runs on the CPU lane.

Estimand and test (MS §9.2): per replicate, the mean accuracy difference of two arms on the same items; across
replicates, a one-sided paired t test with df = R - 1. Item-level tests that pool replicates are not used (their
real alpha is 0.08–0.46 with between-run variance, SCI-07); the clustered bootstrap is descriptive, within one
replicate, clustered by template.

Gatekeeping (MS §9.3, Bretz et al. 2009): branch 1 is H_A -> H_B with alpha_1 = 0.0125, branch 2 is H_C with
alpha_2 = 0.0125, one-sided. H_B starts with weight 0 and receives H_A's alpha once H_A is rejected; when a branch
is fully rejected its alpha moves to the other branch (edges H_A -> H_B, H_B -> H_C, H_C -> H_A, each weight 1).
"Rejected" in the graph means p <= the hypothesis's current alpha; the three-way decision (superior / equivalent /
inconclusive) is reported on top of it and also needs the point estimate to reach the SESOI.

Splits: `confirm` uses test predictions only (and dev predictions only for the H_B precondition); `pilot` and
`futility` use dev predictions only. Records of another split in the same files are left out and counted in the
output's `inputs`, never pooled (MS §9.2: tune on dev, test once).

Command line: `python -m surogate_bilingual stats {power,pilot,futility,confirm} ...` (see `main`).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import DEFAULTS

STATS_FORMAT = "bilingual-stats/1"
_EVAL = DEFAULTS["eval"]  # the one place of these defaults (B-O8, MS §9.3, §9.5; [建议] until the MS §9.7 lock)
ALPHA_BRANCH = _EVAL["alpha_branch"]  # one-sided, per branch
SESOI = dict(_EVAL["sesoi"])
NONINFERIORITY = dict(_EVAL["noninferiority"])
G1P_CHANCE, G1P_MARGIN = _EVAL["g1p"]["chance"], _EVAL["g1p"]["margin"]
POWER_TARGET = _EVAL["power"]
REPLICATES = tuple(_EVAL["replicates"])


# --- distributions -------------------------------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularised incomplete beta function (modified Lentz)."""
    tiny, eps = 1e-300, 1e-15
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ArithmeticError(f"incomplete beta did not converge (a={a}, b={b}, x={x})")


def betainc(a: float, b: float, x: float) -> float:
    """The regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """P(T <= t) for Student's t with df degrees of freedom."""
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    if t * t < df:  # near 0: I_{t^2/(df+t^2)}(1/2, df/2) keeps full precision where df/(df+t^2) rounds to 1
        half = 0.5 * betainc(0.5, df / 2.0, t * t / (df + t * t))
        return 0.5 + half if t > 0 else 0.5 - half
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_sf(t: float, df: float) -> float:
    """P(T > t), the one-sided p-value of t."""
    return t_cdf(-t, df)


def t_ppf(p: float, df: float) -> float:
    """The p-quantile of Student's t (bisection on t_cdf; |error| < 1e-10)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    lo, hi = -1.0, 1.0
    while t_cdf(lo, df) > p:
        lo *= 2.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# --- the run-level paired t test -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class TResult:
    """A one-sided paired t test of H0: mean(diff) <= -margin against H1: mean(diff) > -margin (margin 0 for
    superiority, the non-inferiority margin otherwise)."""

    n: int
    mean: float
    sd: float
    se: float
    t: float
    df: int
    p: float
    margin: float = 0.0

    def ci(self, alpha: float) -> tuple[float, float]:
        """The two-sided (1 - 2 alpha) interval of the mean, i.e. the interval TOST at level alpha uses."""
        if self.n < 2:
            return (-math.inf, math.inf)
        half = t_ppf(1.0 - alpha, self.df) * self.se
        return (self.mean - half, self.mean + half)


def paired_t_one_sided(diffs: Sequence[float], margin: float = 0.0) -> TResult:
    """Run-level one-sided paired t test on per-replicate differences (df = R - 1, MS §9.2).

    A sample with zero spread has no t statistic: its p is 0.0 when the mean beats -margin and 1.0 otherwise."""
    x = np.asarray(list(diffs), dtype=np.float64)
    n = int(x.size)
    if n < 2:
        raise ValueError(f"a paired t test needs at least 2 replicates, got {n}")
    if not np.isfinite(x).all():
        raise ValueError("differences must be finite")
    mean, sd = float(x.mean()), float(x.std(ddof=1))
    if sd <= 1e-12 * max(1.0, abs(mean)):  # equal values up to rounding: no spread
        sd = 0.0
    se = sd / math.sqrt(n)
    shifted = mean + margin
    if se == 0.0:
        t = math.inf if shifted > 0 else (-math.inf if shifted < 0 else 0.0)
        p = 0.0 if shifted > 0 else 1.0
    else:
        t = shifted / se
        p = t_sf(t, n - 1)
    return TResult(n, mean, sd, se, t, n - 1, p, margin)


def t_json(r: TResult) -> dict[str, Any]:
    """A TResult as JSON: a zero-spread sample has an infinite t, recorded as null with `zero_spread` true (canonical
    JSON has no infinities)."""
    out = asdict(r)
    out["zero_spread"] = r.se == 0.0
    if not math.isfinite(out["t"]):
        out["t"] = None
    return out


def tost(diffs: Sequence[float], margin: float, alpha: float) -> bool:
    """Two one-sided tests: True when the (1 - 2 alpha) interval of the mean lies inside (-margin, margin)."""
    if margin <= 0:
        raise ValueError(f"the equivalence margin must be positive, got {margin}")
    r = paired_t_one_sided(diffs)
    lo, hi = r.ci(alpha)
    return -margin < lo and hi < margin


DECISIONS = ("superior", "equivalent", "inconclusive")


def decide(diffs: Sequence[float], alpha: float, sesoi: float) -> dict[str, Any]:
    """The three-way decision of MS §9.3: superior when p <= alpha and the estimate >= SESOI; equivalent when TOST
    holds at +-SESOI with the matching alpha; inconclusive otherwise."""
    superior, equivalent, inconclusive = DECISIONS
    r = paired_t_one_sided(diffs)
    if r.p <= alpha and r.mean >= sesoi:
        decision = superior
    elif tost(diffs, sesoi, alpha):
        decision = equivalent
    else:
        decision = inconclusive
    lo, hi = r.ci(alpha)
    return {
        "decision": decision,
        "p": r.p,
        "estimate": r.mean,
        "sd": r.sd,
        "n": r.n,
        "alpha": alpha,
        "sesoi": sesoi,
        "ci_1m2alpha": [lo, hi],
    }


# --- the clustered bootstrap ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CI:
    estimate: float
    lo: float
    hi: float
    level: float
    n_boot: int
    n_clusters: int
    n_items: int
    seed: int


def clustered_bootstrap(
    diff: np.ndarray, clusters: np.ndarray, n: int = 10_000, seed: int = 0, level: float = 0.95, chunk: int = 2048
) -> CI:
    """Percentile CI of the item mean, resampling whole clusters with replacement (MS §9.2: templates are the
    clusters, 10,000 resamples, fixed seed). Descriptive only, within one replicate."""
    diff = np.asarray(diff, dtype=np.float64)
    clusters = np.asarray(clusters)
    if diff.ndim != 1 or diff.shape != clusters.shape or diff.size == 0:
        raise ValueError("diff and clusters must be equal-length, non-empty 1-D arrays")
    _, first, inverse = np.unique(clusters, return_index=True, return_inverse=True)
    rank = np.empty(first.size, dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size)
    inverse = rank[inverse.reshape(-1)]  # clusters numbered by first appearance, whatever their label type
    sums = np.bincount(inverse, weights=diff)
    sizes = np.bincount(inverse).astype(np.float64)
    c = sums.size
    rng = np.random.default_rng(seed)
    stats = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        idx = rng.integers(0, c, size=(stop - start, c))
        stats[start:stop] = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(stats, [tail, 1.0 - tail])
    return CI(float(diff.mean()), float(lo), float(hi), level, n, c, int(diff.size), seed)


# --- graphical gatekeeping (Bretz et al. 2009) ---------------------------------------------------------------------


@dataclass(frozen=True)
class Graph:
    """Initial alphas per hypothesis and transition weights g[a][b] (rows sum to <= 1)."""

    alpha: dict[str, float]
    edges: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        for a, row in self.edges.items():
            if a not in self.alpha or any(b not in self.alpha or b == a for b in row):
                raise ValueError(f"edges from {a!r} name unknown hypotheses or a self-loop: {row}")
            if sum(row.values()) > 1.0 + 1e-12 or any(w < 0 for w in row.values()):
                raise ValueError(f"the weights out of {a!r} must be >= 0 and sum to <= 1: {row}")


# MS §9.3: branch 1 = H_A -> H_B, branch 2 = H_C; a fully rejected branch passes its alpha to the other one.
MS_GRAPH = Graph(
    alpha={"H_A": ALPHA_BRANCH, "H_B": 0.0, "H_C": ALPHA_BRANCH},
    edges={"H_A": {"H_B": 1.0}, "H_B": {"H_C": 1.0}, "H_C": {"H_A": 1.0}},
)


@dataclass(frozen=True)
class Decision:
    rejected: bool
    alpha: float  # the alpha the hypothesis held when it was rejected, or its final alpha
    order: int | None  # 1-based rejection order


def graph_gatekeeper(pvalues: Mapping[str, float], graph: Graph = MS_GRAPH) -> dict[str, Decision]:
    """The sequentially rejective graph procedure: reject any H with p <= alpha_H, pass its alpha along its edges,
    update the graph, repeat. The result does not depend on which rejectable H is taken first."""
    names = list(graph.alpha)
    missing = [h for h in names if h not in pvalues]
    if missing:
        raise ValueError(f"no p-value for {missing}")
    alpha = {h: float(graph.alpha[h]) for h in names}
    g = {a: {b: float(graph.edges.get(a, {}).get(b, 0.0)) for b in names if b != a} for a in names}
    remaining, order, at = list(names), {}, {}
    while True:
        candidates = [h for h in remaining if alpha[h] > 0 and float(pvalues[h]) <= alpha[h] + 1e-15]
        if not candidates:
            break
        j = candidates[0]
        order[j], at[j] = len(order) + 1, alpha[j]
        remaining.remove(j)
        for b in remaining:
            alpha[b] += alpha[j] * g[j][b]
        new = {a: {} for a in remaining}
        for a in remaining:
            for b in remaining:
                if a == b:
                    continue
                denom = 1.0 - g[a][j] * g[j][a]
                new[a][b] = (g[a][b] + g[a][j] * g[j][b]) / denom if denom > 1e-15 else 0.0
        g = {a: {**{x: 0.0 for x in names if x != a}, **new.get(a, {})} for a in names}
        alpha[j] = 0.0
    return {h: Decision(h in order, at.get(h, alpha[h]), order.get(h)) for h in names}


# --- power, MDE and the replicate rule (MS §9.5) -------------------------------------------------------------------


def _chi_weights(df: int, points: int = 4001) -> tuple[np.ndarray, np.ndarray]:
    """Simpson nodes u and weights w for E[h(sqrt(V))], V ~ chi^2(df), in the variable u = sqrt(v)."""
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    top = math.sqrt(df + 14.0 * math.sqrt(2.0 * df) + 60.0)
    u = np.linspace(0.0, top, points)
    # density of u = sqrt(V): 2 u^(df-1) exp(-u^2/2) / (2^(df/2) Gamma(df/2)); finite at u = 0 for every df >= 1
    log_c = math.log(2.0) - (df / 2.0) * math.log(2.0) - math.lgamma(df / 2.0)
    power = np.ones_like(u) if df == 1 else np.where(u > 0, u, 0.0) ** (df - 1)
    density = np.exp(log_c - u * u / 2.0) * power
    h = u[1] - u[0]
    simpson = np.ones(points)
    simpson[1:-1:2], simpson[2:-1:2] = 4.0, 2.0
    w = simpson * h / 3.0 * density
    return u, w / w.sum()


def power_paired_t(sd: float, delta: float, alpha: float, R: int, sims: int = 0, seed: int = 0) -> float:
    """Power of the one-sided run-level paired t test (df = R - 1) at true mean difference delta and per-replicate
    SD sd. sims = 0 integrates the noncentral t exactly (quadrature over the chi-square); sims > 0 simulates."""
    if R < 2 or sd <= 0:
        raise ValueError(f"power needs R >= 2 and sd > 0, got R={R}, sd={sd}")
    df = R - 1
    crit = t_ppf(1.0 - alpha, df)
    if sims > 0:
        rng = np.random.default_rng(seed)
        x = delta + sd * rng.standard_normal((sims, R))
        t = x.mean(1) / (x.std(1, ddof=1) / math.sqrt(R))
        return float((t > crit).mean())
    ncp = delta * math.sqrt(R) / sd
    u, w = _chi_weights(df)
    phi = np.vectorize(norm_cdf)(ncp - crit * u / math.sqrt(df))
    return float(np.clip((w * phi).sum(), 0.0, 1.0))


def mde(sd: float, alpha: float, R: int, power: float = POWER_TARGET) -> float:
    """The minimal detectable effect: the smallest delta at which the run-level test reaches `power`."""
    lo, hi = 0.0, sd
    while power_paired_t(sd, hi, alpha, R) < power:
        hi *= 2.0
        if hi > 1e6 * sd:
            raise ArithmeticError("no finite MDE")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if power_paired_t(sd, mid, alpha, R) < power:
            lo = mid
        else:
            hi = mid
    return hi


def replicate_rule(
    sd: float,
    sesoi: float,
    alpha: float = ALPHA_BRANCH,
    power: float = POWER_TARGET,
    bounds: tuple[int, int] = REPLICATES,
) -> dict[str, Any]:
    """MS §9.5: the smallest R in bounds whose power at the SESOI is >= `power`; otherwise the MDE at the largest R
    and the label "underpowered, exploratory"."""
    table = {r: power_paired_t(sd, sesoi, alpha, r) for r in range(bounds[0], bounds[1] + 1)}
    chosen = next((r for r, p in table.items() if p >= power), None)
    out: dict[str, Any] = {
        "sd": sd,
        "sesoi": sesoi,
        "alpha": alpha,
        "target_power": power,
        "power_by_R": {str(r): p for r, p in table.items()},
        "R": chosen,
    }
    if chosen is None:
        out.update(mde_at_max=mde(sd, alpha, bounds[1], power), label="underpowered, exploratory")
    else:
        out["label"] = "powered"
    return out


# --- the G1' gate, the futility rule, discordance ------------------------------------------------------------------


def pilot_gate(acc: Mapping[str, Sequence[float]], chance: float = G1P_CHANCE, margin: float = G1P_MARGIN) -> dict:
    """G1' (MS §9.3): on dev, the best of the given arms (B0, B2, B12a) must reach chance + margin. The best arm is
    the one with the highest mean accuracy over replicates; its mean decides."""
    if not acc:
        raise ValueError("the G1' gate needs at least one arm")
    means = {arm: float(np.mean(v)) for arm, v in acc.items()}
    best = max(means, key=lambda a: (means[a], a))
    return {
        "gate": "G1'",
        "chance": chance,
        "margin": margin,
        "threshold": chance + margin,
        "means": means,
        "per_replicate": {a: list(map(float, v)) for a, v in acc.items()},
        "best": best,
        "passed": means[best] >= chance + margin,
    }


def futility(b12a_minus_b4: float, b6_minus_b4: float, sesoi_a: float = SESOI["H_A"]) -> dict[str, Any]:
    """MS §9.6 (B-O18): after family I replicate 0, branch 1 stops for futility when (B12a - B4) and (B6 - B4) on
    dev are both below SESOI_A / 2. Branch 2 is unaffected."""
    half = sesoi_a / 2.0
    stop = b12a_minus_b4 < half and b6_minus_b4 < half
    return {
        "rule": "MS §9.6 family-I-first futility (B-O18)",
        "sesoi_a": sesoi_a,
        "threshold": half,
        "b12a_minus_b4": b12a_minus_b4,
        "b6_minus_b4": b6_minus_b4,
        "decision": "futility stop: branch 1 not tested" if stop else "continue",
        "stop_branch_1": stop,
    }


def discordance(a: Mapping[str, bool], b: Mapping[str, bool]) -> tuple[float, int]:
    """The share of common items on which two arms' correctness differs, and the number of common items."""
    common = sorted(set(a) & set(b))
    if not common:
        return math.nan, 0
    return sum(bool(a[i]) != bool(b[i]) for i in common) / len(common), len(common)


# --- predictions -> per-replicate values -----------------------------------------------------------------------------


def read_predictions(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Every `bilingual-pred/1` line of the given files (or of predictions.jsonl in the given directories)."""
    out = []
    for path in paths:
        path = Path(path)
        files = sorted(path.rglob("predictions.jsonl")) if path.is_dir() else [path]
        for f in files:
            for line in f.read_text("utf-8").splitlines():
                if line.strip():
                    doc = json.loads(line)
                    if doc.get("format") != "bilingual-pred/1":
                        raise ValueError(f"{f}: not a bilingual-pred/1 line: {doc.get('format')!r}")
                    out.append(doc)
    return out


def select_split(preds: Iterable[Mapping[str, Any]], split: str, what: str) -> tuple[list[dict[str, Any]], dict]:
    """(the records of `split`, {"split", "kept", "ignored": {other split: count}}). Records of other splits are left
    out, never pooled: tuning data (dev) must not enter the confirmatory test (MS §9.2, §9.7), and test data must
    not enter a dev decision. A record without a split, or no record of `split` at all, is an error."""
    kept, ignored = [], defaultdict(int)
    for p in preds:
        if p.get("split") is None:
            raise ValueError(f"{what}: a record of {p.get('arm')!r} / {p.get('item_id')!r} has no split")
        if p["split"] == split:
            kept.append(dict(p))
        else:
            ignored[p["split"]] += 1
    if not kept:
        raise ValueError(f"{what}: no {split} records (found {dict(ignored) or 'none'})")
    return kept, {"split": split, "kept": len(kept), "ignored": dict(sorted(ignored.items()))}


NLL_FORMAT = "bilingual-nll/1"


def read_nll(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Every `bilingual-nll/1` document of the given files (or of *.nll.json under the given directories)."""
    out = []
    for path in paths:
        path = Path(path)
        for f in sorted(path.rglob("*.nll.json")) if path.is_dir() else [path]:
            doc = json.loads(f.read_text("utf-8"))
            if doc.get("format") != NLL_FORMAT:
                raise ValueError(f"{f}: not a {NLL_FORMAT} document: {doc.get('format')!r}")
            out.append({**doc, "_path": str(f)})
    return out


def column_bpb(doc: Mapping[str, Any], lang: str) -> float | None:
    """The BPB of evaluation language `lang` on that language's own expert (the cell R_out uses on a monolingual
    document): one column of the 2x2 matrix, compared only with the same column of another arm (MS §6.1)."""
    matrix = doc["matrix"]
    if lang not in matrix["rows"] or lang not in matrix["cols"]:
        return None
    return matrix["bpb"][matrix["rows"].index(lang)][matrix["cols"].index(lang)]


def bpb_diffs(nll_docs: Iterable[Mapping[str, Any]], treatment: str, control: str, lang: str) -> dict[int, float]:
    """{replicate: BPB(control) - BPB(treatment)} in column `lang`; positive means the treatment is better."""
    by: dict[tuple[str, int], float] = {}
    for doc in nll_docs:
        arm, rep = doc.get("arm"), doc.get("replicate")
        value = column_bpb(doc, lang) if arm in (treatment, control) and rep is not None else None
        if value is not None:
            key = (arm, int(rep))
            if key in by:
                raise ValueError(f"two NLL documents of arm {arm} replicate {rep} (one per arm and replicate)")
            by[key] = float(value)
    reps = sorted({r for a, r in by if a == treatment} & {r for a, r in by if a == control})
    return {r: by[(control, r)] - by[(treatment, r)] for r in reps}


def endpoint_of(pred: Mapping[str, Any]) -> str:
    return f"{pred['family']}/{pred['variant']}"


def correctness(
    preds: Iterable[Mapping[str, Any]], arm: str, endpoint: str, value: str = "correct"
) -> dict[int, dict[str, float]]:
    """{replicate: {item_id: value}} for one arm and endpoint, skipped items left out."""
    out: dict[int, dict[str, float]] = defaultdict(dict)
    for p in preds:
        if p["arm"] == arm and endpoint_of(p) == endpoint and p.get("skip_reason") is None:
            v = p.get(value)
            if v is not None:
                out[int(p["replicate"])][p["item_id"]] = float(v)
    return dict(out)


def replicate_diffs(preds, treatment: str, control: str, endpoint: str, value: str = "correct") -> dict[int, float]:
    """{replicate: mean(treatment - control)} over the items both arms scored (the MS §9.2 estimand)."""
    t, c = correctness(preds, treatment, endpoint, value), correctness(preds, control, endpoint, value)
    out = {}
    for r in sorted(set(t) & set(c)):
        common = sorted(set(t[r]) & set(c[r]))
        if common:
            out[r] = float(np.mean([t[r][i] - c[r][i] for i in common]))
    return out


# MS §9.3: the confirmatory hypotheses. H_C is an intersection-union test: the gain AND every non-inferiority part
# (MS §9.1: E1 accuracy, and BPB within one column of the 2x2 matrix, here each language's column).
HYPOTHESES = {
    "H_A": {"treatment": "B9-II", "control": "B4-II", "endpoint": "E3ctx/cross", "value": "correct"},
    "H_B": {"treatment": "B8", "control": "B9-II", "endpoint": "E3ctx/cross", "value": "correct"},
    "H_C": {
        "treatment": "T2",
        "control": "T1'",
        "endpoint": "E4/translate",
        "value": "slot_acc",
        "noninferior": [
            {
                "kind": "accuracy",
                "treatment": "T2",
                "control": "T0",
                "endpoint": "E1/qa",
                "value": "correct",
                "margin": NONINFERIORITY["E1"],
            },
            *(
                {"kind": "bpb", "treatment": "T2", "control": "T0", "lang": lang, "margin": NONINFERIORITY["bpb"]}
                for lang in ("zh", "en")
            ),
        ],
    },
}


def confirmatory(preds, dev_preds=None, nll=None, sesoi: Mapping[str, float] = SESOI, graph: Graph = MS_GRAPH) -> dict:
    """Run-level tests of H_A, H_B, H_C, the H_B precondition on dev, the gatekeeper and the three-way decisions.

    `preds` are test predictions, `dev_preds` dev predictions (the H_B precondition) and `nll` the test eval-nll
    documents of the H_C BPB parts; select them by split first (`select_split`). A part without data makes its
    hypothesis untestable (p = 1): an intersection-union test with a missing part would be anti-conservative."""
    report: dict[str, Any] = {"hypotheses": {}}
    pvalues = {}
    for h, spec in HYPOTHESES.items():
        diffs = replicate_diffs(preds, spec["treatment"], spec["control"], spec["endpoint"], spec["value"])
        entry: dict[str, Any] = {
            **{k: v for k, v in spec.items() if k != "noninferior"},
            "sesoi": sesoi[h],
            "replicates": {str(r): d for r, d in diffs.items()},
        }
        if len(diffs) < 2:
            entry["status"] = f"not testable: {len(diffs)} replicate(s) with both arms"
            pvalues[h] = 1.0
            report["hypotheses"][h] = entry
            continue
        r = paired_t_one_sided(list(diffs.values()))
        p = r.p
        entry["t_test"] = t_json(r)
        if h == "H_B":
            pre = None
            if dev_preds is not None:
                a = correctness(dev_preds, spec["treatment"], spec["endpoint"])
                b = correctness(dev_preds, spec["control"], spec["endpoint"])
                rates = [discordance(a[k], b[k])[0] for k in sorted(set(a) & set(b))]
                pre = float(np.nanmean(rates)) if rates else None
            entry["precondition"] = {
                "discordance_dev": pre,
                "required": sesoi["H_B"],
                "met": pre is not None and pre >= sesoi["H_B"],
            }
            if not entry["precondition"]["met"]:
                entry["status"] = "untestable at this divergence level (MS §9.3)"
                p = 1.0
        for part in spec.get("noninferior", []):
            if part["kind"] == "bpb":
                d = bpb_diffs(nll or [], part["treatment"], part["control"], part["lang"])
                missing = "test eval-nll documents (--nll) of both arms"
            else:
                d = replicate_diffs(preds, part["treatment"], part["control"], part["endpoint"], part["value"])
                missing = "test predictions of both arms"
            part = {**part, "replicates": {str(r): x for r, x in d.items()}}
            if len(d) < 2:
                p = 1.0
                part["status"] = f"not testable: {len(d)} replicate(s) with {missing}"
                entry["status"] = "not testable: a non-inferiority part has no data (intersection-union test)"
            else:
                nr = paired_t_one_sided(list(d.values()), margin=part["margin"])
                p = max(p, nr.p)
                part["t_test"] = t_json(nr)
            entry.setdefault("noninferior", []).append(part)
        if "status" in entry:
            p = 1.0
        entry["p"] = p
        pvalues[h] = p
        report["hypotheses"][h] = entry
    gate = graph_gatekeeper(pvalues, graph)
    for h, dec in gate.items():
        entry = report["hypotheses"][h]
        entry["gatekeeper"] = asdict(dec)
        if "t_test" in entry and "status" not in entry:
            diffs = list(entry["replicates"].values())
            three = decide(diffs, dec.alpha if dec.alpha > 0 else ALPHA_BRANCH, sesoi[h])
            if three["decision"] == DECISIONS[0] and not dec.rejected:
                three["decision"] = DECISIONS[2]  # an H the graph never gave alpha cannot be declared superior
            entry["decision"] = three
        else:
            entry["decision"] = {"decision": DECISIONS[2], "reason": entry.get("status")}
    report["graph"] = {"alpha": graph.alpha, "edges": graph.edges}
    return report


# --- command line ----------------------------------------------------------------------------------------------------


def _write(doc: Mapping[str, Any], out: Path | None, dry_run: bool) -> None:
    from ..canonical import atomic_write_json

    text = json.dumps(doc, indent=1, sort_keys=True, default=float)
    if out is not None and not dry_run:
        atomic_write_json(out, doc)
    print(text)


def item_nuisance(preds, treatment: str, control: str, endpoint: str) -> dict[str, Any]:
    """MS §9.3 pilot nuisance parameters beyond the replicate SD, per replicate then averaged: the item variance of the
    per-item difference (the input of the McNemar-style item count of MS §9.5) and the divergence size, the mean
    absolute difference of the two arms' continuous gold margins on the common items."""
    t, c = correctness(preds, treatment, endpoint), correctness(preds, control, endpoint)
    tm, cm = correctness(preds, treatment, endpoint, "margin"), correctness(preds, control, endpoint, "margin")
    var, div = [], []
    for r in sorted(set(t) & set(c)):
        common = sorted(set(t[r]) & set(c[r]))
        if len(common) >= 2:
            var.append(float(np.var([t[r][i] - c[r][i] for i in common], ddof=1)))
        both = sorted(set(tm.get(r, {})) & set(cm.get(r, {})))
        if both:
            div.append(float(np.mean([abs(tm[r][i] - cm[r][i]) for i in both])))
    return {
        "item_diff_variance": float(np.mean(var)) if var else None,
        "margin_divergence": float(np.mean(div)) if div else None,
    }


def _arm_accuracy(preds, arms: Sequence[str], endpoint: str) -> dict[str, list[float]]:
    out = {}
    for arm in arms:
        by_rep = correctness(preds, arm, endpoint)
        if by_rep:
            out[arm] = [float(np.mean(list(v.values()))) for _, v in sorted(by_rep.items())]
    return out


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual stats", description="confirmatory statistics")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pw = sub.add_parser("power", help="MS §9.5 replicate rule from a per-replicate SD")
    pw.add_argument("--sd", type=float, required=True, help="SD of the per-replicate difference (accuracy units)")
    pw.add_argument("--sesoi", type=float, required=True)
    pw.add_argument("--alpha", type=float, default=ALPHA_BRANCH)
    pi = sub.add_parser("pilot", help="the G1' gate and nuisance parameters from dev predictions")
    pi.add_argument("--pred", type=Path, nargs="+", required=True)
    pi.add_argument("--arms", default="B0,B2,B12a")
    pi.add_argument("--endpoint", default="E3ctx/cross")
    pi.add_argument("--pairs", default="B12a:B4,B2:B4,B6:B4", help="treatment:control pairs for the nuisance SDs")
    fu = sub.add_parser("futility", help="MS §9.6 family-I-first futility rule on dev, one replicate")
    fu.add_argument("--pred", type=Path, nargs="+", required=True)
    fu.add_argument("--replicate", type=int, default=0)
    fu.add_argument("--endpoint", default="E3ctx/cross")
    co = sub.add_parser("confirm", help="H_A, H_B, H_C with the MS §9.3 gatekeeper (test predictions)")
    co.add_argument("--pred", type=Path, nargs="+", required=True, help="predictions; only split test is used")
    co.add_argument("--dev-pred", type=Path, nargs="+", help="predictions for the H_B precondition; split dev only")
    co.add_argument("--nll", type=Path, nargs="+", help="eval-nll documents of T2 and T0 (H_C BPB); split test only")
    for p in (pw, pi, fu, co):
        p.add_argument("--out", type=Path)
        p.add_argument("--dry-run", action="store_true", help="read the inputs and print; write nothing")
    args = ap.parse_args(list(argv))
    try:
        if args.cmd == "power":
            doc = {"format": STATS_FORMAT, "kind": "power", **replicate_rule(args.sd, args.sesoi, args.alpha)}
        elif args.cmd == "pilot":
            preds, inputs = select_split(read_predictions(args.pred), "dev", "stats pilot")
            acc = _arm_accuracy(preds, args.arms.split(","), args.endpoint)
            doc = {
                "format": STATS_FORMAT,
                "kind": "pilot",
                "endpoint": args.endpoint,
                "inputs": inputs,
                "gate": pilot_gate(acc) if acc else None,
                "nuisance": {},
            }
            for pair in filter(None, args.pairs.split(",")):
                t, c = pair.split(":")
                diffs = replicate_diffs(preds, t, c, args.endpoint)
                a, b = correctness(preds, t, args.endpoint), correctness(preds, c, args.endpoint)
                disc = [discordance(a[r], b[r])[0] for r in sorted(set(a) & set(b))]
                sd = float(np.std(list(diffs.values()), ddof=1)) if len(diffs) >= 2 else None
                doc["nuisance"][pair] = {
                    "per_replicate_diff": {str(k): v for k, v in diffs.items()},
                    "between_replicate_sd": sd,
                    "discordance": float(np.mean(disc)) if disc else None,
                    **item_nuisance(preds, t, c, args.endpoint),
                    "replicate_rule": replicate_rule(sd, SESOI["H_A"]) if sd else None,
                }
        elif args.cmd == "futility":
            preds, inputs = select_split(read_predictions(args.pred), "dev", "stats futility")
            d12 = replicate_diffs(preds, "B12a", "B4", args.endpoint).get(args.replicate)
            d6 = replicate_diffs(preds, "B6", "B4", args.endpoint).get(args.replicate)
            if d12 is None or d6 is None:
                raise ValueError(
                    f"futility needs B12a, B6 and B4 predictions of replicate {args.replicate} on {args.endpoint}"
                )
            doc = {
                "format": STATS_FORMAT,
                "kind": "futility",
                "replicate": args.replicate,
                "inputs": inputs,
                **futility(d12, d6),
            }
        else:
            preds, inputs = select_split(read_predictions(args.pred), "test", "stats confirm --pred")
            dev = nll = None
            if args.dev_pred:
                dev, inputs["dev_pred"] = select_split(
                    read_predictions(args.dev_pred), "dev", "stats confirm --dev-pred"
                )
            if args.nll:
                nll, inputs["nll"] = select_split(read_nll(args.nll), "test", "stats confirm --nll")
            doc = {"format": STATS_FORMAT, "kind": "confirm", "inputs": inputs, **confirmatory(preds, dev, nll)}
    except (OSError, ValueError, KeyError) as exc:
        print(f"stats: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    _write(doc, args.out, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
