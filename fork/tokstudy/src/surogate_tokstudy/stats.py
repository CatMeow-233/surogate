"""Statistics and the preregistered verdict of one table (spec §5.5, §6.4, §6.7; STUDY M5, M6, M8, M10, §8.4, §9.2).

**Inputs.** The table's prereg (its sha256 is the verdict's), the `tokstudy-nll/1` files of the table's runs (one per
run and budget point: tokstudy tables from `eval score`, the native table tlc-T from TLC's `eval --dump-nll`), and
optionally the runs root (the prereg lock, run records and training throughput), `tokstudy-probe/1` files (the
bilingual Track S rule), an evidence file (the inputs no file of this package carries: integration sign-off, BPM
cross-boundary rates, probe and EV-BIND improvements, a finished M10 re-sweep, measured throughput), and the
twin-gate record (`verdict twin`, `tokstudy-twin/1`).

**Refusals** (spec §6.7). Files of more than one table or backend, or of another table than the prereg's; files of
different evalsets (another common prefix: spec §6.2) or different documents; two files of one (arm, seed, point);
a file evaluated or trained under a prereg the table did not accept (without a lock, the verdict's prereg only; with
the runs root, the lock's history up to the verdict's prereg, which must be the lock's latest); an unsigned prereg,
unless `--draft` (the verdict is then labelled "draft"). A table whose twin gate failed or did not run is labelled
"internal-only" (a tokstudy table only; the native table needs no twin).

**Statistics** (M6). Per run and domain, BPB = sum nll_nats / (ln 2 * sum bytes). For an arm X against the reference R
at one budget point: the seed means, Delta = mean_X - mean_R (bits per byte) and Delta_rel = Delta / mean_R; the
pooled seed SD s_p over every arm with at least two seeds at that point (sum of squared deviations from each arm's
mean over the sum of (n - 1)), sigma_seed = s_p * sqrt(1/n_X + 1/n_R); the document bootstrap (the same resampled
documents for every run, `bootstrap_resamples` resamples, a generator seeded by `bootstrap_seed` and the comparison's
name, so a verdict is reproducible) gives sigma_boot, the SD of the resampled Delta; sigma = sqrt(sigma_seed^2 +
sigma_boot^2).

**Views** (M5, M8). Equal FLOPs: the tokstudy runs of a table end at the same F_target, so the points compare as
they are. Equal bytes: `interpolate_at` (log-log, between neighbouring budget points; None outside them) of each
arm's (train bytes, BPB) at the reference's train bytes at F_target; a sign that differs between the views gives
"tie". The native table has one checkpoint per run at equal training data (equal bytes, its primary view); its
NLL headers carry no training FLOPs, so its equal-FLOPs view is not computed and its rules read the equal-bytes
view (recorded as `views.equal_flops = "unavailable"`). The trend: a lead that shrinks by more than
`narrowing_fraction` between the last two budget points gives "extend".

**Extension rules** (prereg `[extension]`): M6, a pair whose |Delta| < seed_sigma * sigma gets another seed while an
arm has fewer than seed_max; M10, a challenger within lr_resweep_gap (relative) of the reference needs an LR
re-sweep of lr_resweep_points more points for both before it can be decided (evidence `lr_resweep_done`). Either
gives "extend", with the reason.

**Decisions** (`replace`, `keep`, `tie`, `extend`, `eliminated`, `floor-only`), per table:

- tlc-N (STUDY §8.4(e)): every H-Net arm against N0: Python Delta_rel <= -n1_python_rel_gain beyond
  sigma_multiple * sigma on the preregistered seeds, English Delta_rel <= n1_english_rel_max_loss, no narrowing, the
  same sign at equal bytes, throughput >= n1_throughput_ratio x N0, owner integration sign-off. Bytes arms are
  floor-only; other subword arms (N0b) are reported against N0 with decision keep.
- bilingual-B, stage screen (§9.2): an arm worse than B0 in a language by more than screen_eliminate_rel and
  sigma_multiple * sigma is eliminated; the shortlist is B4 (the H-Net) unless eliminated, then the best survivors
  by mean Delta_rel, at most decision_max_survivors in all. Stage decision: both languages non-inferior
  (Delta <= max(noninferior_abs, noninferior_rel * BPB_B0)), one language better by >= superior_rel_gain beyond
  sigma_multiple * sigma, Track S (gold log-prob per byte) not below B0 - track_s_sigma * sigma, the same sign at equal
  bytes, no narrowing; the H-Net also throughput and sign-off. Of the arms that pass, the largest mean gain
  replaces B0; a tie goes to tie_goes_to (B0).
- tlc-T (§8.4(a)-(c), native): (b) the fine pre-tokenizer arm stays the freqtrim choice unless its Python BPB is worse
  than the Qwen-regex arm's by more than pretok_max_loss, or the evidence says its probes did not improve; (a) an
  own-BPE arm replaces that freqtrim arm only with Python Delta_rel <= -own_python_rel_gain beyond sigma_multiple *
  sigma, English Delta_rel <= own_english_rel_max_loss and a BPM cross-boundary rate <= bpm_cross_boundary_max (an
  arm above it is a control, never selected); (c) the 8K arm replaces its 16K arm only if neither language is worse
  by more than vocab_max_loss and the evidence says EV-BIND improved. (d) digits stay per-digit (no rule to run).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tomllib
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import TokstudyError, atomic_write_json, check_format, read_json, sha256_bytes, sha256_file

VERDICT_FORMAT = "tokstudy-verdict/1"
TWIN_FORMAT = "tokstudy-twin/1"
LN2 = math.log(2.0)
DECISIONS = ("replace", "keep", "tie", "extend", "eliminated", "floor-only")
NATIVE_TABLES = {"tlc-T": "tlc"}
TOKSTUDY_BACKEND = "tokstudy-torch"
NATIVE_BACKEND = "native"
REFERENCE = {"tlc-N": "N0", "bilingual-B": "B0"}
DOMAINS = {"tlc-N": ("py", "en"), "tlc-T": ("py", "en"), "bilingual-B": ("zh", "en")}
EXTENSION_KEYS = {"seed_sigma", "seed_max", "lr_resweep_gap", "lr_resweep_points"}
THRESHOLDS = {
    "tlc-N": {
        "n1_python_rel_gain",
        "n1_english_rel_max_loss",
        "n1_trend_no_narrowing",
        "n1_throughput_ratio",
        "owner_integration_signoff",
        "n2_floor_only",
        "sigma_multiple",
        "narrowing_fraction",
        "bootstrap_resamples",
        "bootstrap_seed",
    },
    "bilingual-B": {
        "noninferior_abs",
        "noninferior_rel",
        "superior_rel_gain",
        "sigma_multiple",
        "track_s_sigma",
        "screen_eliminate_rel",
        "decision_max_survivors",
        "b4_throughput_ratio",
        "owner_integration_signoff",
        "tie_goes_to",
        "narrowing_fraction",
        "bootstrap_resamples",
        "bootstrap_seed",
    },
    "tlc-T": {
        "own_python_rel_gain",
        "own_english_rel_max_loss",
        "bpm_cross_boundary_max",
        "pretok_max_loss",
        "vocab_max_loss",
        "sigma_multiple",
        "narrowing_fraction",
        "bootstrap_resamples",
        "bootstrap_seed",
    },
}
EVIDENCE_KEYS = {
    "integration_signoff",
    "throughput",
    "bpm_cross_boundary",
    "probe_improved",
    "evbind_improved",
    "lr_resweep_done",
    "shortlist",
}


# --- the prereg as the verdict reads it ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rules:
    """What the verdict reads of a prereg: table, backend, identity, seeds per stage, extension rules, thresholds,
    budget points (tokstudy tables) and, for tlc-T, the roles of the native arms."""

    path: Path
    sha256: str
    table: str
    backend: str
    signed: bool
    points: tuple[float, ...]
    stages: tuple[str, ...]
    stop_after: Mapping[str, float | None]
    seeds: Mapping[str, Mapping[str, int]]  # stage -> arm -> planned seeds
    by_rank: Mapping[str, tuple[int, ...]]  # stage -> seeds of the survivors by screening rank (bilingual)
    extension: Mapping[str, float]
    thresholds: Mapping[str, Any]
    roles: Mapping[str, Any] = field(default_factory=dict)

    def planned_seeds(self, stage: str, arm: str, rank: int | None = None) -> int | None:
        table = self.seeds.get(stage, {})
        if arm in table:
            return table[arm]
        ranked = self.by_rank.get(stage, ())
        return ranked[rank] if rank is not None and rank < len(ranked) else None


def _seeds(data: Mapping[str, Any], path: Path) -> tuple[dict[str, dict[str, int]], dict[str, tuple[int, ...]]]:
    """`[seeds] ARM = n` (every stage) or `[seeds.<stage>] ARM = n, by_rank = [n, ...]`."""
    raw = data.get("seeds", {})
    if not isinstance(raw, Mapping):
        raise TokstudyError(f"{path}: [seeds] must be a table")
    flat = {k: v for k, v in raw.items() if not isinstance(v, Mapping)}
    staged = {k: v for k, v in raw.items() if isinstance(v, Mapping)}
    if flat and staged:
        raise TokstudyError(f"{path}: [seeds] is either ARM = n or per-stage tables, not both")
    seeds, by_rank = {}, {}
    for stage, table in (staged or {"*": flat}).items():
        seeds[stage], ranked = {}, table.get("by_rank", [])
        for arm, n in table.items():
            if arm == "by_rank":
                continue
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise TokstudyError(f"{path}: seeds of {arm} must be a positive integer, got {n!r}")
            seeds[stage][arm] = n
        if not isinstance(ranked, list) or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in ranked):
            raise TokstudyError(f"{path}: seeds.{stage}.by_rank must be a list of positive integers")
        by_rank[stage] = tuple(ranked)
    return seeds, by_rank


def load_rules(path: Path) -> Rules:
    """A tokstudy table's prereg (prereg.load_prereg, then its verdict sections) or a native table's
    (`backend = "native"`: only the verdict reads it)."""
    path = Path(path)
    try:
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as err:
        raise TokstudyError(f"{path}: cannot read a prereg: {err}") from None
    check_format(data, "prereg")
    table = data.get("table")
    if table in NATIVE_TABLES:
        if data.get("backend") != NATIVE_BACKEND:
            raise TokstudyError(f'{path}: the native table {table} needs backend = "native"')
        known = {"format", "table", "backend", "notes", "roles", "seeds", "extension", "thresholds", "signature"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise TokstudyError(f"{path}: unknown top-level keys {unknown}")
        signature = data.get("signature", {})
        signed = bool(signature.get("signed_by")) and bool(signature.get("date"))
        points, stages, sha = (1.0,), ("decision",), sha256_bytes(raw)
        stop_after = {"decision": None}
    else:
        from .prereg import load_prereg

        if "backend" in data or "roles" in data:
            raise TokstudyError(f"{path}: backend and roles belong to a native table's prereg")
        prereg = load_prereg(path)
        signed, points, stages, sha = prereg.signed, prereg.points, tuple(prereg.stages), prereg.sha256
        stop_after = {name: prereg.stop_after(name) for name in stages}
    seeds, by_rank = _seeds(data, path)
    if "*" in seeds:
        seeds = {s: seeds["*"] for s in stages}
        by_rank = {s: by_rank["*"] for s in stages}
    elif set(seeds) - set(stages):
        raise TokstudyError(f"{path}: [seeds] names stages {sorted(set(seeds) - set(stages))} the prereg lacks")
    extension = dict(data.get("extension", {}))
    if set(extension) != EXTENSION_KEYS:
        raise TokstudyError(
            f"{path}: [extension] needs exactly {sorted(EXTENSION_KEYS)} (M6, M10), has {sorted(extension)}"
        )
    thresholds = dict(data.get("thresholds", {}))
    if set(thresholds) != THRESHOLDS[table]:
        missing, extra = THRESHOLDS[table] - set(thresholds), set(thresholds) - THRESHOLDS[table]
        raise TokstudyError(f"{path}: [thresholds] of {table}: missing {sorted(missing)}, unknown {sorted(extra)}")
    roles = dict(data.get("roles", {}))
    if table == "tlc-T":
        need = {"reference", "freqtrim", "fine", "own", "vocab"}
        if set(roles) != need:
            raise TokstudyError(f"{path}: [roles] of tlc-T needs {sorted(need)}")
    backend = NATIVE_BACKEND if table in NATIVE_TABLES else TOKSTUDY_BACKEND
    return Rules(
        path, sha, table, backend, signed, points, stages, stop_after, seeds, by_rank, extension, thresholds, roles
    )


# --- NLL files and the table they form -------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunNLL:
    """One NLL file: a run's budget point. `nll` and `bytes` are per document in the table's document order."""

    path: Path
    sha256: str
    header: Mapping[str, Any]
    arm: str
    seed: int
    point: float
    keys: tuple[str, ...]
    domains: tuple[str, ...]
    nll: np.ndarray
    bytes: np.ndarray

    def bpb(self, domain: str, counts: np.ndarray | None = None) -> np.ndarray | float:
        mask = np.array([d == domain for d in self.domains])
        nll, n_bytes = self.nll[mask], self.bytes[mask]
        if counts is None:
            return float(nll.sum() / (LN2 * n_bytes.sum()))
        return (counts @ nll) / (LN2 * (counts @ n_bytes))


def read_run_nll(path: Path) -> RunNLL:
    from .evaluate import read_nll

    path = Path(path)
    header, rows = read_nll(path)
    rows = sorted(rows, key=lambda r: r.key)
    if len({r.key for r in rows}) != len(rows):
        raise TokstudyError(f"{path}: a document key occurs twice")
    seed = header.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TokstudyError(f"{path}: the header names no integer seed")
    return RunNLL(
        path,
        sha256_file(path),
        header,
        str(header.get("arm")),
        seed,
        float(header.get("point", 1.0)),
        tuple(r.key for r in rows),
        tuple(r.domain for r in rows),
        np.array([r.nll_nats for r in rows], dtype=np.float64),
        np.array([r.bytes for r in rows], dtype=np.float64),
    )


@dataclass
class Table:
    rules: Rules
    runs: dict[tuple[str, int], dict[float, RunNLL]]
    context: Mapping[str, Any]

    def arms(self) -> list[str]:
        return sorted({a for a, _ in self.runs})

    def at(self, arm: str, point: float) -> list[RunNLL]:
        return [p[point] for (a, _), p in sorted(self.runs.items()) if a == arm and point in p]

    def points(self, arm: str) -> list[float]:
        found = [set(p) for (a, _), p in self.runs.items() if a == arm]
        return sorted(set.intersection(*found)) if found else []


def accepted_preregs(rules: Rules, runs_root: Path | None) -> set[str]:
    """The prereg sha256s a file of this verdict may have been trained or evaluated under (module docstring)."""
    if runs_root is None or rules.backend == NATIVE_BACKEND:
        return {rules.sha256}
    from . import registry

    path = registry.lock_path(runs_root, rules.table)
    if not path.is_file():
        raise TokstudyError(f"{path}: no prereg lock of table {rules.table} under the runs root")
    history = [e["sha256"] for e in read_json(path, "prereg-lock")["history"]]
    if history[-1] != rules.sha256:
        raise TokstudyError(f"{rules.path}: not the latest prereg of table {rules.table} ({history[-1]} is)")
    return set(history)


def build_table(rules: Rules, paths: Sequence[Path], *, runs_root: Path | None = None) -> Table:
    files = [read_run_nll(p) for p in paths]
    if not files:
        raise TokstudyError("no NLL files")
    tables = {f.header.get("table") for f in files}
    backends = {f.header.get("backend") for f in files}
    if len(tables) != 1 or len(backends) != 1:
        raise TokstudyError(f"the files mix tables {sorted(map(str, tables))} or backends {sorted(map(str, backends))}")
    if tables != {rules.table} or backends != {rules.backend}:
        raise TokstudyError(f"the files are table {tables.pop()} ({backends.pop()}), the prereg is {rules.table}'s")
    same = (
        ("evalset_sha256", "k_eval") if rules.backend == TOKSTUDY_BACKEND else ("corpus_spec_sha256", "split", "keys")
    )
    context = {}
    for key in same:
        values = {json.dumps(f.header.get(key)) for f in files}
        if len(values) != 1 or values == {"null"}:
            raise TokstudyError(f"the files differ in {key} (or lack it): {sorted(values)}; one evalset per table")
        context[key] = files[0].header.get(key)
    first = files[0]
    for f in files[1:]:
        if f.keys != first.keys or f.domains != first.domains or not np.array_equal(f.bytes, first.bytes):
            raise TokstudyError(f"{f.path}: other documents than {first.path} (keys, domains or bytes differ)")
    if rules.backend == TOKSTUDY_BACKEND:
        accepted = accepted_preregs(rules, runs_root)
        for f in files:
            for key in ("prereg_sha256", "run_prereg_sha256"):
                if f.header.get(key) not in accepted:
                    raise TokstudyError(
                        f"{f.path}: {key} {f.header.get(key)} is not a prereg this verdict accepts "
                        f"({sorted(accepted)}; give --runs for the table's amendments)"
                    )
    runs: dict[tuple[str, int], dict[float, RunNLL]] = {}
    for f in files:
        slot = runs.setdefault((f.arm, f.seed), {})
        if f.point in slot:
            raise TokstudyError(
                f"{f.path} and {slot[f.point].path}: two files of arm {f.arm} seed {f.seed} at {f.point:g}"
            )
        slot[f.point] = f
    missing = set(DOMAINS[rules.table]) - set(first.domains)
    if missing:
        raise TokstudyError(f"the evaluation documents lack the domains {sorted(missing)}")
    return Table(rules, runs, context)


# --- statistics ------------------------------------------------------------------------------------------------------


def pooled_seed_sd(groups: Mapping[str, Sequence[float]]) -> float | None:
    """sqrt(sum over groups of squared deviations from the group mean / sum of (n - 1)); None without a group of
    at least two values."""
    ss = dof = 0.0
    for values in groups.values():
        if len(values) >= 2:
            x = np.asarray(values, dtype=np.float64)
            ss += float(((x - x.mean()) ** 2).sum())
            dof += len(x) - 1
    return math.sqrt(ss / dof) if dof else None


def interpolate_at(points: Sequence[tuple[float, float]], x: float) -> float | None:
    """Log-log linear interpolation of (x_i, y_i) at x between the neighbouring points; None outside their range."""
    pts = sorted((float(a), float(b)) for a, b in points)
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=False):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))
    return pts[0][1]  # a single point equal to x


def _rng(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(name.encode("utf-8"))])


@dataclass(frozen=True)
class Delta:
    arm: str
    ref: str
    domain: str
    point: float
    n_arm: int
    n_ref: int
    bpb_arm: float
    bpb_ref: float
    delta: float
    rel: float
    sigma_seed: float
    sigma_boot: float
    sigma: float
    pooled_sd: float | None

    def to_json(self) -> dict[str, Any]:
        return dict(self.__dict__)


def paired_delta(table: Table, arm: str, ref: str, domain: str, point: float) -> Delta:
    """Delta of `arm` against `ref` at one budget point and domain (module docstring: Statistics)."""
    runs_arm, runs_ref = table.at(arm, point), table.at(ref, point)
    if not runs_arm or not runs_ref:
        raise TokstudyError(f"no files of {arm if not runs_arm else ref} at budget point {point:g}")
    t = table.rules.thresholds
    a = np.array([r.bpb(domain) for r in runs_arm])
    b = np.array([r.bpb(domain) for r in runs_ref])
    groups = {x: [r.bpb(domain) for r in table.at(x, point)] for x in table.arms()}
    sd = pooled_seed_sd(groups)
    sigma_seed = 0.0 if sd is None else sd * math.sqrt(1 / len(a) + 1 / len(b))
    n_docs = int(sum(1 for d in runs_arm[0].domains if d == domain))
    resamples = int(t.get("bootstrap_resamples", 10000))
    rng = _rng(int(t.get("bootstrap_seed", 0)), f"{table.rules.table}|{arm}|{ref}|{domain}|{point:g}")
    draws = []
    for start in range(0, resamples, 1000):
        k = min(1000, resamples - start)
        counts = rng.multinomial(n_docs, np.full(n_docs, 1.0 / n_docs), size=k).astype(np.float64)
        da = np.mean([r.bpb(domain, counts) for r in runs_arm], axis=0)
        db = np.mean([r.bpb(domain, counts) for r in runs_ref], axis=0)
        draws.append(da - db)
    sigma_boot = float(np.std(np.concatenate(draws), ddof=1)) if resamples > 1 else 0.0
    delta = float(a.mean() - b.mean())
    return Delta(
        arm,
        ref,
        domain,
        point,
        len(a),
        len(b),
        float(a.mean()),
        float(b.mean()),
        delta,
        delta / float(b.mean()),
        sigma_seed,
        sigma_boot,
        math.hypot(sigma_seed, sigma_boot),
        sd,
    )


def equal_bytes(table: Table, arm: str, ref: str, domain: str) -> dict[str, Any]:
    """The arm's BPB interpolated at the reference's train bytes at F_target (tokstudy tables), against the
    reference's BPB there; None when the reference's bytes lie outside the arm's budget points."""
    final = max(table.rules.points)
    ref_runs = table.at(ref, final)
    if table.rules.backend == NATIVE_BACKEND or any("train_bytes" not in r.header for r in ref_runs):
        return {"available": False, "reason": "no training bytes in the NLL headers"}
    x = float(np.mean([r.header["train_bytes"] for r in ref_runs]))
    y_ref = float(np.mean([r.bpb(domain) for r in ref_runs]))
    curve = []
    for point in table.points(arm):
        runs = table.at(arm, point)
        curve.append(
            (float(np.mean([r.header["train_bytes"] for r in runs])), float(np.mean([r.bpb(domain) for r in runs])))
        )
    y = interpolate_at(curve, x)
    if y is None:
        return {"available": False, "reason": f"the reference's {x:.4g} bytes lie outside the arm's points {curve}"}
    return {
        "available": True,
        "bytes": x,
        "bpb_arm": y,
        "bpb_ref": y_ref,
        "delta": y - y_ref,
        "rel": (y - y_ref) / y_ref,
    }


def narrowing(table: Table, arm: str, ref: str, domain: str) -> dict[str, Any]:
    """The lead of the arm ahead at the final point, at the last two budget points both arms have."""
    common = sorted(set(table.points(arm)) & set(table.points(ref)))
    if len(common) < 2:
        return {"available": False, "reason": "fewer than two budget points"}
    leads = []
    for point in common[-2:]:
        a = np.mean([r.bpb(domain) for r in table.at(arm, point)])
        b = np.mean([r.bpb(domain) for r in table.at(ref, point)])
        leads.append(float(b - a))  # positive: the arm is ahead
    prev, last = leads
    f = float(table.rules.thresholds.get("narrowing_fraction", 0.5))
    shrinks = (
        prev != 0 and last != 0 and math.copysign(1, prev) == math.copysign(1, last) and abs(last) < (1 - f) * abs(prev)
    )
    return {"available": True, "points": common[-2:], "leads": leads, "narrowing": bool(shrinks)}


# --- decisions -------------------------------------------------------------------------------------------------------


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


@dataclass
class ArmVerdict:
    arm: str
    decision: str
    conditions: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        if self.decision not in DECISIONS:
            raise TokstudyError(f"unknown decision {self.decision!r}")
        return {
            "decision": self.decision,
            "conditions": self.conditions,
            "reasons": self.reasons,
            "pending": self.pending,
        }


def _extension(table: Table, d: Delta, evidence: Mapping[str, Any], stage: str, rank: int | None) -> list[str]:
    """The M6, M10 and preregistered-seed reasons to extend one deciding comparison (module docstring)."""
    e, out = table.rules.extension, []
    if abs(d.delta) < e["seed_sigma"] * d.sigma:
        for arm, n in ((d.arm, d.n_arm), (d.ref, d.n_ref)):
            if n < e["seed_max"]:
                out.append(
                    f"M6: |Delta| {abs(d.delta):.4g} < {e['seed_sigma']:g} sigma ({d.sigma:.4g}) in {d.domain}: "
                    f"one more seed of {arm} ({n} of at most {int(e['seed_max'])})"
                )
    if abs(d.rel) < e["lr_resweep_gap"] and d.arm not in evidence.get("lr_resweep_done", []):
        out.append(
            f"M10: {d.arm} is within {e['lr_resweep_gap']:.0%} of {d.ref} in {d.domain}: re-sweep the LR "
            f"({int(e['lr_resweep_points'])} more points for both), then pass lr_resweep_done"
        )
    for arm, n, r in ((d.arm, d.n_arm, rank), (d.ref, d.n_ref, None)):
        planned = table.rules.planned_seeds(stage, arm, r)
        if planned is not None and n < planned:
            out.append(f"seeds: {arm} has {n} of its {planned} preregistered seeds at {d.point:g}")
    return out


def decide(
    table: Table,
    arm: str,
    ref: str,
    point: float,
    evidence: Mapping[str, Any],
    stage: str,
    *,
    gain_domains: Sequence[str],
    gain_rel: float,
    max_rel: Mapping[str, float] | None = None,
    max_abs: Mapping[str, float] | None = None,
    needs: Mapping[str, tuple[bool | None, str]] | None = None,
    rank: int | None = None,
) -> ArmVerdict:
    """A challenger against the reference at one point: the gain in one of `gain_domains` (Delta_rel <= -gain_rel
    beyond sigma_multiple sigma), non-inferiority (`max_rel`, `max_abs` per domain), the further `needs` ({name:
    (pass or None, what is missing)}); then tie on a sign flip between the views, extend on a narrowing lead or an
    extension rule, replace when everything passes, keep otherwise (with what is pending)."""
    k = float(table.rules.thresholds["sigma_multiple"])
    deltas = {dom: paired_delta(table, arm, ref, dom, point) for dom in DOMAINS[table.rules.table]}
    v = ArmVerdict(arm, "keep", conditions={"deltas": {d: x.to_json() for d, x in deltas.items()}})
    gained = [d for d in gain_domains if deltas[d].rel <= -gain_rel and abs(deltas[d].delta) > k * deltas[d].sigma]
    v.conditions["gain"] = {"domains": gained, "need": f"Delta_rel <= -{gain_rel:g} beyond {k:g} sigma"}
    inferior = []
    for dom, limit in (max_rel or {}).items():
        v.conditions[f"noninferior_{dom}"] = {"rel": deltas[dom].rel, "max_rel": limit}
        if deltas[dom].rel > limit:
            inferior.append(dom)
    for dom, limit in (max_abs or {}).items():
        v.conditions[f"noninferior_{dom}"] = {"delta": deltas[dom].delta, "max": limit}
        if deltas[dom].delta > limit:
            inferior.append(dom)
    flips, shrink, unavailable = [], [], []
    for dom in gained:  # the views and the trend matter for a gain the arm would be decided on
        view, trend = equal_bytes(table, arm, ref, dom), narrowing(table, arm, ref, dom)
        v.conditions[f"equal_bytes_{dom}"], v.conditions[f"trend_{dom}"] = view, trend
        if not view.get("available"):
            unavailable.append(dom)
        elif _sign(view["delta"]) != _sign(deltas[dom].delta):
            flips.append(dom)
        if trend.get("narrowing"):
            shrink.append(dom)
    failed = []
    for name, (ok, missing) in (needs or {}).items():
        v.conditions[name] = {"pass": ok}
        if ok is None:
            v.pending.append(missing)
        elif not ok:
            failed.append(name)
    extend = [r for dom in (gained or gain_domains) for r in _extension(table, deltas[dom], evidence, stage, rank)]
    if gained and flips:
        v.decision = "tie"
        v.reasons.append(f"the equal-FLOPs and equal-bytes views disagree in sign in {flips} (M5)")
    elif gained and shrink:
        v.decision = "extend"
        v.reasons.append(f"the lead shrinks by more than half between the last two budget points in {shrink} (M8)")
    elif extend:
        v.decision = "extend"
        v.reasons += extend
    elif gained and not inferior and not failed and not v.pending:
        v.decision = "replace"
    else:
        if not gained:
            v.reasons.append(f"no gain of {gain_rel:.1%} beyond {k:g} sigma in {list(gain_domains)}")
        if inferior:
            v.reasons.append(f"inferior to {ref} beyond the preregistered limit in {inferior}")
        if failed:
            v.reasons.append(f"failed: {failed}")
        if gained and not inferior and not failed:
            v.reasons.append("the BPB conditions hold; pending: " + "; ".join(v.pending))
    if unavailable and v.decision == "replace":
        v.reasons.append(f"equal-bytes view unavailable in {unavailable}: decided on the primary view")
    return v


def _kind(table: Table, arm: str) -> str:
    kinds = {f.header.get("arm_kind") for (a, _), runs in table.runs.items() if a == arm for f in runs.values()}
    if table.rules.backend == NATIVE_BACKEND:
        return "subword"
    if len(kinds) != 1 or None in kinds:
        raise TokstudyError(f"the NLL files of arm {arm} name no single arm_kind ({sorted(map(str, kinds))})")
    return kinds.pop()


def _need_throughput(evidence: Mapping[str, Any], arm: str, ratio: float) -> tuple[bool | None, str]:
    found = evidence.get("throughput", {}).get(arm)
    missing = f"the training bytes/s of {arm} against the reference (--runs, or evidence.throughput)"
    return (None if found is None else found >= ratio), missing


def _need_signoff(evidence: Mapping[str, Any]) -> tuple[bool | None, str]:
    signed = evidence.get("integration_signoff")
    return (None if not signed else True), "the owner's integration sign-off (evidence.integration_signoff)"


def verdict_tlc_n(table: Table, evidence: Mapping[str, Any], stage: str) -> tuple[dict[str, ArmVerdict], dict]:
    t, ref, point = table.rules.thresholds, REFERENCE["tlc-N"], max(table.rules.points)
    out = {ref: ArmVerdict(ref, "keep", reasons=["the reference (the twin of the native TOK-AB leader)"])}
    for arm in table.arms():
        if arm == ref:
            continue
        kind = _kind(table, arm)
        if kind == "hnet":
            needs = {
                "throughput": _need_throughput(evidence, arm, float(t["n1_throughput_ratio"])),
                "integration_signoff": _need_signoff(evidence),
            }
            out[arm] = decide(
                table,
                arm,
                ref,
                point,
                evidence,
                stage,
                gain_domains=["py"],
                gain_rel=float(t["n1_python_rel_gain"]),
                max_rel={"en": float(t["n1_english_rel_max_loss"])},
                needs=needs,
            )
        else:
            deltas = {d: paired_delta(table, arm, ref, d, point).to_json() for d in DOMAINS["tlc-N"]}
            if kind == "bytes":
                out[arm] = ArmVerdict(arm, "floor-only", {"deltas": deltas}, ["flat bytes: a floor, never promoted"])
            else:
                out[arm] = ArmVerdict(arm, "keep", {"deltas": deltas}, ["another twin (N0b): reported against N0"])
    promoted = sorted(a for a, v in out.items() if v.decision == "replace")
    summary = {
        "promote": promoted,
        "note": "STUDY §8.5: even a promoted H-Net waits for the owner's integration route; until then TLC "
        "keeps the subword tokenizer of the native TOK-AB verdict",
    }
    return out, summary


def verdict_bilingual(
    table: Table, evidence: Mapping[str, Any], stage: str, probes: Mapping[str, Any]
) -> tuple[dict[str, ArmVerdict], dict]:
    t, ref = table.rules.thresholds, REFERENCE["bilingual-B"]
    k = float(t["sigma_multiple"])
    out = {ref: ArmVerdict(ref, "keep", reasons=["the reference"])}
    if stage == "screen":
        point = table.rules.stop_after.get("screen") or max(table.rules.points)
        survivors = []
        for arm in table.arms():
            if arm == ref:
                continue
            deltas = {d: paired_delta(table, arm, ref, d, point) for d in DOMAINS["bilingual-B"]}
            v = ArmVerdict(arm, "keep", conditions={"deltas": {d: x.to_json() for d, x in deltas.items()}})
            limit = float(t["screen_eliminate_rel"])
            worse = [d for d, x in deltas.items() if x.rel > limit and x.delta > k * x.sigma]
            if _kind(table, arm) == "bytes":
                v.decision, v.reasons = "floor-only", ["flat bytes: a floor, not entered in the decision"]
            elif worse:
                v.decision = "eliminated"
                v.reasons.append(f"worse than {ref} by more than {limit:.0%} and {k:g} sigma in {worse}")
            else:
                survivors.append((float(np.mean([x.rel for x in deltas.values()])), arm))
            out[arm] = v
        hnet = [a for _, a in survivors if _kind(table, a) == "hnet"]
        ranked = [a for _, a in sorted(survivors) if a not in hnet]
        shortlist = (hnet + ranked)[: int(t["decision_max_survivors"])]
        for arm in hnet:
            if arm in shortlist:
                out[arm].reasons.append("shortlisted (the H-Net enters unless eliminated)")
                out[arm].conditions["decision_seeds"] = table.rules.planned_seeds("decision", arm)
        for rank, arm in enumerate(a for a in ranked if a in shortlist):
            out[arm].reasons.append(f"shortlisted: survivor rank {rank + 1}")
            out[arm].conditions["decision_seeds"] = table.rules.planned_seeds("decision", arm, rank)
        for _, arm in survivors:
            if arm not in shortlist:
                out[arm].reasons.append("survives the screen, not shortlisted")
        summary = {
            "point": point,
            "shortlist": shortlist,
            "eliminated": sorted(a for a, v in out.items() if v.decision == "eliminated"),
        }
        return out, summary
    point = max(table.rules.points)
    challengers = [a for a in table.arms() if a != ref]
    ranked = [a for a in evidence.get("shortlist", []) if a in challengers and _kind(table, a) != "hnet"]
    for arm in challengers:
        kind = _kind(table, arm)
        if kind == "bytes":
            out[arm] = ArmVerdict(arm, "floor-only", reasons=["flat bytes: a floor, never a candidate"])
            continue
        ref_runs = table.at(ref, point)
        bpb_ref = {d: float(np.mean([r.bpb(d) for r in ref_runs])) for d in DOMAINS["bilingual-B"]}
        limits = {d: max(float(t["noninferior_abs"]), float(t["noninferior_rel"]) * bpb_ref[d]) for d in bpb_ref}
        track = probes.get(arm)
        needs = {"track_s": (None if track is None else track["pass"], "Track S probes of every run (--probe)")}
        if kind == "hnet":
            needs["throughput"] = _need_throughput(evidence, arm, float(t["b4_throughput_ratio"]))
            needs["integration_signoff"] = _need_signoff(evidence)
        rank = ranked.index(arm) if arm in ranked else None
        v = decide(
            table,
            arm,
            ref,
            point,
            evidence,
            "decision",
            gain_domains=list(DOMAINS["bilingual-B"]),
            gain_rel=float(t["superior_rel_gain"]),
            max_abs=limits,
            needs=needs,
            rank=rank,
        )
        if track is not None:
            v.conditions["track_s"] = track
        out[arm] = v
    winners = [a for a, v in out.items() if v.decision == "replace"]
    best = None
    if winners:
        gain = {
            a: -float(np.mean([out[a].conditions["deltas"][d]["rel"] for d in DOMAINS["bilingual-B"]])) for a in winners
        }
        best = max(winners, key=lambda a: (gain[a], a))
        for a in winners:
            if a != best:
                out[a].decision = "keep"
                out[a].reasons.append(f"passes, but {best} has the larger mean gain")
        out[ref].reasons.append(f"replaced by {best}")
    ties = sorted(a for a, v in out.items() if v.decision == "tie")
    summary = {"winner": best or ref, "ties_go_to": t["tie_goes_to"], "ties": ties}
    return out, summary


def verdict_tlc_t(table: Table, evidence: Mapping[str, Any], stage: str) -> tuple[dict[str, ArmVerdict], dict]:
    """STUDY §8.4(a)-(c) on the native table, at its one (equal-bytes) checkpoint per run."""
    t, roles, point = table.rules.thresholds, table.rules.roles, 1.0
    freqtrim, fine, own, vocab = roles["freqtrim"], roles["fine"], list(roles["own"]), dict(roles["vocab"])
    out: dict[str, ArmVerdict] = {}
    # (b) the pre-tokenizer of the freqtrim arm: fine unless worse than the Qwen regex or its probes do not improve
    d_fine = paired_delta(table, fine, freqtrim, "py", point)
    probe_ok = evidence.get("probe_improved", {}).get(fine)
    take_fine = d_fine.rel <= float(t["pretok_max_loss"]) and probe_ok is not False
    chosen = fine if take_fine else freqtrim
    other = freqtrim if take_fine else fine
    reason = f"(b) the freqtrim arm is {chosen}"
    out[chosen] = ArmVerdict(chosen, "keep", {"fine_vs_qwen_regex_py": d_fine.to_json(), "probe_improved": probe_ok})
    out[chosen].reasons.append(reason)
    if probe_ok is None:
        out[chosen].pending.append(f"whether {fine}'s probes improved (evidence.probe_improved)")
    out[other] = ArmVerdict(other, "keep", reasons=[reason])
    # (a) an own BPE replaces the chosen freqtrim arm
    bpm = evidence.get("bpm_cross_boundary", {})
    limit = float(t["bpm_cross_boundary_max"])
    for arm in own:
        if arm not in table.arms():
            continue
        rate = bpm.get(arm)
        needs = {
            "bpm_cross_boundary": (
                None if rate is None else rate <= limit,
                f"the BPM cross-boundary rate of {arm} (evidence.bpm_cross_boundary, `align`)",
            )
        }
        v = decide(
            table,
            arm,
            chosen,
            point,
            evidence,
            stage,
            gain_domains=["py"],
            gain_rel=float(t["own_python_rel_gain"]),
            max_rel={"en": float(t["own_english_rel_max_loss"])},
            needs=needs,
        )
        if rate is not None and rate > limit:
            v.reasons.append(f"BPM cross-boundary rate {rate:.2%} > {limit:.0%}: a control arm, never selected")
        out[arm] = v
    gains = {a: out[a].conditions["deltas"]["py"]["rel"] for a in own if a in out and out[a].decision == "replace"}
    best = min(gains, key=lambda a: (gains[a], a)) if gains else None
    for arm in gains:
        if arm != best:
            out[arm].decision = "keep"
            out[arm].reasons.append(f"passes, but {best} gains more")
    if best:
        out[chosen].reasons.append(f"(a) replaced by {best}")
    winner = best or chosen
    # (c) the 8K vocabulary of the winner
    small, size = vocab.get(winner), "16k"
    if small and small in table.arms():
        deltas = {d: paired_delta(table, small, winner, d, point) for d in ("py", "en")}
        ev = evidence.get("evbind_improved", {}).get(small)
        v = ArmVerdict(small, "keep", {"deltas": {d: x.to_json() for d, x in deltas.items()}, "evbind_improved": ev})
        worse = [d for d, x in deltas.items() if x.rel > float(t["vocab_max_loss"])]
        if worse:
            v.reasons.append(f"(c) worse than {winner} by more than {float(t['vocab_max_loss']):.0%} in {worse}")
        elif ev is None:
            v.pending.append(f"whether EV-BIND improved for {small} (evidence.evbind_improved)")
        elif ev:
            v.decision, size = "replace", "8k"
            v.reasons.append("(c) neither language worse by more than vocab_max_loss, and EV-BIND improves")
        else:
            v.reasons.append("(c) EV-BIND does not improve")
        out[small] = v
    summary = {
        "freqtrim_arm": chosen,
        "tokenizer": small if size == "8k" else winner,
        "vocab": size,
        "digits": "per digit (d), unchanged",
    }
    return out, summary


def track_s(rules: Rules, probe_files: Sequence[Path], ref: str) -> dict[str, Any]:
    """Per arm, Track S (gold log-prob per byte, on the items scorable for every file) against the reference:
    pass when mean_X >= mean_ref - track_s_sigma * sigma (seeds pooled, item bootstrap)."""
    from .probes import read_probe

    if not probe_files:
        return {}
    data = []
    for path in probe_files:
        header, rows = read_probe(path)
        if header.get("table") != rules.table:
            raise TokstudyError(f"{path}: a probe file of table {header.get('table')}, not {rules.table}")
        data.append((header, {r.key: r for r in rows}))
    sha = {h.get("items_sha256") for h, _ in data}
    if len(sha) != 1:
        raise TokstudyError("the probe files score different item lists")
    common = sorted(set.intersection(*({k for k, r in rows.items() if r.scorable} for _, rows in data)))
    if not common:
        raise TokstudyError("no Track S item is scorable for every arm")
    values: dict[str, list[np.ndarray]] = {}
    for header, rows in data:
        values.setdefault(header["arm"], []).append(np.array([rows[k].gold_logprob_per_byte for k in common]))
    coverage = {h["arm"]: h.get("coverage") for h, _ in data}
    if ref not in values:
        raise TokstudyError(f"no probe file of the reference {ref}")
    t = rules.thresholds
    sd = pooled_seed_sd({a: [float(v.mean()) for v in vs] for a, vs in values.items()})
    out = {}
    rng = _rng(int(t.get("bootstrap_seed", 0)), "track_s")
    idx = rng.integers(0, len(common), size=(int(t.get("bootstrap_resamples", 10000)), len(common)))
    for arm, vs in values.items():
        if arm == ref:
            continue
        a, b = np.mean(vs, axis=0), np.mean(values[ref], axis=0)
        boot = a[idx].mean(axis=1) - b[idx].mean(axis=1)
        sigma_seed = 0.0 if sd is None else sd * math.sqrt(1 / len(vs) + 1 / len(values[ref]))
        sigma = math.hypot(sigma_seed, float(np.std(boot, ddof=1)))
        delta = float(a.mean() - b.mean())
        out[arm] = {
            "items": len(common),
            "mean_arm": float(a.mean()),
            "mean_ref": float(b.mean()),
            "delta": delta,
            "sigma": sigma,
            "need": f">= ref - {t['track_s_sigma']} sigma",
            "pass": delta >= -float(t["track_s_sigma"]) * sigma,
            "coverage": coverage,
        }
    return out


def throughput_from_runs(runs_root: Path, table: Table) -> dict[str, float]:
    """Median training bytes/s of each arm's main runs (metrics.jsonl) over the reference's."""
    from . import registry

    ref = REFERENCE.get(table.rules.table)
    if ref is None:
        return {}
    ids = {f.header.get("run_id") for runs in table.runs.values() for f in runs.values()}
    medians: dict[str, list[float]] = {}
    for record in registry.list_runs(runs_root, table.rules.table):
        if record["run_id"] not in ids:
            continue
        lines = [json.loads(x) for x in (Path(record["dir"]) / "metrics.jsonl").read_text().splitlines()[1:]]
        speeds = [x["bytes_per_s"] for x in lines[len(lines) // 10 :] if x.get("bytes_per_s")]
        if speeds:
            medians.setdefault(record["arm"]["id"], []).append(float(np.median(speeds)))
    if ref not in medians:
        return {}
    base = float(np.mean(medians[ref]))
    return {arm: float(np.mean(v)) / base for arm, v in medians.items() if arm != ref}


def load_evidence(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        data = (
            json.loads(Path(path).read_text(encoding="utf-8"))
            if Path(path).suffix == ".json"
            else tomllib.loads(Path(path).read_text(encoding="utf-8"))
        )
    except (OSError, ValueError, tomllib.TOMLDecodeError) as err:
        raise TokstudyError(f"{path}: cannot read the evidence: {err}") from None
    unknown = sorted(set(data) - EVIDENCE_KEYS)
    if unknown:
        raise TokstudyError(f"{path}: unknown evidence keys {unknown}; known: {sorted(EVIDENCE_KEYS)}")
    return data


def verdict(
    rules: Rules,
    nll: Sequence[Path],
    *,
    stage: str | None = None,
    runs_root: Path | None = None,
    probes: Sequence[Path] = (),
    evidence: Mapping[str, Any] | None = None,
    twin: Path | None = None,
    draft: bool = False,
) -> dict[str, Any]:
    """The `tokstudy-verdict/1` document of one table (module docstring)."""
    if not rules.signed and not draft:
        raise TokstudyError(f"{rules.path} is not signed: the verdict refuses an unsigned table (--draft for a draft)")
    stage = stage or ("decision" if "decision" in rules.stages else rules.stages[-1])
    if stage not in rules.stages:
        raise TokstudyError(f"{rules.path}: no stage {stage!r} (stages: {list(rules.stages)})")
    table = build_table(rules, nll, runs_root=runs_root)
    evidence = dict(evidence or {})
    if runs_root is not None and "throughput" not in evidence:
        evidence["throughput"] = throughput_from_runs(runs_root, table)
    label = "decision" if rules.signed else "draft"
    twin_record = None
    if rules.backend == TOKSTUDY_BACKEND:
        if twin is None:
            label, twin_record = "internal-only", {"passed": None, "reason": "the twin gate did not run (--twin)"}
        else:
            twin_record = read_json(Path(twin), "twin")
            if twin_record.get("table") != rules.table:
                raise TokstudyError(f"{twin}: the twin gate of table {twin_record.get('table')}, not {rules.table}")
            if not twin_record.get("passed"):
                label = "internal-only"
    if rules.table == "tlc-N":
        arms, summary = verdict_tlc_n(table, evidence, stage)
    elif rules.table == "bilingual-B":
        arms, summary = verdict_bilingual(table, evidence, stage, track_s(rules, probes, REFERENCE["bilingual-B"]))
    else:
        arms, summary = verdict_tlc_t(table, evidence, stage)
    files = [
        {"path": f.path.name, "sha256": f.sha256, "arm": f.arm, "seed": f.seed, "point": f.point}
        for runs in table.runs.values()
        for f in runs.values()
    ]
    native = rules.backend == NATIVE_BACKEND
    return {
        "format": VERDICT_FORMAT,
        "table": rules.table,
        "backend": rules.backend,
        "stage": stage,
        "label": label,
        "prereg": {"path": rules.path.name, "sha256": rules.sha256, "signed": rules.signed},
        "context": dict(table.context),
        "twin_gate": twin_record,
        "views": {
            "equal_flops": "unavailable: native NLL headers carry no training FLOPs" if native else "primary",
            "equal_bytes": "primary" if native else "interpolated at the reference's train bytes at F_target",
        },
        "evidence": evidence,
        "files": sorted(files, key=lambda f: (f["arm"], f["seed"], f["point"])),
        "decision": summary,
        "arms": {a: v.to_json() for a, v in sorted(arms.items())},
    }


# --- the twin gates --------------------------------------------------------------------------------------------------


def twin_gate(
    rules_path: Path, tokstudy: Sequence[Path], native: Sequence[Path], *, draft: bool = False
) -> dict[str, Any]:
    """TW-2 / TW-3: per domain, BPB of the tokstudy twin's files against the production side's, each over its own
    documents (the contexts differ: docs/runbook.md, step 5.3); passed when every |ratio - 1| <= twin_rel_max."""
    from .evaluate import read_nll
    from .prereg import load_prereg

    prereg = load_prereg(rules_path)
    if not prereg.signed and not draft:
        raise TokstudyError(f"{rules_path} is not signed (--draft for a draft)")
    limit = tomllib.loads(Path(rules_path).read_text(encoding="utf-8")).get("thresholds", {}).get("twin_rel_max")
    if not isinstance(limit, int | float) or isinstance(limit, bool):
        raise TokstudyError(f"{rules_path}: [thresholds] twin_rel_max is the twin gate's limit")

    def side(paths):
        rows = [r for p in paths for r in read_nll(p)[1]]
        headers = [read_nll(p)[0] for p in paths]
        out = {}
        for d in sorted({r.domain for r in rows}):
            picked = [r for r in rows if r.domain == d]
            out[d] = {
                "bpb": sum(r.nll_nats for r in picked) / (LN2 * sum(r.bytes for r in picked)),
                "docs": len(picked),
                "bytes": sum(r.bytes for r in picked),
            }
        return out, headers

    ours, ours_h = side(tokstudy)
    theirs, theirs_h = side(native)
    if {h.get("table") for h in ours_h} != {prereg.table}:
        raise TokstudyError(f"the tokstudy files are not table {prereg.table}'s")
    domains = sorted(set(ours) & set(theirs))
    if not domains:
        raise TokstudyError("the two sides share no domain")
    by_domain = {
        d: {"tokstudy": ours[d], "production": theirs[d], "rel": ours[d]["bpb"] / theirs[d]["bpb"] - 1} for d in domains
    }
    return {
        "format": TWIN_FORMAT,
        "table": prereg.table,
        "prereg": {"path": Path(rules_path).name, "sha256": prereg.sha256, "signed": prereg.signed},
        "twin_rel_max": float(limit),
        "by_domain": by_domain,
        "passed": all(abs(v["rel"]) <= limit for v in by_domain.values()),
        "files": {"tokstudy": [sha256_file(p) for p in tokstudy], "production": [sha256_file(p) for p in native]},
    }


# --- command line ----------------------------------------------------------------------------------------------------


def verdict_main(argv: Sequence[str]) -> int:
    argv = list(argv)
    if argv[:1] == ["twin"]:
        parser = argparse.ArgumentParser(prog="python -m surogate_tokstudy verdict twin", description="TW-2 / TW-3")
        parser.add_argument("--prereg", type=Path, required=True, help="the twin prereg (twin_rel_max)")
        parser.add_argument("--tokstudy", type=Path, nargs="+", required=True, help="the twin's NLL files")
        parser.add_argument("--production", type=Path, nargs="+", required=True, help="the production side's")
        parser.add_argument("--draft", action="store_true")
        parser.add_argument("--out", type=Path, required=True)
        args = parser.parse_args(argv[1:])
        try:
            doc = twin_gate(args.prereg, args.tokstudy, args.production, draft=args.draft)
        except (TokstudyError, ValueError) as err:
            print(f"verdict twin: {err}", file=sys.stderr)
            return 1
        atomic_write_json(args.out, doc)
        print(json.dumps({"passed": doc["passed"], "by_domain": {d: v["rel"] for d, v in doc["by_domain"].items()}}))
        return 0
    parser = argparse.ArgumentParser(prog="python -m surogate_tokstudy verdict", description="one table's verdict")
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--nll", type=Path, nargs="+", required=True, help="tokstudy-nll/1 files of the table")
    parser.add_argument("--stage", help="the prereg stage to decide (default: decision)")
    parser.add_argument("--runs", type=Path, help="the runs root: prereg lock, run records, throughput")
    parser.add_argument("--probe", type=Path, nargs="*", default=[], help="tokstudy-probe/1 files (Track S)")
    parser.add_argument("--evidence", type=Path, help="JSON or TOML: " + ", ".join(sorted(EVIDENCE_KEYS)))
    parser.add_argument("--twin", type=Path, help="the twin gate's tokstudy-twin/1 record")
    parser.add_argument("--draft", action="store_true", help="allow an unsigned prereg; the verdict is a draft")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        doc = verdict(
            load_rules(args.prereg),
            args.nll,
            stage=args.stage,
            runs_root=args.runs,
            probes=args.probe,
            evidence=load_evidence(args.evidence),
            twin=args.twin,
            draft=args.draft,
        )
    except (TokstudyError, ValueError) as err:
        print(f"verdict: {err}", file=sys.stderr)
        return 1
    atomic_write_json(args.out, doc)
    print(json.dumps({"label": doc["label"], "arms": {a: v["decision"] for a, v in doc["arms"].items()}}))
    return 0
