"""Preregistration files (`tokstudy-prereg/1`, `prereg/<project>.toml`): the budget, contexts, schedule, stages and
mixture of one table, and its decision thresholds. The sha256 of the file's bytes identifies it: it enters every
run record and every verdict, and the first run of a table locks it (registry.lock_prereg; a change afterwards
needs an amendment with a reason).

    format = "tokstudy-prereg/1"
    table = "tlc-N"
    [budget]       convention = "matmul", F_target = 1.642e17, points = [0.25, 0.5, 1.0]
    [context]      K_train = 2048, K_eval = 2048           # byte positions per training row; text bytes per eval doc
    [schedule]     warmup = 0.02, decay_start = 0.9, floor = 0.1, shape = "linear"   # WSD (default kind)
                   kind = "cosine", warmup = 0.02, floor = 0.1     # the twin gates' production schedule; then
                                                                   # points = [1.0] (no stable phase, no branches)
    [stages.<name>] stop_after = 0.5                        # optional: the main run halts at that point's stable
                                                            # checkpoint (screening); default stage "decision"
    [mixture]      py = 0.7, en = 0.3                       # optional: the stream's shares by domain
    [stream]       measure = "bytes"                        # the mixture's unit: bytes, or a subword arm's tokens
    [seeds], [thresholds], [signature], notes               # read by the verdict (S16); kept as data here

This module checks the parts training and evaluation read; the thresholds are the verdict's. A prereg is signed when
`signature.signed_by` and `signature.date` are both set; run records carry the flag, so the verdict can refuse
unsigned tables.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .arms import TABLES
from .canonical import TokstudyError, check_format, sha256_bytes
from .flops import CONVENTIONS
from .schedule import KINDS, SHAPES, WSD, Cosine, cosine, wsd
from .stream import MIN_K, check_mixture

TOP_KEYS = {
    "format",
    "table",
    "budget",
    "context",
    "schedule",
    "stages",
    "mixture",
    "stream",
    "seeds",
    "thresholds",
    "signature",
    "notes",
}
DEFAULT_STAGES = {"decision": {"stop_after": None}}


@dataclass(frozen=True)
class Prereg:
    path: Path
    sha256: str
    table: str
    convention: str
    f_target: int
    points: tuple[float, ...]
    k_train: int
    k_eval: int
    schedule: Mapping[str, Any]
    stages: Mapping[str, Mapping[str, Any]]
    mixture: Mapping[str, float] | None
    measure: str  # "bytes" or the id of the subword arm whose tokens measure the mixture (STUDY M9)
    signed: bool
    doc: Mapping[str, Any] = field(repr=False)

    @property
    def kind(self) -> str:
        return self.schedule.get("kind", "wsd")

    def lr_schedule(self, peak: float, endpoint: float = 1.0) -> WSD | Cosine:
        """The run's schedule (endpoint 1) or, for WSD, the decay branch's at budget point `endpoint`."""
        if self.kind == "cosine":
            if endpoint != 1:
                raise TokstudyError(f"{self.path}: a cosine schedule has no decay branches")
            return cosine(self.f_target, peak=peak, warmup=self.schedule["warmup"], floor=self.schedule["floor"])
        return wsd(self.f_target, peak=peak, endpoint=endpoint, **self.schedule)

    def stable_flops(self) -> dict[str, int]:
        """{"E<e>": F} of the stable checkpoint of each budget point: the step that reaches decay_start * E_e
        (none under a cosine schedule)."""
        if self.kind == "cosine":
            return {}
        out = {}
        for e in self.points:
            end = self.f_target if e == 1 else round(e * self.f_target)
            out[f"E{e:g}"] = round(self.schedule["decay_start"] * end)
        return out

    def budget(self) -> dict[str, Any]:
        """The prereg's fields besides F_target and the schedule that change a run (registry: run id)."""
        return {"convention": self.convention, "points": list(self.points)}

    def stop_after(self, stage: str) -> float | None:
        if stage not in self.stages:
            raise TokstudyError(f"{self.path}: no stage {stage!r} (stages: {sorted(self.stages)})")
        return self.stages[stage].get("stop_after")

    def record(self) -> dict[str, Any]:
        return {"path": self.path.name, "sha256": self.sha256, "table": self.table, "signed": self.signed}


def _number(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TokstudyError(f"{what} must be a number, got {value!r}")
    return float(value)


def _table(data: Mapping[str, Any], name: str, keys: set[str], path: Path) -> dict[str, Any]:
    table = data.get(name, {})
    if not isinstance(table, Mapping):
        raise TokstudyError(f"{path}: [{name}] must be a table")
    unknown = sorted(set(table) - keys)
    if unknown:
        raise TokstudyError(f"{path}: [{name}] has unknown keys {unknown}; known: {sorted(keys)}")
    return dict(table)


def load_prereg(path: Path) -> Prereg:
    path = Path(path)
    try:
        raw = path.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as err:
        raise TokstudyError(f"{path}: cannot read a prereg: {err}") from None
    check_format(data, "prereg")
    unknown = sorted(set(data) - TOP_KEYS)
    if unknown:
        raise TokstudyError(f"{path}: unknown top-level keys {unknown}")
    table = data.get("table")
    if table not in TABLES:
        raise TokstudyError(f"{path}: table must be one of {sorted(TABLES)}, got {table!r}")
    budget = _table(data, "budget", {"convention", "F_target", "points"}, path)
    convention = budget.get("convention", "matmul")
    if convention not in CONVENTIONS:
        raise TokstudyError(f"{path}: budget.convention must be one of {CONVENTIONS}, got {convention!r}")
    f_target = _number(budget.get("F_target"), f"{path}: budget.F_target")
    if not f_target >= 1:
        raise TokstudyError(f"{path}: budget.F_target must be positive")
    points = tuple(_number(p, f"{path}: budget.points") for p in budget.get("points", [1.0]))
    if not points or list(points) != sorted(set(points)) or points[-1] != 1.0 or points[0] <= 0:
        raise TokstudyError(f"{path}: budget.points must rise strictly in (0, 1] and end at 1.0, got {list(points)}")
    context = _table(data, "context", {"K_train", "K_eval"}, path)
    k_train, k_eval = context.get("K_train"), context.get("K_eval")
    for name, value, low in (("K_train", k_train, MIN_K), ("K_eval", k_eval, 1)):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise TokstudyError(f"{path}: context.{name} must be an integer >= {low}, got {value!r}")
    kind = data.get("schedule", {}).get("kind", "wsd") if isinstance(data.get("schedule"), Mapping) else "wsd"
    if kind not in KINDS:
        raise TokstudyError(f"{path}: schedule.kind must be one of {KINDS}, got {kind!r}")
    if kind == "cosine":
        schedule = {"warmup": 0.02, "floor": 0.1}
        schedule |= {k: v for k, v in _table(data, "schedule", {*schedule, "kind"}, path).items() if k != "kind"}
        for name in ("warmup", "floor"):
            schedule[name] = _number(schedule[name], f"{path}: schedule.{name}")
        if not 0 <= schedule["warmup"] < 1 or not 0 <= schedule["floor"] <= 1:
            raise TokstudyError(f"{path}: a cosine schedule needs 0 <= warmup < 1 and floor in [0, 1]: {schedule}")
        if points != (1.0,):
            raise TokstudyError(f"{path}: a cosine schedule has no stable phase: budget.points must be [1.0]")
        schedule = {"kind": "cosine", **schedule}
    else:
        schedule = {"warmup": 0.02, "decay_start": 0.9, "floor": 0.1, "shape": "linear"}
        schedule |= {k: v for k, v in _table(data, "schedule", {*schedule, "kind"}, path).items() if k != "kind"}
        for name in ("warmup", "decay_start", "floor"):
            schedule[name] = _number(schedule[name], f"{path}: schedule.{name}")
        if not 0 <= schedule["warmup"] < schedule["decay_start"] < 1 or not 0 <= schedule["floor"] <= 1:
            raise TokstudyError(f"{path}: schedule needs 0 <= warmup < decay_start < 1 and floor in [0, 1]: {schedule}")
        if schedule["shape"] not in SHAPES:
            raise TokstudyError(f"{path}: schedule.shape must be one of {SHAPES}")
        if round(schedule["decay_start"] * points[0] * f_target) < round(schedule["warmup"] * f_target):
            raise TokstudyError(f"{path}: the first budget point's decay would start inside the warmup")
    stages = dict(DEFAULT_STAGES) if "stages" not in data else {}
    for name, stage in data.get("stages", {}).items():
        stage = _table(data["stages"], name, {"stop_after"}, path)
        stop = stage.get("stop_after")
        if stop is not None and (_number(stop, f"{path}: stages.{name}.stop_after") not in points[:-1]):
            raise TokstudyError(f"{path}: stages.{name}.stop_after must be a budget point below 1, got {stop}")
        stages[name] = {"stop_after": stop}
    if not stages:
        raise TokstudyError(f"{path}: [stages] names no stage")
    try:
        mixture = check_mixture(data.get("mixture"))
    except TokstudyError as err:
        raise TokstudyError(f"{path}: [mixture]: {err}") from None
    measure = _table(data, "stream", {"measure"}, path).get("measure", "bytes")
    if not isinstance(measure, str) or not measure:
        raise TokstudyError(f'{path}: stream.measure is "bytes" or an arm id')
    signature = _table(data, "signature", {"signed_by", "date"}, path)
    signed = bool(signature.get("signed_by")) and bool(signature.get("date"))
    return Prereg(
        path,
        sha256_bytes(raw),
        table,
        convention,
        round(f_target),
        points,
        k_train,
        k_eval,
        schedule,
        stages,
        mixture,
        measure,
        signed,
        data,
    )
