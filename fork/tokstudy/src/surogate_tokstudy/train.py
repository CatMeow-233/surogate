"""The pilot trainer: one run of one arm of a table at the table's F_target, on the table's stream, resumable; and
the decay branches at the budget points (spec §5.4; STUDY M4, M5, M8).

**Progress.** pi = cumulative realized training FLOPs / F_target (flops.py, the prereg's convention): a token row
costs its positions, an H-Net row its bytes through the outer stage plus its realized chunks through the main
network. The learning rate (schedule.WSD, read at the F before each step), the stable checkpoints and the stop are
functions of F, so every arm of a table ends at F_target within one step, whatever its bytes per token or chunk
ratio; a cheaper arm simply reads further into the same stream (refused at the start when the stream is too short
at the arm's nominal FLOPs per byte, and stopped with a message, its checkpoints kept, if it runs out).

**Data.** Rows come from stream.rows_for_arm in stream order, `batch_rows` per optimizer step, in micro batches of
`micro_rows` right-padded to their longest row; a row's targets are its ids after the first. The loss is the sum of
the step's target NLLs over its target count (so micro batching does not change it), plus, for an H-Net, alpha
times the ratio loss of each micro batch weighted by its share of the step's positions.

**Optimiser and precision.** AdamW, betas (0.9, 0.95), gradient clipping at 1.0; weight decay on every matrix but
the embeddings (the tied heads), none on norms, vectors and embeddings; an H-Net's outer stage runs at lambda_outer
times the learning rate. Precision "bf16" is autocast over fp32 weights (the router and the residual projection run
in fp32 inside the model), "fp32" and "fp64" run without autocast (CPU tests use fp64 or fp32).

**Checkpoints.** `ckpt/step-<n>.pt` every `checkpoint_every` steps (the latest `keep_last` kept), `stable/E<e>.pt`
after the step that reaches round(decay_start * round(e * F_target)) for every budget point e, `final.pt` at
F_target: each holds the trainable weights, the optimiser, the RNG states (torch CPU and CUDA, numpy, python), the
stream cursor, the step and F, written atomically. A run resumes from its latest checkpoint, bit-identically on one
CPU thread (tests/cpu/test_train.py); a run directory whose run.json names another run id is refused.

**Stages.** A stage's `stop_after = e` halts the main run at budget point e's stable checkpoint (a bilingual
screening run: F_target = F_dec, branches at 0.25 and 0.5, cost 0.525 F_dec); running the same arm and seed again
under a stage without a stop continues the same run (same run id) to F_target.

**Branches.** `branch(run_dir, endpoint=e)` starts from stable/E<e>.pt and decays over [D_e, E_e] (schedule.wsd),
writing branches/E<e>/ (its own metrics, checkpoints and final.pt). It equals a run whose F_target is E_e with the
same absolute warmup.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import devices, flops, registry
from .arms import ArmSpec, arm_sha256, codec_for, load_arms
from .canonical import TokstudyError, atomic_write_bytes, atomic_write_json, read_json, sha256_file, sha256_json
from .codecs import ByteCodec, Codec
from .models.build import build_model, flop_spec, positional_limit
from .models.hnet import HNetLM
from .prereg import Prereg, load_prereg
from .schedule import WSD
from .stream import Row, StreamManifest, bytes_per_token, open_stream, rows_for_arm, t_arm

BACKEND = "tokstudy-torch"
CKPT_FORMAT = "tokstudy-ckpt/1"
METRICS_FORMAT = "tokstudy-metrics/1"


@dataclass(frozen=True)
class TrainConfig:
    """What a run computes with; its sha256 enters the run id."""

    lr: float
    batch_rows: int = 32
    micro_rows: int = 8
    lambda_outer: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    precision: str = "bf16"
    backend: str = "torch"

    def __post_init__(self):
        object.__setattr__(self, "betas", tuple(float(b) for b in self.betas))
        if not self.lr > 0 or not self.lambda_outer > 0:
            raise TokstudyError(f"lr and lambda_outer must be positive: {self}")
        if self.batch_rows < 1 or not 1 <= self.micro_rows <= self.batch_rows or self.batch_rows % self.micro_rows:
            raise TokstudyError(f"micro_rows must divide batch_rows: {self.micro_rows} / {self.batch_rows}")
        if self.precision not in devices.PRECISIONS:
            raise TokstudyError(f"precision must be one of {sorted(devices.PRECISIONS)}, got {self.precision!r}")

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["betas"] = list(self.betas)
        return out

    def sha256(self) -> str:
        return sha256_json(self.to_json())


@dataclass
class Runtime:
    """What does not change the computation: checkpoint cadence and test hooks."""

    checkpoint_every: int = 200
    keep_last: int = 2
    on_step: Callable[[dict[str, Any]], None] | None = None  # called after each step's checkpoints (tests)
    device: str | None = None


@dataclass
class _Setup:
    arm: ArmSpec
    codec: Codec
    model: nn.Module
    spec: dict[str, Any]
    stream: StreamManifest
    prereg: Prereg
    config: TrainConfig
    seed: int
    device: torch.device
    record: dict[str, Any] = field(default_factory=dict)


def _codec(arm: ArmSpec) -> Codec:
    codec = codec_for(arm)
    return codec if codec is not None else ByteCodec(arm.specials, name=arm.id)


def setup(arm: ArmSpec, *, stream: StreamManifest, prereg: Prereg, seed: int, config: TrainConfig,
          device: str | None = None, registry_path: Path | None = None,
          code: dict[str, Any] | None = None) -> _Setup:  # fmt: skip
    """Everything a run needs, rebuilt the same way for a start, a resume and a branch; checks the inputs agree.
    `code` replaces the checkout's identity (evaluation of a run recorded under another commit)."""
    if arm.table != prereg.table:
        raise TokstudyError(f"arm {arm.id} is in table {arm.table}, the prereg is table {prereg.table}'s")
    if stream.k_train != prereg.k_train:
        raise TokstudyError(f"the stream has k_train {stream.k_train}, the prereg K_train {prereg.k_train}")
    project = arm.project
    if stream.project != project:
        raise TokstudyError(f"the stream holds {stream.project} documents, arm {arm.id} is a {project} arm")
    if seed not in arm.seeds:
        raise TokstudyError(f"seed {seed} is not one of arm {arm.id}'s registered seeds {list(arm.seeds)}")
    limit = positional_limit(arm)
    if limit is not None and limit < stream.k_train:
        raise TokstudyError(f"arm {arm.id}: its main network takes {limit} positions, a row may hold {stream.k_train}"
                            " (set model.overrides.max_seq)")  # fmt: skip
    codec = _codec(arm)
    dev = devices.device(device)
    model = build_model(arm, vocab_size=codec.vocab_size if arm.kind == "subword" else None, init_seed=seed,
                        backend=config.backend)  # fmt: skip
    model = model.to(dev, devices.PRECISIONS[config.precision])
    spec = flop_spec(model)
    out = _Setup(arm, codec, model, spec, stream, prereg, config, seed, dev)
    schedule = prereg.wsd(config.lr).to_json()
    code = dict(code) if code is not None else registry.code_identity()
    rid = registry.run_id(arm_sha256(arm), stream.sha256, config.sha256(), code["commit"], seed, prereg.f_target,
                          schedule)  # fmt: skip
    c_arm = bytes_per_token(stream, codec) if arm.kind == "subword" else 1.0
    out.record = {
        "format": "tokstudy-run/1",
        "run_id": rid,
        "id_inputs": {"arm_sha256": arm_sha256(arm), "stream_sha256": stream.sha256, "config_hash": config.sha256(),
                      "code_commit": code["commit"], "seed": seed, "f_target": prereg.f_target, "schedule": schedule},
        "backend": BACKEND,
        "table": arm.table,
        "arm": {"id": arm.id, "kind": arm.kind, "main": arm.main, "model": dict(arm.model),
                "tokenizer": None if arm.tokenizer is None else {"path": str(arm.tokenizer.path),
                                                                  "sha256": arm.tokenizer.sha256},
                "registry": None if registry_path is None else {"path": str(Path(registry_path).resolve()),
                                                                "sha256": sha256_file(registry_path)}},
        "codec": codec.identity(),
        "seed": seed,
        "config": config.to_json(),
        "prereg": {**prereg.record(), "path": str(prereg.path.resolve())},
        "stream": {"path": str(stream.root.resolve()), "sha256": stream.sha256, "k_train": stream.k_train},
        "f_target": prereg.f_target,
        "convention": prereg.convention,
        "budget_points": list(prereg.points),
        "stable_flops": {_label(e): _stable_flops(prereg, e) for e in prereg.points},
        "schedule": schedule,
        "c_arm": c_arm,
        "t_arm": t_arm(stream.k_train, c_arm),
        "flop_spec": spec,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "code": code,
    }  # fmt: skip
    _check_stream_length(out, c_arm)
    return out


def _label(e: float) -> str:
    return f"E{e:g}"


def _stable_flops(prereg: Prereg, e: float) -> int:
    end = prereg.f_target if e == 1 else round(e * prereg.f_target)
    return round(prereg.schedule["decay_start"] * end)


def _check_stream_length(s: _Setup, c_arm: float) -> None:
    ratio = s.model.cfg.ratio if s.arm.kind == "hnet" else None
    per_byte = flops.flops_per_byte(s.spec, row_bytes=s.stream.k_train, bytes_per_position=c_arm, ratio=ratio,
                                    convention=s.prereg.convention)  # fmt: skip
    available = per_byte * s.stream.doc["bytes"]
    if available < s.prereg.f_target:
        raise TokstudyError(f"arm {s.arm.id}: the stream's {s.stream.doc['bytes']} bytes give about {available:.4g} "
                            f"training FLOPs at {per_byte:.4g} FLOPs/byte, below F_target {s.prereg.f_target:.4g}")  # fmt: skip


# --- optimiser -------------------------------------------------------------------------------------------------------


def param_groups(model: nn.Module, config: TrainConfig) -> list[dict[str, Any]]:
    """AdamW groups by stage (H-Net: outer at lambda_outer, main at 1) and decay (matrices but embeddings)."""
    embeddings = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    stages = model.parameter_stages() if isinstance(model, HNetLM) else {"main": list(model.parameters())}
    groups = []
    for stage, params in stages.items():
        mult = config.lambda_outer if stage == "outer" else 1.0
        decay = [p for p in params if p.requires_grad and p.ndim >= 2 and id(p) not in embeddings]
        keep = [p for p in params if p.requires_grad and not (p.ndim >= 2 and id(p) not in embeddings)]
        for label, ps, wd in ((f"{stage}/decay", decay, config.weight_decay), (f"{stage}/no_decay", keep, 0.0)):
            if ps:
                groups.append({"params": ps, "group": label, "weight_decay": wd, "lr_mult": mult})
    return groups


def optimizer(s: _Setup) -> torch.optim.AdamW:
    c = s.config
    return torch.optim.AdamW(param_groups(s.model, c), lr=c.lr, betas=c.betas, eps=c.eps, foreach=False)


# --- checkpoints -----------------------------------------------------------------------------------------------------


def _rng_state() -> dict[str, Any]:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    version, internal, gauss = random.getstate()
    state = {"torch": torch.get_rng_state(), "numpy": [name, torch.from_numpy(keys.astype(np.int64)), pos, has_gauss,
             cached], "python": [version, torch.tensor(internal, dtype=torch.int64), gauss]}  # fmt: skip
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, keys.numpy().astype(np.uint32), pos, has_gauss, cached))
    version, internal, gauss = state["python"]
    random.setstate((version, tuple(int(x) for x in internal.tolist()), gauss))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, s: _Setup, opt: torch.optim.Optimizer, state: dict[str, Any], kind: str) -> Path:
    payload = {"format": CKPT_FORMAT, "run_id": s.record["run_id"], "kind": kind, "step": state["step"],
               "flops": state["flops"], "cursor": list(state["cursor"]), "model": s.model.state_dict(),
               "optimizer": opt.state_dict(), "rng": _rng_state()}  # fmt: skip
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return atomic_write_bytes(path, buffer.getvalue())


def load_checkpoint(path: Path, run_id: str, dev: torch.device) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location=dev, weights_only=True)
    except Exception as err:  # torch raises several types for a damaged file
        raise TokstudyError(f"{path}: cannot read a checkpoint: {err}") from None
    if payload.get("format") != CKPT_FORMAT:
        raise TokstudyError(f"{path}: not a {CKPT_FORMAT} file")
    if payload["run_id"] != run_id:
        raise TokstudyError(f"{path}: a checkpoint of run {payload['run_id']}, not of {run_id}")
    return payload


def _restore(s: _Setup, opt: torch.optim.Optimizer, payload: dict[str, Any]) -> dict[str, Any]:
    s.model.load_state_dict(payload["model"])
    opt.load_state_dict(payload["optimizer"])
    _set_rng_state(payload["rng"])
    return {"step": int(payload["step"]), "flops": int(payload["flops"]), "cursor": tuple(payload["cursor"])}


def latest_checkpoint(out: Path) -> Path | None:
    found = sorted((out / "ckpt").glob("step-*.pt"))
    return found[-1] if found else None


def _prune(out: Path, keep: int) -> None:
    for path in sorted((out / "ckpt").glob("step-*.pt"))[:-keep]:
        path.unlink()


# --- the loop --------------------------------------------------------------------------------------------------------


def _batches(rows: Iterator[Row], n: int) -> Iterator[list[Row]]:
    while True:
        batch = []
        for row in rows:
            batch.append(row)
            if len(batch) == n:
                break
        if len(batch) < n:
            return
        yield batch


def _tensors(rows: Sequence[Row], dev: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    width = max(len(r.ids) for r in rows)
    ids = np.zeros((len(rows), width), dtype=np.int64)
    forced = np.zeros((len(rows), width), dtype=bool)
    valid = np.zeros((len(rows), width), dtype=bool)
    for i, r in enumerate(rows):
        n = len(r.ids)
        ids[i, :n], forced[i, :n], valid[i, :n] = r.ids, r.forced, True
    return tuple(torch.from_numpy(a).to(dev) for a in (ids, forced, valid))


def step_losses(s: _Setup, rows: Sequence[Row], n_targets: int, n_positions: int) -> dict[str, Any]:
    """Forward and backward of one micro batch; returns its loss parts (floats) and chunk counts."""
    ids, forced, valid = _tensors(rows, s.device)
    with devices.autocast(s.device, s.config.precision):
        if s.arm.kind == "hnet":
            out = s.model(ids, forced, valid)
            logits = out.logits
        else:
            out, logits = None, s.model(ids)
    head = logits[:, :-1]
    head = head if head.dtype == torch.float64 else head.float()  # the loss in fp32 (fp64 for fp64 tests)
    nll = F.cross_entropy(head.flatten(0, 1), ids[:, 1:].flatten(), reduction="none")
    ce_sum = (nll * valid[:, 1:].flatten().to(nll.dtype)).sum()
    loss = ce_sum / n_targets
    parts: dict[str, Any] = {"ce_sum": float(ce_sum.detach())}
    if out is not None:
        weight = int(valid.sum()) / n_positions
        loss = loss + s.model.cfg.alpha * out.ratio_loss * weight
        parts |= {"ratio": float(out.ratio_loss.detach()) * weight, "F": float(out.F) * weight,
                  "G": float(out.G.detach()) * weight, "chunks": [int(c) for c in out.lengths]}  # fmt: skip
    loss.backward()
    return parts


def _loop(s: _Setup, opt: torch.optim.Optimizer, state: dict[str, Any], *, out: Path, schedule: WSD, end: int,
          stable: dict[str, int], stop_at: int | None, runtime: Runtime, kind: str) -> dict[str, Any]:  # fmt: skip
    c = s.config
    trainable = [p for g in opt.param_groups for p in g["params"]]
    rows = rows_for_arm(s.stream, s.codec, start=state["cursor"])
    batches = _batches(rows, c.batch_rows)
    metrics = (out / "metrics.jsonl").open("a", encoding="utf-8")
    try:
        while state["flops"] < end and (stop_at is None or state["flops"] < stop_at):
            started = time.perf_counter()
            batch = next(batches, None)
            if batch is None:
                raise TokstudyError(f"the stream is exhausted at step {state['step']}, F {state['flops']:.6g} "
                                    f"(progress {state['flops'] / s.prereg.f_target:.4f}); build a longer stream")  # fmt: skip
            lr = schedule.lr(state["flops"])
            for group in opt.param_groups:
                group["lr"] = lr * group["lr_mult"]
            opt.zero_grad(set_to_none=True)
            n_targets = sum(len(r.ids) - 1 for r in batch)
            n_positions = sum(len(r.ids) for r in batch)
            if n_targets <= 0:
                raise TokstudyError(f"step {state['step']}: a batch without targets")
            parts: Counter = Counter()
            chunks: list[int] = []
            for k in range(0, len(batch), c.micro_rows):
                got = step_losses(s, batch[k : k + c.micro_rows], n_targets, n_positions)
                chunks += got.pop("chunks", [])
                parts.update(got)
            norm = float(torch.nn.utils.clip_grad_norm_(trainable, c.grad_clip if c.grad_clip > 0 else math.inf))
            ce = parts["ce_sum"] / n_targets
            if not (math.isfinite(ce) and math.isfinite(norm)):
                raise TokstudyError(f"step {state['step']}: non-finite loss {ce} or gradient norm {norm}")
            opt.step()
            if s.arm.kind == "hnet":
                step_flops = sum(flops.row_train_flops(s.spec, len(r.ids), m, convention=s.prereg.convention)
                                 for r, m in zip(batch, chunks, strict=True))  # fmt: skip
            else:
                step_flops = sum(flops.row_train_flops(s.spec, len(r.ids), convention=s.prereg.convention)
                                 for r in batch)  # fmt: skip
            before = state["flops"]
            state = {"step": state["step"] + 1, "flops": before + step_flops, "cursor": batch[-1].next_cursor}
            seconds = time.perf_counter() - started
            text_bytes = sum(r.text_bytes for r in batch)
            line = {"step": state["step"], "flops": state["flops"], "progress": state["flops"] / s.prereg.f_target,
                    "lr": lr, "rows": len(batch), "positions": n_positions, "targets": n_targets,
                    "bytes": text_bytes, "ce": ce, "loss": ce + (s.model.cfg.alpha * parts["ratio"]
                                                                 if s.arm.kind == "hnet" else 0.0),
                    "grad_norm": norm, "seconds": seconds, "bytes_per_s": text_bytes / seconds,
                    "flops_per_s": step_flops / seconds, "peak_memory": devices.peak_memory(s.device),
                    "cursor": list(state["cursor"])}  # fmt: skip
            if s.arm.kind == "hnet":
                line |= {"ratio_loss": parts["ratio"], "F": parts["F"], "G": parts["G"], "chunks": sum(chunks),
                         "positions_per_chunk": _per_domain(batch, chunks)}  # fmt: skip
            metrics.write(json.dumps(line, sort_keys=True) + "\n")
            metrics.flush()
            for label, threshold in stable.items():
                if before < threshold <= state["flops"]:
                    save_checkpoint(out / "stable" / f"{label}.pt", s, opt, state, f"{kind}/stable/{label}")
            if runtime.checkpoint_every and state["step"] % runtime.checkpoint_every == 0:
                save_checkpoint(out / "ckpt" / f"step-{state['step']:09d}.pt", s, opt, state, kind)
                _prune(out, runtime.keep_last)
            if runtime.on_step is not None:
                runtime.on_step(dict(state))
    finally:
        metrics.close()
    save_checkpoint(out / "ckpt" / f"step-{state['step']:09d}.pt", s, opt, state, kind)
    _prune(out, runtime.keep_last)
    if state["flops"] >= end:
        save_checkpoint(out / "final.pt", s, opt, state, f"{kind}/final")
    return state


def _per_domain(batch: Sequence[Row], chunks: Sequence[int]) -> dict[str, float]:
    positions: Counter = Counter()
    count: Counter = Counter()
    for row, m in zip(batch, chunks, strict=True):
        positions[row.domain] += len(row.ids)
        count[row.domain] += m
    return {d: positions[d] / count[d] for d in sorted(count) if count[d]}


def _event(out: Path, event: str, s: _Setup, state: dict[str, Any], **extra) -> None:
    line = {"event": event, "at": registry.now(), "step": state["step"], "flops": state["flops"],
            "environment": devices.environment(s.device), **extra}  # fmt: skip
    with (out / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")


def _truncate_metrics(out: Path, step: int) -> None:
    """Drop the metric lines of steps after `step` (they are recomputed on resume)."""
    path = out / "metrics.jsonl"
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept = [line for line in lines if "format" in json.loads(line) or json.loads(line)["step"] <= step]
    atomic_write_bytes(path, "".join(kept).encode("utf-8"))


def _start_or_resume(s: _Setup, out: Path, opt: torch.optim.Optimizer, header: dict[str, Any],
                     initial: dict[str, Any], runtime: Runtime, kind: str) -> tuple[dict[str, Any], bool]:  # fmt: skip
    """The state to continue from: the latest checkpoint of `out`, else `initial`; writes the metrics header."""
    latest = latest_checkpoint(out)
    if latest is not None:
        state = _restore(s, opt, load_checkpoint(latest, s.record["run_id"], s.device))
        _truncate_metrics(out, state["step"])
        _event(out, f"{kind}-resume", s, state, checkpoint=latest.name)
        return state, True
    (out / "metrics.jsonl").write_text(json.dumps(header, sort_keys=True) + "\n", encoding="utf-8")
    _event(out, f"{kind}-start", s, initial)
    return initial, False


def train(arm: ArmSpec, runs: Path, *, stream: StreamManifest, prereg: Prereg, seed: int, config: TrainConfig,
          stage: str | None = None, stop_after: float | None = None, amend: str | None = None,
          run_dir: Path | None = None, runtime: Runtime | None = None,
          registry_path: Path | None = None) -> dict[str, Any]:  # fmt: skip
    """Start or resume the run of (arm, seed) under `runs`; returns its record with the state reached."""
    runtime = runtime or Runtime()
    if stage is not None:
        if stage not in arm.stages:
            raise TokstudyError(f"arm {arm.id} takes part in stages {list(arm.stages)}, not {stage!r}")
        stop_after = prereg.stop_after(stage) if stop_after is None else stop_after
    if stop_after is not None and stop_after not in prereg.points[:-1]:
        raise TokstudyError(f"stop_after must be a budget point below 1 ({list(prereg.points[:-1])}), got {stop_after}")
    s = setup(arm, stream=stream, prereg=prereg, seed=seed, config=config, device=runtime.device,
              registry_path=registry_path)  # fmt: skip
    rid = s.record["run_id"]
    expected = registry.run_dir_for(runs, arm.table, arm.id, seed, rid)
    out = Path(run_dir) if run_dir is not None else expected
    if (out / registry.RUN_FILE).exists():
        found = registry.read_run(out)
        if found["run_id"] != rid:
            raise TokstudyError(f"{out} holds run {found['run_id']}; these inputs make run {rid} (refused)")
        registry.lock_prereg(runs, prereg, amend=amend)
    else:
        if out.exists() and any(out.iterdir()):
            raise TokstudyError(f"{out} exists without a run record")
        registry.lock_prereg(runs, prereg, amend=amend)
        registry.write_run(out, s.record)
    opt = optimizer(s)
    header = {"format": METRICS_FORMAT, "run_id": rid, "arm": arm.id, "kind": "main"}
    state, _ = _start_or_resume(s, out, opt, header, {"step": 0, "flops": 0, "cursor": (0, 0)}, runtime, "main")
    stop_at = None if stop_after is None else _stable_flops(prereg, stop_after)
    stable = {_label(e): _stable_flops(prereg, e) for e in prereg.points}
    unfinished = state["flops"] < prereg.f_target and (stop_at is None or state["flops"] < stop_at)
    if unfinished or (state["flops"] >= prereg.f_target and not (out / "final.pt").is_file()):
        state = _loop(s, opt, state, out=out, schedule=prereg.wsd(config.lr), end=prereg.f_target, stable=stable,
                      stop_at=stop_at, runtime=runtime, kind="main")  # fmt: skip
    done = state["flops"] >= prereg.f_target
    _event(out, "main-done" if done else "main-stop", s, state)
    return {**s.record, "dir": str(out), "state": _state_json(state, prereg), "done": done}


def _state_json(state: dict[str, Any], prereg: Prereg) -> dict[str, Any]:
    return {"step": state["step"], "flops": state["flops"], "progress": state["flops"] / prereg.f_target,
            "cursor": list(state["cursor"])}  # fmt: skip


def reopen(run_dir: Path, *, arms: Path | None = None, stream: Path | None = None, prereg: Path | None = None,
           device: str | None = None, verify_stream: bool = True, same_code: bool = True) -> _Setup:  # fmt: skip
    """The setup of an existing run from its record (inputs at their recorded paths unless given), the run id
    recomputed from them (so every input is checked: tokenizer sha256s, the stream manifest, the prereg's budget and
    schedule, the config, the code); `verify_stream` also re-hashes the stream's parts (evaluation reads none).
    Continuing a run (resume, branch) needs the same code commit; evaluation passes same_code=False, which takes the
    recorded commit for the id and leaves the evaluating commit to the NLL header."""
    record = registry.read_run(run_dir)
    registry_path = Path(arms or (record["arm"]["registry"] or {}).get("path", ""))
    if not registry_path.is_file():
        raise TokstudyError(f"{run_dir}: the run's arm registry is not at {registry_path}; pass --arms")
    found = {a.id: a for a in load_arms(registry_path)}
    if record["arm"]["id"] not in found:
        raise TokstudyError(f"{registry_path} has no arm {record['arm']['id']!r}")
    p = load_prereg(Path(prereg or record["prereg"]["path"]))
    st = open_stream(Path(stream or record["stream"]["path"]), verify=verify_stream)
    config = TrainConfig(**record["config"])
    s = setup(found[record["arm"]["id"]], stream=st, prereg=p, seed=record["seed"], config=config, device=device,
              registry_path=registry_path, code=None if same_code else record["code"])  # fmt: skip
    if s.record["run_id"] != record["run_id"]:
        raise TokstudyError(f"{run_dir}: run {record['run_id']}, but its inputs now make run {s.record['run_id']} "
                            "(arm, stream, prereg schedule, config or code changed)")  # fmt: skip
    return s


def branch(run_dir: Path, *, endpoint: float, runtime: Runtime | None = None, arms: Path | None = None,
           stream: Path | None = None, prereg: Path | None = None) -> dict[str, Any]:  # fmt: skip
    """The decay branch of a run at budget point `endpoint` (< 1), from its stable checkpoint (module docstring)."""
    runtime = runtime or Runtime()
    run_dir = Path(run_dir)
    s = reopen(run_dir, arms=arms, stream=stream, prereg=prereg, device=runtime.device)
    p = s.prereg
    if endpoint not in p.points[:-1]:
        raise TokstudyError(f"a branch ends at a budget point below 1 ({list(p.points[:-1])}), got {endpoint}")
    label = _label(endpoint)
    source = run_dir / "stable" / f"{label}.pt"
    if not source.is_file():
        raise TokstudyError(f"{run_dir}: no stable checkpoint {source.name} yet (the run has not reached it)")
    out = run_dir / "branches" / label
    schedule = p.wsd(s.config.lr, endpoint=endpoint)
    end = schedule.end_flops
    opt = optimizer(s)
    info = {"format": "tokstudy-branch/1", "run_id": s.record["run_id"], "endpoint": endpoint,
            "schedule": schedule.to_json(), "from": source.name, "from_sha256": sha256_file(source)}  # fmt: skip
    if (out / "branch.json").is_file():
        if read_json(out / "branch.json", "branch") != info:
            raise TokstudyError(f"{out}: branch.json differs from this branch (other schedule or stable checkpoint)")
    else:
        atomic_write_json(out / "branch.json", info)
    initial = _restore(s, opt, load_checkpoint(source, s.record["run_id"], s.device))
    header = {"format": METRICS_FORMAT, "run_id": s.record["run_id"], "arm": s.arm.id, "kind": f"branch/{label}"}
    state, _ = _start_or_resume(s, out, opt, header, initial, runtime, "branch")
    if state["flops"] < end or not (out / "final.pt").is_file():  # a stable step may already reach the endpoint
        state = _loop(s, opt, state, out=out, schedule=schedule, end=end, stable={}, stop_at=None, runtime=runtime,
                      kind=f"branch/{label}")  # fmt: skip
    _event(out, "branch-done", s, state)
    return {**info, "dir": str(out), "state": _state_json(state, p), "done": state["flops"] >= end}


def point_checkpoint(run_dir: Path, endpoint: float) -> Path:
    """The final checkpoint of a run's budget point: final.pt at 1, branches/E<e>/final.pt below."""
    run_dir = Path(run_dir)
    path = run_dir / "final.pt" if endpoint == 1 else run_dir / "branches" / _label(endpoint) / "final.pt"
    if not path.is_file():
        raise TokstudyError(f"{run_dir}: budget point {endpoint:g} has no final checkpoint ({path} missing)")
    return path


# --- command line ----------------------------------------------------------------------------------------------------


def _config_args(parser: argparse.ArgumentParser) -> None:
    d = TrainConfig(lr=1.0)
    parser.add_argument(
        "--lr", type=float, required=True, help="peak learning rate (M10: fixed per family after the sweep)"
    )
    parser.add_argument("--batch-rows", type=int, default=d.batch_rows)
    parser.add_argument("--micro-rows", type=int, default=d.micro_rows)
    parser.add_argument("--lambda-outer", type=float, default=d.lambda_outer, help="H-Net outer-stage LR multiplier")
    parser.add_argument("--weight-decay", type=float, default=d.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=d.grad_clip)
    parser.add_argument("--precision", choices=sorted(devices.PRECISIONS), default=d.precision)
    parser.add_argument("--backend", choices=["torch", "triton"], default=d.backend)


def _runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint-every", type=int, default=Runtime.checkpoint_every)
    parser.add_argument("--device", default="auto")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tokstudy train",
                                     description="one run of one arm at the table's F_target; resumes")  # fmt: skip
    parser.add_argument("--arms", type=Path, required=True, help="arm registry (tokstudy-arms/1)")
    parser.add_argument("--arm", required=True, help="arm id")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stream", type=Path, required=True, help="stream directory (python -m ... stream)")
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True, help="runs root (registry.py)")
    parser.add_argument("--stage", help="a prereg stage (its stop_after halts the main run)")
    parser.add_argument("--amend", help="accept a changed prereg for this table, with this reason")
    _config_args(parser)
    _runtime_args(parser)
    args = parser.parse_args(list(argv))
    try:
        arms = {a.id: a for a in load_arms(args.arms)}
        if args.arm not in arms:
            raise TokstudyError(f"{args.arms} has no arm {args.arm!r} (arms: {sorted(arms)})")
        config = TrainConfig(lr=args.lr, batch_rows=args.batch_rows, micro_rows=args.micro_rows,
                             lambda_outer=args.lambda_outer, weight_decay=args.weight_decay,
                             grad_clip=args.grad_clip, precision=args.precision, backend=args.backend)  # fmt: skip
        result = train(arms[args.arm], args.runs, stream=open_stream(args.stream), prereg=load_prereg(args.prereg),
                       seed=args.seed, config=config, stage=args.stage, amend=args.amend,
                       runtime=Runtime(checkpoint_every=args.checkpoint_every, device=args.device),
                       registry_path=args.arms)  # fmt: skip
    except (TokstudyError, ValueError) as err:
        print(f"train: {err}", file=sys.stderr)
        return 1
    print(json.dumps({"run_id": result["run_id"], "dir": result["dir"], "done": result["done"], **result["state"]}))
    return 0


def branch_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tokstudy branch",
                                     description="decay branch of a run from its stable checkpoint")  # fmt: skip
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--at", type=float, required=True, help="the budget point (below 1)")
    parser.add_argument("--arms", type=Path, help="the arm registry, when it moved")
    parser.add_argument("--stream", type=Path, help="the stream directory, when it moved")
    parser.add_argument("--prereg", type=Path, help="the prereg, when it moved")
    _runtime_args(parser)
    args = parser.parse_args(list(argv))
    try:
        result = branch(args.run_dir, endpoint=args.at, arms=args.arms, stream=args.stream, prereg=args.prereg,
                        runtime=Runtime(checkpoint_every=args.checkpoint_every, device=args.device))  # fmt: skip
    except (TokstudyError, ValueError) as err:
        print(f"branch: {err}", file=sys.stderr)
        return 1
    print(json.dumps({"run_id": result["run_id"], "dir": result["dir"], "done": result["done"], **result["state"]}))
    return 0
