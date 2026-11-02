"""The single training loop (docs/spec.md, I5; MS §8): one run of a row config, resumable bit for bit on the CPU.

`train(run, recipe)`:

1. **Inputs.** The shard (opened with the run's tokenizer hash and vocabulary, every file hash verified), the seed
   pack and the expert packs the recipe reads (every hash verified), the tokenizer. A shard of the dev or test split
   is refused: training never reads evaluation data.
2. **Model.** The recipe builds it: from `init_seed` through one torch.Generator (IMPL-01) for the seed, from the
   seed pack and expert packs otherwise. The recipe's FreezePlan fixes what may change; the optimizer is built on
   exactly that set (audited by storage) and every frozen tensor is hashed.
3. **Identity.** The run config (model, train, gqa, data window, recipe and its arguments, steps, precision, the
   shard header hash, the hashes of the packs read) is hashed; with the data manifest, tokenizer,
   seed fingerprint, code commit, recipe and both seeds it gives the run_id. `registry.run.open_run` creates the run
   or reopens the same one; another id under the same name is refused.
4. **Resume.** Stale temporary state directories are removed. The latest state is loaded (its run_id, file hashes
   and frozen hashes checked); the trainable tensors, optimizer, schedule position, sampler cursor and RNG states
   are restored before the first batch is drawn (IMPL-02), and metrics.jsonl is cut back to that step.
5. **Steps.** Per optimizer step: the scheduled lr; `accum` micro batches of `microbatch` windows from the
   EpochChunkSampler; plans from the windows' shard routes through the recipe's route map (`routing.policies`);
   forward under bf16 autocast when the precision is bf16 (parameters stay fp32); the loss summed over the loss-mask
   targets of the whole step and divided by their count, so every target weighs the same across micro batches; KL
   to the seed when train.anchor_kl > 0 (no reference forward at all when it is 0); gradient clipping; the
   decoupled decay to the seed; AdamW. A non-finite loss or gradient norm fails the run.
6. **State and packs.** A state every `save_every` steps and at the last step, with retention; at the end the frozen
   tensors are re-hashed (nothing outside the trainable set moved) and the recipe writes its packs.

Only `run.json`, `environment*.json`, `metrics.jsonl`, `state/` and `packs/` under data/runs/<name>/ (and the
registry's index.json) are written.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..assembly.freeze import FreezeError, FreezePlan
from ..assembly.hashing import content_hash
from ..assembly.packs import TOKENIZER, AssemblyError, ExpertPack, SeedPack, read_pack
from ..audit.params import p_update
from ..canonical import sha256_file, sha256_json
from ..config import RunConfig, config_hash, load_toml, stage_names
from ..data.sampler import EpochChunkSampler
from ..data.shards import ShardReader
from ..registry import env as registry_env
from ..registry import run as registry
from ..routing.plan import RoutePlan
from ..routing.policies import plan_from_route_ids
from . import metrics, objectives, schedule, state
from .recipes import Inputs, Recipe, RecipeError, SeedRecipe, expected_packs, recipe_from_spec

EVAL_SPLITS = ("dev", "test")
UPDATE_BUDGET = 100_000_000  # P_update <= 100M (MS §5.1); recorded, not enforced (B12a is over it by design)


class TrainError(RuntimeError):
    pass


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise TrainError("run.device = cuda, but CUDA is not available")
    return torch.device(name)


def precision(train_dtype: str, device: torch.device) -> str:
    """bf16 autocast on CUDA, fp32 on the CPU, unless train.dtype names one."""
    if train_dtype == "auto":
        return "bf16" if device.type == "cuda" else "fp32"
    return train_dtype


# --- inputs ---------------------------------------------------------------------------------------------------------


@dataclass
class Resolved:
    root: Path
    shard_path: Path
    shard: ShardReader
    inputs: Inputs
    tokenizer_sha256: str


def resolve_inputs(run: RunConfig, recipe: Recipe, root: Path) -> Resolved:
    spec = run.run
    seed = seed_path = None
    if isinstance(recipe, SeedRecipe):
        if spec.tokenizer is None:
            raise TrainError("the seed recipe needs run.tokenizer (the tokenizer.json the shard was built with)")
        tokenizer = root / spec.tokenizer
        tok_sha = sha256_file(tokenizer)
        if spec.seed_pack or spec.packs:
            raise TrainError("the seed recipe reads no packs: unset run.seed_pack and run.packs")
    else:
        if spec.seed_pack is None:
            raise TrainError(f"the {recipe.kind} recipe forks from a seed: set run.seed_pack")
        seed_path = root / spec.seed_pack
        seed = read_pack(seed_path)
        if not isinstance(seed, SeedPack):
            raise TrainError(f"run.seed_pack {seed_path} is not a seed pack")
        tokenizer = seed_path / TOKENIZER
        tok_sha = seed.meta["tokenizer_sha256"]
        if spec.tokenizer is not None and sha256_file(root / spec.tokenizer) != tok_sha:
            raise TrainError("run.tokenizer differs from the seed pack's tokenizer")
    wanted = set(expected_packs(recipe))
    if set(spec.packs) != wanted:
        raise TrainError(
            f"the {recipe.kind} recipe reads the packs {sorted(wanted)}; run.packs names {sorted(spec.packs)}"
        )
    packs = {}
    for name, rel in sorted(spec.packs.items()):
        pack = read_pack(root / rel)
        if not isinstance(pack, ExpertPack):
            raise TrainError(f"run.packs[{name!r}] = {rel} is not an expert pack")
        packs[name] = (root / rel, pack)
    shard_path = root / spec.shard
    shard = ShardReader(shard_path, tokenizer_sha256=tok_sha, vocab_size=run.model.vocab_size)
    if set(str(shard.header["split"]).split("+")) & set(EVAL_SPLITS):  # "zh_train+en_train" names two splits
        raise TrainError(f"{shard_path} holds the {shard.header['split']!r} split; training never reads dev or test")
    if run.data["window"] != "packed":
        raise TrainError(f"data.window = {run.data['window']!r} is deferred; P0 trains on packed windows")
    route_map = recipe.route_map()
    if route_map is not None:
        missing = sorted(set(route_map) - set(shard.route_names))
        if missing:
            raise TrainError(f"{shard_path} has no shard routes {missing} (route_names {shard.route_names})")
    if len(shard) < run.train.seq + 1:
        raise TrainError(f"{shard_path} holds {len(shard)} tokens, fewer than one window of seq {run.train.seq}")
    inputs = Inputs(seed, seed_path, packs, tokenizer)
    return Resolved(root, shard_path, shard, inputs, tok_sha)


# --- one optimizer step ---------------------------------------------------------------------------------------------


class Trainer:
    """The per-step computation of a run: plans, forward and backward over the micro batches, clipping, decay to
    the seed, AdamW. Owns the optimizer; reads nothing from disk."""

    def __init__(
        self,
        model,
        plan: FreezePlan,
        recipe: Recipe,
        run: RunConfig,
        *,
        device: torch.device,
        precision: str,
        route_names: Sequence[str] = (),
        reference=None,
        anchors: Mapping[str, torch.Tensor] | None = None,
    ):
        self.model, self.plan, self.recipe, self.cfg = model, plan, recipe, run.train
        self.device, self.precision = device, precision
        self.reference = reference
        self.route_names = list(route_names)
        self.route_map = recipe.route_map()
        self.params = plan.parameters()
        lr0 = run.train.lr if run.train.lr is not None else 1.0
        self.optimizer = schedule.adamw(
            recipe.param_groups(model, plan, run.train), lr0, run.train.adam_betas, run.train.adam_eps
        )
        plan.check_optimizer(self.optimizer)
        self.decay = None
        if not isinstance(recipe, SeedRecipe) and run.train.expert_decay == "to_seed":
            anchors = anchors or {}
            missing = [n for n in plan.trainable_names if n not in anchors]
            if missing:
                raise TrainError(f"decay to the seed needs an anchor for every trained tensor; missing {missing[:3]}")
            self.decay = schedule.DecayToSeed(
                self.params, [anchors[n] for n in plan.trainable_names], run.train.expert_decay_lambda
            )

    def plan_for(self, routes: np.ndarray) -> RoutePlan | None:
        if self.route_map is None:
            return None
        return plan_from_route_ids(
            routes,
            self.route_names,
            self.model.expert_names,
            self.route_map,
            f"{self.recipe.kind}:{','.join(f'{k}>{v}' for k, v in self.route_map.items())}",
        )

    def _autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.precision == "bf16")

    def gradients(self, windows: Sequence[Mapping[str, np.ndarray]]) -> dict[str, Any]:
        """Accumulate the gradients of one optimizer step over its micro-batch windows."""
        masks = [torch.from_numpy(np.asarray(w["loss_mask"], dtype=np.bool_)) for w in windows]
        n = int(sum(int(m.sum()) for m in masks))
        if n == 0:
            raise TrainError("an optimizer step holds no loss target (every loss_mask is 0)")
        ce_total = kl_total = 0.0
        anchor = self.cfg.anchor_kl
        for window, mask in zip(windows, masks, strict=True):
            inputs = torch.from_numpy(np.asarray(window["inputs"], dtype=np.int64)).to(self.device)
            targets = torch.from_numpy(np.asarray(window["targets"], dtype=np.int64)).to(self.device)
            mask = mask.to(self.device)
            plan = self.plan_for(window["routes"])
            with self._autocast():
                logits = self.model(inputs, plan, checkpointing=self.cfg.activation_checkpointing)
            ce = objectives.target_ce_sum(logits, targets, mask)
            loss = ce
            if anchor > 0:
                with torch.no_grad(), self._autocast():
                    ref_logits = self.reference(inputs)
                kl = objectives.kl_sum(logits, ref_logits, mask)
                loss = loss + anchor * kl
                kl_total += float(kl.detach())
            (loss / n).backward()
            ce_total += float(ce.detach())
        kl_mean = kl_total / n if anchor > 0 else None
        return {
            "targets": n,
            "ce": ce_total / n,
            "kl": kl_mean,
            "loss": (ce_total + (anchor * kl_total if anchor > 0 else 0.0)) / n,
        }

    def step(self, windows: Sequence[Mapping[str, np.ndarray]], lr: float) -> dict[str, Any]:
        stats = self.gradients(windows)
        max_norm = self.cfg.grad_clip if self.cfg.grad_clip > 0 else math.inf
        norm = float(torch.nn.utils.clip_grad_norm_(self.params, max_norm))
        if not math.isfinite(stats["loss"]) or not math.isfinite(norm):
            raise TrainError(f"non-finite loss {stats['loss']} or gradient norm {norm}")
        self.apply(lr)
        return {**stats, "grad_norm": norm}

    def apply(self, lr: float) -> None:
        """At learning rate `lr`: decay to the seed (if configured), then the AdamW step on the accumulated
        gradients (AdamW's own decoupled decay, for "to_zero", uses the same lr)."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        if self.decay is not None:
            self.decay.apply(lr)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)


# --- preparing a run ------------------------------------------------------------------------------------------------


@dataclass
class Prepared:
    run: RunConfig
    recipe: Recipe
    resolved: Resolved
    model: Any
    plan: FreezePlan
    trainer: Trainer
    schedule: schedule.Schedule
    config: dict[str, Any]
    identity: dict[str, Any]
    fields: dict[str, Any]
    frozen_hashes: dict[str, str]
    device: torch.device


def run_document(run: RunConfig, recipe: Recipe, resolved: Resolved, total: int, prec: str) -> dict[str, Any]:
    """The hashed run config: everything that changes the trajectory, and nothing that does not (device, save and
    log cadence). Training plans are one-hot, so the fusion and the gate policy, which act only on dual rows at
    assembly and evaluation, are not part of a training run: B8 and B9-II evaluate the same checkpoints."""
    return {
        "model": recipe.model_config(run.model).to_dict(),
        "train": run.train.to_dict(),
        "gqa": run.gqa,
        "data": {"window": run.data["window"]},
        "recipe": {"kind": recipe.kind, **recipe.args()},
        "steps": total,
        "precision": prec,
        "shard": {"split": resolved.shard.header["split"], "header_sha256": sha256_json(resolved.shard.header)},
        "packs": {name: pack.expert_hash for name, (_, pack) in sorted(resolved.inputs.packs.items())},
    }


def prepare(
    run: RunConfig, recipe: Recipe | None = None, *, root: Path | None = None, device: str | None = None
) -> Prepared:
    spec = run.run
    if spec is None:
        raise TrainError("the config has no [run] table")
    recipe = recipe or recipe_from_spec(spec, run.train)
    if run.train.lr is None:
        raise TrainError("train.lr must be set by the row config (MS §8 grid)")
    if getattr(recipe, "anchor_kl", run.train.anchor_kl) != run.train.anchor_kl:
        raise TrainError(
            f"the recipe's anchor_kl {recipe.anchor_kl} differs from train.anchor_kl "
            f"{run.train.anchor_kl}; the run config is what the run id records"
        )
    root = Path.cwd() if root is None else Path(root)
    total = spec.total_steps(run.train)
    dev = resolve_device(device or spec.device)
    prec = precision(run.train.dtype, dev)
    resolved = resolve_inputs(run, recipe, root)
    cfg = recipe.model_config(run.model)
    built = recipe.build(
        cfg, resolved.inputs, fusion=run.fusion, gqa=run.gqa, init_seed=spec.init_seed, anchor_kl=run.train.anchor_kl
    )
    model = built.model.to(dev).train()
    reference = built.reference.to(dev) if built.reference is not None else None
    plan = recipe.freeze_plan(model).apply()
    trainer = Trainer(
        model,
        plan,
        recipe,
        run,
        device=dev,
        precision=prec,
        route_names=resolved.shard.route_names,
        reference=reference,
        anchors=built.anchors,
    )
    trainable = dict(zip(plan.trainable_names, plan.parameters(), strict=True))
    config_doc = run_document(run, recipe, resolved, total, prec)
    seed = resolved.inputs.seed
    identity = {
        "config_hash": config_hash(config_doc),
        "data_manifest_sha256": resolved.shard.header.get("manifest_sha256"),
        "tokenizer_sha256": resolved.tokenizer_sha256,
        "seed_fingerprint": seed.seed_fingerprint if seed is not None else None,
        "recipe": recipe.kind,
        "init_seed": spec.init_seed,
        "data_seed": spec.data_seed,
    }
    updated = p_update(trainer.optimizer)
    fields = {
        "matrix_row": spec.matrix_row,
        "replicate": spec.replicate,
        "config": config_doc,
        "inputs": {
            "shard": str(spec.shard),
            "seed_pack": spec.seed_pack,
            "packs": dict(spec.packs),
            "tokenizer": str(resolved.inputs.tokenizer) if spec.tokenizer is None else spec.tokenizer,
        },
        "total_steps": total,
        "init_hash": content_hash({k: v.detach() for k, v in trainable.items()}),
        "p_update": updated,
        "within_update_budget": updated <= UPDATE_BUDGET,
        "experts": list(model.expert_names),
    }
    sched = schedule.Schedule(total, run.train.lr, run.train.warmup_frac, run.train.min_lr_frac)
    return Prepared(
        run, recipe, resolved, model, plan, trainer, sched, config_doc, identity, fields, plan.frozen_hashes(), dev
    )


# --- the run --------------------------------------------------------------------------------------------------------


@dataclass
class RunResult:
    run_name: str
    run_id: str
    run_dir: Path
    status: str
    step: int
    total_steps: int
    tokens: int
    losses: list[float] = field(default_factory=list)  # this invocation's steps
    packs: dict[str, Path] = field(default_factory=dict)
    resumed_from: int | None = None
    model: Any = None
    optimizer: Any = None
    sampler: Any = None


def _cuda_peaks(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    return torch.cuda.max_memory_allocated(device) / 2**30, torch.cuda.max_memory_reserved(device) / 2**30


def _environment(run_dir: Path, step: int) -> None:
    name = "environment.json" if not (run_dir / "environment.json").exists() else f"environment.step_{step:08d}.json"
    registry_env.write_environment(run_dir / name)


def _resume(p: Prepared, run_dir: Path, run_id: str, sampler: EpochChunkSampler, resume: bool) -> tuple[int, int]:
    state.clean_stale(run_dir)
    latest = state.latest_state(run_dir)
    if latest is None:
        return 0, 0
    if not resume:
        raise TrainError(f"{run_dir} already holds states (latest {latest.name}); resume it or use another run name")
    loaded = state.load_state(latest, run_id=run_id, map_location=p.device)
    stored = loaded.extra["frozen_hashes"]
    if stored != p.frozen_hashes:
        changed = sorted(n for n in set(stored) | set(p.frozen_hashes) if stored.get(n) != p.frozen_hashes.get(n))
        raise FreezeError(f"{latest}: {len(changed)} frozen tensors differ from the live model, e.g. {changed[:3]}")
    if loaded.extra["trainable_names"] != sorted(p.plan.trainable_names):
        raise TrainError(f"{latest} holds another trainable set than this recipe's")
    params = dict(zip(p.plan.trainable_names, p.plan.parameters(), strict=True))
    with torch.no_grad():
        for name, value in loaded.trainable.items():
            params[name].copy_(value)
    p.trainer.optimizer.load_state_dict(loaded.optimizer)
    if p.schedule.check_state(loaded.extra["scheduler"]) != loaded.step:
        raise TrainError(f"{latest}: the schedule step differs from the state step")
    sampler.load_state_dict(loaded.extra["sampler"])
    state.restore_torch_rng(loaded.torch_rng)
    state.restore_rng(loaded.extra["rng"])
    return loaded.step, int(loaded.extra["tokens"])


def _save(p: Prepared, run_dir: Path, run_id: str, step: int, tokens: int, sampler: EpochChunkSampler) -> None:
    trainable = dict(zip(p.plan.trainable_names, p.plan.parameters(), strict=True))
    extra = {
        "run_id": run_id,
        "tokens": tokens,
        "frozen_hashes": p.frozen_hashes,
        "scheduler": p.schedule.state_dict(step),
        "sampler": sampler.state_dict(),
    }
    state.save_state(
        run_dir, step, trainable=trainable, optimizer=p.trainer.optimizer, extra=extra, keep_every=p.run.run.keep_every
    )


def _packs(p: Prepared, run_dir: Path, doc: Mapping[str, Any], step: int, tokens: int) -> dict[str, Path]:
    p.plan.check_frozen(p.frozen_hashes)
    out_dir = run_dir / "packs"
    trained = p.recipe.trained
    existing = out_dir / trained
    if existing.exists():  # a rerun of a finished run: the pack must be this model's
        pack = read_pack(existing)
        live = p.model.seed_ffn_state() if isinstance(pack, SeedPack) else p.model.expert_state(trained)
        recorded = pack.meta["hashes"]["seed_ffn" if isinstance(pack, SeedPack) else "expert"]
        if content_hash({k: v.detach().cpu() for k, v in live.items()}) != recorded:
            raise TrainError(f"{existing} exists and holds other weights than this run's final state")
        return {trained: existing}
    t = p.run.train
    meta = {
        "run_name": doc["run_name"],
        "run_id": doc["run_id"],
        "recipe": p.recipe.kind,
        "recipe_args": p.recipe.args(),
        "matrix_row": doc.get("matrix_row"),
        "replicate": doc.get("replicate"),
        "step": step,
        "tokens": tokens,
        "data_manifest_sha256": doc["data_manifest_sha256"],
        "expert_decay": t.expert_decay,
        "expert_decay_lambda": t.expert_decay_lambda,
        "anchor_kl": t.anchor_kl,
        "bridge_variant": p.run.train.bridge_variant if p.recipe.kind == "bridge" else None,
        "code_commit": doc["code_commit"],
        "code_dirty": doc["code_dirty"],
    }
    return p.recipe.write_packs(p.model, run_dir, p.resolved.inputs, meta)


def train(
    run: RunConfig,
    recipe: Recipe | None = None,
    resume: bool = True,
    *,
    root: Path | None = None,
    runs: Path | None = None,
    device: str | None = None,
    stop_after: int | None = None,
) -> RunResult:
    """Run (or resume) one training run; returns when it is done, or after `stop_after` completed steps."""
    p = prepare(run, recipe, root=root, device=device)
    spec = run.run
    runs = Path(runs) if runs is not None else p.resolved.root / registry.RUNS_DIR
    run_dir, doc, _ = registry.open_run(
        runs, spec.name, p.identity, p.fields, allow_dirty=bool(run.registry["allow_dirty"])
    )
    run_id = doc["run_id"]
    t = run.train
    sampler = EpochChunkSampler(len(p.resolved.shard), t.seq, spec.data_seed)
    metrics_path = run_dir / metrics.FILE
    total = p.schedule.total
    try:
        start, tokens = _resume(p, run_dir, run_id, sampler, resume)
        metrics.truncate(metrics_path, start)
        _environment(run_dir, start)
        result = RunResult(
            spec.name,
            run_id,
            run_dir,
            "running",
            start,
            total,
            tokens,
            resumed_from=start if start else None,
            model=p.model,
            optimizer=p.trainer.optimizer,
            sampler=sampler,
        )
        registry.update_run(run_dir, status="running", failure=None)
        if p.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(p.device)
        for index in range(start, total):
            if stop_after is not None and index >= stop_after:
                return result
            began = time.perf_counter()
            lr = p.schedule.lr(index)
            windows = [p.resolved.shard.windows(sampler.next_batch(t.microbatch)[0], t.seq) for _ in range(t.accum)]
            stats = p.trainer.step(windows, lr)
            if p.device.type == "cuda":
                torch.cuda.synchronize(p.device)
            elapsed = time.perf_counter() - began
            step, tokens = index + 1, tokens + t.global_batch_tokens
            result.step, result.tokens = step, tokens
            result.losses.append(stats["loss"])
            if step % spec.log_every == 0 or step == total:
                alloc, reserved = _cuda_peaks(p.device)
                metrics.append(
                    metrics_path,
                    metrics.line(
                        step=step,
                        tokens=tokens,
                        targets=stats["targets"],
                        loss=stats["loss"],
                        kl=stats["kl"],
                        lr=lr,
                        grad_norm=stats["grad_norm"],
                        tokens_per_s=t.global_batch_tokens / max(elapsed, 1e-9),
                        step_s=elapsed,
                        peak_alloc_gib=alloc,
                        peak_reserved_gib=reserved,
                    ),
                )
            if step == total or step == stop_after or (spec.save_every and step % spec.save_every == 0):
                _save(p, run_dir, run_id, step, tokens, sampler)
        result.packs = _packs(p, run_dir, doc, result.step, result.tokens)
    except BaseException as exc:
        registry.update_run(run_dir, status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    registry.update_run(
        run_dir,
        status="done",
        failure=None,
        finished_step=result.step,
        packs={k: str(v.relative_to(run_dir)) for k, v in result.packs.items()},
    )
    result.status = "done"
    return result


# --- command line ---------------------------------------------------------------------------------------------------


def _dry_run(p: Prepared) -> dict[str, Any]:
    commit, dirty = registry_env.code_identity()
    identity = {**p.identity, "code_commit": commit}
    return {
        "run_name": p.run.run.name,
        "run_id": registry.run_id(identity),
        "recipe": p.recipe.kind,
        "experts": list(p.model.expert_names),
        "trainable": len(p.plan.trainable_names),
        "p_update": p.fields["p_update"],
        "within_update_budget": p.fields["within_update_budget"],
        "total_steps": p.schedule.total,
        "warmup_steps": p.schedule.warmup,
        "tokens": p.schedule.total * p.run.train.global_batch_tokens,
        "precision": p.config["precision"],
        "device": str(p.device),
        "shard": {
            "path": str(p.resolved.shard_path),
            "split": p.resolved.shard.header["split"],
            "n_tokens": len(p.resolved.shard),
        },
        "code_commit": commit,
        "code_dirty": dirty,
        "allow_dirty": bool(p.run.registry["allow_dirty"]),
        "identity": identity,
    }


def main(argv: Sequence[str]) -> int:
    """`train ROW.toml [--stage S] [--list] [--root DIR] [--runs DIR] [--device D] [--set K=V]... [--name N]
    [--stop-after N] [--no-resume] [--dry-run]`"""
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual train", description="one run of a row config")
    ap.add_argument("config", type=Path)
    ap.add_argument("--stage", help="the stage of a row file with [stage.*] tables (see --list)")
    ap.add_argument("--list", action="store_true", help="print the file's stages and exit")
    ap.add_argument("--root", type=Path, help="where the row file's relative paths start (default: the cwd)")
    ap.add_argument("--runs", type=Path, help="the runs directory (default: ROOT/data/runs)")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], help="overrides run.device")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override one config value (TOML syntax), e.g. train.lr=3e-4",
    )
    ap.add_argument("--name", help="the run name (e.g. one lr trial of a row)")
    ap.add_argument("--stop-after", type=int, help="return after this many completed steps (a state is saved)")
    ap.add_argument("--no-resume", action="store_true", help="refuse to continue a run that already has states")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve the config, read and verify every input, build the model; write nothing",
    )
    args = ap.parse_args(list(argv))
    try:
        if args.list:
            print("\n".join(stage_names(args.config)))
            return 0
        overrides = tuple(args.set) + ((f'run.name="{args.name}"',) if args.name else ())
        run = load_toml(args.config, stage=args.stage, overrides=overrides)
        if args.dry_run:
            print(json.dumps(_dry_run(prepare(run, root=args.root, device=args.device)), indent=1, sort_keys=True))
            return 0
        result = train(
            run,
            resume=not args.no_resume,
            root=args.root,
            runs=args.runs,
            device=args.device,
            stop_after=args.stop_after,
        )
    except (ValueError, TrainError, RecipeError, AssemblyError, FreezeError, OSError) as exc:
        print(f"train: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except NotImplementedError as exc:
        print(f"train: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "run_name": result.run_name,
                "run_id": result.run_id,
                "status": result.status,
                "step": result.step,
                "total_steps": result.total_steps,
                "tokens": result.tokens,
                "packs": {k: str(v) for k, v in result.packs.items()},
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0
