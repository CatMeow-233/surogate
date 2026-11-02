"""The G0.5 throughput and memory grid (`bench`; docs/runbook.md, G0.5) and the SDPA backend probe of G0 (g3).

For every (family, recipe, seq, microbatch, checkpointing) point the bench builds the profile's model (random init,
no data or packs needed) and times the training step itself: `train.loop.Trainer.step` with the recipe's freeze
plan, parameter groups, objectives (masked CE on the loss-mask targets, the KL-anchor reference forward when the
recipe has one), gradient clipping and AdamW, on synthetic windows of the shard layout (inputs, targets, loss mask,
routes). There is no second copy of the step, so a change of the training step is measured as it is. Recipes:

- `seed`: every parameter trains (the seed run); `expert`: the backbone is frozen and one expert trains (B1, the
  family-II experts); `expert_kl`: the same with the KL anchor to a frozen seed model (train.anchor_kl, P1);
- `joint_wide` (family I): the 2f-wide B12a expert; `bridge_c` (family I): ZH and EN frozen in the forward, the
  Bridge trained, the first half of each window the source prefix (loss 0), the second half the target (T2, T3').
  T1' trains one copy beside one frozen expert, a smaller layout than `bridge_c`;
- bf16 autocast on CUDA (parameters fp32);
- recorded (`bilingual-bench/1`, one JSON line per point): tokens/s, mean step time, max_memory_allocated and
  max_memory_reserved, the nvidia-smi memory.used after the measured steps (the caching allocator keeps its peak
  reservation, so this is the process's footprint at the peak plus the CUDA context), the analytic weight MACs per
  token and MFU = c * N_matmul * tokens/s / PEAK (N_matmul = weight MACs per token; c = 6, plus 2 for the KL
  reference forward; frozen weights skip their weight gradients, so the MFU of the frozen-backbone recipes
  understates what the GPU does), and "oom" for points that do not fit.
- a last line (kind "choice") picks, per (family, recipe), the point at the training seq whose microbatch divides
  `global_batch_tokens / seq` with the highest tokens/s; accum = global_batch_tokens / (microbatch * seq).

PEAK_TFLOPS (209.5, dense BF16 with fp32 accumulation) is the value to verify against NVIDIA's spec sheet at G0 (g6);
every number this module produces on a GPU is a measurement of that host only. Without CUDA the bench refuses to
run unless `--device cpu` is given (CPU runs are for tests; their numbers say nothing about the 5090).
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from ..canonical import atomic_write_bytes, canonical_json
from ..config import FAMILIES, N_SPECIAL, PROFILES, profile, run_config
from ..model.lm import RoutedLM
from .flops import macs_per_token
from .memory import CudaPeak

BENCH_FORMAT = "bilingual-bench/1"
PEAK_TFLOPS = 209.5  # [估计] to verify against the NVIDIA spec sheet (G0 g6)
RECIPES = ("seed", "expert", "expert_kl", "joint_wide", "bridge_c")
FAMILY_I_ONLY = ("joint_wide", "bridge_c")
BENCH_KL = 0.1  # the anchor weight of the expert_kl points; its value does not change the cost
ROUTE_NAMES = ("zh", "en", "bridge", "seed")  # the shard default (data.shards)
BACKENDS = ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH")


def _csv(text: str, cast=str) -> list:
    return [cast(x) for x in text.split(",") if x]


def _onoff(text: str) -> bool:
    if text not in ("on", "off"):
        raise ValueError(f"checkpointing is on or off, got {text!r}")
    return text == "on"


def grid(families, recipes, seqs, microbatches, checkpointing) -> list[dict[str, Any]]:
    """The product of the axes, without the family-I-only recipes on family II."""
    bad = [f for f in families if f not in FAMILIES] + [r for r in recipes if r not in RECIPES]
    if bad:
        raise ValueError(f"unknown families or recipes: {bad}")
    return [
        {"family": f, "recipe": r, "seq": s, "microbatch": m, "checkpointing": c}
        for f, r, s, m, c in itertools.product(families, recipes, seqs, microbatches, checkpointing)
        if f == "swiglu" or r not in FAMILY_I_ONLY
    ]


def sdpa_probe(cfg, device: torch.device, dtype: torch.dtype = torch.bfloat16, seq: int = 256) -> dict[str, str]:
    """Each SDPA backend forced in turn with the model's GQA shape and enable_gqa: "ok" or the error (G0 g3)."""
    if device.type != "cuda":
        return {b: "not probed (CUDA only)" for b in BACKENDS}
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(2, cfg.n_heads, seq, cfg.head_dim, device=device, dtype=dtype)
    kv = torch.randn(2, cfg.n_kv_heads, seq, cfg.head_dim, device=device, dtype=dtype)
    out = {}
    for name in BACKENDS:
        try:
            with sdpa_kernel([getattr(SDPBackend, name)]):
                F.scaled_dot_product_attention(q, kv, kv, is_causal=True, enable_gqa=True)
            torch.cuda.synchronize(device)
            out[name] = "ok"
        except (RuntimeError, AttributeError) as exc:
            out[name] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    return out


def _smi_used_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(index)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
        return float(out.strip().splitlines()[0]) / 1024.0
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def bench_recipe(name: str):
    """(recipe, KL anchor weight) of a bench recipe name."""
    from ..train.recipes import BridgeRecipe, ExpertRecipe, JointWideRecipe, SeedRecipe

    return {
        "seed": (SeedRecipe(), 0.0),
        "expert": (ExpertRecipe("zh"), 0.0),
        "expert_kl": (ExpertRecipe("zh", anchor_kl=BENCH_KL), BENCH_KL),
        "joint_wide": (JointWideRecipe("joint"), 0.0),
        "bridge_c": (BridgeRecipe("bridge", variant="c"), 0.0),
    }[name]


def synthetic_window(cfg, recipe_name: str, mb: int, seq: int) -> dict[str, np.ndarray]:
    """One shard window (data.shards.ShardReader.windows layout) of uniform random tokens: every target has loss 1
    and the route of the trained expert, except under `bridge_c`, whose first half is a zh source prefix (loss 0)."""
    rng = np.random.default_rng(0)
    batch = rng.integers(N_SPECIAL, cfg.vocab_size, size=(mb, seq + 1), dtype=np.int64)
    routes = np.full((mb, seq), ROUTE_NAMES.index("seed"), dtype=np.uint8)
    mask = np.ones((mb, seq), dtype=np.bool_)
    if recipe_name == "bridge_c":
        half = seq // 2
        routes[:, :half], routes[:, half:] = ROUTE_NAMES.index("zh"), ROUTE_NAMES.index("bridge")
        mask[:, :half] = False
    return {"inputs": batch[:, :-1], "targets": batch[:, 1:], "loss_mask": mask, "routes": routes}


def bench_point(
    cfg,
    point: dict[str, Any],
    *,
    warmup: int,
    steps: int,
    device: torch.device,
    gqa: str = "sdpa_gqa",
    peak_tflops: float = PEAK_TFLOPS,
) -> dict[str, Any]:
    from ..train.loop import Trainer, precision

    seq, mb, ckpt, name = point["seq"], point["microbatch"], point["checkpointing"], point["recipe"]
    prec = precision("auto", device)
    record: dict[str, Any] = {
        "format": BENCH_FORMAT,
        "kind": "point",
        **point,
        "device": str(device),
        "precision": prec,
        "gqa": gqa,
        "warmup": warmup,
        "steps": steps,
    }
    model = trainer = reference = None
    try:
        recipe, kl = bench_recipe(name)
        model_cfg = recipe.model_config(cfg)
        run = run_config(
            {
                "model": cfg.to_dict(),
                "train": {
                    "microbatch": mb,
                    "seq": seq,
                    "accum": 1,
                    "global_batch_tokens": mb * seq,
                    "lr": 1e-4,
                    "activation_checkpointing": ckpt,
                    "anchor_kl": kl,
                },
            }
        )
        model = RoutedLM(model_cfg, recipe.names(), gqa=gqa, device=device, init_seed=0).train()
        if kl > 0:
            reference = RoutedLM(cfg, ["seed"], gqa=gqa, device=device, init_seed=1).requires_grad_(False).eval()
        plan = recipe.freeze_plan(model).apply()
        trainer = Trainer(
            model, plan, recipe, run, device=device, precision=prec, route_names=ROUTE_NAMES, reference=reference
        )
        window = synthetic_window(model_cfg, name, mb, seq)
        for _ in range(warmup):
            trainer.step([window], 1e-4)
        with CudaPeak(device) as peak:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            began = time.perf_counter()
            for _ in range(steps):
                stats = trainer.step([window], 1e-4)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - began
        tokens_per_s = mb * seq * steps / elapsed
        macs = macs_per_token(model_cfg, "single")
        factor = 6.0 + (2.0 if kl > 0 else 0.0)
        record.update(
            status="ok",
            tokens_per_s=tokens_per_s,
            step_s=elapsed / steps,
            max_memory_allocated_gib=peak.allocated_gib,
            max_memory_reserved_gib=peak.reserved_gib,
            nvidia_smi_used_gib=_smi_used_gib(device),
            weight_macs_per_token=macs,
            trainable_parameters=sum(p.numel() for p in plan.parameters()),
            final_loss=float(stats["loss"]),
            mfu=(factor * macs * tokens_per_s / (peak_tflops * 1e12)) if device.type == "cuda" else None,
            mfu_factor=factor,
            peak_tflops=peak_tflops,
        )
    except torch.cuda.OutOfMemoryError as exc:
        record.update(status="oom", error=str(exc).splitlines()[0][:300])
    finally:
        model = trainer = reference = None  # free the point's memory before the next one
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return record


def choose(records: Sequence[dict[str, Any]], global_batch_tokens: int, seq: int) -> list[dict[str, Any]]:
    """Per (family, recipe): the fastest point at `seq` that reaches global_batch_tokens with an integer accum."""
    out = []
    keys = sorted({(r["family"], r["recipe"]) for r in records if r.get("kind") == "point"})
    for family, recipe in keys:
        ok = [
            r
            for r in records
            if r.get("kind") == "point"
            and r["family"] == family
            and r["recipe"] == recipe
            and r["status"] == "ok"
            and r["seq"] == seq
            and global_batch_tokens % (r["microbatch"] * seq) == 0
        ]
        best = max(ok, key=lambda r: r["tokens_per_s"], default=None)
        out.append(
            {
                "format": BENCH_FORMAT,
                "kind": "choice",
                "family": family,
                "recipe": recipe,
                "seq": seq,
                "global_batch_tokens": global_batch_tokens,
                "microbatch": best["microbatch"] if best else None,
                "accum": global_batch_tokens // (best["microbatch"] * seq) if best else None,
                "checkpointing": best["checkpointing"] if best else None,
                "tokens_per_s": best["tokens_per_s"] if best else None,
                "status": "ok" if best else "no point fits",
            }
        )
    return out


def main(argv: Sequence[str]) -> int:
    """`bench --family F,.. --seq S,.. --microbatch M,.. --checkpointing off,on --warmup N --steps N --out FILE`"""
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual bench", description="G0.5 throughput and memory")
    ap.add_argument("--family", default="swiglu,slot")
    ap.add_argument("--recipe", default="seed,expert")
    ap.add_argument("--seq", default="256,512,1024")
    ap.add_argument("--microbatch", default="1,4,8,16,32")
    ap.add_argument("--checkpointing", default="off,on")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="pilot")
    ap.add_argument("--gqa", choices=["sdpa_gqa", "repeat"], default="sdpa_gqa")
    ap.add_argument("--device", default="auto", help="auto = cuda; cpu only when given explicitly (tests)")
    ap.add_argument("--peak-tflops", type=float, default=PEAK_TFLOPS)
    ap.add_argument("--global-batch-tokens", type=int, default=65536)
    ap.add_argument("--choice-seq", type=int, help="the training seq of the choice (default: the largest --seq)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true", help="validate the grid and print it; build nothing")
    args = ap.parse_args(list(argv))
    try:
        points = grid(
            _csv(args.family),
            _csv(args.recipe),
            _csv(args.seq, int),
            _csv(args.microbatch, int),
            [_onoff(c) for c in _csv(args.checkpointing)],
        )
        cfgs = {f: profile(args.profile, f) for f in {p["family"] for p in points}}
        too_long = sorted({p["seq"] for p in points if p["seq"] > cfgs[p["family"]].max_seq})
        if too_long:
            raise ValueError(f"seq {too_long} exceeds max_seq of the {args.profile} profile")
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "dry_run": True,
                        "points": len(points),
                        "profile": args.profile,
                        "out": str(args.out),
                        "grid": points,
                    },
                    indent=1,
                )
            )
            return 0
        if args.device == "auto":
            if not torch.cuda.is_available():
                raise ValueError("the G0.5 bench measures the GPU; no CUDA device here (use --device cpu for a test)")
            device = torch.device("cuda")
        else:
            device = torch.device(args.device)
    except ValueError as exc:
        print(f"bench: {exc}", file=sys.stderr)
        return 1
    from ..registry.env import environment

    env = environment()
    header = {
        "format": BENCH_FORMAT,
        "kind": "header",
        "profile": args.profile,
        "environment": env,
        "sdpa_backends": sdpa_probe(cfgs[points[0]["family"]], device),
        "peak_tflops": args.peak_tflops,
    }
    records = [header]
    for point in points:
        records.append(
            bench_point(
                cfgs[point["family"]],
                point,
                warmup=args.warmup,
                steps=args.steps,
                device=device,
                gqa=args.gqa,
                peak_tflops=args.peak_tflops,
            )
        )
        print(
            json.dumps(
                {
                    k: records[-1].get(k)
                    for k in (
                        "family",
                        "recipe",
                        "seq",
                        "microbatch",
                        "checkpointing",
                        "status",
                        "tokens_per_s",
                        "max_memory_reserved_gib",
                    )
                }
            )
        )
        atomic_write_bytes(args.out, b"".join(canonical_json(r) + b"\n" for r in records))
    seq = args.choice_seq or max(p["seq"] for p in points)
    records += choose(records, args.global_batch_tokens, seq)
    atomic_write_bytes(args.out, b"".join(canonical_json(r) + b"\n" for r in records))
    for r in records:
        if r["kind"] == "choice":
            print(json.dumps(r))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
