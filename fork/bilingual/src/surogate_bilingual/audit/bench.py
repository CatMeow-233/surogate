"""The G0.5 throughput and memory grid (`bench`; docs/runbook.md, G0.5) and the SDPA backend probe of G0 (g3).

For every (family, recipe, seq, microbatch, checkpointing) point the bench builds the profile's model (random init,
no data or packs needed), runs `warmup` optimizer steps on a fixed random batch and times `steps` more:

- `seed`: every parameter trains (the seed run); `expert`: the backbone is frozen and one expert trains;
- bf16 autocast on CUDA (parameters fp32), AdamW, masked-free CE over all positions;
- recorded (`bilingual-bench/1`, one JSON line per point): tokens/s, mean step time, max_memory_allocated and
  max_memory_reserved, the nvidia-smi memory.used after the measured steps (the caching allocator keeps its peak
  reservation, so this is the process's footprint at the peak plus the CUDA context), the analytic weight MACs per
  token and MFU = 6 * N_matmul * tokens/s / PEAK (N_matmul = weight MACs per token, the 6N convention; frozen weights
  skip their weight gradients, so the expert recipe's MFU understates what the GPU does), and "oom" for points that
  do not fit.
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

import torch
from torch.nn import functional as F

from ..canonical import atomic_write_bytes, canonical_json
from ..config import FAMILIES, N_SPECIAL, PROFILES, profile
from ..model.lm import RoutedLM
from .flops import macs_per_token
from .memory import CudaPeak

BENCH_FORMAT = "bilingual-bench/1"
PEAK_TFLOPS = 209.5  # [估计] to verify against the NVIDIA spec sheet (G0 g6)
RECIPES = ("seed", "expert")
BACKENDS = ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION", "MATH")


def _csv(text: str, cast=str) -> list:
    return [cast(x) for x in text.split(",") if x]


def _onoff(text: str) -> bool:
    if text not in ("on", "off"):
        raise ValueError(f"checkpointing is on or off, got {text!r}")
    return text == "on"


def grid(families, recipes, seqs, microbatches, checkpointing) -> list[dict[str, Any]]:
    bad = [f for f in families if f not in FAMILIES] + [r for r in recipes if r not in RECIPES]
    if bad:
        raise ValueError(f"unknown families or recipes: {bad}")
    return [{"family": f, "recipe": r, "seq": s, "microbatch": m, "checkpointing": c}
            for f, r, s, m, c in itertools.product(families, recipes, seqs, microbatches, checkpointing)]  # fmt: skip


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
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i",
                              str(index)], capture_output=True, text=True, timeout=30, check=True).stdout  # fmt: skip
        return float(out.strip().splitlines()[0]) / 1024.0
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def bench_point(cfg, point: dict[str, Any], *, warmup: int, steps: int, device: torch.device, gqa: str = "sdpa_gqa",
                peak_tflops: float = PEAK_TFLOPS) -> dict[str, Any]:  # fmt: skip
    seq, mb, ckpt, recipe = point["seq"], point["microbatch"], point["checkpointing"], point["recipe"]
    names = ["seed"] if recipe == "seed" else ["zh"]
    record: dict[str, Any] = {"format": BENCH_FORMAT, "kind": "point", **point, "device": str(device),
                              "precision": "bf16" if device.type == "cuda" else "fp32", "gqa": gqa,
                              "warmup": warmup, "steps": steps}  # fmt: skip
    model = optimizer = None
    try:
        model = RoutedLM(cfg, names, gqa=gqa, device=device, init_seed=0).train()
        if recipe == "expert":
            for p in model.backbone.parameters():
                p.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        generator = torch.Generator().manual_seed(0)
        batch = torch.randint(N_SPECIAL, cfg.vocab_size, (mb, seq + 1), generator=generator).to(device)
        x, y = batch[:, :-1], batch[:, 1:]

        def step():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x, None, checkpointing=ckpt)
            loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            return loss

        for _ in range(warmup):
            step()
        with CudaPeak(device) as peak:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            began = time.perf_counter()
            for _ in range(steps):
                loss = step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - began
        tokens_per_s = mb * seq * steps / elapsed
        macs = macs_per_token(cfg, "single")
        record.update(status="ok", tokens_per_s=tokens_per_s, step_s=elapsed / steps,
                      max_memory_allocated_gib=peak.allocated_gib, max_memory_reserved_gib=peak.reserved_gib,
                      nvidia_smi_used_gib=_smi_used_gib(device), weight_macs_per_token=macs,
                      trainable_parameters=sum(p.numel() for p in trainable), final_loss=float(loss.detach()),
                      mfu=(6.0 * macs * tokens_per_s / (peak_tflops * 1e12)) if device.type == "cuda" else None,
                      peak_tflops=peak_tflops)  # fmt: skip
    except torch.cuda.OutOfMemoryError as exc:
        record.update(status="oom", error=str(exc).splitlines()[0][:300])
    finally:
        model = optimizer = None  # free the point's memory before the next one
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return record


def choose(records: Sequence[dict[str, Any]], global_batch_tokens: int, seq: int) -> list[dict[str, Any]]:
    """Per (family, recipe): the fastest point at `seq` that reaches global_batch_tokens with an integer accum."""
    out = []
    keys = sorted({(r["family"], r["recipe"]) for r in records if r.get("kind") == "point"})
    for family, recipe in keys:
        ok = [r for r in records if r.get("kind") == "point" and r["family"] == family and r["recipe"] == recipe
              and r["status"] == "ok" and r["seq"] == seq and global_batch_tokens % (r["microbatch"] * seq) == 0]  # fmt: skip
        best = max(ok, key=lambda r: r["tokens_per_s"], default=None)
        out.append({"format": BENCH_FORMAT, "kind": "choice", "family": family, "recipe": recipe, "seq": seq,
                    "global_batch_tokens": global_batch_tokens,
                    "microbatch": best["microbatch"] if best else None,
                    "accum": global_batch_tokens // (best["microbatch"] * seq) if best else None,
                    "checkpointing": best["checkpointing"] if best else None,
                    "tokens_per_s": best["tokens_per_s"] if best else None,
                    "status": "ok" if best else "no point fits"})  # fmt: skip
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
        points = grid(_csv(args.family), _csv(args.recipe), _csv(args.seq, int), _csv(args.microbatch, int),
                      [_onoff(c) for c in _csv(args.checkpointing)])  # fmt: skip
        cfgs = {f: profile(args.profile, f) for f in {p["family"] for p in points}}
        too_long = sorted({p["seq"] for p in points if p["seq"] > cfgs[p["family"]].max_seq})
        if too_long:
            raise ValueError(f"seq {too_long} exceeds max_seq of the {args.profile} profile")
        if args.dry_run:
            print(json.dumps({"dry_run": True, "points": len(points), "profile": args.profile, "out": str(args.out),
                              "grid": points}, indent=1))  # fmt: skip
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
    header = {"format": BENCH_FORMAT, "kind": "header", "profile": args.profile, "environment": env,
              "sdpa_backends": sdpa_probe(cfgs[points[0]["family"]], device), "peak_tflops": args.peak_tflops}  # fmt: skip
    records = [header]
    for point in points:
        records.append(bench_point(cfgs[point["family"]], point, warmup=args.warmup, steps=args.steps, device=device,
                                   gqa=args.gqa, peak_tflops=args.peak_tflops))  # fmt: skip
        print(json.dumps({k: records[-1].get(k) for k in ("family", "recipe", "seq", "microbatch", "checkpointing",
                                                          "status", "tokens_per_s", "max_memory_reserved_gib")}))  # fmt: skip
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
