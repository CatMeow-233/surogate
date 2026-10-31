"""Ledger T22 GPU check: which softmax scale did one attention layer's kernel apply?

Run on the GPU host, after a native forward (trainer.validate) or step (trainer.step) with
SUROGATE_DEBUG_ATTENTION_DUMP_LAYER=<layer> and SUROGATE_DEBUG_DUMP_DIR=<dir>. flash_attention.cpp then writes the
layer's attention tensors as float32 (attn_live.layer<N>.<name>.bin, shape in the matching .json); every call
overwrites the previous one:

    forward   qkv [B, T, Hq + 2 Hkv, Hs] (Q heads, then K, then V)   out [B, T, Hq, Hs]   lse [B, Hq, T]
    backward  qkv_bwd (as qkv)   d_out (as out)   d_qkv (as qkv)

From the dumped qkv (and d_out) this computes causal GQA attention in float64 at the declared scale and at the
kernel default 1/sqrt(Hs), forward and the analytic backward, and reports how far the kernel's out, lse (natural
log) and d_qkv are from each. A bf16 kernel lands within bf16 rounding of one reference; on the errata's tiny
Gemma3 (layer-5 q/k_norm gain 2) the other is about 7% (out) and 14% (d_qkv) away. lse is read as cuDNN's
[B, Hq, T]; flash_varlen's [Hq, B*T] is the same layout only for B=1.

    python3 fork/patches/tests/native/t22_attention_dump_check.py DIR --layer 5 --hq 4 --hkv 2 --scale 0.07715167498104596

Exit status 0 when every dumped quantity matches the declared scale and not the default, 1 otherwise, 2 when the
forward dumps are missing. Needs numpy only.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# A bf16 kernel's out and d_qkv sit ~0.2-0.3% (relative L2) from the reference at the scale it applied, its fp32 lse
# ~1e-4 nats; the other scale is ~7% / ~14% / ~0.9 nats (mean) away on the errata's model.
MATCH_REL_L2 = {"out": 2e-2, "d_qkv": 3e-2}
MATCH_LSE_MAX_ABS = 5e-2


def load(dump_dir, layer, name):
    stem = Path(dump_dir) / f"attn_live.layer{layer}.{name}"
    bin_path, json_path = Path(f"{stem}.bin"), Path(f"{stem}.json")
    if not bin_path.exists() or not json_path.exists():
        return None
    shape = json.loads(json_path.read_text())["shape"]
    data = np.fromfile(bin_path, dtype=np.float32)
    if data.size != math.prod(shape):
        raise SystemExit(f"{bin_path}: {data.size} floats, the .json says {shape}")
    return data.astype(np.float64), shape


def split_qkv(flat, B, T, hq, hkv, hs):
    qkv = flat.reshape(B, T, hq + 2 * hkv, hs).transpose(0, 2, 1, 3)  # [B, H, T, Hs]
    return qkv[:, :hq], qkv[:, hq : hq + hkv], qkv[:, hq + hkv :]


def attention(q, k, v, scale):
    """Causal GQA attention: q [B, Hq, T, Hs], k/v [B, Hkv, T, Hs] -> (out [B, Hq, T, Hs], lse [B, Hq, T], p, k, v)."""
    rep = q.shape[1] // k.shape[1]
    k, v = np.repeat(k, rep, axis=1), np.repeat(v, rep, axis=1)  # query head h reads KV head h // rep
    s = scale * (q @ k.transpose(0, 1, 3, 2))
    T = q.shape[2]
    s = np.where(np.triu(np.ones((T, T), dtype=bool), 1), -np.inf, s)
    m = s.max(-1, keepdims=True)
    lse = (m + np.log(np.exp(s - m).sum(-1, keepdims=True)))[..., 0]
    p = np.exp(s - lse[..., None])
    return p @ v, lse, p, k, v


def attention_backward(q, k, v, d_out, scale):
    """d(q, k, v) of attention() for the upstream gradient d_out [B, Hq, T, Hs]; dK and dV summed over each group."""
    out, _, p, ke, ve = attention(q, k, v, scale)
    d_v = p.transpose(0, 1, 3, 2) @ d_out
    d_p = d_out @ ve.transpose(0, 1, 3, 2)
    d_s = p * (d_p - (d_out * out).sum(-1, keepdims=True))
    d_q = scale * (d_s @ ke)
    d_k = scale * (d_s.transpose(0, 1, 3, 2) @ q)
    B, hkv, T, hs = k.shape
    rep = q.shape[1] // hkv
    d_k = d_k.reshape(B, hkv, rep, T, hs).sum(2)
    d_v = d_v.reshape(B, hkv, rep, T, hs).sum(2)
    return np.concatenate([d_q, d_k, d_v], axis=1).transpose(0, 2, 1, 3)  # [B, T, Hq + 2 Hkv, Hs]


def rel_l2(x, ref):
    return float(np.linalg.norm(x - ref) / np.linalg.norm(ref))


def compare(dump_dir, layer, hq, hkv, scale):
    """(Hs, default scale, rows of (quantity, error vs declared, error vs default, metric)) for the dumps present."""
    fwd = [load(dump_dir, layer, n) for n in ("qkv", "out", "lse")]
    if any(x is None for x in fwd):
        return None
    (qkv, qkv_shape), (out, _), (lse, _) = fwd
    B, T = qkv_shape[0], qkv_shape[1]  # [B, T, H, Hs] or [B, T, H * Hs]
    hs = qkv.size // (B * T * (hq + 2 * hkv))
    if B * T * (hq + 2 * hkv) * hs != qkv.size or out.size != B * T * hq * hs or lse.size != B * hq * T:
        raise SystemExit(f"dump sizes do not fit Hq={hq}, Hkv={hkv}: qkv {qkv_shape}, out {out.size}, lse {lse.size}")
    default = 1.0 / math.sqrt(hs)
    scales = (scale, default)
    q, k, v = split_qkv(qkv, B, T, hq, hkv, hs)
    refs = [attention(q, k, v, s) for s in scales]
    out = out.reshape(B, T, hq, hs).transpose(0, 2, 1, 3)
    lse = lse.reshape(B, hq, T)
    rows = [
        ("out", *(rel_l2(out, r[0]) for r in refs), "rel L2"),
        ("lse", *(float(np.abs(lse - r[1]).max()) for r in refs), "max |diff| (nats)"),
    ]
    bwd = [load(dump_dir, layer, n) for n in ("qkv_bwd", "d_out", "d_qkv")]
    if all(x is not None for x in bwd):
        (qkv_b, _), (d_out, _), (d_qkv, _) = bwd
        qb, kb, vb = split_qkv(qkv_b, B, T, hq, hkv, hs)
        g = d_out.reshape(B, T, hq, hs).transpose(0, 2, 1, 3)
        d_qkv = d_qkv.reshape(B, T, hq + 2 * hkv, hs)
        rows.append(("d_qkv", *(rel_l2(d_qkv, attention_backward(qb, kb, vb, g, s)) for s in scales), "rel L2"))
    return hs, default, rows


def verdict(name, err_declared, err_default):
    limit = MATCH_LSE_MAX_ABS if name == "lse" else MATCH_REL_L2[name]
    if err_declared < limit <= err_default:
        return "declared"
    if err_default < limit <= err_declared:
        return "default"
    return "neither" if min(err_declared, err_default) >= limit else "both"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dump_dir")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--hq", type=int, required=True, help="query heads of that layer")
    ap.add_argument("--hkv", type=int, required=True, help="KV heads of that layer")
    ap.add_argument("--scale", type=float, required=True, help="the op's softmax_scale, e.g. 168**-0.5")
    args = ap.parse_args(argv)
    result = compare(args.dump_dir, args.layer, args.hq, args.hkv, args.scale)
    if result is None:
        print(f"no attn_live.layer{args.layer}.qkv/out/lse dumps in {args.dump_dir}", file=sys.stderr)
        return 2
    hs, default, rows = result
    print(f"layer {args.layer}: Hs={hs}, declared scale {args.scale:.9g}, default 1/sqrt(Hs) {default:.9g}")
    verdicts = []
    for name, err_declared, err_default, metric in rows:
        verdicts.append(verdict(name, err_declared, err_default))
        print(
            f"  {name:6s} {metric:18s} vs declared {err_declared:.3e}  vs default {err_default:.3e}  -> {verdicts[-1]}"
        )
    if not any(name == "d_qkv" for name, *_ in rows):
        print("  (no qkv_bwd/d_out/d_qkv dumps: forward only)")
    return 0 if all(v == "declared" for v in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
