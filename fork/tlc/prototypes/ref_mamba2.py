"""Mamba-2 numerical reference for TLC's Mamba rows (A0, A0-split), in plain PyTorch, for gates G1 and G2.

It reads a TLC model directory (config.json, model.safetensors), or takes weights directly (Reference.from_weights),
and computes what the native graph computes, op for op: the ops LAYER_OPS lists, as upstream Mamba2Mixer emits them
inside TLCMambaFFNBlock, with the details of csrc/src/runtime/ops/mamba_*.cpp and the selective-scan kernel:

- in_proj output order [z | x B C | dt]; causal depthwise conv over x B C with bias, then SiLU;
- dt = clamp(softplus(dt_raw + dt_bias), TIME_STEP_LIMIT), softplus linear above 20; A = -exp(A_log);
- h_t = exp(dt A) h_{t-1} + (dt u_t) B_t, y_t = C_t . h_t + D u_t, head h reading group h // (H / G);
- gated norm RMSNorm(y * silu(z)) with one RMS per group of H*P/G channels (norm before the gate is off);
- fused residual + RMSNorm: the norm reads the sum, the stored residual is rounded;
- SwiGLU on the fused [up | gate] weight: up * silu(gate).

The scan has three evaluation orders, which agree (g1_suite.py, G1.1):
- "scan": the recurrence, one token after another, as the native kernel does; the goldens use it;
- "chunked": the chunked SSD form (Dao & Gu 2024, ssd_minimal_discrete), sharing no code with the recurrence;
- prefill() then step(): the whole model one token at a time with conv and SSM caches, the decode form.

`positions` (as compute_logprobs takes them: a 0 after the first token starts a new document) make every path
document-aware: the SSM state and the conv window reset at a document start. The native mamba_ssm_scan does not
reset, which is why the shards keep one document per row; the document-aware form is the reference for that kernel
(gate G1.6). The functions causal_conv, conv_window, decay_matrix, ssd_chunked and same_document are the points
g1_suite.py mutates to show that its tests catch each bug.

With bf16=True every tensor the native graph stores in bf16 is rounded to bf16 at the same point; the gap between
the bf16 and fp32 results sets the G1(b) tolerance. dtype=torch.float64 runs everything in fp64. Attention layers are
not modelled.

Usage: python ref_mamba2.py MODEL_DIR OUT.npz [--batch 2] [--seq-len 512] [--seed 0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

# dt clamp of the native scan when Mamba2Mixer gets no time_step_limit, as in TLCMambaFFNBlock.
TIME_STEP_LIMIT = (0.0, 1e9)
# Native ops of one Mamba layer in graph order, views left out.
LAYER_OPS = (
    "fused_residual_rmsnorm",
    "matmul",
    "mamba_split_proj",
    "mamba_conv1d",
    "mamba_split_conv_out",
    "mamba_ssm_scan",
    "mamba_gated_rmsnorm",
    "matmul",
    "fused_residual_rmsnorm",
    "matmul",
    "swiglu",
    "matmul",
)
LAYER_FIELDS = (
    "ln1_weight",
    "in_proj_weight",
    "conv_weight",
    "conv_bias",
    "A_log",
    "D_param",
    "dt_bias",
    "gated_norm_weight",
    "out_proj_weight",
    "ln2_weight",
    "mlp_up_weight",
    "mlp_down_weight",
)
EMBEDDING, FINAL_NORM, LM_HEAD = "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"
SPLIT_KEY, FIRST_LAYER = "tlc_split_first_layer", "first_layer"
IGNORE = -100  # target id compute_logprobs skips (surogate_tlc.data.shards.IGNORE; prototypes do not import it)


def layer_key(cfg: dict, layer: int, field: str) -> str:
    """safetensors name of `field` of layer `layer` (from 0): blocks[i].X, or first_layer_X for A0-split."""
    if cfg.get(SPLIT_KEY, False):
        return f"{FIRST_LAYER}_{field}" if layer == 0 else f"blocks[{layer - 1}].{field}"
    return f"blocks[{layer}].{field}"


def weight_names(cfg: dict) -> set[str]:
    """Every tensor name model.safetensors must hold for `cfg`, and nothing else."""
    names = {EMBEDDING, FINAL_NORM} | (set() if cfg["tie_word_embeddings"] else {LM_HEAD})
    return names | {layer_key(cfg, i, f) for i in range(cfg["num_hidden_layers"]) for f in LAYER_FIELDS}


def documents(positions: torch.Tensor | None) -> torch.Tensor | None:
    """Document index per token [b, T] from position ids: a position 0 after the first token starts a document."""
    if positions is None:
        return None
    starts = positions == 0
    starts[:, 0] = False
    return starts.long().cumsum(-1)


def same_document(doc: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Whether two document indices (broadcast) name the same document."""
    return doc == other


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, groups: int = 1) -> torch.Tensor:
    xs = x.unflatten(-1, (groups, -1))
    return (xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + eps)).flatten(-2) * weight


def causal_conv(xbc: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, doc: torch.Tensor | None) -> torch.Tensor:
    """Depthwise causal conv of xbc [b,T,D] with weight [D,K] and bias [D], before the activation. Output t reads
    inputs t-K+1..t (weight[:, K-1-k] for t-k); with `doc`, no tap reaches into an earlier document."""
    T, D, K = xbc.shape[1], xbc.shape[2], weight.shape[1]
    if doc is None:
        conv = F.conv1d(xbc.transpose(1, 2), weight[:, None, :], bias, padding=K - 1, groups=D)
        return conv[..., :T].transpose(1, 2)
    out = bias.expand_as(xbc)
    for k in range(K):
        earlier = F.pad(xbc, (0, 0, k, 0))[:, :T]
        same = same_document(F.pad(doc, (k, 0), value=-1)[:, :T], doc)
        out = out + earlier * same[..., None] * weight[:, K - 1 - k]
    return out


def conv_window(xbc: torch.Tensor, kernel: int, doc: torch.Tensor | None) -> torch.Tensor:
    """The conv cache after a prefill: the last `kernel` inputs [b, D, kernel], zeros before the sequence or, with
    `doc`, before the last document's start."""
    if doc is not None:
        xbc = xbc * same_document(doc, doc[:, -1:])[..., None]
    return F.pad(xbc, (0, 0, kernel, 0))[:, -kernel:].transpose(1, 2)


def ssm_step(
    h: torch.Tensor, u: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One token of the recurrence: h [b,H,P,N], u [b,H,P], dt [b,H], A [H], B and C [b,H,N] -> (h, C.h)."""
    h = torch.exp(dt * A)[..., None, None] * h + (dt[..., None] * u)[..., None] * B[:, :, None, :]
    return h, (h * C[:, :, None, :]).sum(-1)


def ssm_scan(
    u: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, doc: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The recurrence over a sequence: u [b,T,H,P], dt [b,T,H], B and C [b,T,H,N] -> (y [b,T,H,P], final state)."""
    h = u.new_zeros(*u.shape[:1], *u.shape[2:], B.shape[-1])
    ys = []
    for t in range(u.shape[1]):
        if doc is not None and t:
            h = h * same_document(doc[:, t], doc[:, t - 1])[:, None, None, None]
        h, y = ssm_step(h, u[:, t], dt[:, t], A, B[:, t], C[:, t])
        ys.append(y)
    return torch.stack(ys, 1), h


def decay_matrix(a: torch.Tensor) -> torch.Tensor:
    """a [..., L] (log decays of one chunk) -> [..., L, L]: exp(a[j+1] + ... + a[i]) for j <= i, 0 above the
    diagonal. The sums are masked to -inf before exp, so no exp of a large positive sum reaches the backward."""
    L = a.shape[-1]
    lower = torch.ones(L, L, dtype=torch.bool, device=a.device)
    sums = a[..., None].expand(*a.shape, L).masked_fill(~lower.tril(-1), 0).cumsum(-2)
    return torch.exp(sums.masked_fill(~lower.tril(0), -torch.inf))


def ssd_chunked(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk: int,
    doc: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The map of ssm_scan in the chunked SSD form: within a chunk the quadratic form, between chunks the state.

    The sequence is zero-padded to a multiple of `chunk` (padding has no input and no decay, so the final state is
    that of the last real token). With `doc`, a pair of positions in different documents does not interact, and the
    state carried into a chunk reaches only its positions before the chunk's first document start.
    """
    b, T, H, P = u.shape
    pad = -T % chunk
    c = (T + pad) // chunk
    X = F.pad(u * dt[..., None], (0, 0, 0, 0, 0, pad)).reshape(b, c, chunk, H, P)
    a = F.pad(dt * A, (0, 0, 0, pad)).reshape(b, c, chunk, H).permute(0, 3, 1, 2)  # b H c l
    B, C = (F.pad(t, (0, 0, 0, 0, 0, pad)).reshape(b, c, chunk, H, -1) for t in (B, C))
    doc = torch.zeros(b, T, dtype=torch.long, device=u.device) if doc is None else doc
    doc = torch.cat([doc, doc[:, -1:].expand(b, pad)], 1).reshape(b, c, chunk)
    pairs = same_document(doc[..., :, None], doc[..., None, :])  # b c l s
    y_diag = torch.einsum("bclhn,bcshn,bhcls,bcshp->bclhp", C, B, decay_matrix(a) * pairs[:, None], X)
    a_cum = a.cumsum(-1)
    state = u.new_zeros(b, H, P, B.shape[-1])
    before = doc[:, 0, 0]  # the document the carried state belongs to
    ys = []
    for k in range(c):
        reach = same_document(doc[:, k], before[:, None])  # b l: positions the carried state reaches
        y_off = torch.einsum("blhn,bhpn,bhl->blhp", C[:, k], state, torch.exp(a_cum[:, :, k]) * reach[:, None])
        ys.append(y_diag[:, k] + y_off)
        last = doc[:, k, -1]
        to_end = torch.exp(a_cum[:, :, k, -1:] - a_cum[:, :, k]) * same_document(doc[:, k], last[:, None])[:, None]
        carried = torch.exp(a_cum[:, :, k, -1])[..., None, None] * state * reach[:, -1, None, None, None]
        state = carried + torch.einsum("blhn,bhl,blhp->bhpn", B[:, k], to_end, X[:, k])
        before = last
    return torch.stack(ys, 1).reshape(b, c * chunk, H, P)[:, :T], state


class Reference:
    """A0 or A0-split; fp32 throughout (or `dtype`), or with the native bf16 stores when bf16=True."""

    def __init__(self, model_dir: Path, *, bf16: bool = False, dtype: torch.dtype = torch.float32) -> None:
        model_dir = Path(model_dir)
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self._setup(cfg, load_file(model_dir / "model.safetensors"), bf16=bf16, dtype=dtype)

    @classmethod
    def from_weights(
        cls, cfg: dict, weights: dict[str, torch.Tensor], *, bf16: bool = False, dtype: torch.dtype = torch.float32
    ) -> Reference:
        reference = cls.__new__(cls)
        reference._setup(cfg, weights, bf16=bf16, dtype=dtype)
        return reference

    def _setup(self, cfg: dict, weights: dict[str, torch.Tensor], *, bf16: bool, dtype: torch.dtype) -> None:
        if set(cfg["tlc_layer_pattern"]) != {"M"}:
            raise ValueError(f"only Mamba layers are modelled, not {cfg['tlc_layer_pattern']!r}")
        missing, extra = weight_names(cfg) - set(weights), set(weights) - weight_names(cfg)
        if missing or extra:
            raise ValueError(f"model.safetensors: missing {sorted(missing)}, unexpected {sorted(extra)}")
        self.cfg, self.dtype = cfg, dtype
        self.w = {name: t.detach().to(dtype) for name, t in weights.items()}
        self.round = (lambda t: t.to(torch.bfloat16).to(dtype)) if bf16 else (lambda t: t)
        self.eps = cfg["rms_norm_eps"]
        self.heads, self.head_dim = cfg["mamba_num_heads"], cfg["mamba_head_dim"]
        self.groups, self.state = cfg["n_groups"], cfg["ssm_state_size"]
        self.inner = self.heads * self.head_dim
        self.conv_dim = self.inner + 2 * self.groups * self.state

    @property
    def head(self) -> torch.Tensor:
        return self.w.get(LM_HEAD, self.w[EMBEDDING])

    def p(self, layer: int, field: str) -> torch.Tensor:
        return self.w[layer_key(self.cfg, layer, field)]

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.w[EMBEDDING][ids]

    def logits(self, ids: torch.Tensor, mode: str = "scan", positions: torch.Tensor | None = None) -> torch.Tensor:
        """ids [b,T] -> logits [b,T,V], the full sequence at once; mode "scan" or "chunked"."""
        return self.forward(self.embed(ids), mode, positions)[0]

    def forward(
        self, x: torch.Tensor, mode: str = "scan", positions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Input embeddings [b,T,d] -> (logits, per-layer decode caches (conv window, SSM state))."""
        if mode not in ("scan", "chunked"):
            raise ValueError(f"mode must be 'scan' or 'chunked', got {mode!r}")
        doc = documents(positions)
        residual, caches = torch.zeros_like(x), []
        for layer in range(self.cfg["num_hidden_layers"]):
            residual, h = self._add_norm(residual, x, self.p(layer, "ln1_weight"))
            x, cache = self._mixer(layer, h, mode, doc)
            caches.append(cache)
            residual, h = self._add_norm(residual, x, self.p(layer, "ln2_weight"))
            x = self._mlp(layer, h)
        _, h = self._add_norm(residual, x, self.w[FINAL_NORM])
        return self.round(h @ self.head.T), caches

    def prefill(
        self, ids: torch.Tensor, mode: str = "scan", pad_id: int | None = None
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Logits of a prompt and the caches step() continues from. Rows must not be left-padded: a decode cache
        cannot tell padding from text."""
        if pad_id is not None and ((ids[:, :-1] == pad_id) & (ids[:, 1:] != pad_id)).any():
            raise ValueError("left padding: a pad token before a real token")
        return self.forward(self.embed(ids), mode)

    def step(
        self, ids: torch.Tensor, caches: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """One token per row: ids [b] -> (logits [b,V], the caches after it)."""
        x = self.embed(ids)
        residual, out = torch.zeros_like(x), []
        for layer, cache in enumerate(caches):
            residual, h = self._add_norm(residual, x, self.p(layer, "ln1_weight"))
            x, cache = self._mixer_step(layer, h, cache)
            out.append(cache)
            residual, h = self._add_norm(residual, x, self.p(layer, "ln2_weight"))
            x = self._mlp(layer, h)
        _, h = self._add_norm(residual, x, self.w[FINAL_NORM])
        return self.round(h @ self.head.T), out

    def empty_caches(self, batch: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        kernel = self.cfg["conv_kernel"]
        window = torch.zeros(batch, self.conv_dim, kernel, dtype=self.dtype)
        state = torch.zeros(batch, self.heads, self.head_dim, self.state, dtype=self.dtype)
        return [(window, state) for _ in range(self.cfg["num_hidden_layers"])]

    def step_logits(self, ids: torch.Tensor) -> torch.Tensor:
        """The logits of ids [b,T] computed one token at a time from empty caches."""
        caches, out = self.empty_caches(ids.shape[0]), []
        for t in range(ids.shape[1]):
            logits, caches = self.step(ids[:, t], caches)
            out.append(logits)
        return torch.stack(out, 1)

    def logprobs(
        self, ids: torch.Tensor, targets: torch.Tensor, mode: str = "scan", positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """log p(targets[b, t] | ids[b, :t+1]) as compute_logprobs returns it: 0 where the target is IGNORE."""
        logp = torch.log_softmax(self.logits(ids, mode, positions), -1)
        out = logp.gather(-1, targets.clamp(min=0)[..., None])[..., 0]
        return out.masked_fill(targets == IGNORE, 0.0)

    def _add_norm(self, residual, x, weight):
        total = residual + x
        return self.round(total), self.round(rms_norm(total, weight, self.eps))

    def _split_proj(self, layer, h):
        proj = self.round(h @ self.p(layer, "in_proj_weight").T)
        return proj.split([self.inner, self.conv_dim, self.heads], -1)

    def _ssm_inputs(self, layer, xbc, dt_raw):
        """Conv output [..., conv_dim] and raw dt [..., H] -> u [..., H, P], dt [..., H], B and C [..., H, N]."""
        u, B, C = xbc.split([self.inner, self.groups * self.state, self.groups * self.state], -1)
        per_group = self.heads // self.groups
        B, C = (t.unflatten(-1, (self.groups, self.state)).repeat_interleave(per_group, -2) for t in (B, C))
        dt = F.softplus(dt_raw + self.p(layer, "dt_bias")).clamp(*TIME_STEP_LIMIT)
        return u.unflatten(-1, (self.heads, self.head_dim)), dt, B, C

    def _gate_out(self, layer, y, u, z):
        """D skip (added before the scan's bf16 store), gated group RMSNorm and out_proj."""
        y = self.round(y + u * self.p(layer, "D_param")[:, None]).flatten(-2)
        gated = self.round(y * self.round(F.silu(z)))
        normed = self.round(rms_norm(gated, self.p(layer, "gated_norm_weight"), self.eps, self.groups))
        return self.round(normed @ self.p(layer, "out_proj_weight").T)

    def _mixer(self, layer, h, mode, doc):
        z, xbc_in, dt_raw = self._split_proj(layer, h)
        conv = causal_conv(xbc_in, self.p(layer, "conv_weight"), self.p(layer, "conv_bias"), doc)
        xbc = self.round(F.silu(conv))
        u, dt, B, C = self._ssm_inputs(layer, xbc, dt_raw)
        A = -torch.exp(self.p(layer, "A_log"))
        if mode == "scan":
            y, state = ssm_scan(u, dt, A, B, C, doc)
        else:
            y, state = ssd_chunked(u, dt, A, B, C, self.cfg["chunk_size"], doc)
        window = conv_window(xbc_in, self.cfg["conv_kernel"], doc)
        return self._gate_out(layer, y, u, z), (window, state)

    def _mixer_step(self, layer, h, cache):
        window, state = cache
        z, xbc, dt_raw = self._split_proj(layer, h)
        window = torch.cat([window[..., 1:], xbc[..., None]], -1)
        conv = (window * self.p(layer, "conv_weight")).sum(-1) + self.p(layer, "conv_bias")
        u, dt, B, C = self._ssm_inputs(layer, self.round(F.silu(conv)), dt_raw)
        state, y = ssm_step(state, u, dt, -torch.exp(self.p(layer, "A_log")), B, C)
        return self._gate_out(layer, y, u, z), (window, state)

    def _mlp(self, layer, h):
        up, gate = self.round(h @ self.p(layer, "mlp_up_weight").T).chunk(2, -1)
        return self.round(self.round(up * F.silu(gate)) @ self.p(layer, "mlp_down_weight").T)


def write_golden(model_dir: Path, out: Path, *, batch: int = 2, seq_len: int = 512, seed: int = 0) -> Path:
    """Write the G1(b) goldens for `model_dir` to `out` (.npz), arguments as compute_logprobs takes them.

    input_ids, targets (int32 [batch, seq_len]; targets are the next ids, IGNORE last), logprobs (fp32 reference,
    scan order), logprobs_bf16 (bf16 stores), tolerance = 2 * max |logprobs_bf16 - logprobs|, and meta (JSON).
    """
    model_dir, out = Path(model_dir), Path(out)
    if out.suffix != ".npz":
        raise ValueError(f"{out}: the goldens file name must end in .npz")
    vocab = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["vocab_size"]
    ids = torch.randint(0, vocab, (batch, seq_len), generator=torch.Generator().manual_seed(seed))
    targets = torch.cat([ids[:, 1:], torch.full((batch, 1), IGNORE)], 1)
    with torch.no_grad():
        fp32 = Reference(model_dir).logprobs(ids, targets).numpy()
        bf16 = Reference(model_dir, bf16=True).logprobs(ids, targets).numpy()
    weights = hashlib.sha256((model_dir / "model.safetensors").read_bytes()).hexdigest()
    meta = {"model_safetensors_sha256": weights, "seed": seed, "time_step_limit": TIME_STEP_LIMIT, "order": "scan"}
    np.savez(
        out,
        input_ids=ids.numpy().astype(np.int32),
        targets=targets.numpy().astype(np.int32),
        logprobs=fp32,
        logprobs_bf16=bf16,
        tolerance=np.float32(2 * np.abs(bf16 - fp32).max()),
        meta=np.array(json.dumps(meta)),
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("out", type=Path, help="goldens file (.npz)")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    # The recurrence runs thousands of small ops; intra-op threads only add synchronisation to them.
    torch.set_num_threads(1)
    out = write_golden(args.model_dir, args.out, batch=args.batch, seq_len=args.seq_len, seed=args.seed)
    print(out)


if __name__ == "__main__":
    main()
