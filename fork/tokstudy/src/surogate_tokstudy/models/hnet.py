"""H-Net, one stage: learned dynamic chunking of bytes around a main network (Hwang, Wang & Gu, 2507.07955, §2.1-2.3,
eqs. (1)-(10) [paper]; details the paper leaves open follow the authors' code, goombalab/hnet modules/dc.py and
models/hnet.py [verified]). Imports only torch, the package's kernels and Mamba-2 layers (and the codec constants).

    x_hat  = E(embed(bytes))                                 encoder: Mamba-2 layers, final RMSNorm
    p_t    = (1 - cos(W_q x_hat_t, W_k x_hat_{t-1})) / 2,  p_1 = 1;   b_t = [p_t >= 0.5]       (eq. 4)
    x_c    = x_hat at the boundaries, P = p there            downsample (right-padded to the batch's chunk count)
    z_c    = WidthDown(M(WidthUp(x_c)))                      main network, final RMSNorm included
    zbar_k = P_k z_c,k + (1 - P_k) zbar_{k-1}                smoothing EMA over chunks (eq. 5)
    z_t    = STE(c_t) zbar_{chunk(t)} + Linear(x_hat_t)      c_t = p_t if b_t else 1 - p_t   (eqs. 6-9, 3)
    logits = D(z) E_byte^T                                   decoder: Mamba-2 layers, final RMSNorm; tied byte head
    L      = CE + alpha * N/(N-1) ((N-1) F G + (1 - F)(1 - G)),  F = mean b, G = mean p        (eq. 10, alpha 0.03)

Choices where the paper is silent, as the authors' code makes them: W_q and W_k start as the identity; the dechunk
clamps P to [1e-4, 1 - 1e-4] (EMA_P_MIN) before the EMA; the residual Linear starts at zero and runs in fp32; the
widening vector starts at zero and belongs to the main stage (its LR multiplier). Where they differ: the residual
Linear has no bias (the code's has one, default-initialised; the paper writes Linear(x_hat) "initialized close to
0"); b_t uses p_t >= 0.5 as the paper writes (the code's argmax makes p_t = 0.5 no boundary); STE returns exactly
1 in the forward pass (an autograd function, as in the code; c + (1 - c).detach() can round to 1 - ulp).

Forced boundaries (`forced`, built by the training stream from the special ids, route changes or lexeme starts) set
p_t = 1, hence b_t = 1, with no gradient through p_t there. Rows are right-padded: `valid` marks the real positions;
padding is never a boundary, is left out of F and G, and its outputs are not to be read. The router runs in fp32
(fp64 for fp64 weights) with autocast off, because b_t is discontinuous at 0.5.

Causality (spec §6.5, tests/cpu/test_models.py): b_t reads x_hat_t and x_hat_{t-1}; chunk k's main output reads
chunks <= k, whose positions are <= t for every t in chunk k; the EMA and the repeat are causal; so the logits at t
(which predict byte t + 1) read bytes <= t, with or without forced boundaries, alone or in a batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .. import kernels
from ..codecs import BYTE_VOCAB
from .mamba2 import Mamba2Geometry, Mamba2Stack

ALPHA = 0.03
EMA_P_MIN = 1e-4
EMBEDDING_STD = 0.02


@dataclass(frozen=True)
class HNetConfig:
    """The outer stage. The main network is passed separately (A0Twin or BilingualTrunk, built without a vocabulary);
    its width d1 must be >= d0 (widths are monotone, §2.1.2)."""

    d0: int
    encoder_layers: int
    decoder_layers: int
    ratio: float  # target bytes per chunk, N > 1
    alpha: float = ALPHA
    vocab_size: int = BYTE_VOCAB
    expand: int = 2  # outer Mamba-2 layers: inner width expand * d0 in heads of head_dim, state, kernel 4
    head_dim: int = 64
    state: int = 64
    ssd_chunk: int = 128
    eps: float = 1e-5
    chunk_multiple: int = 1  # pad the chunk axis to a multiple (fewer distinct shapes, e.g. RoPE tables)
    backend: str = "torch"

    def __post_init__(self):
        if not self.ratio > 1:
            raise ValueError(f"the target ratio N must exceed 1, got {self.ratio}")
        if self.encoder_layers < 1 or self.decoder_layers < 1:
            raise ValueError("the encoder and the decoder have at least one layer each")
        if self.vocab_size < BYTE_VOCAB:
            raise ValueError(f"an H-Net reads byte ids: vocab_size >= {BYTE_VOCAB}, got {self.vocab_size}")
        if self.chunk_multiple < 1:
            raise ValueError(f"chunk_multiple must be positive, got {self.chunk_multiple}")
        kernels._check_backend(self.backend)

    def geometry(self) -> Mamba2Geometry:
        return Mamba2Geometry.for_width(
            self.d0, expand=self.expand, head_dim=self.head_dim, state=self.state, chunk=self.ssd_chunk
        )


class Router(nn.Module):
    """p_t = (1 - cos(q_t, k_{t-1})) / 2 with q = W_q x_hat, k = W_k x_hat; p_1 = 1 (eq. 4). W_q, W_k start at I."""

    def __init__(self, d: int):
        super().__init__()
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for linear in (self.W_q, self.W_k):
            linear.weight.copy_(torch.eye(linear.weight.shape[0]))

    def forward(self, x_hat: torch.Tensor) -> torch.Tensor:
        ct = kernels.compute_dtype(self.W_q.weight.dtype)
        with torch.autocast(device_type=x_hat.device.type, enabled=False):
            x = x_hat.to(ct)
            q = F.normalize(F.linear(x[:, 1:], self.W_q.weight.to(ct)), dim=-1)
            k = F.normalize(F.linear(x[:, :-1], self.W_k.weight.to(ct)), dim=-1)
            p = ((1 - (q * k).sum(-1)) / 2).clamp(0.0, 1.0)
            return F.pad(p, (1, 0), value=1.0)


def boundaries(
    p: torch.Tensor, forced: torch.Tensor | None = None, valid: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """(b, p): b = p >= 0.5, with p := 1 where forced and at the first position (p_1 = 1 by definition, so b_1 = 1);
    padding (not valid) is never a boundary."""
    first = torch.zeros_like(p, dtype=torch.bool)
    first[:, 0] = True
    if forced is not None:
        if forced.shape != p.shape or forced.dtype != torch.bool:
            raise ValueError(f"forced must be a bool tensor of shape {list(p.shape)}")
        first = first | forced
    p = torch.where(first, torch.ones_like(p), p)
    b = p >= 0.5
    if valid is not None:
        b = b & valid
    return b, p


def check_valid(valid: torch.Tensor, shape: torch.Size) -> None:
    """`valid` [B, L] marks real positions: each row a non-empty prefix (right padding)."""
    if valid.shape != shape or valid.dtype != torch.bool:
        raise ValueError(f"valid must be a bool tensor of shape {list(shape)}")
    if not bool(valid[:, 0].all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("valid must mark a non-empty prefix of every row (rows are right-padded)")


def chunk(
    x_hat: torch.Tensor, b: torch.Tensor, p: torch.Tensor, *, multiple: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the vectors at the boundaries, in order: (x_c [B, M, d], P_c [B, M], lengths [B]), M the batch's largest
    chunk count rounded up to `multiple`; padding chunks are zero vectors with P = 1 and come after every real one."""
    lengths = b.sum(-1)
    L = b.shape[1]
    M = int(lengths.max())
    M = -(-M // multiple) * multiple
    order = torch.argsort(torch.arange(L, device=b.device) + (~b).long() * L, dim=1, stable=True)
    index = F.pad(order, (0, max(0, M - L)))[:, :M]
    real = torch.arange(M, device=b.device)[None, :] < lengths[:, None]
    x_c = torch.gather(x_hat, 1, index[..., None].expand(-1, -1, x_hat.shape[-1])) * real[..., None]
    P_c = torch.where(real, torch.gather(p, 1, index), torch.ones_like(p[:, :1]))
    return x_c, P_c, lengths


def upsample(zbar: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """z_t = zbar of the most recent chunk at or before t (eq. 8): the chunk index is cumsum(b)_t - 1."""
    index = (b.long().cumsum(-1) - 1).clamp(min=0)
    return torch.gather(zbar, 1, index[..., None].expand(-1, -1, zbar.shape[-1]))


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c):
        return torch.ones_like(c)

    @staticmethod
    def backward(ctx, grad):
        return grad


def ste(c: torch.Tensor) -> torch.Tensor:
    """Forward value exactly 1, gradient that of c (eq. 7)."""
    return _STE.apply(c)


def dechunk(
    z_c: torch.Tensor, P_c: torch.Tensor, b: torch.Tensor, p: torch.Tensor, *, backend: str = "torch"
) -> torch.Tensor:
    """STE(c_t) * upsample(EMA(z_c, P_c))_t, with P clamped to [EMA_P_MIN, 1 - EMA_P_MIN] (module docstring)."""
    zbar = kernels.ema_scan(z_c, P_c.clamp(EMA_P_MIN, 1 - EMA_P_MIN), backend=backend)
    c = torch.where(b, p, 1 - p)
    return upsample(zbar, b) * ste(c).to(zbar.dtype)[..., None]


def ratio_stats(
    b: torch.Tensor, p: torch.Tensor, valid: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """F (fraction of boundaries, no gradient) and G (mean p) over the valid positions."""
    if valid is None:
        return b.to(p.dtype).mean(), p.mean()
    weight = valid.to(p.dtype)
    count = weight.sum()
    return (b.to(p.dtype) * weight).sum() / count, (p * weight).sum() / count


def ratio_loss(b: torch.Tensor, p: torch.Tensor, N: float, valid: torch.Tensor | None = None) -> torch.Tensor:
    """N/(N-1) ((N-1) F G + (1-F)(1-G)) (eq. 10); 1 at F = G = 1/N."""
    return ratio_from_stats(*ratio_stats(b, p, valid), N)


def ratio_from_stats(F_: torch.Tensor, G: torch.Tensor, N: float) -> torch.Tensor:
    return N / (N - 1) * ((N - 1) * F_ * G + (1 - F_) * (1 - G))


class WidthUp(nn.Module):
    """d0 -> d1: append one shared trainable vector of size d1 - d0 (zero at the start)."""

    def __init__(self, d0: int, d1: int):
        super().__init__()
        if d1 < d0:
            raise ValueError(f"widths are monotone: the main width {d1} is below the outer width {d0}")
        self.extra = nn.Parameter(torch.zeros(d1 - d0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.extra.to(x.dtype).expand(*x.shape[:-1], -1)], -1)


class WidthDown(nn.Module):
    """d1 -> d0: the first d0 dimensions."""

    def __init__(self, d0: int):
        super().__init__()
        self.d0 = d0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : self.d0]


@dataclass
class HNetOut:
    logits: torch.Tensor  # [B, L, V]; position t predicts byte t + 1
    ratio_loss: torch.Tensor  # scalar, eq. 10 (the training loss adds alpha times it)
    F: torch.Tensor  # fraction of valid positions that are boundaries (no gradient)
    G: torch.Tensor  # mean boundary probability over valid positions
    lengths: torch.Tensor  # [B] chunk count of each row: the main network's realized positions
    b: torch.Tensor  # [B, L] boundaries
    p: torch.Tensor  # [B, L] boundary probabilities (1 where forced)


class HNetLM(nn.Module):
    def __init__(self, main: nn.Module, cfg: HNetConfig, *, init_seed: int = 0):
        super().__init__()
        self.cfg = cfg
        geometry = cfg.geometry()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d0)
        self.encoder = Mamba2Stack(cfg.d0, cfg.encoder_layers, geometry, eps=cfg.eps, backend=cfg.backend)
        self.router = Router(cfg.d0)
        self.width_up = WidthUp(cfg.d0, main.d_model)
        self.main = main
        self.width_down = WidthDown(cfg.d0)
        self.residual = nn.Linear(cfg.d0, cfg.d0, bias=False)
        self.decoder = Mamba2Stack(cfg.d0, cfg.decoder_layers, geometry, eps=cfg.eps, backend=cfg.backend)
        self.reset_outer(init_seed)

    @torch.no_grad()
    def reset_outer(self, init_seed: int) -> None:
        """Initialise the outer stage (the main network keeps its own initialisation): byte embedding N(0, 0.02),
        encoder and decoder by the Mamba-2 scheme (models.mamba2), router identity, residual and widening zero."""
        generator = torch.Generator().manual_seed(int(init_seed))
        embedding = torch.empty(self.embedding.weight.shape).normal_(0.0, EMBEDDING_STD, generator=generator)
        self.embedding.weight.copy_(embedding)
        self.encoder.reset_parameters(generator)
        self.decoder.reset_parameters(generator)
        self.router.reset_parameters()
        self.residual.weight.zero_()
        self.width_up.extra.zero_()

    def forward(
        self, ids: torch.Tensor, forced: torch.Tensor | None = None, valid: torch.Tensor | None = None
    ) -> HNetOut:
        if valid is not None:
            check_valid(valid, ids.shape)
        x_hat = self.encoder(self.embedding(ids))
        ct = kernels.compute_dtype(self.residual.weight.dtype)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            residual = F.linear(x_hat.to(ct), self.residual.weight.to(ct))
        b, p = boundaries(self.router(x_hat), forced, valid)
        x_c, P_c, lengths = chunk(x_hat, b, p, multiple=self.cfg.chunk_multiple)
        z_c = self.width_down(self.main.forward_hidden(self.width_up(x_c)))
        z = dechunk(z_c, P_c, b, p, backend=self.cfg.backend).to(ct) + residual
        h = self.decoder(z.to(x_hat.dtype))
        weight = self.embedding.weight
        logits = F.linear(h.to(weight.dtype), weight)
        F_, G = ratio_stats(b, p, valid)
        return HNetOut(logits, ratio_from_stats(F_, G, self.cfg.ratio), F_.detach(), G, lengths, b, p)

    def parameter_stages(self) -> dict[str, list[nn.Parameter]]:
        """Trainable parameters by stage, for per-stage LR multipliers: "main" is the main network and the widening
        vector (the authors' code gives it the inner stage's multiplier), "outer" everything else."""
        main = {id(p) for p in self.main.parameters()} | {id(self.width_up.extra)}
        stages: dict[str, list[nn.Parameter]] = {"outer": [], "main": []}
        for param in self.parameters():
            if param.requires_grad:
                stages["main" if id(param) in main else "outer"].append(param)
        return stages
