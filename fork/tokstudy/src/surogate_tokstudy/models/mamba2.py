"""Mamba-2 layers in plain PyTorch: the mixer of TLC's Mamba rows, the SwiGLU MLP, pre-norm layers and stacks with
a network-final RMSNorm. Imports only torch and the package's kernels, so the file moves whole into a project that
promotes an H-Net arm (spec §1.3).

The mixer computes what fork/tlc/prototypes/ref_mamba2.py computes (gate TW-1, tests/cpu/test_models.py):
in_proj output [z | x B C | dt]; depthwise causal conv over x B C with bias, then SiLU; dt = clamp(softplus(dt_raw +
dt_bias), TIME_STEP_LIMIT) (softplus linear above 20); A = -exp(A_log); y = SSD(x, dt, A, B, C) + D x; RMSNorm of
y * silu(z) with one RMS per group of H*P/G channels; out_proj. The SwiGLU reads the fused [up | gate] weight:
down(up * silu(gate)). A layer is x + mixer(norm1(x)), then x + mlp(norm2(x)) when it has an MLP (the H-Net encoder
and decoder layers have none, as the lower-case `m` layers of the H-Net code [verified, goombalab/hnet
modules/isotropic.py]); a stack ends with an RMSNorm. Norms evaluate in fp32 (fp64 for fp64) and return that dtype.

Initialisation is TLC's scheme (surogate_tlc.init, after state-spaces/mamba): dt_bias = softplus^-1(dt) with dt =
exp(U(ln 1e-3, ln 1e-1)) floored at 1e-4; A_log = ln U(1, 16); D = 1; conv weight and bias U(+-1/sqrt(K)); linear
weights U(+-1/sqrt(fan_in)), divided by sqrt(residual writes of the stack) for the two that write into the residual
stream (out_proj, the MLP's down); norm weights 1. The draws come from one torch.Generator in the stack's parameter
order, so they are not TLC's per-tensor draws; a twin that must start from a native initialisation loads it from a
TLC model directory (twin_a0.A0Twin.from_tlc_model_dir).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .. import kernels

# dt clamp of the native scan (ref_mamba2.TIME_STEP_LIMIT); kept equal so the twin computes the native function.
TIME_STEP_LIMIT = (0.0, 1e9)
DT_MIN, DT_MAX, DT_INIT_FLOOR = 1e-3, 1e-1, 1e-4
A_INIT_RANGE = (1.0, 16.0)


@dataclass(frozen=True)
class Mamba2Geometry:
    heads: int
    head_dim: int
    state: int
    groups: int = 1
    conv_kernel: int = 4
    chunk: int = 128

    def __post_init__(self):
        for name in ("heads", "head_dim", "state", "groups", "conv_kernel", "chunk"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"Mamba2Geometry.{name} must be a positive int, got {value!r}")
        if self.heads % self.groups:
            raise ValueError(f"{self.heads} heads do not split into {self.groups} groups")

    @property
    def inner(self) -> int:
        return self.heads * self.head_dim

    @property
    def conv_dim(self) -> int:
        return self.inner + 2 * self.groups * self.state

    @classmethod
    def for_width(
        cls, d: int, *, expand: int = 2, head_dim: int = 64, state: int = 64, groups: int = 1, chunk: int = 128
    ) -> Mamba2Geometry:
        """inner width = expand * d in heads of head_dim, conv kernel 4 (TLC's A0: d 384 -> 12 heads of 64)."""
        if (expand * d) % head_dim:
            raise ValueError(f"inner width {expand * d} is not a multiple of head_dim {head_dim}")
        return cls(heads=expand * d // head_dim, head_dim=head_dim, state=state, groups=groups, chunk=chunk)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float, groups: int = 1):
        super().__init__()
        if d % groups:
            raise ValueError(f"width {d} does not split into {groups} groups")
        self.eps, self.groups = float(eps), groups
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ct = kernels.compute_dtype(x.dtype)
        xs = x.to(ct).unflatten(-1, (self.groups, -1))
        normed = (xs * torch.rsqrt(xs.pow(2).mean(-1, keepdim=True) + self.eps)).flatten(-2)
        return normed * self.weight.to(ct)


class Mamba2Mixer(nn.Module):
    def __init__(self, d: int, geometry: Mamba2Geometry, eps: float, *, backend: str = "torch"):
        super().__init__()
        kernels._check_backend(backend)
        g = self.geometry = geometry
        self.backend = backend
        self.in_proj = nn.Linear(d, 2 * g.inner + 2 * g.groups * g.state + g.heads, bias=False)
        self.conv_weight = nn.Parameter(torch.empty(g.conv_dim, g.conv_kernel))
        self.conv_bias = nn.Parameter(torch.empty(g.conv_dim))
        self.dt_bias = nn.Parameter(torch.empty(g.heads))
        self.A_log = nn.Parameter(torch.empty(g.heads))
        self.D = nn.Parameter(torch.empty(g.heads))
        self.norm = RMSNorm(g.inner, eps, groups=g.groups)
        self.out_proj = nn.Linear(g.inner, d, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        g = self.geometry
        z, xbc, dt_raw = self.in_proj(h).split([g.inner, g.conv_dim, g.heads], -1)
        xbc = F.silu(kernels.causal_conv(xbc, self.conv_weight.to(xbc.dtype), self.conv_bias.to(xbc.dtype)))
        u, B, C = xbc.split([g.inner, g.groups * g.state, g.groups * g.state], -1)
        u = u.unflatten(-1, (g.heads, g.head_dim))
        B, C = (t.unflatten(-1, (g.groups, g.state)) for t in (B, C))
        ct = kernels.compute_dtype(h.dtype)
        dt = F.softplus(dt_raw.to(ct) + self.dt_bias.to(ct)).clamp(*TIME_STEP_LIMIT)
        A = -torch.exp(self.A_log.to(ct))
        y = kernels.ssd(u, dt, A, B, C, chunk=g.chunk, backend=self.backend).to(ct)
        y = (y + u.to(ct) * self.D.to(ct)[:, None]).flatten(-2)
        gated = self.norm(y * F.silu(z.to(ct)))
        return self.out_proj(gated.to(h.dtype))


class SwiGLU(nn.Module):
    def __init__(self, d: int, ffn: int):
        super().__init__()
        self.up = nn.Linear(d, 2 * ffn, bias=False)  # fused [up | gate], TLC's mlp_up_weight
        self.down = nn.Linear(ffn, d, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        up, gate = self.up(h).chunk(2, -1)
        return self.down(up * F.silu(gate))


class Mamba2Layer(nn.Module):
    def __init__(self, d: int, geometry: Mamba2Geometry, eps: float, ffn: int | None, *, backend: str = "torch"):
        super().__init__()
        self.norm1 = RMSNorm(d, eps)
        self.mixer = Mamba2Mixer(d, geometry, eps, backend=backend)
        self.norm2 = RMSNorm(d, eps) if ffn else None
        self.mlp = SwiGLU(d, ffn) if ffn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self._in(self.norm1, x))
        if self.mlp is not None:
            x = x + self.mlp(self._in(self.norm2, x))
        return x

    @staticmethod
    def _in(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
        # the norm evaluates in fp32; a matmul reads it in the residual stream's dtype (autocast decides under it)
        return norm(x).to(x.dtype)


class Mamba2Stack(nn.Module):
    """`n_layers` Mamba-2 layers (with a SwiGLU of width `ffn`, or none) and a final RMSNorm."""

    def __init__(
        self,
        d: int,
        n_layers: int,
        geometry: Mamba2Geometry,
        *,
        eps: float = 1e-5,
        ffn: int | None = None,
        backend: str = "torch",
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"a stack has at least one layer, got {n_layers}")
        self.d_model, self.geometry, self.ffn = d, geometry, ffn
        self.layers = nn.ModuleList(Mamba2Layer(d, geometry, eps, ffn, backend=backend) for _ in range(n_layers))
        self.final_norm = RMSNorm(d, eps)

    @property
    def residual_writes(self) -> int:
        return len(self.layers) * (2 if self.ffn else 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h)
        return self.final_norm(h)

    @torch.no_grad()
    def reset_parameters(self, generator: torch.Generator) -> None:
        """TLC's initialisation scheme (module docstring), drawn from `generator` in parameter order."""
        scale = 1.0 / math.sqrt(self.residual_writes)
        for layer in self.layers:
            init_mixer(layer.mixer, generator, residual_scale=scale)
            layer.norm1.weight.fill_(1.0)
            if layer.mlp is not None:
                layer.norm2.weight.fill_(1.0)
                uniform_linear(layer.mlp.up.weight, generator)
                uniform_linear(layer.mlp.down.weight, generator, scale=scale)
        self.final_norm.weight.fill_(1.0)


def uniform_linear(weight: torch.Tensor, generator: torch.Generator, *, scale: float = 1.0) -> None:
    bound = scale / math.sqrt(weight.shape[1])
    _fill(weight, torch.empty(weight.shape, dtype=torch.float32).uniform_(-bound, bound, generator=generator))


def init_mixer(mixer: Mamba2Mixer, generator: torch.Generator, *, residual_scale: float) -> None:
    g = mixer.geometry
    uniform_linear(mixer.in_proj.weight, generator)
    bound = g.conv_kernel**-0.5
    for p in (mixer.conv_weight, mixer.conv_bias):
        _fill(p, torch.empty(p.shape, dtype=torch.float32).uniform_(-bound, bound, generator=generator))
    log_dt = torch.empty(g.heads, dtype=torch.float64).uniform_(math.log(DT_MIN), math.log(DT_MAX), generator=generator)
    dt = log_dt.exp().clamp(min=DT_INIT_FLOOR)
    _fill(mixer.dt_bias, dt + torch.log(-torch.expm1(-dt)))  # softplus^-1
    _fill(mixer.A_log, torch.empty(g.heads, dtype=torch.float64).uniform_(*A_INIT_RANGE, generator=generator).log())
    mixer.D.fill_(1.0)
    mixer.norm.weight.fill_(1.0)
    uniform_linear(mixer.out_proj.weight, generator, scale=residual_scale)


def _fill(param: torch.Tensor, value: torch.Tensor) -> None:
    param.copy_(value.to(param.dtype))
