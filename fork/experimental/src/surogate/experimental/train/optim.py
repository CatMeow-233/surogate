"""Independent low-state optimizer experiments.

``RankScaledAdamW`` is inspired by the random-projection statistics of
APOLLO-Mini, but implements the mathematical construction below independently.
It is an experimental variant, not the official APOLLO implementation: each
projection refresh also resets its moments and their bias-correction clock.
Only projected statistics persist; the parameter has no FP32 master copy.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from collections.abc import Callable, Iterable
from numbers import Real
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer

_SEED_MODULUS = 1 << 63
_MAX_SEED = _SEED_MODULUS - 1
_PARAMETER_SEED_STRIDE = 0x9E3779B97F4A7C15
_REFRESH_SEED_STRIDE = 0xD1B54A32D192ED03
_ROUNDING_SEED_STRIDE = 0x94D049BB133111EB
_FLOAT_STATE_KEYS = ("exp_avg", "exp_avg_sq", "projector")
_BF16_UPDATE_CHUNK_SIZE = 1 << 20


def _stochastic_round_bfloat16(values: Tensor, generator: torch.Generator) -> Tensor:
    """Independently round FP32 to adjacent BF16 values using random low bits.

    Adding uniform 16-bit integer noise before truncation rounds away from zero
    with probability equal to the discarded mantissa fraction, for either sign.
    Exact BF16 values (including signed zeros) stay exact. NaNs/infinities keep
    their nonfinite class. Finite values outside BF16's finite range saturate;
    unbiased finite rounding is impossible beyond that range.
    """
    if values.dtype != torch.float32:
        raise TypeError("stochastic BF16 rounding requires FP32 input")
    limit = torch.finfo(torch.bfloat16).max
    finite_values = values.clamp(min=-limit, max=limit)
    noise = torch.randint(0, 1 << 16, values.shape, device=values.device, dtype=torch.int32, generator=generator)
    bits = finite_values.view(torch.int32)
    rounded_bits = noise.add_(bits).bitwise_and_(-65536)
    rounded = rounded_bits.view(torch.float32).to(dtype=torch.bfloat16)
    return torch.where(torch.isfinite(values), rounded, values.to(dtype=torch.bfloat16))


def _adam_direction(first: Tensor, second: Tensor, gradient: Tensor, step: int, group: dict[str, Any]) -> Tensor:
    beta1, beta2 = group["betas"]
    first.mul_(beta1).add_(gradient, alpha=1 - beta1)
    second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    corrected_first = first / (1 - beta1**step)
    denominator = (second / (1 - beta2**step)).sqrt_().add_(group["eps"])
    return corrected_first.div_(denominator)


def _project_bfloat16_chunked(gradient: Tensor, basis: Tensor, side: str) -> Tensor:
    """Sketch contiguous BF16 gradients with bounded FP32 input conversions."""
    with torch.autocast(device_type=gradient.device.type, enabled=False):
        if gradient.numel() <= _BF16_UPDATE_CHUNK_SIZE:
            full = gradient.to(dtype=torch.float32)
            return full @ basis if side == "right" else basis.transpose(-2, -1) @ full
        rows, columns = gradient.shape[-2:]
        matrices = math.prod(gradient.shape[:-2])
        rank = basis.shape[-1]
        sketch_shape = (*gradient.shape[:-2], rows, rank) if side == "right" else (*gradient.shape[:-2], rank, columns)
        sketch = torch.zeros(sketch_shape, dtype=torch.float32, device=gradient.device)
        inputs = gradient.view(matrices, rows, columns)
        projectors = basis.view(matrices, basis.shape[-2], rank)
        outputs = sketch.view(matrices, rows, rank) if side == "right" else sketch.view(matrices, rank, columns)
        column_chunk = min(columns, _BF16_UPDATE_CHUNK_SIZE)
        row_chunk = max(1, _BF16_UPDATE_CHUNK_SIZE // column_chunk)
        for matrix in range(matrices):
            for row in range(0, rows, row_chunk):
                row_end = min(rows, row + row_chunk)
                for column in range(0, columns, column_chunk):
                    column_end = min(columns, column + column_chunk)
                    block = inputs[matrix, row:row_end, column:column_end].to(dtype=torch.float32)
                    if side == "right":
                        outputs[matrix, row:row_end].add_(block @ projectors[matrix, column:column_end])
                    else:
                        outputs[matrix, :, column:column_end].add_(projectors[matrix, row:row_end].T @ block)
        return sketch


def _apply_bfloat16_chunk(param: Tensor, direction: Tensor, group: dict[str, Any], generator: torch.Generator) -> None:
    # Combine decay and update before the one BF16 rounding. No persistent master
    # or full-size FP32 parameter/update buffer is constructed.
    updated = param.to(dtype=torch.float32)
    updated.mul_(1 - group["lr"] * group["weight_decay"])
    updated.add_(direction, alpha=-group["lr"])
    param.copy_(_stochastic_round_bfloat16(updated, generator))


def _finite_number(name: str, value: Any, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")


def _validate_group(group: dict[str, Any]) -> None:
    for name in ("lr", "weight_decay", "scale"):
        _finite_number(name, group[name])
    _finite_number("eps", group["eps"], positive=True)
    betas = group["betas"]
    if not isinstance(betas, (tuple, list)) or len(betas) != 2:
        raise ValueError("betas must contain two numbers in [0, 1)")
    for beta in betas:
        _finite_number("beta", beta)
        if beta >= 1:
            raise ValueError("betas must be in [0, 1)")
    for name in ("rank", "update_interval"):
        if isinstance(group[name], bool) or not isinstance(group[name], int) or group[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if isinstance(group["seed"], bool) or not isinstance(group["seed"], int) or not 0 <= group["seed"] <= _MAX_SEED:
        raise ValueError(f"seed must be an integer in [0, {_MAX_SEED}]")
    if not isinstance(group.get("stochastic_rounding", False), bool):
        raise ValueError("stochastic_rounding must be a boolean")


class RankScaledAdamW(Optimizer):
    """Scale each full matrix gradient using Adam statistics of a random sketch.

    For a gradient ``G[..., M, N]`` and rank ``r < min(M, N)``, choose the
    projection side minimizing ``basis.numel() + 2 * sketch.numel()``. A right
    projector has shape ``[..., N, r]`` and produces ``G @ P``; a left projector
    has shape ``[..., M, r]`` and produces ``P.transpose(-2, -1) @ G``. Every
    matrix, including every MoE expert, receives an independent Gaussian basis
    with entries distributed as ``N(0, 1/r)``.

    FP32 Adam moments of the sketch produce a bias-corrected update ``U``.
    The parameter update uses the *original full gradient*, multiplied by the
    per-matrix scalar ``scale * ||U||_F / (||sketch||_F + eps)``. It does not lift
    the sketch back into parameter space. Vectors/scalars and ranks at least
    ``min(M, N)`` use ordinary AdamW with FP32 moments; ``scale`` applies only
    to the projected variant. Weight decay is decoupled in both paths.

    Refreshes occur on gradient-bearing parameter steps 1, 1+update_interval,
    etc. They reset projected moments and their bias-correction step, while
    preserving the total step count. CPU generators derived from parameter and
    refresh seeds leave the global RNG unchanged. Inactive parameters neither
    advance clocks nor receive weight decay. CPU-offloaded state is restored
    lazily when that parameter next has a gradient.

    ``stochastic_rounding=True`` combines each BF16 update with decay in FP32
    chunks of at most 1,048,576 elements before stochastic BF16 rounding. It
    avoids losing sub-ULP updates without retaining FP32 master weights. This
    path requires contiguous BF16 parameters and gradients on CPU or CUDA;
    other parameter dtypes retain their usual update. Its RNG is per parameter
    and step and is independent of the projection and global generators.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 2e-5,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        rank: int = 1,
        update_interval: int = 200,
        scale: float = 1.0,
        seed: int = 42,
        stochastic_rounding: bool = False,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rank=rank,
            update_interval=update_interval,
            scale=scale,
            seed=seed,
            stochastic_rounding=stochastic_rounding,
        )
        _validate_group(defaults)
        self._parameter_indices: dict[Tensor, int] = {}
        self._parameter_groups: dict[Tensor, dict[str, Any]] = {}
        super().__init__(params, defaults)

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        options = {**self.defaults, **param_group}
        _validate_group(options)
        # Materialize once so validation does not consume a generator before
        # Optimizer.add_param_group receives it.
        param_group = dict(param_group)
        params = param_group["params"]
        if isinstance(params, Tensor):
            params = [params]
        elif isinstance(params, set):
            raise TypeError("optimizer parameters must have a deterministic order")
        else:
            params = list(params)
        for param in params:
            if not isinstance(param, Tensor) or not param.is_floating_point():
                raise TypeError("RankScaledAdamW requires real floating-point parameters")
        param_group["params"] = params
        super().add_param_group(param_group)
        group = self.param_groups[-1]
        for param in group["params"]:
            if param not in self._parameter_indices:
                self._parameter_indices[param] = len(self._parameter_indices)
            self._parameter_groups[param] = group

    @staticmethod
    def _projection_shape(param: Tensor, rank: int) -> tuple[str, tuple[int, ...], tuple[int, ...]]:
        rows, columns = param.shape[-2:]
        right_elements = columns * rank + 2 * rows * rank
        left_elements = rows * rank + 2 * columns * rank
        batch = tuple(param.shape[:-2])
        if right_elements <= left_elements:
            return "right", (*batch, columns, rank), (*batch, rows, rank)
        return "left", (*batch, rows, rank), (*batch, rank, columns)

    def _initialize_state(self, param: Tensor, group: dict[str, Any]) -> dict[str, Any]:
        state = self.state[param]
        if not state:
            projected = param.ndim >= 2 and group["rank"] < min(param.shape[-2:])
            state.update(
                step=0,
                moment_step=0,
                seed=(group["seed"] + self._parameter_indices[param] * _PARAMETER_SEED_STRIDE) % _SEED_MODULUS,
                parameter_shape=tuple(param.shape),
                rank=group["rank"],
                projected=projected,
                projection_index=-1,
            )
            if not projected:
                state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                state["exp_avg_sq"] = torch.zeros_like(param, dtype=torch.float32)
        elif tuple(state["parameter_shape"]) != tuple(param.shape) or state["rank"] != group["rank"]:
            raise RuntimeError("parameter shape or projection rank changed; create a new optimizer")
        for key in _FLOAT_STATE_KEYS:
            value = state.get(key)
            if isinstance(value, Tensor):
                state[key] = value.to(device=param.device, dtype=torch.float32)
        return state

    @staticmethod
    def _refresh_projection(param: Tensor, state: dict[str, Any], rank: int) -> None:
        side, basis_shape, moment_shape = RankScaledAdamW._projection_shape(param, rank)
        state["projection_index"] += 1
        refresh_seed = (state["seed"] + state["projection_index"] * _REFRESH_SEED_STRIDE) % _SEED_MODULUS
        generator = torch.Generator(device="cpu")
        generator.manual_seed(refresh_seed)
        basis = torch.randn(basis_shape, generator=generator, dtype=torch.float32, device="cpu")
        basis.div_(math.sqrt(rank))
        state["projector"] = basis.to(device=param.device)
        state["projection_side"] = side
        state["exp_avg"] = torch.zeros(moment_shape, dtype=torch.float32, device=param.device)
        state["exp_avg_sq"] = torch.zeros(moment_shape, dtype=torch.float32, device=param.device)
        state["moment_step"] = 0

    def _step_parameter(self, param: Tensor, group: dict[str, Any]) -> None:
        gradient = param.grad
        if gradient is None:
            return
        if gradient.layout != torch.strided:
            raise RuntimeError("RankScaledAdamW supports dense gradients only")
        if gradient.shape != param.shape:
            raise ValueError("gradient shape must match its parameter")
        if not gradient.is_floating_point():
            raise TypeError("RankScaledAdamW requires real floating-point gradients")
        stochastic = group.get("stochastic_rounding", False) and param.dtype == torch.bfloat16
        if stochastic and (not param.is_contiguous() or not gradient.is_contiguous()):
            raise ValueError("stochastic rounding requires contiguous BF16 parameters and gradients")
        if stochastic and param.device.type not in {"cpu", "cuda"}:
            raise ValueError("stochastic BF16 rounding supports CPU and CUDA parameters")
        state = self._initialize_state(param, group)
        state["step"] += 1
        generator = None
        if stochastic:
            generator = torch.Generator(device=param.device)
            generator.manual_seed((state["seed"] + state["step"] * _ROUNDING_SEED_STRIDE) % _SEED_MODULUS)
        if stochastic and not state["projected"]:
            state["moment_step"] += 1
            param_flat, gradient_flat = param.view(-1), gradient.view(-1)
            first, second = state["exp_avg"].view(-1), state["exp_avg_sq"].view(-1)
            for start in range(0, param.numel(), _BF16_UPDATE_CHUNK_SIZE):
                end = min(param.numel(), start + _BF16_UPDATE_CHUNK_SIZE)
                direction = _adam_direction(
                    first[start:end],
                    second[start:end],
                    gradient_flat[start:end].to(dtype=torch.float32),
                    state["moment_step"],
                    group,
                )
                _apply_bfloat16_chunk(param_flat[start:end], direction, group, generator)
            return

        grad = None if stochastic else gradient.to(dtype=torch.float32)
        if state["projected"]:
            if (state["step"] - 1) % group["update_interval"] == 0:
                self._refresh_projection(param, state, group["rank"])
            basis = state["projector"]
            # A post-accumulate hook can execute inside the caller's autocast
            # context. Keep the sketch statistics in FP32 there as well.
            if stochastic:
                sketch = _project_bfloat16_chunked(gradient, basis, state["projection_side"])
            else:
                with torch.autocast(device_type=param.device.type, enabled=False):
                    sketch = grad @ basis if state["projection_side"] == "right" else basis.transpose(-2, -1) @ grad
        else:
            sketch = grad

        first, second = state["exp_avg"], state["exp_avg_sq"]
        state["moment_step"] += 1
        adam_update = _adam_direction(first, second, sketch, state["moment_step"], group)
        if state["projected"]:
            update_norm = torch.linalg.vector_norm(adam_update, dim=(-2, -1), keepdim=True)
            sketch_norm = torch.linalg.vector_norm(sketch, dim=(-2, -1), keepdim=True)
            factor = update_norm / (sketch_norm + group["eps"])
            if stochastic:
                factors = (factor * group["scale"]).view(-1)
                matrix_elements = math.prod(param.shape[-2:])
                param_flat, gradient_flat = param.view(-1), gradient.view(-1)
                for matrix in range(factors.numel()):
                    matrix_start = matrix * matrix_elements
                    for offset in range(0, matrix_elements, _BF16_UPDATE_CHUNK_SIZE):
                        start = matrix_start + offset
                        end = matrix_start + min(matrix_elements, offset + _BF16_UPDATE_CHUNK_SIZE)
                        # Device scalar broadcast: do not .item() each expert's
                        # factor and introduce a host synchronization.
                        direction = gradient_flat[start:end].to(dtype=torch.float32) * factors[matrix]
                        _apply_bfloat16_chunk(param_flat[start:end], direction, group, generator)
                return
            # Out-of-place multiplication preserves the caller's .grad even
            # when it already has FP32 dtype and grad shares its storage.
            update = grad * (factor * group["scale"])
        else:
            update = adam_update
        if group["weight_decay"]:
            param.mul_(1 - group["lr"] * group["weight_decay"])
        param.add_(update, alpha=-group["lr"])

    @torch.no_grad()
    def step_parameter(self, param: Tensor) -> None:
        """Update one registered parameter, for a post-accumulate-grad hook."""
        try:
            group = self._parameter_groups[param]
        except KeyError as error:
            raise ValueError("parameter is not registered with this optimizer") from error
        self._step_parameter(param, group)

    @torch.no_grad()
    def step(self, closure: Callable[[], Any] | None = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for param in group["params"]:
                self._step_parameter(param, group)
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Preserve FP32 statistics, CPU offload, and scalar/string metadata.

        The base loader casts tensors to the parameter device/dtype and can
        transform iterable metadata such as strings. Use its group/ID validation
        with empty parameter states, then clone the full original state. CPU
        checkpoints remain on CPU until a gradient-bearing step restores them.
        Run load hooks around the complete restored state, in their usual order.
        """
        checkpoint = state_dict.copy()
        pre_hooks = self._optimizer_load_state_dict_pre_hooks
        post_hooks = self._optimizer_load_state_dict_post_hooks
        for hook in pre_hooks.values():
            replacement = hook(self, checkpoint)
            if replacement is not None:
                checkpoint = replacement
        saved_ids = [identifier for group in checkpoint["param_groups"] for identifier in group["params"]]
        params = [param for group in self.param_groups for param in group["params"]]
        parameter_ids = set(saved_ids)
        metadata_only = {
            **checkpoint,
            "state": {
                identifier: {} if identifier in parameter_ids else value
                for identifier, value in checkpoint["state"].items()
            },
        }
        # Avoid both duplicate pre-hooks and post-hooks observing incomplete state.
        self._optimizer_load_state_dict_pre_hooks = OrderedDict()
        self._optimizer_load_state_dict_post_hooks = OrderedDict()
        try:
            super().load_state_dict(metadata_only)
        finally:
            self._optimizer_load_state_dict_pre_hooks = pre_hooks
            self._optimizer_load_state_dict_post_hooks = post_hooks
        for identifier, param in zip(saved_ids, params, strict=True):
            if identifier not in checkpoint["state"]:
                continue
            saved = checkpoint["state"][identifier]
            self.state[param] = {
                key: value.detach().to(dtype=torch.float32).clone()
                if key in _FLOAT_STATE_KEYS and isinstance(value, Tensor)
                else copy.deepcopy(value)
                for key, value in saved.items()
            }
        self._parameter_groups = {param: group for group in self.param_groups for param in group["params"]}
        for group in self.param_groups:
            _validate_group(group)
        for hook in post_hooks.values():
            hook(self)
