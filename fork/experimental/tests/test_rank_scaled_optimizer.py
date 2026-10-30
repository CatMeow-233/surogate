"""Numerical and checkpoint contracts for the independent optimizer prototype."""

from __future__ import annotations

import copy
import io
import math

import numpy as np
import pytest
import torch
from surogate.experimental.train import optim as optim_module
from surogate.experimental.train.optim import RankScaledAdamW


def _values(shape, offset=0.0):
    return (torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) % 11 - 5) / 7 + offset


def _numpy_sketch(gradient, basis, side):
    if side == "right":
        return np.matmul(gradient, basis)
    return np.matmul(np.swapaxes(basis, -2, -1), gradient)


@pytest.mark.parametrize("shape", [(3, 7), (7, 3), (2, 3, 7), (2, 7, 3)])
def test_projected_statistics_scale_the_original_full_gradient(shape):
    parameter = torch.nn.Parameter(_values(shape, offset=0.3))
    options = dict(lr=0.03, betas=(0.7, 0.8), eps=1e-5, weight_decay=0.2, rank=2, scale=0.6)
    optimizer = RankScaledAdamW([parameter], update_interval=10, **options)
    expected = parameter.detach().double().numpy().copy()
    first = second = None
    for step in range(1, 4):
        gradient = _values(shape, offset=step * 0.17)
        original_gradient = gradient.clone()
        parameter.grad = gradient
        optimizer.step()
        state = optimizer.state[parameter]
        basis = state["projector"].double().numpy()
        grad_np = original_gradient.double().numpy()
        sketch = _numpy_sketch(grad_np, basis, state["projection_side"])
        if first is None:
            first, second = np.zeros_like(sketch), np.zeros_like(sketch)
        beta1, beta2 = options["betas"]
        first = beta1 * first + (1 - beta1) * sketch
        second = beta2 * second + (1 - beta2) * sketch**2
        corrected = first / (1 - beta1**step)
        adam_update = corrected / (np.sqrt(second / (1 - beta2**step)) + options["eps"])
        factor = np.linalg.norm(adam_update, axis=(-2, -1), keepdims=True) / (
            np.linalg.norm(sketch, axis=(-2, -1), keepdims=True) + options["eps"]
        )
        expected = expected * (1 - options["lr"] * options["weight_decay"])
        expected -= options["lr"] * options["scale"] * grad_np * factor
        np.testing.assert_allclose(parameter.detach().numpy(), expected, rtol=2e-6, atol=2e-7)
        # The optimizer must not overwrite a FP32 gradient through an aliased cast.
        torch.testing.assert_close(parameter.grad, original_gradient, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("shape", "side"), [((8, 32), "right"), ((32, 8), "left"), ((4, 8, 32), "right"), ((4, 32, 8), "left")]
)
def test_state_counts_include_basis_and_two_moments_for_every_expert(shape, side):
    parameter = torch.nn.Parameter(_values(shape))
    optimizer = RankScaledAdamW([parameter], rank=2)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["projection_side"] == side
    float_state = [value for value in state.values() if isinstance(value, torch.Tensor)]
    assert len(float_state) == 3  # the basis and two moments; no full parameter master
    assert all(value.dtype == torch.float32 for value in float_state)
    matrices = math.prod(shape[:-2])
    rows, columns = shape[-2:]
    expected = matrices * 2 * (max(rows, columns) + 2 * min(rows, columns))
    assert sum(value.numel() for value in float_state) == expected
    assert expected < 2 * parameter.numel()


def test_square_projection_tie_chooses_right_and_rank_one_is_default():
    parameter = torch.nn.Parameter(_values((8, 8)))
    optimizer = RankScaledAdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["rank"] == 1
    assert state["projection_side"] == "right"
    assert state["projector"].shape == (8, 1)


@pytest.mark.parametrize("shape", [(), (7,), (2, 3), (2, 2, 3)])
def test_fallback_matches_adamw_and_uses_fp32_state(shape):
    parameter = torch.nn.Parameter(_values(shape, offset=0.5))
    reference = torch.nn.Parameter(parameter.detach().clone())
    options = dict(lr=0.03, betas=(0.6, 0.9), eps=1e-6, weight_decay=0.1)
    optimizer = RankScaledAdamW([parameter], rank=2, **options)
    baseline = torch.optim.AdamW([reference], **options)
    for step in range(1, 5):
        gradient = _values(shape, offset=0.1 * step)
        parameter.grad, reference.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        baseline.step()
        torch.testing.assert_close(parameter, reference, rtol=1e-6, atol=2e-7)
    state = optimizer.state[parameter]
    assert not state["projected"]
    assert "projector" not in state
    assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32
    assert state["exp_avg"].numel() + state["exp_avg_sq"].numel() == 2 * parameter.numel()


def test_refresh_resets_projected_moments_and_bias_correction_only():
    parameter = torch.nn.Parameter(_values((5, 9)))
    optimizer = RankScaledAdamW([parameter], update_interval=2, betas=(0.8, 0.9), rank=2)
    for step in range(1, 3):
        parameter.grad = _values((5, 9), offset=step)
        optimizer.step()
    old_basis = optimizer.state[parameter]["projector"].clone()
    gradient = _values((5, 9), offset=-0.25)
    parameter.grad = gradient
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["step"] == 3
    assert state["moment_step"] == 1
    assert state["projection_index"] == 1
    assert not torch.equal(old_basis, state["projector"])
    projected = gradient @ state["projector"]
    torch.testing.assert_close(state["exp_avg"], 0.2 * projected)
    torch.testing.assert_close(state["exp_avg_sq"], 0.1 * projected.square())


def test_zero_gradients_are_finite_even_after_nonzero_moments():
    parameter = torch.nn.Parameter(_values((6, 10), offset=0.2))
    optimizer = RankScaledAdamW([parameter], lr=0.1, weight_decay=0.3)
    parameter.grad = torch.zeros_like(parameter)
    before = parameter.detach().clone()
    optimizer.step()
    torch.testing.assert_close(parameter, before * 0.97)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = parameter.detach().clone()
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    assert torch.isfinite(parameter).all()
    torch.testing.assert_close(parameter, before * 0.97)


def test_no_gradient_does_not_decay_or_advance_its_schedule():
    parameter = torch.nn.Parameter(_values((4, 9)))
    inactive = torch.nn.Parameter(_values((4, 9), offset=1))
    optimizer = RankScaledAdamW([parameter, inactive], lr=0.1, weight_decay=0.2, update_interval=2)
    inactive_before = inactive.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert inactive not in optimizer.state
    torch.testing.assert_close(inactive, inactive_before, rtol=0, atol=0)
    parameter.grad = None
    state_before = copy.deepcopy(optimizer.state[parameter])
    before = parameter.detach().clone()
    optimizer.step()
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    assert optimizer.state[parameter]["step"] == state_before["step"]
    assert optimizer.state[parameter]["moment_step"] == state_before["moment_step"]
    torch.testing.assert_close(optimizer.state[parameter]["projector"], state_before["projector"], rtol=0, atol=0)


def test_projection_uses_seeded_cpu_generator_without_affecting_global_rng():
    parameter = torch.nn.Parameter(_values((4, 9)))
    optimizer = RankScaledAdamW([parameter], update_interval=1, seed=123)
    rng_before = torch.random.get_rng_state().clone()
    bases = []
    for _ in range(3):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        bases.append(optimizer.state[parameter]["projector"].clone())
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    other = torch.nn.Parameter(_values((4, 9)))
    repeated = RankScaledAdamW([other], update_interval=1, seed=123)
    for basis in bases:
        other.grad = torch.ones_like(other)
        repeated.step()
        torch.testing.assert_close(repeated.state[other]["projector"], basis, rtol=0, atol=0)
    assert not torch.equal(bases[0], bases[1])


def test_gaussian_basis_has_explicit_rank_scaling_and_parameter_seeds():
    parameters = [torch.nn.Parameter(_values((4, 9))) for _ in range(2)]
    optimizer = RankScaledAdamW(parameters, rank=2, seed=123)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for parameter in parameters:
        state = optimizer.state[parameter]
        generator = torch.Generator(device="cpu").manual_seed(state["seed"])
        expected = torch.randn((9, 2), generator=generator) / math.sqrt(2)
        torch.testing.assert_close(state["projector"], expected, rtol=0, atol=0)
    assert optimizer.state[parameters[0]]["seed"] != optimizer.state[parameters[1]]["seed"]
    assert not torch.equal(optimizer.state[parameters[0]]["projector"], optimizer.state[parameters[1]]["projector"])


def test_bf16_checkpoint_preserves_fp32_state_and_resumes_across_refresh():
    parameters = [torch.nn.Parameter(_values(shape).bfloat16()) for shape in ((8, 12), (5,))]
    optimizer = RankScaledAdamW(parameters, lr=0.03, update_interval=2)
    for step in range(2):
        for parameter in parameters:
            parameter.grad = _values(tuple(parameter.shape), offset=0.123 * step).bfloat16()
        optimizer.step()
    checkpoint = optimizer.state_dict()
    restarted_params = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    restarted = RankScaledAdamW(restarted_params, seed=999)
    restarted.load_state_dict(checkpoint)
    for parameter, new_parameter in zip(parameters, restarted_params, strict=True):
        for key in ("exp_avg", "exp_avg_sq", "projector"):
            if key not in optimizer.state[parameter]:
                continue
            original = optimizer.state[parameter][key]
            restored = restarted.state[new_parameter][key]
            assert restored.dtype == torch.float32
            torch.testing.assert_close(restored, original, rtol=0, atol=0)
            assert restored.data_ptr() != original.data_ptr()
    assert not torch.equal(
        optimizer.state[parameters[0]]["exp_avg"],
        optimizer.state[parameters[0]]["exp_avg"].bfloat16().float(),
    ), "the test needs moment values that a BF16 cast would change"
    for step in range(2, 6):
        for parameter, new_parameter in zip(parameters, restarted_params, strict=True):
            gradient = _values(tuple(parameter.shape), offset=0.123 * step).bfloat16()
            parameter.grad, new_parameter.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        restarted.step()
        for parameter, new_parameter in zip(parameters, restarted_params, strict=True):
            torch.testing.assert_close(new_parameter, parameter, rtol=0, atol=0)
            assert restarted.state[new_parameter]["step"] == optimizer.state[parameter]["step"]


def test_serialized_checkpoint_resumes_without_refreshing_projection_metadata():
    parameter = torch.nn.Parameter(_values((4, 9)).bfloat16())
    optimizer = RankScaledAdamW([parameter], update_interval=10, lr=0.03)
    parameter.grad = _values((4, 9), offset=0.1).bfloat16()
    optimizer.step()
    stream = io.BytesIO()
    torch.save({"parameter": parameter.detach(), "optimizer": optimizer.state_dict()}, stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True)
    resumed_parameter = torch.nn.Parameter(checkpoint["parameter"].clone())
    resumed = RankScaledAdamW([resumed_parameter])
    resumed.load_state_dict(checkpoint["optimizer"])
    assert resumed.state[resumed_parameter]["projection_side"] == "right"
    assert resumed.state[resumed_parameter]["parameter_shape"] == (4, 9)
    for step in range(2, 5):
        gradient = _values((4, 9), offset=0.1 * step).bfloat16()
        parameter.grad, resumed_parameter.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        resumed.step()
        torch.testing.assert_close(parameter, resumed_parameter, rtol=0, atol=0)
        assert resumed.state[resumed_parameter]["projection_index"] == 0


def test_loading_cpu_statistics_does_not_eagerly_hydrate_parameter_device():
    parameter = torch.nn.Parameter(_values((4, 9)).bfloat16())
    optimizer = RankScaledAdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    # A meta parameter lets this CPU test distinguish device hydration from
    # deferred state restoration without allocating any CUDA storage.
    placeholder = torch.nn.Parameter(torch.empty((4, 9), device="meta", dtype=torch.bfloat16))
    resumed = RankScaledAdamW([placeholder])
    resumed.load_state_dict(optimizer.state_dict())
    for key in ("exp_avg", "exp_avg_sq", "projector"):
        assert resumed.state[placeholder][key].device.type == "cpu"
        assert resumed.state[placeholder][key].dtype == torch.float32
        torch.testing.assert_close(resumed.state[placeholder][key], optimizer.state[parameter][key], rtol=0, atol=0)


def test_load_hooks_transform_full_checkpoint_and_observe_final_fp32_state_once():
    parameter = torch.nn.Parameter(_values((4, 9)).bfloat16())
    optimizer = RankScaledAdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    resumed_parameter = torch.nn.Parameter(parameter.detach().clone())
    resumed = RankScaledAdamW([resumed_parameter])
    calls = []

    def preprocess(_optimizer, state_dict):
        calls.append("pre")
        replacement = copy.deepcopy(state_dict)
        replacement["state"][0]["exp_avg"].zero_()
        return replacement

    def postprocess(loaded):
        calls.append("post")
        state = loaded.state[resumed_parameter]
        assert state["exp_avg"].dtype == torch.float32
        assert state["exp_avg"].count_nonzero() == 0
        assert state["projection_side"] == "right"
        assert state["projector"].dtype == torch.float32

    resumed.register_load_state_dict_pre_hook(preprocess)
    resumed.register_load_state_dict_post_hook(postprocess)
    resumed.load_state_dict(optimizer.state_dict())
    assert calls == ["pre", "post"]
    assert optimizer.state[parameter]["exp_avg"].count_nonzero() > 0


def test_reset_keeps_clocks_so_reactivation_draws_a_fresh_basis_and_rounding_noise():
    matrix = torch.nn.Parameter(torch.ones(16, 4096, dtype=torch.bfloat16))
    vector = torch.nn.Parameter(torch.ones(1 << 16, dtype=torch.bfloat16))
    optimizer = RankScaledAdamW(
        [matrix, vector], lr=1e-2, weight_decay=1e-2, update_interval=1000, stochastic_rounding=True
    )

    def decay_step():
        # Zero gradients make both paths a pure decay 1 -> 1 - 1e-4, a sub-ULP
        # update that stochastic rounding resolves element by element.
        for parameter in (matrix, vector):
            with torch.no_grad():
                parameter.fill_(1)
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        return [(parameter != 1).clone() for parameter in (matrix, vector)]

    first = decay_step()
    first_basis = optimizer.state[matrix]["projector"].clone()
    for parameter in (matrix, vector):
        optimizer.reset_parameter_state(parameter)
        state = optimizer.state[parameter]
        assert not any(isinstance(value, torch.Tensor) for value in state.values())
        assert state["step"] == 1 and state["moment_step"] == 0
    optimizer.reset_parameter_state(matrix)  # idempotent while inactive
    second = decay_step()
    for parameter in (matrix, vector):
        state = optimizer.state[parameter]
        assert state["step"] == 2 and state["moment_step"] == 1
        assert state["exp_avg"].dtype == torch.float32
    assert optimizer.state[matrix]["projection_index"] == 1
    assert not torch.equal(optimizer.state[matrix]["projector"], first_basis)
    for before, after in zip(first, second, strict=True):
        # Independent noise rounds down a fraction p ~= 1e-4 / 2**-8 of the elements
        # each time, overlapping in ~p**2 of them; replayed noise repeats the same set.
        probability = 1e-4 / 2**-8
        assert abs(before.float().mean().item() - probability) < 0.005
        assert abs(after.float().mean().item() - probability) < 0.005
        assert (before & after).float().mean().item() < 2 * probability**2
    with pytest.raises(ValueError, match="not registered"):
        optimizer.reset_parameter_state(torch.nn.Parameter(torch.ones(2)))


def test_deleting_state_instead_of_resetting_restarts_the_clocks():
    parameter = torch.nn.Parameter(_values((4, 9)))
    optimizer = RankScaledAdamW([parameter], update_interval=1)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    original_basis = optimizer.state[parameter]["projector"].clone()
    optimizer.step()
    assert not torch.equal(original_basis, optimizer.state[parameter]["projector"])
    del optimizer.state[parameter]
    optimizer.step()
    assert optimizer.state[parameter]["step"] == 1
    torch.testing.assert_close(optimizer.state[parameter]["projector"], original_basis, rtol=0, atol=0)


def test_offloaded_state_is_restored_only_when_a_gradient_arrives():
    parameter = torch.nn.Parameter(_values((4, 9)))
    optimizer = RankScaledAdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    # CPU execution still exercises the FP32 restoration path. The CUDA test
    # below additionally checks an actual CPU -> CUDA device migration.
    for key in ("exp_avg", "exp_avg_sq", "projector"):
        state[key] = state[key].double()
    parameter.grad = None
    optimizer.step()
    assert state["exp_avg"].dtype == torch.float64
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert all(state[key].dtype == torch.float32 for key in ("exp_avg", "exp_avg_sq", "projector"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for a CPU-offloaded device migration")
def test_cpu_offloaded_state_returns_to_cuda_lazily():
    parameter = torch.nn.Parameter(_values((4, 9)).cuda())
    optimizer = RankScaledAdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    for key in ("exp_avg", "exp_avg_sq", "projector"):
        state[key] = state[key].cpu()
    parameter.grad = None
    optimizer.step()
    assert state["exp_avg"].device.type == "cpu"
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert all(state[key].device == parameter.device for key in ("exp_avg", "exp_avg_sq", "projector"))


def test_step_parameter_matches_normal_step_with_group_learning_rates():
    values = [_values((4, 9)), _values((7,)), _values((2, 5, 3))]
    ordinary_params = [torch.nn.Parameter(value.clone()) for value in values]
    hooked_params = [torch.nn.Parameter(value.clone()) for value in values]
    ordinary = RankScaledAdamW(
        [{"params": ordinary_params[:2], "lr": 0.03}, {"params": ordinary_params[2:], "lr": 0.07}], update_interval=2
    )
    hooked = RankScaledAdamW(
        [{"params": hooked_params[:2], "lr": 0.03}, {"params": hooked_params[2:], "lr": 0.07}], update_interval=2
    )
    for step in range(4):
        ordinary.param_groups[0]["lr"] = hooked.param_groups[0]["lr"] = 0.03 / (step + 1)
        for i, (parameter, hooked_parameter) in enumerate(zip(ordinary_params, hooked_params, strict=True)):
            gradient = _values(tuple(parameter.shape), offset=step / 10)
            parameter.grad, hooked_parameter.grad = gradient.clone(), gradient.clone()
            if i == 1 and step == 2:
                parameter.grad = hooked_parameter.grad = None
        ordinary.step()
        # Hook order need not be parameter-registration order.
        for parameter in reversed(hooked_params):
            hooked.step_parameter(parameter)
        for parameter, hooked_parameter in zip(ordinary_params, hooked_params, strict=True):
            torch.testing.assert_close(parameter, hooked_parameter, rtol=0, atol=0)


def test_step_runs_closure_with_grad_enabled_and_returns_its_loss():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = RankScaledAdamW([parameter], lr=0.1)

    def closure():
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    assert loss.item() == 5.0
    assert not torch.equal(parameter, torch.tensor([1.0, 2.0]))


def test_autocast_does_not_lower_projection_precision():
    parameter = torch.nn.Parameter(_values((4, 9)))
    reference = torch.nn.Parameter(parameter.detach().clone())
    optimizer = RankScaledAdamW([parameter], lr=0.01)
    baseline = RankScaledAdamW([reference], lr=0.01)
    for step in range(3):
        gradient = _values((4, 9), offset=step * 0.09)
        parameter.grad, reference.grad = gradient.clone(), gradient.clone()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            optimizer.step_parameter(parameter)
        baseline.step()
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        torch.testing.assert_close(
            optimizer.state[parameter]["exp_avg"], baseline.state[reference]["exp_avg"], rtol=0, atol=0
        )


@pytest.mark.parametrize("step", [1, 37])
def test_adam_temporary_reuse_matches_original_formula_bitwise(step):
    gradient = _values((2, 3, 7), offset=0.123)
    first = _values(gradient.shape, offset=-0.19)
    second = _values(gradient.shape, offset=0.4).square()
    original_gradient = gradient.clone()
    expected_first, expected_second = first.clone(), second.clone()
    group = dict(betas=(0.7, 0.95), eps=1e-5)
    beta1, beta2 = group["betas"]
    expected_first.mul_(beta1).add_(gradient, alpha=1 - beta1)
    expected_second.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
    corrected_first = expected_first / (1 - beta1**step)
    denominator = (expected_second / (1 - beta2**step)).sqrt().add_(group["eps"])
    expected = corrected_first / denominator
    actual = optim_module._adam_direction(first, second, gradient, step, group)
    for value, reference in (
        (actual, expected),
        (first, expected_first),
        (second, expected_second),
        (gradient, original_gradient),
    ):
        assert torch.equal(value.view(torch.int32), reference.view(torch.int32))


@pytest.mark.parametrize("transposed", [False, True])
def test_rounding_noise_reuse_matches_original_formula_bits_and_rng(transposed):
    raw_generator = torch.Generator(device="cpu").manual_seed(401)
    raw = torch.randint(-(1 << 31), (1 << 31) - 1, (16, 64), dtype=torch.int32, generator=raw_generator)
    values = raw.view(torch.float32)
    values[0, :8] = torch.tensor(
        [
            0.0,
            -0.0,
            float("nan"),
            float("inf"),
            -float("inf"),
            torch.finfo(torch.float32).max,
            torch.finfo(torch.float32).tiny,
            torch.finfo(torch.bfloat16).max,
        ]
    )
    if transposed:
        values = values.T
    before = values.clone()
    generator = torch.Generator(device="cpu").manual_seed(123)
    reference_generator = torch.Generator(device="cpu").manual_seed(123)
    global_rng = torch.random.get_rng_state().clone()
    finite_values = values.clamp(min=-torch.finfo(torch.bfloat16).max, max=torch.finfo(torch.bfloat16).max)
    noise = torch.randint(0, 1 << 16, values.shape, dtype=torch.int32, generator=reference_generator)
    rounded_bits = (finite_values.view(torch.int32) + noise).bitwise_and_(-65536)
    rounded = rounded_bits.view(torch.float32).bfloat16()
    expected = torch.where(torch.isfinite(values), rounded, values.bfloat16())
    actual = optim_module._stochastic_round_bfloat16(values, generator)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(values.view(torch.int32), before.view(torch.int32))
    assert torch.equal(generator.get_state(), reference_generator.get_state())
    assert torch.equal(torch.random.get_rng_state(), global_rng)


def test_stochastic_rounding_preserves_exact_values_and_nonfinite_classes():
    exact = torch.tensor(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.125,
            torch.finfo(torch.bfloat16).tiny,
            torch.finfo(torch.bfloat16).max,
            -torch.finfo(torch.bfloat16).max,
        ],
        dtype=torch.bfloat16,
    ).float()
    generator = torch.Generator(device="cpu").manual_seed(123)
    rounded = optim_module._stochastic_round_bfloat16(exact, generator)
    torch.testing.assert_close(rounded.float(), exact, rtol=0, atol=0)
    assert torch.equal(torch.signbit(rounded[:2]), torch.signbit(exact[:2]))
    extreme = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max]
    )
    rounded = optim_module._stochastic_round_bfloat16(extreme, generator)
    assert torch.isnan(rounded[0])
    assert rounded[1] == float("inf") and rounded[2] == -float("inf")
    assert rounded[3] == torch.finfo(torch.bfloat16).max
    assert rounded[4] == -torch.finfo(torch.bfloat16).max


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_stochastic_rounding_population_mean_matches_the_fp32_value(sign):
    count = 100_000
    gap = 1 / 128
    target = sign * (1.0 + 0.25 * gap)
    values = torch.full((count,), target, dtype=torch.float32)
    rng_before = torch.random.get_rng_state().clone()
    rounded = optim_module._stochastic_round_bfloat16(values, torch.Generator(device="cpu").manual_seed(713))
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert set(rounded.float().unique().tolist()) == {sign, sign * (1.0 + gap)}
    # Six standard errors for a Bernoulli mixture. Nearest rounding misses the
    # mean by 1/4 ULP, far outside this interval at this population size.
    tolerance = 6 * gap * math.sqrt(0.25 * 0.75 / count)
    assert abs(rounded.double().mean().item() - target) < tolerance


def test_stochastic_bf16_updates_can_accumulate_below_one_ulp():
    nearest_parameter = torch.nn.Parameter(torch.ones(8192, dtype=torch.bfloat16))
    stochastic_parameter = torch.nn.Parameter(nearest_parameter.detach().clone())
    nearest = RankScaledAdamW([nearest_parameter], lr=2e-5)
    stochastic = RankScaledAdamW([stochastic_parameter], lr=2e-5, stochastic_rounding=True)
    nearest_parameter.grad = torch.ones_like(nearest_parameter)
    stochastic_parameter.grad = torch.ones_like(stochastic_parameter)
    nearest.step()
    stochastic.step()
    assert torch.equal(nearest_parameter, torch.ones_like(nearest_parameter)), "the update must be sub-ULP"
    assert (stochastic_parameter != 1).any()
    assert abs(stochastic_parameter.double().mean().item() - (1 - 2e-5)) < 1.5e-5
    assert stochastic.state[stochastic_parameter]["exp_avg"].dtype == torch.float32


@pytest.mark.parametrize("shape", [(2, 3, 7), (2, 7, 3), (19,), ()])
def test_stochastic_update_combines_decay_and_hand_computed_direction_before_rounding(shape, monkeypatch):
    monkeypatch.setattr(optim_module, "_BF16_UPDATE_CHUNK_SIZE", 5)
    incoming = []

    def record_round(values, generator):
        incoming.append(values.detach().clone())
        return values.bfloat16()

    monkeypatch.setattr(optim_module, "_stochastic_round_bfloat16", record_round)
    parameter = torch.nn.Parameter(_values(shape, offset=0.6).bfloat16())
    gradient = _values(shape, offset=0.19).bfloat16()
    before = parameter.detach().float().numpy().astype(np.float64)
    original_gradient = gradient.float().numpy().astype(np.float64)
    options = dict(lr=0.07, betas=(0.6, 0.8), eps=1e-5, weight_decay=0.3, scale=0.4)
    optimizer = RankScaledAdamW([parameter], stochastic_rounding=True, **options)
    parameter.grad = gradient
    optimizer.step()
    state = optimizer.state[parameter]
    if state["projected"]:
        basis = state["projector"].numpy().astype(np.float64)
        sketch = _numpy_sketch(original_gradient, basis, state["projection_side"])
        # On the first step Adam's bias corrections cancel both moment factors.
        update = sketch / (np.abs(sketch) + options["eps"])
        factor = np.linalg.norm(update, axis=(-2, -1), keepdims=True) / (
            np.linalg.norm(sketch, axis=(-2, -1), keepdims=True) + options["eps"]
        )
        direction = original_gradient * options["scale"] * factor
    else:
        direction = original_gradient / (np.abs(original_gradient) + options["eps"])
    expected = before * (1 - options["lr"] * options["weight_decay"]) - options["lr"] * direction
    captured = torch.cat([part.reshape(-1) for part in incoming]).numpy().reshape(shape)
    np.testing.assert_allclose(captured, expected, rtol=3e-6, atol=2e-7)
    torch.testing.assert_close(parameter, torch.from_numpy(captured).bfloat16(), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(3, 13, 17), (3, 17, 13)])
def test_chunked_bf16_sketch_matches_the_matrix_formula(shape, monkeypatch):
    monkeypatch.setattr(optim_module, "_BF16_UPDATE_CHUNK_SIZE", 7)
    gradient = _values(shape, offset=0.123).bfloat16()
    rows, columns = shape[-2:]
    side = "right" if rows <= columns else "left"
    dimension = columns if side == "right" else rows
    basis = torch.randn((*shape[:-2], dimension, 2), generator=torch.Generator(device="cpu").manual_seed(19))
    expected = gradient.float() @ basis if side == "right" else basis.transpose(-2, -1) @ gradient.float()
    actual = optim_module._project_bfloat16_chunked(gradient, basis, side)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("shape", [(3, 13, 17), (3, 17, 13), (241,)])
def test_stochastic_update_fp32_conversions_and_rounding_stay_chunk_bounded(shape, monkeypatch):
    from torch.utils._python_dispatch import TorchDispatchMode

    chunk_size = 64
    monkeypatch.setattr(optim_module, "_BF16_UPDATE_CHUNK_SIZE", chunk_size)
    conversions = []
    rounding_sizes = []
    original_round = optim_module._stochastic_round_bfloat16

    class ConversionRecorder(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func == torch.ops.aten._to_copy.default and kwargs.get("dtype") == torch.float32:
                if args[0].dtype == torch.bfloat16:
                    conversions.append(args[0].numel())
            return func(*args, **kwargs)

    def record_round(values, generator):
        rounding_sizes.append(values.numel())
        return original_round(values, generator)

    monkeypatch.setattr(optim_module, "_stochastic_round_bfloat16", record_round)
    parameter = torch.nn.Parameter(_values(shape, offset=1).bfloat16())
    parameter.grad = _values(shape, offset=0.2).bfloat16()
    original_gradient = parameter.grad.clone()
    optimizer = RankScaledAdamW([parameter], stochastic_rounding=True)
    with ConversionRecorder():
        optimizer.step()
    assert parameter.numel() > chunk_size
    assert len(conversions) > 1 and max(conversions) <= chunk_size
    assert len(rounding_sizes) > 1 and max(rounding_sizes) <= chunk_size
    torch.testing.assert_close(parameter.grad, original_gradient, rtol=0, atol=0)


def test_stochastic_rounding_resume_and_parameter_hook_order_are_deterministic():
    parameters = [torch.nn.Parameter(_values(shape, offset=0.4).bfloat16()) for shape in ((3, 4, 9), (13,))]
    optimizer = RankScaledAdamW(parameters, lr=0.01, update_interval=3, stochastic_rounding=True)
    for parameter in parameters:
        parameter.grad = _values(tuple(parameter.shape), offset=0.125).bfloat16()
    rng_before = torch.random.get_rng_state().clone()
    optimizer.step()
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    stream = io.BytesIO()
    torch.save(optimizer.state_dict(), stream)
    stream.seek(0)
    resumed_parameters = [torch.nn.Parameter(parameter.detach().clone()) for parameter in parameters]
    resumed = RankScaledAdamW(resumed_parameters)
    resumed.load_state_dict(torch.load(stream, weights_only=True))
    assert resumed.param_groups[0]["stochastic_rounding"] is True
    for step in range(2, 7):
        for parameter, new_parameter in zip(parameters, resumed_parameters, strict=True):
            gradient = _values(tuple(parameter.shape), offset=0.125 * step).bfloat16()
            parameter.grad, new_parameter.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        for parameter in reversed(resumed_parameters):
            resumed.step_parameter(parameter)
        for parameter, new_parameter in zip(parameters, resumed_parameters, strict=True):
            torch.testing.assert_close(parameter, new_parameter, rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_stochastic_rounding_leaves_projection_rng_and_statistics_unchanged():
    parameter = torch.nn.Parameter(_values((5, 9), offset=1).bfloat16())
    nearest_parameter = torch.nn.Parameter(parameter.detach().clone())
    stochastic = RankScaledAdamW([parameter], stochastic_rounding=True, update_interval=1)
    nearest = RankScaledAdamW([nearest_parameter], update_interval=1)
    for step in range(3):
        gradient = _values((5, 9), offset=step * 0.2).bfloat16()
        parameter.grad, nearest_parameter.grad = gradient.clone(), gradient.clone()
        stochastic.step()
        nearest.step()
        for key in ("projector", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                stochastic.state[parameter][key], nearest.state[nearest_parameter][key], rtol=0, atol=0
            )


def test_stochastic_rounding_rejects_noncontiguous_bf16_before_mutating_state():
    parameter = torch.nn.Parameter(_values((9, 4)).T.bfloat16())
    parameter.grad = torch.ones_like(parameter)
    optimizer = RankScaledAdamW([parameter], stochastic_rounding=True)
    original = parameter.detach().clone()
    with pytest.raises(ValueError, match="contiguous BF16"):
        optimizer.step()
    assert not optimizer.state
    torch.testing.assert_close(parameter, original, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for device-local BF16 rounding RNG")
@pytest.mark.parametrize("shape", [(3, 4, 9), (3, 9, 4), (31,)])
def test_cuda_stochastic_rounding_resume_preserves_global_rng_and_fp32_state(shape, monkeypatch):
    monkeypatch.setattr(optim_module, "_BF16_UPDATE_CHUNK_SIZE", 17)
    parameter = torch.nn.Parameter(_values(shape, offset=1).bfloat16().cuda())
    optimizer = RankScaledAdamW([parameter], lr=0.01, stochastic_rounding=True, update_interval=3)
    parameter.grad = _values(shape, offset=0.1).bfloat16().cuda()
    global_rng = torch.cuda.get_rng_state().clone()
    cpu_rng = torch.random.get_rng_state().clone()
    optimizer.step()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    for state in checkpoint["state"].values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.cpu()
    restarted_parameter = torch.nn.Parameter(parameter.detach().clone())
    restarted = RankScaledAdamW([restarted_parameter])
    restarted.load_state_dict(checkpoint)
    assert all(
        value.device.type == "cpu"
        for value in restarted.state[restarted_parameter].values()
        if isinstance(value, torch.Tensor)
    )
    for step in range(2, 5):
        gradient = _values(shape, offset=step * 0.1).bfloat16().cuda()
        parameter.grad, restarted_parameter.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        restarted.step_parameter(restarted_parameter)
        torch.testing.assert_close(restarted_parameter, parameter, rtol=0, atol=0)
    assert torch.equal(torch.cuda.get_rng_state(), global_rng)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    for value in restarted.state[restarted_parameter].values():
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cuda" and value.dtype == torch.float32


@pytest.mark.parametrize(
    "option",
    [
        {"lr": -1},
        {"lr": float("nan")},
        {"weight_decay": -1},
        {"scale": -1},
        {"eps": 0},
        {"eps": float("inf")},
        {"rank": 0},
        {"rank": 1.5},
        {"rank": True},
        {"update_interval": 0},
        {"update_interval": 2.5},
        {"betas": (0.9,)},
        {"betas": (1.0, 0.9)},
        {"betas": (0.9, -0.1)},
        {"seed": -1},
        {"seed": 1.5},
        {"stochastic_rounding": "yes"},
    ],
)
def test_invalid_options_are_rejected(option):
    with pytest.raises(ValueError):
        RankScaledAdamW([torch.nn.Parameter(torch.ones(2, 3))], **option)


def test_sparse_gradients_and_unknown_parameters_are_rejected_before_state_changes():
    parameter = torch.nn.Parameter(torch.ones(3, 4))
    optimizer = RankScaledAdamW([parameter])
    parameter.grad = torch.sparse_coo_tensor(
        torch.tensor([[0, 2], [1, 3]]), torch.tensor([1.0, 2.0]), (3, 4), check_invariants=True
    )
    with pytest.raises(RuntimeError, match="dense gradients"):
        optimizer.step()
    assert not optimizer.state
    with pytest.raises(ValueError, match="not registered"):
        optimizer.step_parameter(torch.nn.Parameter(torch.ones(3, 4)))
