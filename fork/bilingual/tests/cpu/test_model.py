"""The routed model (docs/spec.md, I2-A1 to I2-A5, I2-A7): parity with the owner's v1.0 model, dispatch against a
dense reference, the bf16-autocast regression NEW-01, explicit plans for multi-expert models, causality, and
gradients with activation checkpointing.
"""

import pytest
import torch

from surogate_bilingual.assembly.hashing import content_hash
from surogate_bilingual.config import profile
from surogate_bilingual.model.executor import Executor, dense_reference_ffn
from surogate_bilingual.model.lm import RoutedLM
from surogate_bilingual.routing.plan import RoutePlan
from surogate_bilingual.train.recipes import ExpertRecipe


def next_token_ce(logits, tokens):
    """Mean CE of predicting tokens[:, t] from logits[:, t - 1] (a test objective; training uses target_ce_sum)."""
    pred = logits[:, :-1].float()
    return torch.nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]), tokens[:, 1:].reshape(-1))


NAMES = ("zh", "en", "bridge")
FAMILY_FUSIONS = [("swiglu", "output"), ("swiglu", "concat"), ("slot", "pre"), ("slot", "post")]


def random_plan(names, B, T, kind, seed, dtype=torch.float32):
    """Distinct per-row plans: `one_hot`, `dual` (two random experts per token with random simplex gates), or
    `mixed` (one-hot and dual tokens interleaved)."""
    g = torch.Generator().manual_seed(seed)
    E = len(names)
    gates = torch.zeros(B, T, E, dtype=dtype)
    ids = torch.randint(0, E, (B, T), generator=g)
    gates.scatter_(-1, ids[..., None], 1.0)
    dual = torch.ones(B, T, dtype=torch.bool) if kind == "dual" else torch.rand(B, T, generator=g) < 0.5
    if kind == "one_hot":
        dual[:] = False
    for b, t in torch.nonzero(dual).tolist():
        a, c = torch.randperm(E, generator=g)[:2].tolist()
        w = float(torch.rand((), generator=g, dtype=torch.float64)) * 0.8 + 0.1
        gates[b, t] = 0
        gates[b, t, a], gates[b, t, c] = w, 1 - w
    plan = RoutePlan(tuple(names), gates, f"test-{kind}")
    plan.validate()
    return plan


def model_for(name, family, fusion, dtype=torch.float32, names=NAMES, seed=0):
    return RoutedLM(profile(name, family), names, fusion, dtype=dtype, init_seed=seed).eval()


# --- I2-A1: parity with v1.0 -------------------------------------------------------------------------------------


def v10_into_new(ref) -> RoutedLM:
    """The v1.0 smoke model's weights under the new key names (experts.* -> bank.experts.*, the rest -> backbone.*)."""
    model = RoutedLM(profile("smoke", "swiglu"), ref.names).eval()
    state = {("bank." if k.startswith("experts.") else "backbone.") + k: v for k, v in ref.state_dict().items()}
    model.load_state_dict(state, strict=True)
    return model


@pytest.mark.parametrize("T", [128, 37])
def test_i2_a1_logits_match_the_v10_model_on_forced_and_mixed_routes(moe_pilot, T):
    torch.manual_seed(0)
    ref = moe_pilot.RoutedLM(moe_pilot.cfg_for_profile("smoke", 256), list(NAMES)).eval()
    model = v10_into_new(ref)
    assert ref.cfg.max_seq == model.cfg.max_seq == 128
    x = torch.randint(0, 256, (2, T))
    with torch.no_grad():
        for e, name in enumerate(NAMES):
            diff = (ref(x, torch.full_like(x, e)) - model(x, RoutePlan.forced(name, NAMES, 2, T))).abs().max()
            assert float(diff) <= 1e-6, (name, float(diff))
        routes = torch.randint(0, 3, (2, T))
        assert not torch.equal(routes[0], routes[1])
        diff = (ref(x, routes) - model(x, RoutePlan.one_hot(routes, NAMES, "mixed"))).abs().max()
        assert float(diff) <= 1e-6, float(diff)


# --- I2-A2: dispatch against the dense reference -----------------------------------------------------------------

TOLERANCE = {torch.float32: 1e-6, torch.float64: 1e-12}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("profile_name", ["tiny", "smoke"])
@pytest.mark.parametrize("family,fusion", FAMILY_FUSIONS)
def test_i2_a2_executor_matches_the_dense_reference(profile_name, family, fusion, dtype):
    model = model_for(profile_name, family, fusion, dtype)
    cfg = model.cfg
    g = torch.Generator().manual_seed(1)
    for T in (cfg.max_seq, cfg.max_seq - 1):  # max_seq is even in both profiles: T = max_seq and an odd T
        u = torch.randn(3, T, cfg.d_model, generator=g, dtype=dtype)
        for kind in ("one_hot", "dual", "mixed"):
            plan = random_plan(NAMES, 3, T, kind, seed=T + len(kind), dtype=dtype)
            with torch.no_grad():
                for layer in range(cfg.n_layers):
                    readout = model.backbone.layers[layer].readout
                    got = model.executor(layer, u, plan, model.bank, readout)
                    want = dense_reference_ffn(layer, u, plan, model.bank, readout, fusion)
                    assert got.dtype == u.dtype
                    diff = float((got - want).abs().max())
                    assert diff <= TOLERANCE[dtype], (kind, T, layer, diff)


class DenseExecutor(Executor):
    """Every expert on every token: the model-level reference."""

    def __call__(self, layer, u, plan, bank, readout, *, dispatch=None):
        return dense_reference_ffn(layer, u, plan, bank, readout, self.fusion)


@pytest.mark.parametrize("family,fusion", FAMILY_FUSIONS)
def test_i2_a2_model_logits_match_a_dense_model(family, fusion):
    model = model_for("smoke", family, fusion)
    dense = model_for("smoke", family, fusion)
    dense.load_state_dict(model.state_dict())
    dense.executor = DenseExecutor(fusion)
    x = torch.randint(5, model.cfg.vocab_size, (2, 37))
    plan = random_plan(NAMES, 2, 37, "mixed", seed=3)
    with torch.no_grad():
        assert float((model(x, plan) - dense(x, plan)).abs().max()) <= 1e-6


# --- I2-A3: bf16 autocast (NEW-01) -------------------------------------------------------------------------------


def single_expert_copy(model, name):
    single = RoutedLM(model.cfg, [name], model.fusion).eval()
    single.load_backbone_state(model.backbone_state())
    single.load_expert_state(name, model.expert_state(name))
    return single


@pytest.mark.parametrize("family,fusion", FAMILY_FUSIONS)
def test_i2_a3_every_dispatch_path_runs_under_bf16_autocast(family, fusion):
    model = model_for("smoke", family, fusion)
    x = torch.randint(5, model.cfg.vocab_size, (2, 37))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        for kind in ("one_hot", "dual", "mixed"):
            out = model(x, random_plan(NAMES, 2, 37, kind, seed=5))
            assert out.shape == (2, 37, model.cfg.vocab_size) and bool(torch.isfinite(out).all()), kind
        for name in NAMES:
            forced = model(x, RoutePlan.forced(name, NAMES, 2, 37))
            alone = single_expert_copy(model, name)(x)
            assert torch.equal(forced, alone), name


@pytest.mark.parametrize("family,fusion", FAMILY_FUSIONS)
def test_every_dispatch_path_runs_with_bf16_parameters_and_no_autocast(family, fusion):
    """CR1 review: with bf16 parameters (RoutedLM(dtype=...), load_assembly(dtype=...), verify --dtype bfloat16) and
    autocast off, the fp32-promoted gate products of concat, pre and post met bf16 weights in F.linear and raised.
    The dual rows now run and stay close to the fp32 model."""
    model = model_for("smoke", family, fusion)
    half = RoutedLM(model.cfg, NAMES, fusion, dtype=torch.bfloat16).eval()
    half.load_state_dict({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()})
    x = torch.randint(5, model.cfg.vocab_size, (2, 37))
    with torch.no_grad():
        for kind in ("one_hot", "dual", "mixed"):
            plan = random_plan(NAMES, 2, 37, kind, seed=5)
            out = half(x, plan)
            assert out.dtype == torch.bfloat16 and bool(torch.isfinite(out).all()), kind
            assert float((out.float() - model(x, plan)).abs().max()) <= 0.25, kind  # bf16 rounding only


def test_i2_a3_the_v10_dispatch_is_the_regression_it_guards(moe_pilot):
    """Documented negative: v1.0 `_ffn` scatters bf16 expert outputs into an fp32 buffer and fails under autocast."""
    torch.manual_seed(0)
    ref = moe_pilot.RoutedLM(moe_pilot.cfg_for_profile("smoke", 256), list(NAMES)).eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        ref(x)  # routes=None runs the first expert directly
        with pytest.raises(RuntimeError, match="same"):
            ref(x, torch.zeros_like(x))


# --- I2-A4 -------------------------------------------------------------------------------------------------------


def test_i2_a4_a_multi_expert_model_requires_a_plan():
    model = model_for("tiny", "swiglu", "output")
    x = torch.randint(5, 64, (2, 8))
    with pytest.raises(ValueError, match="explicit RoutePlan"):
        model(x)
    single = model_for("tiny", "swiglu", "output", names=("zh",))
    assert single(x).shape == (2, 8, 64)
    with pytest.raises(ValueError, match="not experts of the model"):
        model(x, RoutePlan.forced("t3", ("zh", "t3"), 2, 8))


# --- I2-A5: causality --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("family,fusion", FAMILY_FUSIONS)
def test_i2_a5_later_tokens_never_change_earlier_logits(family, fusion):
    model = model_for("tiny", family, fusion, torch.float64)
    T = model.cfg.max_seq
    full = random_plan(NAMES, 2, T, "mixed", seed=7, dtype=torch.float64)
    g = torch.Generator().manual_seed(8)
    x = torch.randint(5, model.cfg.vocab_size, (2, T), generator=g)
    with torch.no_grad():
        reference = model(x, full)
        for t0 in (1, T // 2, T - 1):
            changed = x.clone()
            changed[:, t0:] = torch.randint(5, model.cfg.vocab_size, (2, T - t0), generator=g)
            assert not torch.equal(changed, x)
            assert torch.equal(model(changed, full)[:, :t0], reference[:, :t0]), t0
            window = model(x[:, :t0], full.slice(0, t0))  # the window's plan is the slice of the full plan
            assert float((window - reference[:, :t0]).abs().max()) <= 1e-12, t0


# --- I2-A7: activation checkpointing -----------------------------------------------------------------------------


@pytest.mark.parametrize("family,fusion", FAMILY_FUSIONS)
def test_i2_a7_checkpointing_leaves_gradients_unchanged_and_frozen_grads_none(family, fusion):
    grads = []
    for checkpointing in (False, True):
        model = model_for("smoke", family, fusion).train()
        plan_ = ExpertRecipe("en").freeze_plan(model).apply()
        x = torch.randint(5, model.cfg.vocab_size, (2, 37), generator=torch.Generator().manual_seed(9))
        loss = next_token_ce(model(x, random_plan(NAMES, 2, 37, "mixed", seed=11), checkpointing=checkpointing), x)
        loss.backward()
        plan_.check_grads()
        assert all(p.grad is not None for p in plan_.parameters())
        grads.append({n: p.grad.clone() for n, p in zip(plan_.trainable_names, plan_.parameters(), strict=True)})
    assert grads[0].keys() == grads[1].keys()
    worst = max(float((grads[0][n] - grads[1][n]).abs().max()) for n in grads[0])
    assert worst <= 1e-7, worst


# --- construction ------------------------------------------------------------------------------------------------


def test_initialisation_is_a_function_of_init_seed():
    def init_hash(seed):
        return content_hash(model_for("tiny", "slot", "post", seed=seed).state_dict())

    assert init_hash(3) == init_hash(3) != init_hash(4)
    model = model_for("tiny", "slot", "post")
    assert torch.equal(model.backbone.layers[0].readout.b, torch.zeros(model.cfg.ffn_width))
    assert torch.equal(model.backbone.final_norm.weight, torch.ones(model.cfg.d_model))
    assert abs(float(model.backbone.emb.weight.detach().std()) - 0.02) < 0.002


def test_state_views_use_the_pack_key_names():
    from surogate_bilingual.model.backbone import backbone_keys
    from surogate_bilingual.model.bank import expert_keys

    for family in ("swiglu", "slot"):
        cfg = profile("tiny", family, expert_layers=(0,))
        model = RoutedLM(cfg, ["zh", "en"], device="meta")
        assert sorted(model.backbone_state()) == backbone_keys(cfg)
        assert sorted(model.expert_state("zh")) == expert_keys(cfg) == sorted(model.expert_state("en"))
        with pytest.raises(KeyError, match="missing"):
            model.load_expert_state("zh", {})
