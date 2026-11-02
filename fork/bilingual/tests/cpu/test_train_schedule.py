"""Schedule, parameter groups and the expert decay (docs/spec.md, I5-T7, I5-T8; SCI-12, IMPL-09)."""

import math

import pytest
import torch

from surogate_bilingual.assembly.packs import read_pack
from surogate_bilingual.config import profile
from surogate_bilingual.model.lm import RoutedLM
from surogate_bilingual.train import schedule
from surogate_bilingual.train.loop import Trainer
from surogate_bilingual.train.recipes import ExpertRecipe, Inputs, SeedRecipe


def closed_form(step, total, peak, warmup, floor_frac):
    """Written from MS §8.1 independently of train.schedule: linear warmup to the peak over `warmup` steps, then a
    half cosine from the peak to floor_frac * peak, reaching it at the last step."""
    if step < warmup:
        return peak * (step + 1) / warmup
    low = floor_frac * peak
    return low + 0.5 * (peak - low) * (1 + math.cos(math.pi * (step - warmup) / (total - 1 - warmup)))


def test_i5_t8_lr_at_the_four_points_equals_the_closed_form():
    peak = 6e-4
    sched = schedule.Schedule(total=1001, peak=peak, warmup_frac=0.02, min_lr_frac=0.10)
    assert sched.warmup == 20  # 2 % of 1001 steps
    points = {0: peak / 20, 19: peak, 20: peak, 510: (peak + 0.1 * peak) / 2, 1000: 0.1 * peak}
    for step, expected in points.items():
        assert abs(sched.lr(step) - expected) <= 1e-12, step
        assert abs(sched.lr(step) - closed_form(step, 1001, peak, 20, 0.10)) <= 1e-12, step
    lrs = [sched.lr(s) for s in range(1001)]
    assert all(a < b for a, b in zip(lrs[:19], lrs[1:20], strict=True))  # warmup rises
    assert all(a >= b for a, b in zip(lrs[20:], lrs[21:], strict=False))  # cosine falls
    with pytest.raises(schedule.ScheduleError):
        sched.lr(1001)
    no_warmup = schedule.Schedule(total=10, peak=1.0, warmup_frac=0.0, min_lr_frac=0.0)
    assert no_warmup.warmup == 0 and no_warmup.lr(0) == 1.0 and abs(no_warmup.lr(9)) <= 1e-12


def test_schedule_state_refuses_another_schedule():
    sched = schedule.Schedule(total=100, peak=1e-3, warmup_frac=0.02, min_lr_frac=0.1)
    assert sched.check_state(sched.state_dict(37)) == 37
    other = schedule.Schedule(total=101, peak=1e-3, warmup_frac=0.02, min_lr_frac=0.1)
    with pytest.raises(schedule.ScheduleError, match="differs"):
        other.check_state(sched.state_dict(37))


def _fp64_expert_trainer(tiny_trained, tiny_stage, decay: str, lam: float):
    """The trainer of tiny-zh in fp64, with the expert moved away from the seed so that w - w_seed != 0."""
    run = tiny_stage("tiny-zh", f'train.expert_decay="{decay}"', f"train.expert_decay_lambda={lam}")
    seed = read_pack(tiny_trained["seed"])
    recipe = ExpertRecipe("zh")
    built = recipe.build(
        run.model,
        Inputs(seed=seed, seed_path=tiny_trained["seed"]),
        fusion=run.fusion,
        gqa=run.gqa,
        init_seed=0,
        anchor_kl=0.0,
    )
    model = built.model.to(torch.float64)
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in model.bank.experts["zh"].parameters():
            p.add_(0.01 * torch.randn(p.shape, generator=g, dtype=torch.float64))
    plan = recipe.freeze_plan(model).apply()
    trainer = Trainer(model, plan, recipe, run, device=torch.device("cpu"), precision="fp32", anchors=built.anchors)
    anchors = [built.anchors[n].to(torch.float64) for n in plan.trainable_names]
    return trainer, plan, anchors


def test_i5_t7_decay_to_seed_shrinks_the_distance_by_exactly_one_minus_lr_lambda(tiny_trained, tiny_stage):
    lam = 0.1
    trainer, plan, anchors = _fp64_expert_trainer(tiny_trained, tiny_stage, "to_seed", lam)
    sched = schedule.Schedule(total=6, peak=0.05, warmup_frac=0.2, min_lr_frac=0.1)
    for step in range(6):
        lr = sched.lr(step)
        before = [(p - a).detach().clone() for p, a in zip(plan.parameters(), anchors, strict=True)]
        for p in plan.parameters():
            p.grad = torch.zeros_like(p)  # zero gradients: only the decay may move the weights
        trainer.apply(lr)
        for p, a, d in zip(plan.parameters(), anchors, before, strict=True):
            assert float(((p - a) - (1 - lr * lam) * d).abs().max().detach()) <= 1e-12
    assert all(group["weight_decay"] == 0.0 for group in trainer.optimizer.param_groups)  # not AdamW's own decay


@pytest.mark.parametrize("decay,lam", [("none", 0.0), ("none", 0.3), ("to_zero", 0.1)])
def test_i5_t7_none_leaves_weights_unchanged_and_to_zero_is_adamw_decay(tiny_trained, tiny_stage, decay, lam):
    trainer, plan, _ = _fp64_expert_trainer(tiny_trained, tiny_stage, decay, lam)
    before = [p.detach().clone() for p in plan.parameters()]
    for p in plan.parameters():
        p.grad = torch.zeros_like(p)
    trainer.apply(0.02)
    for p, b in zip(plan.parameters(), before, strict=True):
        if decay == "none":
            assert torch.equal(p, b)
        else:
            assert float((p - (1 - 0.02 * lam) * b).abs().max().detach()) <= 1e-15


@pytest.mark.parametrize("family", ["swiglu", "slot"])
def test_i5_t7_seed_recipe_never_decays_norms_or_the_embedding(family):
    from surogate_bilingual.config import TrainConfig

    model = RoutedLM(profile("tiny", family), ["seed"], device="meta")
    plan = SeedRecipe().freeze_plan(model)
    groups = {g["group"]: g for g in SeedRecipe().param_groups(model, plan, TrainConfig(weight_decay=0.1))}
    assert set(groups) == {"decay", "no_decay"}
    assert groups["decay"]["weight_decay"] == 0.1 and groups["no_decay"]["weight_decay"] == 0.0
    no_decay = set(groups["no_decay"]["names"])
    assert "backbone.emb.weight" in no_decay  # the embedding is also the tied head
    norms = {n for n, _ in model.named_parameters() if "norm" in n}
    assert norms and norms <= no_decay
    assert all(model.get_parameter(n).dim() == 2 for n in groups["decay"]["names"])
    assert not any("norm" in n or n == "backbone.emb.weight" for n in groups["decay"]["names"])
    if family == "slot":
        assert any(n.endswith("readout.b") for n in no_decay)
        assert any(n.endswith("readout.Q.weight") for n in groups["decay"]["names"])
    assert set(groups["decay"]["names"]) | no_decay == set(plan.trainable_names)
