"""Freeze plans (docs/spec.md, I2-A11; IMPL-06, MA-12): one AdamW step of the expert recipe changes the allowed
expert only, the optimizer holds exactly the allowed storages, and frozen tensors get no gradient.
"""

import pytest
import torch

from surogate_bilingual.assembly.freeze import FreezeError, FreezePlan
from surogate_bilingual.config import profile
from surogate_bilingual.model.lm import RoutedLM
from surogate_bilingual.routing.plan import RoutePlan
from surogate_bilingual.train.objectives import masked_ce
from surogate_bilingual.train.recipes import ExpertRecipe, SeedRecipe

NAMES = ("zh", "en", "bridge")


@pytest.mark.parametrize("family", ["swiglu", "slot"])
def test_i2_a11_one_adamw_step_changes_only_the_trained_expert(family):
    model = RoutedLM(profile("smoke", family), NAMES).train()
    plan = ExpertRecipe("en").freeze_plan(model).apply()
    assert set(plan.trainable_names) == {n for n, _ in model.named_parameters() if n.startswith("bank.experts.en.")}
    before = plan.frozen_hashes()
    trainable_before = {n: p.detach().clone() for n, p in zip(plan.trainable_names, plan.parameters(), strict=True)}
    opt = torch.optim.AdamW(plan.parameters(), lr=1e-2, weight_decay=0.1)
    plan.check_optimizer(opt)
    g = torch.Generator().manual_seed(0)
    x = torch.randint(5, model.cfg.vocab_size, (2, 33), generator=g)
    routes = torch.randint(0, 3, (2, 33), generator=g)  # every expert sees tokens; only en may learn
    masked_ce(model(x, RoutePlan.one_hot(routes, NAMES, "mixed")), x).backward()
    plan.check_grads()
    assert all(model.get_parameter(n).grad is None for n in plan.frozen_names)
    opt.step()
    plan.check_frozen(before)
    moved = [
        n
        for n, p in zip(plan.trainable_names, plan.parameters(), strict=True)
        if not torch.equal(p, trainable_before[n])
    ]
    assert moved == list(plan.trainable_names)


def test_i2_a11_audits_catch_violations():
    model = RoutedLM(profile("tiny", "swiglu"), NAMES)
    plan = ExpertRecipe("zh").freeze_plan(model).apply()
    hashes = plan.frozen_hashes()
    with torch.no_grad():
        model.backbone.final_norm.weight[0] += 1
    with pytest.raises(FreezeError, match="frozen tensors changed"):
        plan.check_frozen(hashes)
    with pytest.raises(FreezeError, match="outside the trainable set"):
        plan.check_optimizer(torch.optim.AdamW(model.parameters()))
    with pytest.raises(FreezeError, match="lacks"):
        plan.check_optimizer(torch.optim.AdamW(plan.parameters()[:-1]))
    alias = torch.nn.Parameter(model.backbone.emb.weight.detach())  # another object over a frozen storage
    with pytest.raises(FreezeError, match="outside the trainable set"):
        plan.check_optimizer(torch.optim.SGD(plan.parameters() + [alias], lr=0.1))
    model.backbone.emb.weight.grad = torch.zeros_like(model.backbone.emb.weight)
    with pytest.raises(FreezeError, match="have a .grad"):
        plan.check_grads()
    with pytest.raises(FreezeError, match="matches no parameter"):
        FreezePlan(model, ["bank.experts.t3."])


def test_recipes_select_their_parameters():
    seed = RoutedLM(profile("tiny", "slot"), ["seed"], device="meta")
    plan = SeedRecipe().freeze_plan(seed)
    assert plan.frozen_names == () and len(plan.trainable_names) == len(list(seed.parameters()))
    with pytest.raises(ValueError, match="seed model"):
        SeedRecipe().trainable(RoutedLM(profile("tiny", "slot"), NAMES, device="meta"))
    with pytest.raises(ValueError, match="no expert"):
        ExpertRecipe("t3").trainable(seed)
