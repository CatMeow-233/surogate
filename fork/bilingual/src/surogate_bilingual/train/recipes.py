"""Recipes: what a training run may change (step I2 stub).

Step I2 defines only the trainable sets, which the freeze audits and the budget's P_update need; the loop,
objectives beyond masked CE, schedules and the joint-wide, Bridge, target-copy and router recipes land in step I5.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..assembly.freeze import FreezePlan


class Recipe:
    name = "recipe"

    def trainable(self, model) -> list[str]:
        raise NotImplementedError

    def freeze_plan(self, model) -> FreezePlan:
        return FreezePlan(model, self.trainable(model))


@dataclass(frozen=True)
class SeedRecipe(Recipe):
    """The seed S: every parameter (backbone and the one FFN, named "seed") is trained."""

    name = "seed"

    def trainable(self, model) -> list[str]:
        if model.expert_names != ("seed",):
            raise ValueError(f"the seed recipe trains a seed model (experts ('seed',)), not {model.expert_names}")
        return ["backbone.", "bank.experts.seed."]


@dataclass(frozen=True)
class ExpertRecipe(Recipe):
    """One expert forked from the seed: only bank.experts.<expert> is trained; theta_S and every other expert are
    frozen. `anchor_kl` (KL to the seed) is applied by the loop in step I5."""

    expert: str
    anchor_kl: float = 0.0
    name = "expert"

    def trainable(self, model) -> list[str]:
        if self.expert not in model.expert_names:
            raise ValueError(f"no expert {self.expert!r} in {model.expert_names}")
        return [f"bank.experts.{self.expert}."]
