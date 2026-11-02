"""Recipes: what one training run builds, trains, freezes, routes and writes (MS §8; docs/spec.md, I5).

A recipe owns the layout of its run's model (which experts it holds, which one is trained, where each starts),
the mapping from shard routes to those experts, its parameter groups and the packs it writes. The loop
(`train.loop`) owns everything else: inputs, schedule, optimizer steps, state, registry.

| recipe | experts in the model | trained | starts from | shard route -> expert |
|---|---|---|---|---|
| seed | seed | everything | `init_seed` | every token on seed |
| expert (B1, B2, LP-k) | [context...,] E | E | the seed FFN | every token on E; with context packs, `run.routes` |
| joint_wide (B12a) | E, width 2f | E | the seed FFN stacked twice, down halved | every token on E |
| bridge (a) | bridge | bridge | the seed FFN | every token on bridge |
| bridge (b) | seed, bridge | bridge | the seed FFN; the prefix runs a frozen seed-FFN copy | zh, en -> seed; bridge -> bridge |
| bridge (c), B-O13 | zh, en, bridge | bridge | the seed FFN; the prefix runs the frozen ZH/EN packs | zh -> zh, en -> en, bridge -> bridge |
| target_copy (T1') | S, T | a copy of pack T | pack T (the original is never written) | S -> S, bridge -> T |
| router | - | - | - | raises NotImplementedError: deferred (Section 8) |

The theta_S of every non-seed recipe is the seed pack's, frozen, and every frozen expert is a pack loaded with its
hashes verified. The loss covers the shard's loss-mask positions only, so the Bridge and T1' recipes, trained on
parallel shards whose source prefix has loss 0, learn on the target span only (MS §8.4). The first target token is
predicted at the last source position, whose route is the source expert's, so its loss is counted but no
trainable parameter can lower it: the same causal routing as at inference (R_out).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..assembly.freeze import FreezePlan
from ..assembly.packs import (
    ExpertPack,
    SeedPack,
    check_extends,
    seed_backbone_for,
    seed_expert_init,
    write_expert_pack,
    write_seed_pack,
)
from ..config import ModelConfig, TrainConfig
from ..model.lm import RoutedLM
from . import schedule

DEFERRED = (
    "learned-router training is deferred to P1 (IMPLEMENTATION_SPEC Section 8; docs/spec.md 推迟项): pick R-early,"
    " R-late or LM-head tag emission first"
)


class RecipeError(ValueError):
    pass


@dataclass(frozen=True)
class Inputs:
    """The packs a run starts from, read and verified by the loop."""

    seed: SeedPack | None = None
    seed_path: Path | None = None
    packs: Mapping[str, tuple[Path, ExpertPack]] = field(default_factory=dict)
    tokenizer: Path | None = None


@dataclass
class Built:
    model: RoutedLM
    anchors: dict[str, torch.Tensor] = field(default_factory=dict)  # trained parameter -> w_seed (to_seed decay)
    reference: RoutedLM | None = None  # the seed model, for the KL anchor


class Recipe:
    kind = "recipe"
    trained: str = ""

    def trainable(self, model) -> list[str]:
        raise NotImplementedError

    def freeze_plan(self, model) -> FreezePlan:
        return FreezePlan(model, self.trainable(model))

    def names(self) -> tuple[str, ...]:
        raise NotImplementedError

    def route_map(self) -> dict[str, str] | None:
        """shard route name -> expert, or None when the model holds one expert and needs no plan."""
        return None

    def args(self) -> dict[str, Any]:
        """The recipe's arguments, part of the run config (and so of the run id)."""
        return {}

    def model_config(self, cfg: ModelConfig) -> ModelConfig:
        return cfg

    def build(
        self, cfg: ModelConfig, inputs: Inputs, *, fusion: str, gqa: str, init_seed: int, anchor_kl: float
    ) -> Built:
        raise NotImplementedError

    def param_groups(self, model, plan: FreezePlan, train: TrainConfig) -> list[dict[str, Any]]:
        named = list(zip(plan.trainable_names, plan.parameters(), strict=True))
        return schedule.expert_groups(named, train.expert_decay, train.expert_decay_lambda)

    def depends_on(self, inputs: Inputs) -> list[tuple[str, Path]]:
        return []

    def write_packs(self, model, run_dir: Path, inputs: Inputs, meta: Mapping[str, Any]) -> dict[str, Path]:
        out = Path(run_dir) / "packs" / self.trained
        write_expert_pack(out, model, self.trained, inputs.seed_path, meta, depends_on=self.depends_on(inputs))
        return {self.trained: out}


def _need_seed(inputs: Inputs, kind: str) -> SeedPack:
    if inputs.seed is None:
        raise RecipeError(f"the {kind} recipe forks from a seed pack: set run.seed_pack")
    return inputs.seed


def _pack(inputs: Inputs, name: str, kind: str) -> ExpertPack:
    if name not in inputs.packs:
        raise RecipeError(f"the {kind} recipe needs the expert pack {name!r} in run.packs")
    return inputs.packs[name][1]


def _check_pack_against(seed: SeedPack, pack: ExpertPack, cfg: ModelConfig, slot: str) -> None:
    if pack.seed_fingerprint != seed.seed_fingerprint:
        raise RecipeError(f"pack {pack.path} ({slot}) was forked from another seed than {seed.path}")
    if pack.config != cfg:
        raise RecipeError(f"pack {pack.path} ({slot}) has another config than this run's model")
    if pack.name != slot:
        raise RecipeError(f"run.packs[{slot!r}] holds the pack of expert {pack.name!r}")


def _model(cfg, names, fusion, gqa, init_seed=0) -> RoutedLM:
    return RoutedLM(cfg, names, fusion, gqa, init_seed=init_seed)


def _seed_reference(seed: SeedPack, cfg: ModelConfig, fusion: str, gqa: str) -> RoutedLM:
    """The seed model on this run's geometry (LP-k included): theta_S plus the seed FFN, frozen, for KL."""
    ref = _model(cfg, ["seed"], fusion, gqa)
    ref.load_backbone_state(seed_backbone_for(seed, cfg))
    ref.load_expert_state("seed", seed_expert_init(seed, cfg))
    ref.requires_grad_(False)
    return ref.eval()


def _anchors(model: RoutedLM, name: str, state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {f"bank.experts.{name}.{k.removeprefix('ffn.')}": v.detach().clone() for k, v in state.items()}


@dataclass(frozen=True)
class SeedRecipe(Recipe):
    """The seed S (MS §8.1): every parameter (theta_S and the one FFN, named "seed") is trained on D_seed."""

    kind = "seed"
    trained = "seed"

    def trainable(self, model) -> list[str]:
        if model.expert_names != ("seed",):
            raise ValueError(f"the seed recipe trains a seed model (experts ('seed',)), not {model.expert_names}")
        return ["backbone.", "bank.experts.seed."]

    def names(self) -> tuple[str, ...]:
        return ("seed",)

    def build(self, cfg, inputs, *, fusion, gqa, init_seed, anchor_kl) -> Built:
        if cfg.expert_layers is not None:
            raise RecipeError("a seed holds its FFN on every layer: model.expert_layers must be unset")
        if anchor_kl:
            raise RecipeError("train.anchor_kl applies to experts; the seed recipe has no seed to anchor to")
        return Built(_model(cfg, ["seed"], fusion, gqa, init_seed))

    def param_groups(self, model, plan, train) -> list[dict[str, Any]]:
        named = list(zip(plan.trainable_names, plan.parameters(), strict=True))
        return schedule.seed_groups(named, train.weight_decay)

    def write_packs(self, model, run_dir, inputs, meta) -> dict[str, Path]:
        if inputs.tokenizer is None:
            raise RecipeError("the seed recipe records its tokenizer in the seed pack: set run.tokenizer")
        out = Path(run_dir) / "packs" / "seed"
        write_seed_pack(out, model, inputs.tokenizer, meta)
        return {"seed": out}


@dataclass(frozen=True)
class ExpertRecipe(Recipe):
    """One expert forked from the seed FFN (MS §8.2): only bank.experts.<expert> trains; theta_S and every context
    expert are frozen. `context` names frozen expert packs the forward also runs, routed by `routes`."""

    expert: str
    anchor_kl: float = 0.0
    context: tuple[str, ...] = ()
    routes: tuple[tuple[str, str], ...] = ()
    kind = "expert"

    @property
    def trained(self) -> str:
        return self.expert

    def trainable(self, model) -> list[str]:
        if self.expert not in model.expert_names:
            raise ValueError(f"no expert {self.expert!r} in {model.expert_names}")
        return [f"bank.experts.{self.expert}."]

    def names(self) -> tuple[str, ...]:
        if self.expert in self.context or len(set(self.context)) != len(self.context):
            raise RecipeError(
                f"context experts {self.context} must be distinct from each other and from {self.expert!r}"
            )
        return (*self.context, self.expert)

    def route_map(self) -> dict[str, str] | None:
        if not self.context:
            if self.routes:
                raise RecipeError("run.routes routes tokens between experts; a lone expert takes every token")
            return None
        if not self.routes:
            raise RecipeError("an expert trained among context experts needs run.routes (shard route -> expert)")
        return dict(self.routes)

    def args(self) -> dict[str, Any]:
        return {"expert": self.expert, "context": list(self.context), "routes": dict(self.routes)}

    def _start(self, seed: SeedPack, cfg: ModelConfig) -> dict[str, torch.Tensor]:
        return seed_expert_init(seed, cfg)

    def build(self, cfg, inputs, *, fusion, gqa, init_seed, anchor_kl) -> Built:
        seed = _need_seed(inputs, self.kind)
        check_extends(seed, cfg)
        model = _model(cfg, self.names(), fusion, gqa)
        model.load_backbone_state(seed_backbone_for(seed, cfg))
        for name in self.context:
            pack = _pack(inputs, name, self.kind)
            _check_pack_against(seed, pack, cfg, name)
            model.load_expert_state(name, pack.weights)
        start = self._start(seed, cfg)
        model.load_expert_state(self.expert, start)
        reference = _seed_reference(seed, cfg, fusion, gqa) if anchor_kl > 0 else None
        return Built(model, _anchors(model, self.expert, seed_expert_init(seed, cfg)), reference)

    def depends_on(self, inputs: Inputs) -> list[tuple[str, Path]]:
        return [("context", inputs.packs[name][0]) for name in self.context]


@dataclass(frozen=True)
class JointWideRecipe(ExpertRecipe):
    """B12a (MS §7.1): the non-partitioned, capacity-matched ceiling. One family-I expert of width copies x f,
    built from the seed FFN with gate/up rows stacked and down columns scaled by 1/copies, so it starts as the seed's
    function; trained on D_zh u D_en. Over the update budget (2 paths), reported as such."""

    expert: str = "joint"
    copies: int = 2
    kind = "joint_wide"

    def args(self) -> dict[str, Any]:
        return {"expert": self.expert, "copies": self.copies}

    def model_config(self, cfg: ModelConfig) -> ModelConfig:
        if cfg.family != "swiglu" or cfg.expert_layers is not None:
            raise RecipeError("B12a is a family-I arm with the expert on every layer (MS §7.1)")
        if self.copies < 2:
            raise RecipeError("joint_wide needs copies >= 2")
        return cfg.replace(ffn_width=self.copies * cfg.ffn_width)

    def _start(self, seed: SeedPack, cfg: ModelConfig) -> dict[str, torch.Tensor]:
        return stacked_seed_ffn(seed.seed_ffn_state(), self.copies)

    def build(self, cfg, inputs, *, fusion, gqa, init_seed, anchor_kl) -> Built:
        if self.context or self.routes:
            raise RecipeError("joint_wide trains one expert on every token; it takes no context experts")
        seed = _need_seed(inputs, self.kind)
        base = cfg.replace(ffn_width=seed.config.ffn_width)
        model = _model(cfg, [self.expert], fusion, gqa)
        model.load_backbone_state(seed_backbone_for(seed, cfg))
        start = self._start(seed, cfg)
        model.load_expert_state(self.expert, start)
        reference = _seed_reference(seed, base, fusion, gqa) if anchor_kl > 0 else None
        return Built(model, _anchors(model, self.expert, start), reference)


def stacked_seed_ffn(seed_ffn: Mapping[str, torch.Tensor], copies: int = 2) -> dict[str, torch.Tensor]:
    """The width-(copies x f) SwiGLU equal to the seed FFN: gate/up rows repeated, down columns divided by copies
    (a power of two divides exactly, so only the final sum rounds)."""
    out = {}
    for key, value in seed_ffn.items():
        if key.endswith(".down.weight"):
            out[key] = torch.cat([value / copies] * copies, dim=1)
        elif key.endswith((".gate.weight", ".up.weight")):
            out[key] = torch.cat([value] * copies, dim=0)
        else:
            raise RecipeError(f"stacked_seed_ffn takes family-I FFN keys, got {key!r}")
    return out


@dataclass(frozen=True)
class BridgeRecipe(ExpertRecipe):
    """The Bridge (MS §8.4, B-O13): a third expert from the seed FFN, trained on parallel data with the loss on
    the target span only. (a) every token on the Bridge; (b) the source prefix on a frozen copy of the seed FFN;
    (c, default) the source prefix on the frozen ZH/EN experts, whose hashes the Bridge pack records."""

    expert: str = "bridge"
    variant: str = "c"
    kind = "bridge"

    def __post_init__(self):
        if self.variant not in ("a", "b", "c"):
            raise RecipeError(f"bridge variant must be a, b or c: {self.variant!r}")

    def names(self) -> tuple[str, ...]:
        return {"a": (self.expert,), "b": ("seed", self.expert), "c": ("zh", "en", self.expert)}[self.variant]

    def route_map(self) -> dict[str, str] | None:
        return {
            "a": None,
            "b": {"zh": "seed", "en": "seed", "bridge": self.expert},
            "c": {"zh": "zh", "en": "en", "bridge": self.expert},
        }[self.variant]

    def args(self) -> dict[str, Any]:
        return {"expert": self.expert, "variant": self.variant}

    def build(self, cfg, inputs, *, fusion, gqa, init_seed, anchor_kl) -> Built:
        seed = _need_seed(inputs, self.kind)
        check_extends(seed, cfg)
        model = _model(cfg, self.names(), fusion, gqa)
        model.load_backbone_state(seed_backbone_for(seed, cfg))
        if self.variant == "b":
            model.load_expert_state("seed", seed_expert_init(seed, cfg))
        if self.variant == "c":
            for name in ("zh", "en"):
                pack = _pack(inputs, name, "bridge (c)")
                _check_pack_against(seed, pack, cfg, name)
                model.load_expert_state(name, pack.weights)
        start = seed_expert_init(seed, cfg)
        model.load_expert_state(self.expert, start)
        reference = _seed_reference(seed, cfg, fusion, gqa) if anchor_kl > 0 else None
        return Built(model, _anchors(model, self.expert, start), reference)

    def depends_on(self, inputs: Inputs) -> list[tuple[str, Path]]:
        if self.variant != "c":
            return []
        return [("source", inputs.packs["zh"][0]), ("source", inputs.packs["en"][0])]


@dataclass(frozen=True)
class TargetCopyRecipe(Recipe):
    """T1' (MS §7.3; SCI-06): fine-tune a **copy** of the target-language expert on the same parallel tokens as the
    Bridge, one direction per run. The source prefix runs the frozen source expert; the shard's "bridge" route (the
    target span) goes to the copy; the loss covers the target span only. The original pack is only read."""

    target: str
    source: str
    kind = "target_copy"

    @property
    def trained(self) -> str:
        return self.target

    def trainable(self, model) -> list[str]:
        if self.target not in model.expert_names:
            raise ValueError(f"no expert {self.target!r} in {model.expert_names}")
        return [f"bank.experts.{self.target}."]

    def names(self) -> tuple[str, ...]:
        if self.target == self.source:
            raise RecipeError("T1' needs two different experts: target and source")
        return (self.source, self.target)

    def route_map(self) -> dict[str, str] | None:
        return {self.source: self.source, "bridge": self.target}

    def args(self) -> dict[str, Any]:
        return {"target": self.target, "source": self.source}

    def build(self, cfg, inputs, *, fusion, gqa, init_seed, anchor_kl) -> Built:
        seed = _need_seed(inputs, self.kind)
        check_extends(seed, cfg)
        model = _model(cfg, self.names(), fusion, gqa)
        model.load_backbone_state(seed_backbone_for(seed, cfg))
        for name in (self.source, self.target):
            pack = _pack(inputs, name, "target_copy")
            _check_pack_against(seed, pack, cfg, name)
            model.load_expert_state(name, {k: v.clone() for k, v in pack.weights.items()})
        reference = _seed_reference(seed, cfg, fusion, gqa) if anchor_kl > 0 else None
        return Built(model, _anchors(model, self.target, seed_expert_init(seed, cfg)), reference)

    def depends_on(self, inputs: Inputs) -> list[tuple[str, Path]]:
        return [("source", inputs.packs[self.source][0]), ("origin", inputs.packs[self.target][0])]


@dataclass(frozen=True)
class RouterRecipe(Recipe):
    """Learned-router training (B5) is deferred; the recipe exists so that a row naming it fails loudly."""

    kind = "router"

    def __post_init__(self):
        raise NotImplementedError(DEFERRED)


def recipe_from_spec(spec, train: TrainConfig) -> Recipe:
    """The recipe a [run] table names, with its arguments."""
    kind = spec.recipe
    if kind == "router":
        return RouterRecipe()
    if kind == "seed":
        return SeedRecipe()
    if kind in ("expert", "joint_wide"):
        if not spec.expert and kind == "expert":
            raise RecipeError("the expert recipe needs run.expert (the trained expert's name)")
        if kind == "joint_wide":
            return JointWideRecipe(expert=spec.expert or "joint", anchor_kl=train.anchor_kl)
        context = tuple(sorted(spec.packs))
        return ExpertRecipe(spec.expert, train.anchor_kl, context, tuple(sorted(spec.routes.items())))
    if kind == "bridge":  # T2, and T3' (a third expert, e.g. "t3", on unpaired parallel data)
        return BridgeRecipe(expert=spec.expert or "bridge", anchor_kl=train.anchor_kl, variant=train.bridge_variant)
    if kind == "target_copy":
        if not spec.target or not spec.source:
            raise RecipeError("the target_copy recipe needs run.target and run.source")
        return TargetCopyRecipe(spec.target, spec.source)
    raise RecipeError(f"unknown recipe {kind!r}")


def expected_packs(recipe: Recipe) -> Sequence[str]:
    """The run.packs entries a recipe reads."""
    if isinstance(recipe, BridgeRecipe):
        return ("zh", "en") if recipe.variant == "c" else ()
    if isinstance(recipe, TargetCopyRecipe):
        return (recipe.source, recipe.target)
    if isinstance(recipe, ExpertRecipe) and not isinstance(recipe, JointWideRecipe):
        return recipe.context
    return ()
