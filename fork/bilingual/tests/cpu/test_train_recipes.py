"""Recipes (docs/spec.md, I5-T6, I5-T9, I5-T10, I5-T15; MS §8; SCI-06, IMPL-13)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from surogate_bilingual.__main__ import main as cli
from surogate_bilingual.assembly import assemble as asm
from surogate_bilingual.assembly.hashing import content_hash
from surogate_bilingual.assembly.packs import read_pack, seed_expert_init
from surogate_bilingual.audit.params import p_update
from surogate_bilingual.config import profile
from surogate_bilingual.data.sampler import EpochChunkSampler
from surogate_bilingual.model.lm import RoutedLM
from surogate_bilingual.train import schedule
from surogate_bilingual.train.loop import prepare, train
from surogate_bilingual.train.recipes import (
    BridgeRecipe,
    Inputs,
    JointWideRecipe,
    RouterRecipe,
    TargetCopyRecipe,
    stacked_seed_ffn,
)

PATH = 63_700_992  # one pilot path (MS §5.1)


def _frozen_hashes(model, names):
    return {n: content_hash({"t": model.get_parameter(n).detach()}) for n in names}


def test_i5_t6_an_expert_run_changes_only_its_expert(tiny_trained, tiny_stage, tmp_path):
    """EN trains while the frozen ZH pack runs on the zh tokens of a mixed shard (routes from the shard)."""
    run = tiny_stage(
        "tiny-en",
        'run.shard="shards/d_zh_en"',
        'run.packs={zh = "data/runs/tiny-zh/packs/zh"}',
        'run.routes={zh = "zh", en = "en"}',
    )
    result = train(run, root=tiny_trained["root"], runs=tmp_path / "runs")
    model = result.model
    assert model.expert_names == ("zh", "en") and result.step == 20
    seed, zh = read_pack(tiny_trained["seed"]), read_pack(tiny_trained["zh"])
    assert content_hash({k: v.detach() for k, v in model.backbone_state().items()}) == seed.meta["hashes"]["backbone"]
    assert content_hash({k: v.detach() for k, v in model.expert_state("zh").items()}) == zh.expert_hash
    held = {p.untyped_storage().data_ptr() for g in result.optimizer.param_groups for p in g["params"]}
    allowed = {p.untyped_storage().data_ptr() for n, p in model.named_parameters() if n.startswith("bank.experts.en.")}
    assert held == allowed
    assert content_hash({k: v.detach() for k, v in model.expert_state("en").items()}) != content_hash(
        seed_expert_init(seed, model.cfg)
    )  # the expert did move
    pack = json.loads((result.packs["en"] / "pack.json").read_text())
    assert [(d["name"], d["role"], d["expert_hash"]) for d in pack["depends_on"]] == [("zh", "context", zh.expert_hash)]


def _ce_only_reference(tiny_trained, run, steps):
    """CE-only training written here without train.loop: the seed FFN copied into the expert, AdamW (wd 0) over the
    expert alone, the closed-form schedule, the same sampler windows, clipping at 1.0, loss = sum CE / targets."""
    from surogate_bilingual.data.shards import ShardReader

    seed = read_pack(tiny_trained["seed"])
    model = asm.model_from_seed(seed, ["zh"])
    params = [p for n, p in sorted(model.named_parameters()) if n.startswith("bank.experts.zh.")]
    for n, p in model.named_parameters():
        p.requires_grad_(n.startswith("bank.experts.zh."))
    t = run.train
    opt = torch.optim.AdamW(params, lr=t.lr, betas=t.adam_betas, eps=t.adam_eps, weight_decay=0.0)
    reader = ShardReader(tiny_trained["root"] / run.run.shard, tokenizer_sha256=None, vocab_size=None)
    sampler = EpochChunkSampler(len(reader), t.seq, run.run.data_seed)
    sched = schedule.Schedule(steps, t.lr, t.warmup_frac, t.min_lr_frac)
    trace = []
    for step in range(steps):
        windows = [reader.windows(sampler.next_batch(t.microbatch)[0], t.seq) for _ in range(t.accum)]
        n = sum(int(w["loss_mask"].sum()) for w in windows)
        total = 0.0
        for w in windows:
            mask = torch.from_numpy(w["loss_mask"].astype(bool))
            logits = model(torch.from_numpy(w["inputs"]))
            ce = F.cross_entropy(logits[mask].float(), torch.from_numpy(w["targets"])[mask], reduction="sum")
            (ce / n).backward()
            total += float(ce.detach())
        torch.nn.utils.clip_grad_norm_(params, t.grad_clip)
        for g in opt.param_groups:
            g["lr"] = sched.lr(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        trace.append(total / n)
    return trace, model.expert_state("zh")


def test_i5_t6_anchor_kl_zero_is_bitwise_ce_only_training(tiny_trained, tiny_stage, tmp_path):
    run = tiny_stage("tiny-zh", "train.anchor_kl=0.0")
    result = train(run, root=tiny_trained["root"], runs=tmp_path / "zero")
    trace, expert = _ce_only_reference(tiny_trained, run, 20)
    assert result.losses == trace
    assert all(torch.equal(result.model.expert_state("zh")[k], expert[k]) for k in expert)
    anchored = train(tiny_stage("tiny-zh", "train.anchor_kl=0.5"), root=tiny_trained["root"], runs=tmp_path / "kl")
    lines = [json.loads(x) for x in (anchored.run_dir / "metrics.jsonl").read_text().splitlines()]
    assert lines[0]["kl"] == 0.0  # the expert starts as the seed FFN: p_i = p_seed exactly
    assert all(line["kl"] >= 0 for line in lines) and any(line["kl"] > 0 for line in lines)
    assert anchored.losses != trace  # the anchor is live when it is on


def test_i5_t9_the_width_2f_expert_built_from_the_seed_reproduces_the_seed(tiny_trained):
    seed = read_pack(tiny_trained["seed"])
    seed_model = asm.model_from_seed(seed, ["seed"]).eval()
    recipe = JointWideRecipe()
    cfg = recipe.model_config(seed.config)
    assert cfg.ffn_width == 2 * seed.config.ffn_width
    wide = recipe.build(
        cfg,
        Inputs(seed=seed, seed_path=tiny_trained["seed"]),
        fusion="output",
        gqa="sdpa_gqa",
        init_seed=0,
        anchor_kl=0.0,
    ).model.eval()
    tokens = torch.randint(5, cfg.vocab_size, (2, cfg.max_seq), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        diff = float((wide(tokens) - seed_model(tokens)).abs().max())
    assert diff <= 1e-6
    stacked = stacked_seed_ffn(seed.seed_ffn_state())
    assert torch.equal(
        stacked["ffn.0.down.weight"][:, : seed.config.ffn_width] * 2, seed.seed_ffn_state()["ffn.0.down.weight"]
    )
    with pytest.raises(ValueError, match="family-I"):
        recipe.model_config(profile("tiny", "slot"))
    # B12a updates two paths: over the 100M update budget by design, recorded as such
    pilot = RoutedLM(recipe.model_config(profile("pilot", "swiglu")), ["joint"], device="meta")
    plan = recipe.freeze_plan(pilot)
    assert p_update(torch.optim.AdamW(plan.parameters())) == 2 * PATH


def test_i5_t9_b12a_trains_and_packs_a_wide_expert(tiny_trained, tiny_stage, tmp_path):
    result = train(tiny_stage("tiny-b12a"), root=tiny_trained["root"], runs=tmp_path / "runs")
    pack = read_pack(result.packs["joint"])
    assert pack.config.ffn_width == 96 and pack.meta["train"]["recipe"] == "joint_wide"
    manifest = asm.assemble(tiny_trained["seed"], {"joint": result.packs["joint"]}, None, tmp_path / "a.json")
    assert asm.verify_fidelity(manifest).passed


def _bridge(tiny_trained, tiny_stage, *overrides):
    return prepare(tiny_stage("tiny-bridge", *overrides), root=tiny_trained["root"])


def test_i5_t10_only_loss_mask_positions_contribute(tiny_trained, tiny_stage):
    p = _bridge(tiny_trained, tiny_stage)
    reader = p.resolved.shard
    windows = [reader.windows([3, 40], 16), reader.windows([90, 120], 16)]
    masked_out = sum(int((w["loss_mask"] == 0).sum()) for w in windows)
    assert masked_out > 0 and all(int(w["loss_mask"].sum()) > 0 for w in windows)
    stats = p.trainer.gradients(windows)
    grads = [q.grad.clone() for q in p.plan.parameters()]
    p.trainer.optimizer.zero_grad(set_to_none=True)
    changed = []
    for w in windows:
        targets = w["targets"].copy()
        targets[w["loss_mask"] == 0] = (targets[w["loss_mask"] == 0] + 7) % 64  # other labels where mask = 0
        changed.append({**w, "targets": targets})
    assert any(not np.array_equal(a["targets"], b["targets"]) for a, b in zip(windows, changed, strict=True))
    again = p.trainer.gradients(changed)
    assert again["loss"] == stats["loss"] and again["targets"] == stats["targets"]
    assert all(torch.equal(a, b.grad) for a, b in zip(grads, p.plan.parameters(), strict=True))


def test_i5_t10_variant_c_routes_the_prefix_to_the_frozen_source_expert(tiny_trained, tiny_stage):
    p = _bridge(tiny_trained, tiny_stage)
    assert p.model.expert_names == ("zh", "en", "bridge") and p.plan.trainable_names == tuple(
        sorted(n for n, _ in p.model.named_parameters() if n.startswith("bank.experts.bridge."))
    )
    reader = p.resolved.shard
    window = reader.windows(np.arange(0, len(reader) - 17, 16)[:12], 16)  # windows spanning many pairs
    plan = p.trainer.plan_for(window["routes"])
    experts = np.asarray(p.model.expert_names)[plan.route_ids().numpy()]
    shard_routes = np.asarray(reader.route_names)[window["routes"]]
    assert (experts == shard_routes).all()  # zh prefix -> frozen zh, en prefix -> frozen en, target -> bridge
    assert {"zh", "en", "bridge"} <= set(experts.ravel())
    # the loss covers the target span only: no loss target is a token of a source prefix
    prefix = shard_routes != "bridge"
    assert not (prefix[:, 1:] & (window["loss_mask"][:, :-1] == 1)).any() and window["loss_mask"].any()
    b = BridgeRecipe(variant="b")
    assert b.names() == ("seed", "bridge") and b.route_map() == {"zh": "seed", "en": "seed", "bridge": "bridge"}
    assert BridgeRecipe(variant="a").route_map() is None and BridgeRecipe(variant="a").names() == ("bridge",)


def test_i5_t10_variant_c_records_the_zh_en_hashes(tiny_trained, tiny_stage, tmp_path):
    result = train(tiny_stage("tiny-bridge"), root=tiny_trained["root"], runs=tmp_path / "runs")
    meta = json.loads((result.packs["bridge"] / "pack.json").read_text())
    zh, en = read_pack(tiny_trained["zh"]), read_pack(tiny_trained["en"])
    assert [(d["name"], d["expert_hash"], d["role"]) for d in meta["depends_on"]] == [
        ("zh", zh.expert_hash, "source"),
        ("en", en.expert_hash, "source"),
    ]
    assert all(
        (result.packs["bridge"] / d["path"]).resolve() == tiny_trained[d["name"]].resolve() for d in meta["depends_on"]
    )
    assert meta["train"]["bridge_variant"] == "c" and meta["train"]["recipe"] == "bridge"
    manifest = asm.assemble(
        tiny_trained["seed"],
        {"zh": tiny_trained["zh"], "en": tiny_trained["en"], "bridge": result.packs["bridge"]},
        None,
        tmp_path / "a.json",
    )
    assert asm.verify_fidelity(manifest).passed


@pytest.mark.parametrize("variant", ["a", "b"])
def test_bridge_variants_a_and_b_train(tiny_trained, tiny_stage, tmp_path, variant):
    run = tiny_stage("tiny-bridge", f'train.bridge_variant="{variant}"', "run.steps=2", "run.packs={}")
    result = train(run, root=tiny_trained["root"], runs=tmp_path / "runs")
    assert (
        result.status == "done" and json.loads((result.packs["bridge"] / "pack.json").read_text())["depends_on"] == []
    )


def test_i5_t10_target_copy_trains_a_copy_and_leaves_the_original(tiny_trained, tiny_stage, tmp_path):
    original = tiny_trained["en"]
    before = (read_pack(original).expert_hash, sorted((p.name, p.read_bytes()) for p in original.iterdir()))
    result = train(tiny_stage("tiny-t1p-en"), root=tiny_trained["root"], runs=tmp_path / "runs")
    after = (read_pack(original).expert_hash, sorted((p.name, p.read_bytes()) for p in original.iterdir()))
    assert before == after  # the original pack is untouched, byte for byte
    copy = read_pack(result.packs["en"])
    assert copy.name == "en" and copy.expert_hash != before[0]
    roles = [(d["role"], d["name"], d["expert_hash"]) for d in copy.meta["depends_on"]]
    assert roles == [("source", "zh", read_pack(tiny_trained["zh"]).expert_hash), ("origin", "en", before[0])]
    doc = json.loads((result.run_dir / "run.json").read_text())
    one_path = sum(p.numel() for n, p in result.model.named_parameters() if n.startswith("bank.experts.en."))
    assert doc["p_update"] == one_path and doc["experts"] == ["zh", "en"]
    # at the pilot geometry one T1' run updates exactly one path: 63,700,992 <= 100M (two runs adapt 127.4M)
    pilot = RoutedLM(profile("pilot", "swiglu"), ["zh", "en"], device="meta")
    recipe = TargetCopyRecipe("en", "zh")
    plan = recipe.freeze_plan(pilot)
    groups = recipe.param_groups(pilot, plan, tiny_stage("tiny-t1p-en").train)
    assert p_update(torch.optim.AdamW(groups)) == PATH
    # a window holding the other direction's prefix (shard route "en") is refused, not guessed
    p = prepare(tiny_stage("tiny-t1p-en", 'run.shard="shards/d_parallel"'), root=tiny_trained["root"])
    with pytest.raises(ValueError, match="does not map"):
        p.trainer.plan_for(p.resolved.shard.windows(np.arange(0, 400, 16), 16)["routes"])


def test_i5_t15_the_router_recipe_is_deferred(tiny_trained, tmp_path, capsys):
    with pytest.raises(NotImplementedError, match="Section 8"):
        RouterRecipe()
    from pathlib import Path

    tiny = Path(__file__).resolve().parents[2] / "examples" / "tiny.toml"
    args = [
        "train",
        str(tiny),
        "--stage",
        "tiny-zh",
        "--root",
        str(tiny_trained["root"]),
        "--runs",
        str(tmp_path),
        "--set",
        'run.recipe="router"',
        "--set",
        "registry.allow_dirty=true",
    ]
    assert cli(args) == 1 and "Section 8" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


def test_lp_k_expert_keeps_the_seed_ffn_on_the_other_layers(tiny_trained, tiny_stage, tmp_path):
    result = train(tiny_stage("tiny-lp-zh"), root=tiny_trained["root"], runs=tmp_path / "runs")
    seed = read_pack(tiny_trained["seed"])
    layer0 = {
        k.removeprefix("backbone.layers.0.ffn."): v
        for k, v in result.model.backbone_state().items()
        if k.startswith("backbone.layers.0.ffn.")
    }
    assert layer0 and all(torch.equal(v, seed.seed_ffn_state()[f"ffn.0.{k}"]) for k, v in layer0.items())
    assert read_pack(result.packs["zh"]).config.expert_layers == (1,)


# --- the row file of the minimal set (examples/pilot_rows.toml) ------------------------------------------------------

PILOT_ROWS = Path(__file__).resolve().parents[2] / "examples" / "pilot_rows.toml"
P_UPDATE = {  # MS §5.1 / I2-A12: what one run of each kind may update at its geometry
    ("pilot", "swiglu", "expert"): PATH,
    ("pilot", "swiglu", "lp4"): 21_233_664,
    ("pilot", "slot", "expert"): 42_416_640,
    ("pilot", "swiglu", "joint_wide"): 2 * PATH,
}


def test_pilot_rows_are_consistent_with_the_minimal_set():
    from collections import defaultdict

    from surogate_bilingual.config import load_toml, stage_names
    from surogate_bilingual.train.recipes import recipe_from_spec

    runs = {name: load_toml(PILOT_ROWS, stage=name) for name in stage_names(PILOT_ROWS)}
    assert len(runs) == 60 and all(name == run.run.name for name, run in runs.items())
    seeds = [s for run in runs.values() for s in (run.run.init_seed, run.run.data_seed)]
    assert len(set(seeds)) == len(seeds)
    tokens = defaultdict(int)
    for name, run in runs.items():
        spec, recipe = run.run, recipe_from_spec(run.run, run.train)
        assert run.train.global_batch_tokens == 65536 and run.train.lr in (6e-4, 2e-4)
        block = (
            "g1p"
            if name.startswith("g1p")
            else "II"
            if run.model.family == "slot"
            else ("bridge" if spec.matrix_row.startswith("T") else "I")
        )
        tokens[(block, spec.replicate)] += spec.tokens
        if recipe.kind == "seed":
            assert spec.seed_pack is None and run.model.expert_layers is None
            continue
        seed_run = runs[spec.seed_pack.split("/")[2]]  # data/runs/<seed run>/packs/seed
        assert seed_run.run.recipe == "seed" and seed_run.run.replicate == spec.replicate
        assert run.model.replace(expert_layers=None) == seed_run.model
        for slot, path in spec.packs.items():
            source = runs[path.split("/")[2]]
            assert source.run.expert == slot and source.run.replicate == spec.replicate
            assert source.run.seed_pack == spec.seed_pack
    for r in (0, 1, 2):
        assert (tokens[("g1p", r)], tokens[("I", r)], tokens[("II", r)], tokens[("bridge", r)]) == (
            900_000_000,
            2_400_000_000,
            1_200_000_000,
            400_000_000,
        )


def test_pilot_rows_update_what_ms_5_1_budgets():
    from surogate_bilingual.config import load_toml, stage_names
    from surogate_bilingual.train.recipes import recipe_from_spec

    seen = {}
    for name in stage_names(PILOT_ROWS):
        run = load_toml(PILOT_ROWS, stage=name)
        recipe = recipe_from_spec(run.run, run.train)
        if not name.endswith("-r0") or name.startswith("g1p") or recipe.kind == "seed":
            continue
        kind = "lp4" if run.model.expert_layers else ("joint_wide" if recipe.kind == "joint_wide" else "expert")
        key = ("pilot", run.model.family, kind)
        if key not in seen:
            model = RoutedLM(recipe.model_config(run.model), recipe.names(), device="meta")
            seen[key] = recipe.freeze_plan(model).numel()
        assert seen[key] == P_UPDATE[key], name
    assert set(seen) == set(P_UPDATE)


# --- the Bridge layout of parallel shards (data.parallel, `shard --bridge`) -----------------------------------------


def test_bridge_layout_unpair_and_direction_on_track_s_pairs(track_s_ci):
    from surogate_bilingual.data import parallel
    from surogate_bilingual.data.docs import iter_file_docs

    docs = list(iter_file_docs(track_s_ci[0] / "parallel" / "track_s.jsonl", source=None, kind=None))
    assert {parallel.direction(d) for d in docs} == {"zh>en", "en>zh"}
    laid = [parallel.bridge_layout(d) for d in docs]
    for before, after in zip(docs, laid, strict=True):
        src, tgt = after.segments
        assert (src.route, src.loss, tgt.route, tgt.loss) == (src.lang, False, "bridge", True)
        assert [s.text for s in after.segments] == [s.text for s in before.segments if s.text is not None]
    zh_en = parallel.select_direction(docs, "zh>en")
    assert zh_en and all(parallel.direction(d) == "zh>en" for d in zh_en)
    broken = parallel.unpair(docs, seed=5)
    assert broken == parallel.unpair(docs, seed=5) and broken != parallel.unpair(docs, seed=6)
    for before, after in zip(docs, broken, strict=True):
        assert after.segments[0].text == before.segments[0].text  # the source stays
        assert after.segments[1].text != before.segments[1].text  # its own target never does
        assert parallel.direction(after) == parallel.direction(before)
    assert sorted(d.segments[1].text for d in broken) == sorted(d.segments[1].text for d in docs)  # same sentences


def test_shard_bridge_command_writes_the_layout_the_recipes_read(data_pipeline, track_s_ci, tmp_path):
    from surogate_bilingual.data.shards import ShardReader

    base = [
        "shard",
        "--manifest",
        str(data_pipeline["base"] / "splits" / "manifest.json"),
        "--split",
        "parallel_train",
        "--tokenizer",
        str(data_pipeline["tokenizer_path"]),
        "--add",
        str(track_s_ci[0] / "parallel" / "track_s.jsonl"),
        "--seq",
        "64",
        "--bridge",
    ]
    assert cli([*base, "--out", str(tmp_path / "both")]) == 0
    assert cli([*base, "--direction", "zh>en", "--out", str(tmp_path / "zh_en")]) == 0
    assert cli([*base, "--unpair-seed", "3", "--out", str(tmp_path / "unpaired")]) == 0
    for name in ("both", "zh_en", "unpaired"):
        reader = ShardReader(tmp_path / name, tokenizer_sha256=data_pipeline["tokenizer_sha256"], vocab_size=1024)
        names = np.asarray(reader.route_names)[np.asarray(reader.routes)]
        loss = np.asarray(reader.loss_mask)
        assert set(names) == ({"zh", "bridge"} if name == "zh_en" else {"zh", "en", "bridge"})
        assert (loss[names == "bridge"] == 1).all() and (loss[names != "bridge"] == 0).all()
    both = [
        "shard",
        "--manifest",
        str(data_pipeline["base"] / "splits" / "manifest.json"),
        "--tokenizer",
        str(data_pipeline["tokenizer_path"]),
        "--seq",
        "64",
    ]
    assert cli([*both, "--split", "zh_train,en_train", "--out", str(tmp_path / "mono")]) == 0  # B2, B12a, T3''
    header = json.loads((tmp_path / "mono" / "shard.json").read_text())
    assert header["split"] == "zh_train+en_train" and len(header["inputs"]) >= 2
    with pytest.raises(SystemExit):  # --direction needs --bridge
        cli([*base[:-1], "--direction", "zh>en", "--out", str(tmp_path / "x")])
