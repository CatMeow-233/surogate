"""I4-D7: Track S (SCI-01, SCI-08, SCI-11; MS §6.2)."""

import copy
import json
import tomllib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from surogate_bilingual.canonical import sha256_file
from surogate_bilingual.data.synth import build as B
from surogate_bilingual.data.synth import leak
from surogate_bilingual.data.synth.checker import Checker
from surogate_bilingual.data.synth.entities import Person, by_group, draw_people
from surogate_bilingual.data.synth.render import Renderer
from surogate_bilingual.data.synth.tasks import K, ctx_items, par_items, qa_items
from surogate_bilingual.data.synth.templates import TemplateError, load_registry
from surogate_bilingual.data.synth.values import ATTRIBUTES, LISTS, find_attributes, lexicon

TRACK_S_SPEC = Path(__file__).resolve().parents[2] / "examples" / "track_s.toml"


def _load(out, rel):
    return [json.loads(line) for line in (out / rel).read_text(encoding="utf-8").splitlines()]


def _people(out):
    return [Person.from_json(d) for d in _load(out, "entities.jsonl")]


def _items(out):
    return [it for f in sorted((out / "items").glob("*.jsonl")) for it in _load(out, f"items/{f.name}")]


@pytest.fixture(scope="module")
def full():
    spec, prof, _ = B.load_spec(TRACK_S_SPEC, "full")
    registry = load_registry(spec["templates"])
    return spec, prof, registry, B.lexicons_for(prof)


# --- determinism ---------------------------------------------------------------------------------------------------


def test_i4_d7_same_version_and_seed_give_identical_bytes(track_s_ci, tmp_path):
    out, card = track_s_ci
    again = tmp_path / "again"
    B.build_track_s(TRACK_S_SPEC, again, seed=20261102, profile="ci")
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(again).as_posix() for p in again.rglob("*") if p.is_file())
    for rel in files:
        assert (out / rel).read_bytes() == (again / rel).read_bytes(), rel
    assert set(card["files"]) == set(files) - {"datacard.json"}
    for rel, digest in card["files"].items():
        assert sha256_file(out / rel) == digest
    assert card["generator"]["version"] == B.TRACK_S_VERSION and card["generator"]["seed"] == 20261102
    assert card["generator"]["spec_sha256"] == sha256_file(TRACK_S_SPEC)
    other = tmp_path / "other"
    B.build_track_s(TRACK_S_SPEC, other, seed=1, profile="ci")
    assert (other / "items" / "E3ctx.test.jsonl").read_bytes() != (out / "items" / "E3ctx.test.jsonl").read_bytes()
    with pytest.raises(B.BuildError, match="exists"):
        B.build_track_s(TRACK_S_SPEC, other, seed=1, profile="ci")


# --- gold answers --------------------------------------------------------------------------------------------------


def test_i4_d7_checker_rederives_every_gold_and_finds_the_facts(track_s_ci):
    out, card = track_s_ci
    people, items = _people(out), _items(out)
    checker = Checker(people, B.lexicons_for(card["profile"]))
    for component in B.COMPONENTS:
        checker.scan_bios(_load(out, f"{component}/track_s.jsonl"), component)
    report = checker.check_items(items, corpora={"zh": "zh", "en": "en"})
    assert report.ok, report.errors[:5]
    assert not checker.doc_errors
    assert report.items == len(items) == sum(card["checker"]["by_family"].values())
    assert set(report.by_family) == {"E1", "E2", "E3par", "E3ctx", "E4"}
    assert card["status"]["checker"] is True


def test_i4_d7_checker_catches_corrupted_items(track_s_ci):
    out, card = track_s_ci
    people, items = _people(out), _items(out)
    checker = Checker(people, B.lexicons_for(card["profile"]))
    first = {f: next(it for it in items if it["family"] == f) for f in ("E1", "E2", "E3par", "E3ctx", "E4")}
    bad = []
    for family, item in first.items():
        broken = copy.deepcopy(item)
        if family == "E4":
            key = next(k for k in broken["slots"] if k != "name")
            broken["slots"][key] = "wrong"
            broken["stages"][-1]["slots"][key] = "wrong"
        else:
            broken["gold_index"] = (broken["gold_index"] + 1) % len(broken["candidates"])
        bad.append(broken)
    ctx = copy.deepcopy(first["E3ctx"])  # the text, not the structured fields, decides the E3-ctx gold
    fact = next(s for s in ctx["segments"] if s["role"] == "fact")
    color = lexicon("color")
    target = ctx["structured"]["category"]
    lang = fact["lang"]
    other = next(c for c in color.surfaces(lang) if c != color.surface(color.index_of(target, "en"), lang))
    fact["text"] = fact["text"].replace(color.surface(color.index_of(target, "en"), lang), other)
    bad.append(ctx)
    report = checker.check_items(bad)
    failed = {e.split(":")[0] for e in report.errors}
    assert failed == {it["id"] for it in bad}


# --- disjointness, sizes, partitions, co-occurrence -------------------------------------------------------------------


def test_i4_d7_template_ids_are_disjoint_across_splits(track_s_ci):
    out, card = track_s_ci
    disjoint = card["templates"]["disjointness"]
    assert set(disjoint) == {"E1", "E2", "E3par", "E3ctx", "E4"}
    for family, info in disjoint.items():
        assert {"train", "dev", "test"} <= set(info["counts"]), family
        assert all(v == [] for v in info["overlap"].values()), family
        assert all(v == [] for v in info["part_overlap"].values()), family  # no part of a test template trains
    # independently: the registry pools never share a text between splits of one role
    spec = tomllib.loads(TRACK_S_SPEC.read_text(encoding="utf-8"))
    registry = load_registry(spec["templates"])
    texts = {}
    for t in registry.templates:
        texts.setdefault((t.lang, t.role, t.text), set()).add(t.split)
    assert all(len(splits) == 1 for splits in texts.values())


def test_i4_d7_registry_refuses_a_text_shared_by_two_splits():
    spec = tomllib.loads(TRACK_S_SPEC.read_text(encoding="utf-8"))
    bad = copy.deepcopy(spec["templates"])
    bad["en"]["frame"]["test"].append(bad["en"]["frame"]["train"][0])
    with pytest.raises(TemplateError, match="in both train and test"):
        load_registry(bad)
    bad = copy.deepcopy(spec["templates"])
    bad["zh"]["qa"]["city"]["train"].append("{name}出生在哪里？")  # a name glued to Han characters
    with pytest.raises(TemplateError, match="touches a Han character"):
        load_registry(bad)
    bad = copy.deepcopy(spec["templates"])
    bad["en"]["ctx_fact"]["train"].append("Group {entity} is {category}, not red.")
    with pytest.raises(TemplateError, match="lexicon surface 'red'"):
        load_registry(bad)


def test_i4_d7_full_size_build_has_100_test_templates_and_2000_items_per_family_and_direction(full):
    spec, prof, registry, lex = full
    for family in ("E1", "E2", "E3par", "E3ctx", "E4"):
        assert registry.family_count(family, "test", tuple(prof["e4_attributes"])) >= 100, family
    assert prof["items"]["test"] >= 2000 and prof["items"]["dev"] >= 500
    # generate the full-size test items (no training documents) and count what they use
    people = draw_people(
        prof["partitions"],
        {a: len(lex[a]) for a in ATTRIBUTES},
        {k: prof[k] for k in ("heldout_fraction", "eval_dev_fraction", "par_train_fraction")},
        B.stream(5, "people"),
    )
    small = dict(prof, items={"test": prof["items"]["test"], "dev": 0})
    from surogate_bilingual.data.synth.entities import CodeAllocator

    items = B.eval_items(people, Renderer(registry, lex), small, 5, CodeAllocator(B.stream(5, "codes")), {"seed": 5})
    test = [it for (_, split), v in items.items() if split == "test" for it in v]
    per_direction = Counter((it["family"], it["variant"], it["direction"]) for it in test)
    assert len(per_direction) == 16 and min(per_direction.values()) >= 2000
    used = leak.template_sets(test)
    for family in ("E1", "E2", "E3par", "E3ctx", "E4"):
        assert len(used[family]["test"]) >= 100, family
    report = Checker(people, lex).check_items(test)
    assert report.ok, report.errors[:3]


def test_i4_d7_entity_partitions_are_disjoint_and_sized(track_s_ci):
    out, card = track_s_ci
    people = _people(out)
    assert len({p.id for p in people}) == len({p.name for p in people}) == len(people)
    sizes = Counter(p.partition for p in people)
    assert dict(sizes) == card["profile"]["partitions"]
    groups = {(p.partition, p.group) for p in people}
    assert groups == {
        ("seed", "train"),
        ("zh", "qa_train"),
        ("zh", "dev"),
        ("zh", "test"),
        ("en", "qa_train"),
        ("en", "dev"),
        ("en", "test"),
        ("par", "train"),
        ("par", "dev"),
        ("par", "test"),
    }
    # items only ask about held-out people of the right split; training documents never name them in QA
    by_id = {p.id: p for p in people}
    for it in _items(out):
        s = it["structured"]
        for pid in [s.get("person"), s.get("a"), s.get("b")]:
            if pid:
                assert by_id[pid].group == it["split"], it["id"]
    for component in B.COMPONENTS:
        for doc in _load(out, f"{component}/track_s.jsonl"):
            if doc["meta"]["family"] == "qa":
                assert all(by_id[p].group in ("train", "qa_train") for p in doc["meta"]["persons"])
            if doc["meta"]["family"] == "parallel":
                assert all(by_id[p].group == "train" for p in doc["meta"]["persons"])


def test_i4_d7_no_test_pair_cooccurs_and_codes_are_fresh(track_s_ci):
    out, card = track_s_ci
    people = _people(out)
    by_id = {p.id: p for p in people}
    checker = Checker(people, B.lexicons_for(card["profile"]))
    texts = [d for c in B.COMPONENTS for d in _load(out, f"{c}/track_s.jsonl")]
    named = [checker.names_in("\n".join(s["text"] for s in d["segments"] if "text" in s)) for d in texts]
    pairs = [
        (by_id[it["structured"]["a"]].name, by_id[it["structured"]["b"]].name)
        for it in _items(out)
        if it["family"] == "E3par" and it["split"] == "test"
    ]
    assert pairs and leak.cooccurrence(pairs, named) == []
    assert leak.cooccurrence([("A B", "C D")], [{"A B", "C D", "E F"}]) == [("A B", "C D")]  # the check can fire
    assert card["codes"]["shared_between_items"] == [] and card["codes"]["in_training"] == []
    codes = [c for it in _items(out) if it["family"] == "E3ctx" for c, _ in it["structured"]["facts"]]
    assert len(codes) == len(set(codes)) == card["codes"]["n_codes"]
    training_codes = set().union(*(leak.codes_in(s["text"]) for d in texts for s in d["segments"] if "text" in s))
    assert training_codes and not training_codes & set(codes)


# --- independence and leak tests ------------------------------------------------------------------------------------


def test_i4_d7_chi2_survival_function_matches_known_quantiles():
    for x, df, p in (
        (3.841459, 1, 0.05),
        (18.307038, 10, 0.05),
        (9.210340, 2, 0.01),
        (6.634897, 1, 0.01),
        (124.342113, 100, 0.05),
        (0.0, 3, 1.0),
    ):
        assert leak.chi2_sf(x, df) == pytest.approx(p, abs=1e-6)
    assert leak.chi2_sf(2.0, 2) == pytest.approx(np.exp(-1.0), abs=1e-12)  # df 2: exp(-x/2)


def test_i4_d7_attribute_values_are_independent_of_name_and_partition():
    n_values = {"year": 100, "month": 12, "city": 200, "occupation": 100, "employer": 200, "hobby": 60}
    stats = leak.attribute_independence(20_000, n_values, seed=1)
    assert len(stats) == 6 * 3 and min(s["n"] for s in stats.values()) == 20_000
    for key, s in stats.items():
        assert s["p"] >= 0.001, (key, s)


@pytest.fixture(scope="module")
def large_pools():
    rng = np.random.default_rng(2)
    people = draw_people(
        {"seed": 0, "zh": 3000, "en": 3000, "par": 0},
        {"year": 100, "month": 12, "city": 200, "occupation": 100, "employer": 200, "hobby": 60},
        {"heldout_fraction": 1.0, "eval_dev_fraction": 0.0, "par_train_fraction": 0.5},
        rng,
    )
    return people, {p.id: p for p in people}


def test_i4_d7_e3par_single_facts_carry_no_information(large_pools, full):
    people, by_id = large_pools
    _, _, registry, lex = full
    r = Renderer(registry, lex)
    a_side, b_side = by_group(people, "zh"), by_group(people, "en")
    for variant, attr, n_labels in (("same_city", "city", 2), ("month_sum", "month", 12)):
        items = par_items(variant, a_side, b_side, "en", "test", 6000, r, np.random.default_rng(3), {})
        for side in ("a", "b"):
            s = leak.single_fact_leak(items, by_id, side, attr, len(lex[attr]), n_labels)
            assert s["p"] >= 0.001, (variant, side, s)
            assert s["bayes_accuracy"] < 1 / n_labels + 0.05, (variant, side, s)
        labels = Counter(it["structured"]["label"] for it in items)
        assert max(labels.values()) - min(labels.values()) <= 1  # balanced by construction


def test_i4_d7_born_earlier_control_is_flagged(large_pools):
    people, by_id = large_pools
    items = leak.born_earlier_items(by_group(people, "zh"), by_group(people, "en"), 6000, np.random.default_rng(4))
    s = leak.single_fact_leak(items, by_id, "a", "year", 100, 2)
    assert s["bayes_accuracy"] >= 0.7 and s["p"] < 1e-6, s
    assert leak.leak_flagged(s)


# --- E3-ctx ablation, candidates, labels, exposures -------------------------------------------------------------------


def test_i4_d7_e3ctx_either_segment_alone_leaves_the_gold_uniform(track_s_ci, full):
    out, card = track_s_ci
    ctx = [it for it in _items(out) if it["family"] == "E3ctx"]
    assert card["checker"]["ablation_uniform"] == len(ctx) > 0
    checker = Checker(_people(out), B.lexicons_for(card["profile"]))
    day = lexicon("weekday")
    for it in ctx:
        parsed = checker.parse_ctx(it)
        answer = [c for i, c in enumerate(it["candidates"]) if i != it["distractor_index"]]
        # without facts: the target's colour is any of the K rule colours -> one weekday each
        without_fact = Counter(day.surface(d, it["answer_lang"]) for d in parsed["rules"].values())
        assert sorted(without_fact) == sorted(answer) and set(without_fact.values()) == {1}
        assert len(answer) == K and it["difficulty"]["n_rules"] == K
    # a five-rule item is caught: its rule segment alone narrows the gold to five weekdays
    item = copy.deepcopy(ctx[0])
    rule = next(s for s in item["segments"] if s["role"] == "rule")
    sentences = [s for s in rule["text"].replace("。", "。|").replace(". ", ".|").split("|") if s.strip()]
    target_color = item["structured"]["category"]
    keep, dropped = [], 0
    for s in sentences:
        colors = lexicon("color").find(s, rule["lang"])
        if colors and lexicon("color").surface(colors[0], "en") != target_color and dropped < 2:
            dropped += 1
            continue
        keep.append(s)
    rule["text"] = ("" if rule["lang"] == "zh" else " ").join(x.strip() for x in keep) + (
        "\n" if rule["text"].endswith("\n") else ""
    )
    report = checker.check_items([item])
    assert any("not a bijection" in e or "disagree" in e for e in report.errors), report.errors


def test_i4_d7_candidates_hold_the_other_language_form_and_labels_balance(full):
    _, prof, registry, lex = full
    r = Renderer(registry, lex)
    people = draw_people(
        {"seed": 0, "zh": 400, "en": 400, "par": 0},
        {a: len(lex[a]) for a in ATTRIBUTES},
        {"heldout_fraction": 1.0, "eval_dev_fraction": 0.0, "par_train_fraction": 0.5},
        np.random.default_rng(8),
    )
    from surogate_bilingual.data.synth.entities import CodeAllocator

    codes = CodeAllocator(np.random.default_rng(9))
    n = 1600
    groups = {
        "E2": qa_items("E2", by_group(people, "zh"), "en", "test", n, r, np.random.default_rng(10), {}),
        "E3ctx": ctx_items(
            "cross",
            ("zh", "en", "zh"),
            "test",
            n,
            r,
            codes,
            np.random.default_rng(11),
            {},
            n_colors=10,
            distractors=(0, 3),
        ),
    }
    for family, items in groups.items():
        for it in items:
            gold = it["candidates"][it["gold_index"]]
            other = it["candidates"][it["distractor_index"]]
            name = "weekday" if family == "E3ctx" else it["structured"]["attribute"]
            idx = lexicon(name, len(lex[name]) if name in lex else None).index_of(gold, it["answer_lang"])
            assert other == lexicon(name, len(lex[name])).surface(idx, "zh" if it["answer_lang"] == "en" else "en")
            assert len(it["candidates"]) == K + 1 and len(set(it["candidates"])) == K + 1
        shares = Counter(it["gold_index"] for it in items)
        assert all(abs(c / n - 1 / (K + 1)) <= 0.01 for c in shares.values()) and len(shares) == K + 1
    golds = Counter(it["candidates"][it["gold_index"]] for it in groups["E3ctx"])
    assert len(golds) == K and all(abs(c / n - 1 / K) <= 0.01 for c in golds.values())


def test_i4_d7_exposures_per_fact_equal_the_declared_number(track_s_ci):
    out, card = track_s_ci
    prof = card["profile"]
    people = _people(out)
    lex = B.lexicons_for(prof)
    declared = {
        "seed": sum(prof["bio_exposures"]["seed"].values()),
        "zh": prof["bio_exposures"]["zh"]["zh"],
        "en": prof["bio_exposures"]["en"]["en"],
        "parallel": 2 * prof["parallel_pairs_per_person"],
    }
    # count directly from the text, without the checker: name + every attribute value of that person
    counts = Counter()
    by_name = {p.name: p for p in people}
    for component in B.COMPONENTS:
        for doc in _load(out, f"{component}/track_s.jsonl"):
            if doc["meta"]["family"] not in ("bio", "parallel"):
                continue
            for seg in doc["segments"]:
                person = next(p for name, p in by_name.items() if name in seg["text"])
                stated = find_attributes(seg["text"], seg["lang"], lex)
                for a in ATTRIBUTES:
                    assert stated[a] == [person.attrs[a]]
                    counts[(component, person.id, a)] += 1
    part = {"seed": ("seed", None), "zh": ("zh", None), "en": ("en", None), "parallel": ("par", "train")}
    for component, want in declared.items():
        members = by_group(people, *part[component])
        assert {counts[(component, p.id, a)] for p in members for a in ATTRIBUTES} == {want}, component
    assert card["status"]["exposures_exact"] is True
    for key, info in card["rewrites"]["distinct_per_person"].items():
        assert info["min"] >= prof["min_rewrites"], key


def test_i4_d7_lexicons_have_fixed_distinct_pairs():
    for name in LISTS:
        lex = lexicon(name)
        assert len(set(lex.surfaces("zh"))) == len(lex) == len(set(lex.surfaces("en")))
    assert len(lexicon("city")) >= 200 and len(lexicon("occupation")) >= 100 and len(lexicon("employer")) == 200
    assert len(lexicon("hobby")) >= 60 and len(lexicon("year")) == 100 and len(lexicon("weekday")) == K
    for lang in ("zh", "en"):  # no surface belongs to two attribute lexicons
        seen = Counter(s for a in ATTRIBUTES for s in lexicon(a).surfaces(lang))
        assert max(seen.values()) == 1, lang
    # joint matching keeps an occupation that contains a hobby from stating the hobby too
    lex = {a: lexicon(a) for a in ATTRIBUTES}
    assert find_attributes("X 当陶艺家已经多年。", "zh", lex)["hobby"] == []
    assert lexicon("hobby").find("X 当陶艺家已经多年。", "zh") != []  # what a per-lexicon match would wrongly read


def test_i4_d7_dry_run_plans_without_writing(tmp_path, capsys):
    out = tmp_path / "never"
    assert B.main(["--spec", str(TRACK_S_SPEC), "--seed", "1", "--out", str(out), "--dry-run"]) == 0
    assert not out.exists()
    assert "test templates per family" in capsys.readouterr().out
    plan = B.plan(TRACK_S_SPEC, "full")
    assert plan["test_templates"]["E3ctx"] >= 100 and plan["items"]["E3ctx/cross"]["test"] == 4000
