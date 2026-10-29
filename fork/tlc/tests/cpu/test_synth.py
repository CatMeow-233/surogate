"""Synthetic binding programs (surogate_tlc.data.synth, README: 合成绑定任务): executed golds, the grid, the four
split axes between the training stream and the eval halves, and the heuristic floors per pattern."""

import itertools
import json
import random
import re
import shutil
from collections import defaultdict

import pytest
from surogate_tlc.data import suites, synth


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One item per grid cell and half, and about 20k characters of training programs."""
    out = tmp_path_factory.mktemp("synth") / "out"
    manifest = synth.build(out, seed=3, per_task=1, train_chars=20000)
    halves = {half: suites.read_items(out / suites.SUITE_FILES[half]) for half in suites.HALVES}
    return out, manifest, halves


def _train_records(out):
    return [json.loads(line) for line in (out / synth.TRAIN_FILE).read_text(encoding="utf-8").splitlines()]


def test_build_writes_disjoint_halves_and_a_training_stream(built):
    out, manifest, halves = built
    assert manifest["disjoint"] == synth.check(out)
    assert manifest["disjoint"]["eval_items_with_a_training_ast"] == 0
    for half, items in halves.items():
        assert manifest["files"][half]["items"] == len(items) > 0
    assert not {i.program for i in halves["dev"]} & {i.program for i in halves["test"]}
    records = _train_records(out)
    assert sum(len(r["text"]) for r in records) >= 20000 and manifest["files"]["train"]["documents"] == len(records)
    for record in records:
        synth.run(record["text"])  # every readout of the module holds
        assert record["text"].count("assert ") == len(record["items"])
        assert all(i["pattern"] not in synth.HELD_OUT_PATTERNS for i in record["items"])
        assert all(i["family"] not in synth.HELD_OUT_FAMILIES for i in record["items"])


KEYWORDS = {"assert", "def", "return", "lambda", "str"}


@pytest.mark.parametrize("leak", ["eval literal", "held-out identifier", "held-out pattern", "eval ast"])
def test_check_finds_leaks(built, tmp_path, leak):
    out, _, halves = built
    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    record = {"text": "y = 'zzzz'\n", "items": []}
    if leak == "eval literal":
        words = ("".join(w) for w in itertools.product("abcd", repeat=4))
        record["text"] = next(f"x = '{w}'\n" for w in words if synth.eval_literal(w))
    elif leak == "held-out identifier":
        record["text"] = f"{synth.POOLS['novel'][0]} = 'zzzz'\n"
    elif leak == "held-out pattern":
        record["items"] = [{"task": "T1", "pattern": "shadow-closure", "family": "assign", "hash": "0"}]
    else:
        record["items"] = [{"task": "T1", "pattern": "flat", "family": "assign", "hash": halves["test"][0].hash}]
    with (copy / synth.TRAIN_FILE).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    with pytest.raises(AssertionError):
        synth.check(copy)


def test_items_are_executed_and_complete(built):
    _, _, halves = built
    for item in halves["dev"][::7]:
        assert not item.problems()
        synth.run(item.program)
        wrong = next(c for c in item.candidates if c != item.gold) if len(item.candidates) > 1 else None
        if wrong:  # the readout checks the value
            with pytest.raises(AssertionError):
                synth.run(item.prompt + wrong + item.program[len(item.prompt) + len(item.gold) :])
        assert item.program.startswith(item.prompt + item.gold + item.tail)
        assert item.prompt[item.anchor : item.anchor + len(item.gold)] == item.gold
        if item.query:
            name = item.prompt[item.query[0] : item.query[1]]
            assert re.fullmatch(r"\w+", name) and item.prompt.endswith(f"assert {name} == ")


def test_every_cell_is_covered_and_fits_2048_tokens(built):
    _, _, halves = built
    for task in ("T1", "T2"):
        keys = synth.cells(task)[0].keys()
        found = {tuple(i.strata[k] for k in keys) for i in halves["test"] if i.task == task}
        assert found == {tuple(c.values()) for c in synth.cells(task)}
    # v0.7: every cell fits T=2048; synthetic code averages well over two characters per token.
    assert max(len(i.prompt) for items in halves.values() for i in items) < 2 * 2048


def test_split_axes(built):
    out, _, halves = built
    items = [i for v in halves.values() for i in v]
    assert {i.strata["pool"] for i in items} == set(synth.POOLS)
    assert {i.strata["family"] for i in items if i.task == "T1"} == set(synth.FAMILIES["T1"])
    for item in items:
        assert all(synth.eval_literal(c[1:-1]) for c in item.candidates)
        names = set(synth.IDENTIFIER.findall(synth.LITERAL.sub("''", item.program))) - KEYWORDS
        assert names <= set(synth.POOLS[item.strata["pool"]]), item.id
        held = {"family"} if item.strata["family"] in synth.HELD_OUT_FAMILIES else set()
        held |= {"pattern"} if item.strata["pattern"] in synth.HELD_OUT_PATTERNS else set()
        held |= {"pool"} if item.strata["pool"] != "seen" else set()
        assert set(item.strata["heldout"]) == held
    for record in _train_records(out):
        literals = synth.LITERAL.findall(record["text"])
        assert literals and not any(synth.eval_literal(x) for x in literals)


def test_floors_follow_the_patterns(built):
    """same_name is the algorithm where nothing shadows the name read and wrong where a shadowing binding comes
    last; first_bound is right exactly without overwrites; position is the algorithm for positional calls."""
    _, _, halves = built
    acc = defaultdict(list)
    for item in halves["dev"] + halves["test"]:
        s = item.strata
        for h, v in item.heuristics.items():
            acc[item.task, s["pattern"], h].append(v == item.gold)
        if item.task == "T1" and s["pattern"] in ("flat", "shadow-out"):
            acc["first_bound", s["overwrites"] == 0].append(item.heuristics["first_bound"] == item.gold)
        if s["pattern"] == "shadow-out":
            acc["stale", s["depth"]].append(item.heuristics["stale"] == item.gold)
    rate = {k: sum(v) / len(v) for k, v in acc.items()}
    for pattern in ("flat", "closure", "shadow-in", "call-kw"):
        assert rate["T1", pattern, "same_name"] == 1, pattern
    for pattern in ("shadow-out", "shadow-closure", "call-pos"):
        assert rate["T1", pattern, "same_name"] == 0, pattern
    for pattern in ("keyword", "keyword-reordered"):
        assert rate["T2", pattern, "same_name"] == 1, pattern
    for pattern in ("positional", "mixed", "names-reordered"):
        assert rate["T2", pattern, "position"] == 1, pattern
    assert rate["T1", "call-pos", "position"] == 1 and rate["T2", "keyword-reordered", "position"] < 1
    assert rate["first_bound", True] == 1 and rate["first_bound", False] == 0
    assert rate["stale", 1] == 1 and rate["stale", 2] == 0  # one shadowing function: the previous value is the outer
    assert rate["T1", "flat", "recent"] < 0.5 and rate["T2", "positional", "same_name"] == 0


def test_generation_is_deterministic():
    a, _ = synth.training_documents(5, 3000, set())
    b, _ = synth.training_documents(5, 3000, set())
    c, _ = synth.training_documents(6, 3000, set())
    assert a == b and a != c
    cell = {**synth.cells("T1", ["shadow-in"])[50], "family": "assign"}
    one = synth.make_item("T1", cell, random.Random("x"), "rare", "eval", "i")
    assert one == synth.make_item("T1", cell, random.Random("x"), "rare", "eval", "i")


@pytest.mark.parametrize("family", ["unpack", "annotated"])
def test_families_keep_the_meaning(family):
    lines = ["a = 'x1'", "b = 'x2'", "c = b", "def f():", "    d = 'x3'", "    e = 'x4'", "    __probe__(c)", "f()"]
    changed = synth.apply_family(lines, family, random.Random(0))
    assert changed != lines and synth.run("\n".join(changed)) == synth.run("\n".join(lines)) == ["x2"]
    if family == "unpack":
        assert "a, b = 'x1', 'x2'" in changed and "    d, e = 'x3', 'x4'" in changed
