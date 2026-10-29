"""Synthetic binding programs (surogate_tlc.data.synth, docs/data.md: 合成绑定任务): executed golds, the grid, the four
split axes between the training stream and the eval halves, and the heuristic floors per pattern."""

import ast
import itertools
import json
import random
import re
import shutil
from collections import Counter, defaultdict

import pytest
from surogate_tlc.data import synth
from surogate_tlc.data.decontam import ast_hash
from surogate_tlc.data.items import HALVES, SUITE_FILES, Item, read_items, write_items


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One item per grid cell and half, and about 20k characters of training programs."""
    out = tmp_path_factory.mktemp("synth") / "out"
    manifest = synth.build(out, seed=3, per_task=1, train_chars=20000)
    halves = {half: read_items(out / SUITE_FILES[half]) for half in HALVES}
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


def _as_training_program(item) -> str:
    """The item's program with every identifier renamed into the training pool and every literal into the training
    literal space: the same alpha-renamed AST, as a generator could have drawn it for the training stream. One pass,
    so that no renamed identifier lands inside a literal."""
    names = iter(n for n in synth.POOLS["seen"] if not set(n.split("_")) & KEYWORDS)
    renamed: dict[str, str] = {}
    words = ("".join(w) for w in itertools.product("efgh", repeat=4))
    literals = (w for w in words if not synth.eval_literal(w))

    def rename(match: re.Match) -> str:
        token = match.group(0)
        if token.startswith("'"):
            return f"'{next(literals)}'"
        return token if token in KEYWORDS else renamed.setdefault(token, next(names))

    return re.sub(rf"{synth.LITERAL.pattern}|{synth.IDENTIFIER.pattern}", rename, item.program)


def _with_record(out, tmp_path, record):
    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    with (copy / synth.TRAIN_FILE).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return copy


@pytest.mark.parametrize(
    "leak, message",
    [
        ("eval literal", "an eval literal"),
        ("novel identifier", "held-out identifiers"),
        ("rare identifier", "held-out identifiers"),
        ("held-out pattern", "held-out family or pattern"),
        ("held-out family", "held-out family or pattern"),
        ("recorded hash", "its record says"),
    ],
)
def test_check_finds_leaks(built, tmp_path, leak, message):
    """One leak per case, on one axis: the others stay at training values."""
    out, _, halves = built
    text = "y = 'zzzz'\n"
    assert not synth.eval_literal("zzzz")
    info = {"task": "T1", "pattern": "flat", "family": "assign", "hash": ast_hash(ast.parse(text))}
    if leak == "eval literal":
        words = ("".join(w) for w in itertools.product("abcd", repeat=4))
        text = next(f"x = '{w}'\n" for w in words if synth.eval_literal(w))
    elif leak.endswith("identifier"):
        text = f"{synth.POOLS[leak.split()[0]][0]} = 'zzzz'\n"
    elif leak == "held-out pattern":
        info["pattern"] = "shadow-closure"
    elif leak == "held-out family":
        info["family"] = "annotated"
    info["hash"] = ast_hash(ast.parse(text))
    if leak == "recorded hash":
        info["hash"] = halves["test"][0].hash  # a recorded hash cannot stand in for the program's
    with pytest.raises(synth.SplitLeak, match=message) as error:
        synth.check(_with_record(out, tmp_path, {"text": text, "items": [info]}))
    assert len(error.value.problems) == 1, error.value.problems


def test_check_hashes_a_program_without_a_record(built, tmp_path):
    """A program the item records do not list still meets the AST rule; the count mismatch is reported too."""
    out, _, halves = built
    item = next(i for i in halves["dev"] if i.task == "T2" and i.strata["family"] not in synth.HELD_OUT_FAMILIES)
    benign = "y = 'zzzz'\n"
    info = {"task": "T1", "pattern": "flat", "family": "assign", "hash": ast_hash(ast.parse(benign))}
    text = benign + "\n" + _as_training_program(item)
    with pytest.raises(synth.SplitLeak) as error:
        synth.check(_with_record(out, tmp_path, {"text": text, "items": [info]}))
    problems = error.value.problems
    assert any(p.endswith("2 programs, 1 item records") for p in problems), problems
    assert any("eval items have a training program's AST" in p for p in problems), problems


def _with_halves(out, tmp_path, edit):
    copy = tmp_path / "copy"
    shutil.copytree(out, copy)
    halves = {half: read_items(copy / SUITE_FILES[half]) for half in HALVES}
    edit(halves)
    for half, items in halves.items():
        write_items(copy / SUITE_FILES[half], items)
    return copy


def _training_literal_candidate(halves):
    word = next(w for w in ("".join(t) for t in itertools.product("pqrs", repeat=4)) if not synth.eval_literal(w))
    halves["dev"][0].candidates.append(f"'{word}'")


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda h: h["test"].append(h["dev"][0]), "programs are in both halves"),
        (_training_literal_candidate, "a candidate outside the eval literal space"),
        (lambda h: setattr(h["dev"][0], "tail", ""), "empty tail"),
        (lambda h: h["dev"][0].candidates.remove(h["dev"][0].gold), "the gold must be one of distinct candidates"),
    ],
    ids=["shared program", "training literal", "empty tail", "gold not a candidate"],
)
def test_check_finds_bad_eval_items(built, tmp_path, edit, message):
    out, _, _ = built
    with pytest.raises(synth.SplitLeak, match=message) as error:
        synth.check(_with_halves(out, tmp_path, edit))
    assert len(error.value.problems) == 1, error.value.problems


def test_items_name_their_problems():
    good = {"id": "i", "task": "T1", "prompt": "x = ", "candidates": ["'a'", "'b'"], "gold": "'a'", "tail": "\n"}
    good |= {"heuristics": {}, "strata": {}}
    assert Item(**good).problems() == []
    for change, problem in [
        ({"task": "T9"}, "task 'T9'"),
        ({"gold": "'c'"}, "the gold must be one of distinct candidates"),
        ({"candidates": ["'a'", "'a'"]}, "the gold must be one of distinct candidates"),
        ({"tail": ""}, "empty tail"),
    ]:
        assert Item(**(good | change)).problems() == [f"i: {problem}"], change


def test_training_stream_redraws_excluded_asts():
    first = list(itertools.islice(synth.train_programs(5, set(), Counter()), 6))
    exclude = {info["hash"] for _, info in first}
    redrawn = Counter()
    again = list(itertools.islice(synth.train_programs(5, exclude, redrawn), 6))
    assert all(info["hash"] not in exclude for _, info in again) and sum(redrawn.values()) > 0


def test_build_excludes_both_halves_and_keeps_test_programs_out_of_dev(tmp_path, monkeypatch):
    """A test item whose program dev already holds is dropped, and the training stream avoids the ASTs of both
    halves, the test half's too."""
    cells = synth.cells("T1", ["flat"])
    a, b = (
        synth.make_item("T1", {**cells[i], "family": "assign"}, random.Random(i), "seen", "eval", f"T1-{i}")
        for i in (0, 1)
    )
    assert a.program != b.program and a.hash != b.hash
    picked = {"dev": [a], "test": [a, b]}
    monkeypatch.setattr(synth, "suite", lambda task, half, per_task, seed: picked[half] if task == "T1" else [])
    seen = {}
    real = synth.training_documents

    def spy(seed, chars, exclude, *args, **kwargs):
        seen["exclude"] = set(exclude)
        return real(seed, chars, exclude, *args, **kwargs)

    monkeypatch.setattr(synth, "training_documents", spy)
    synth.build(tmp_path / "out", seed=3, per_task=1, train_chars=2000)
    assert [i.id for i in read_items(tmp_path / "out" / SUITE_FILES["test"])] == [b.id]
    assert seen["exclude"] == {a.hash, b.hash}


@pytest.mark.parametrize("pattern", ["keyword", "mixed", "default"])
def test_check_finds_an_eval_ast_called_by_keyword(built, tmp_path, pattern):
    """A training program equal to an eval item up to renaming, keyword arguments included, is a leak."""
    out, _, halves = built
    item = next(
        i
        for items in halves.values()
        for i in items
        if i.task == "T2" and i.strata["pattern"] == pattern and i.strata["family"] not in synth.HELD_OUT_FAMILIES
    )
    text = _as_training_program(item)
    assert "=" in text.splitlines()[-1].split("(", 1)[1]  # the readout call passes a keyword argument
    assert ast_hash(ast.parse(text)) == item.hash
    info = {"task": "T2", "pattern": pattern, "family": item.strata["family"], "hash": item.hash}
    with pytest.raises(synth.SplitLeak, match="eval items have a training program's AST"):
        synth.check(_with_record(out, tmp_path, {"text": text, "items": [info]}))


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
    # Every cell fits a 2048-token context (formal training); synthetic code has well over two characters a token.
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


def test_annotated_family_keeps_the_ast():
    """The held-out `annotated` family is the `assign` form with annotations (ast_hash drops them), so build() and
    check() keep it apart from `assign` training programs."""
    lines = ["a = 'x1'", "b = a", "def f():", "    c = 'x3'", "    __probe__(c)", "f()"]
    annotated = synth.apply_family(lines, "annotated", random.Random(0))
    assert annotated != lines and ": str = " in "\n".join(annotated)
    assert ast_hash(ast.parse("\n".join(annotated))) == ast_hash(ast.parse("\n".join(lines)))


def test_check_finds_an_annotated_eval_ast_written_plainly(built, tmp_path):
    out, _, halves = built
    item = next(
        i
        for items in halves.values()
        for i in items
        if i.task == "T1" and i.strata["family"] == "annotated" and i.strata["pattern"] not in synth.HELD_OUT_PATTERNS
    )
    text = _as_training_program(item).replace(": str = ", " = ")
    assert ": str" not in text and ast_hash(ast.parse(text)) == item.hash
    info = {"task": "T1", "pattern": item.strata["pattern"], "family": "assign", "hash": item.hash}
    with pytest.raises(synth.SplitLeak, match="eval items have a training program's AST"):
        synth.check(_with_record(out, tmp_path, {"text": text, "items": [info]}))


@pytest.mark.parametrize("family", ["unpack", "annotated"])
def test_families_keep_the_meaning(family):
    lines = ["a = 'x1'", "b = 'x2'", "c = b", "def f():", "    d = 'x3'", "    e = 'x4'", "    __probe__(c)", "f()"]
    changed = synth.apply_family(lines, family, random.Random(0))
    assert changed != lines and synth.run("\n".join(changed)) == synth.run("\n".join(lines)) == ["x2"]
    if family == "unpack":
        assert "a, b = 'x1', 'x2'" in changed and "    d, e = 'x3', 'x4'" in changed
