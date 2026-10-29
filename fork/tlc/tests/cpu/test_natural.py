"""T3 items from natural code (surogate_tlc.data.natural, docs/training.md: 绑定探针): prefix-resolvable local loads in
the static subset, functions left out by their AST, the val split, and no items from generated sources."""

import json
import random
import textwrap
from types import SimpleNamespace

import pytest
from surogate_tlc.data import corpus, decontam, natural
from surogate_tlc.data.items import HALVES, read_items

SOURCE = '''\
import os


def scale(values, factor, offset=0):
    """Scale and shift."""
    total = offset
    for value in values:
        total = total + value * factor
    if total > factor and offset < 0:
        total = total - offset * factor
    return total


def uses_eval(code, scope):
    result = eval(code, scope)
    return result + code


def joined(parts, sep):
    text = sep.join(parts)
    return text + sep
'''


def test_natural_items_are_prefix_resolvable_local_loads():
    items = natural.natural_items(SOURCE, "doc", rng=random.Random(0), limit=100)
    assert items and all(item.task == "T3" for item in items)
    for item in items:
        at = len(item.prompt)
        assert SOURCE[:at] == item.prompt and SOURCE[at:].startswith(item.gold + item.tail)
        assert item.gold in item.candidates and len(item.candidates) >= 2
        assert SOURCE[item.anchor : item.anchor + len(item.gold)] == item.gold  # a binding of the gold
        assert set(item.heuristics) == {"recent", "frequent"} and item.tail.endswith("\n")
    assert not any("eval" in item.prompt.rsplit("def ", 1)[-1][:12] for item in items)  # not in the static subset
    # Bindings count once Python performs them: `total` after its statement, `value` after the iterable.
    first = next(i for i in items if i.prompt.endswith("    total = "))
    assert first.gold == "offset" and first.candidates == ["factor", "offset", "values"]
    loop = next(i for i in items if i.prompt.endswith("    for value in "))
    assert loop.candidates == ["factor", "offset", "total", "values"]
    total = next(i for i in items if i.prompt.endswith("        total = "))
    assert total.candidates == ["factor", "offset", "total", "value", "values"]
    assert total.heuristics == {"recent": "value", "frequent": "values"}


def test_natural_items_leave_out_excluded_functions():
    [digest] = [h for name, _, h in decontam.function_hashes(SOURCE) if name == "scale"]
    items = natural.natural_items(SOURCE, "doc", rng=random.Random(0), limit=100, exclude={digest})
    assert items and not any("def scale" in item.prompt.rsplit("\ndef ", 1)[-1] for item in items)


def test_static_subset():
    import ast

    cases = (("x = 1\nreturn x", True), ("global y\ny = 1", False), ("nonlocal y\ny = 1", False), ("del x", False))
    for body, inside in cases:
        fn = ast.parse("def f(x):\n" + textwrap.indent(body, "    ")).body[0]
        assert natural.static_subset(fn, natural._own_nodes(fn)) is inside


FUNCTIONS = (
    "def first(items, key):\n    found = None\n    for item in items:\n        if item == key:\n"
    "            found = item\n    return found\n",
    "def pairs(left, right):\n    out = []\n    for a in left:\n        for b in right:\n"
    "            out.append((a, b))\n    return out\n",
    "def count(text, sep):\n    parts = text.split(sep)\n    size = len(parts)\n    return size + len(sep)\n",
    "def clamp(value, low, high):\n    if value < low:\n        value = low\n    if value > high:\n"
    "        value = high\n    return value\n",
)
CLONE = (  # at least decontam.MIN_AST_NODES nodes, so its structure has a hash
    "def merge(base, extra, limit):\n    result = dict(base)\n    for key, value in extra.items():\n"
    "        if key in result and len(result) < limit:\n            result[key] = result[key] + value\n"
    "        elif key not in result:\n            result[key] = value\n    return result\n"
)


def test_build_natural_uses_the_val_split(tmp_path):
    (tmp_path / "code").mkdir()
    for i in range(24):
        # CLONE with other names in every file, so in both splits: it gives no item. Which split's functions count is
        # test_t3_excludes_the_train_split_and_keeps_a_group_in_one_half.
        clone = CLONE.replace("merge", f"merge{i}").replace("result", f"out{i}")
        (tmp_path / "code" / f"m{i:02d}.py").write_text(FUNCTIONS[i % 4] + "\n\n" + clone)
    # Synthetic binding programs as a generated source: they are code of the val split too, but give no T3 item.
    synthetic = [{"text": FUNCTIONS[i % 4].replace("def ", f"def synth{i}_")} for i in range(24)]
    (tmp_path / "synth.jsonl").write_text("".join(json.dumps(r) + "\n" for r in synthetic))
    spec = tmp_path / "corpus.toml"
    source = '[[source]]\nname = "{}"\nkind = "code"\nfiles = ["{}"]\nurl = "https://example.invalid"\n'
    source += 'license = "MIT"\ndownloaded = 2026-10-01\nfilters = []\n'
    spec.write_text(
        "seed = 3\nval_fraction = 0.5\n\n"
        + source.format("code", "code/*.py")
        + source.format("synth-binding", "synth.jsonl")
        + "generated = true\n"
    )
    c = corpus.load_corpus(spec)
    assert [s.generated for s in c.sources] == [False, True]
    manifest = natural.build_natural(c, tmp_path / "t3", seed=1)
    val = {doc.key for doc in corpus.documents(c) if corpus.split_of(c, doc) == "val"}
    halves = {half: read_items(tmp_path / "t3" / f"{half}.jsonl") for half in HALVES}
    keys = {half: {item.id.split(":")[1] for item in items} for half, items in halves.items()}
    assert halves["dev"] and halves["test"] and not keys["dev"] & keys["test"]
    assert keys["dev"] | keys["test"] <= val
    assert not any(key.startswith("synth-binding/") for half in keys.values() for key in half)
    assert any(key.startswith("synth-binding/") for key in val)  # it had val documents to draw from
    assert manifest["corpus"]["sources"] == ["code"]
    assert manifest["files"]["dev"]["items"] == len(halves["dev"])
    assert len(decontam.function_hashes(CLONE)) == 1
    assert not any(item.prompt.rsplit("\ndef ", 1)[-1].startswith("merge") for v in halves.values() for item in v)
    with pytest.raises(FileExistsError):
        natural.build_natural(c, tmp_path / "t3")


VAL_ONLY = (
    "def tally(rows, width, fill):\n    grid = []\n    for row in rows:\n        line = list(row)\n"
    "        while len(line) < width:\n            line.append(fill)\n        if len(line) > width:\n"
    "            line = line[:width]\n        grid.append(line)\n    return grid\n"
)
NAT_TRAIN = (
    "def window(seq, size, step):\n    out = []\n    index = 0\n    while index + size <= len(seq):\n"
    "        chunk = seq[index : index + size]\n        if chunk and size > step:\n            out.append(chunk)\n"
    "        index = index + step\n    return out\n"
)


def test_t3_excludes_the_train_split_and_keeps_a_group_in_one_half(tmp_path):
    """The excluded ASTs are those of every train-split code document, generated sources included, and only those: a
    function that is only in the val split stays. A group's documents (one repository) stay in one half."""
    seed, fraction = 3, 0.5
    probe = corpus.Corpus(tmp_path / "unused.toml", seed, fraction, ())
    groups = {"train": [], "val": []}
    for i in range(200):
        split = corpus.split_of(probe, SimpleNamespace(group=f"group:repo{i}"))
        groups[split].append(f"repo{i}")
    train, val = groups["train"][:4], groups["val"][:6]
    hashes = {}
    for name, text in (("clone", CLONE), ("val", VAL_ONLY), ("natural", NAT_TRAIN)):
        [(_, _, hashes[name])] = decontam.function_hashes(text)
    natural_docs = [{"text": NAT_TRAIN.replace("window", f"window{i}"), "group": g} for i, g in enumerate(train)]
    natural_docs += [
        {"text": NAT_TRAIN + "\n\n" + CLONE.replace("result", f"out{i}") + "\n\n" + VAL_ONLY, "group": g}
        for i, g in enumerate(val * 3)
    ]
    generated = [{"text": CLONE.replace("merge", f"gen{i}"), "group": g} for i, g in enumerate(train)]
    for name, records in (("code.jsonl", natural_docs), ("gen.jsonl", generated)):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in records))
    source = '[[source]]\nname = "{}"\nkind = "code"\nfiles = ["{}"]\nurl = "https://example.invalid"\n'
    source += 'license = "MIT"\ndownloaded = 2026-10-01\nfilters = []\n'
    spec = tmp_path / "corpus.toml"
    spec.write_text(
        f"seed = {seed}\nval_fraction = {fraction}\n\n"
        + source.format("code", "code.jsonl")
        + source.format("synth", "gen.jsonl")
        + "generated = true\n"
    )
    c = corpus.load_corpus(spec)
    assert natural.train_function_hashes(c) == {hashes["clone"], hashes["natural"]}
    natural.build_natural(c, tmp_path / "t3", seed=1)
    halves = {half: read_items(tmp_path / "t3" / f"{half}.jsonl") for half in HALVES}
    assert all(item.prompt.rsplit("\ndef ", 1)[-1].startswith("tally") for v in halves.values() for item in v)
    group_of = {f"code/code.jsonl:{n}": r["group"] for n, r in enumerate(natural_docs, 1)}
    used = {half: {group_of[i.id[len("T3:") :].rsplit(":", 1)[0]] for i in items} for half, items in halves.items()}
    assert used["dev"] and used["test"] and not used["dev"] & used["test"], used
