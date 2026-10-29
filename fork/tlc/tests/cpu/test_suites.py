"""Probe items and heuristic floors (surogate_tlc.data.suites, README: 绑定探针): the scope-blind reader on
hand-written programs, and T3 items from natural code."""

import random
import textwrap

import pytest
from surogate_tlc.data import corpus, decontam, suites

# (program, heuristics the reader must give; the readout is the program's last line)
CASES = [
    (  # a copy chain rebound: the previous value is stale
        """
        a = 'x1'
        b = a
        a = 'x2'
        b = a
        c = 'x3'
        assert b == 'x2'
        """,
        {"recent": "'x3'", "line1": "'x1'", "same_name": "'x2'", "first_bound": "'x1'", "stale": "'x1'"},
    ),
    (  # a copy keeps the value it was made with
        """
        b = 'v1'
        q = b
        b = 'v2'
        assert q == 'v1'
        """,
        {"same_name": "'v1'", "first_bound": "'v1'", "stale": None, "position": None},
    ),
    (  # read after a shadowing call: same_name sees the inner binding
        """
        q = 'a1'
        def g():
            q = 'a2'
        g()
        assert q == 'a1'
        """,
        {"recent": "'a2'", "same_name": "'a2'", "first_bound": "'a1'", "stale": "'a1'"},
    ),
    (  # keywords bind by name; position ignores them
        """
        def f(a, b, c):
            return b
        assert f(c='z1', a='z2', b='z3') == 'z3'
        """,
        {"recent": "'z3'", "line1": "'z1'", "same_name": "'z3'", "position": "'z2'"},
    ),
    (  # module variables named like the formals, passed in another order
        """
        def f(a, b):
            return a
        a = 'p1'
        b = 'p2'
        assert f(b, a) == 'p2'
        """,
        {"same_name": "'p1'", "position": "'p2'", "stale": None},
    ),
    (  # a default through a local copy
        """
        def f(a, b='d1'):
            r = b
            return r
        assert f('p1') == 'd1'
        """,
        {"same_name": "'d1'", "position": "'d1'", "recent": "'p1'"},
    ),
    (
        """
        f = lambda a, b: a
        assert f('l1', 'l2') == 'l1'
        """,
        {"same_name": None, "position": "'l1'", "recent": "'l2'"},
    ),
    (  # the value passes through a call's result
        """
        def h(u, w):
            return w
        x = 'k1'
        r = h(u=x, w='k2')
        assert r == 'k2'
        """,
        {"same_name": "'k2'", "position": "'k2'", "first_bound": "'k2'"},
    ),
]


@pytest.mark.parametrize(("source", "expected"), CASES)
def test_reader_heuristics(source, expected):
    program = textwrap.dedent(source).lstrip()
    line = len(program.splitlines())
    found = suites.literal_heuristics(program, line)
    assert set(found) == {"recent", "line1", "same_name", "first_bound", "stale", "position"}
    assert {k: found[k] for k in expected} == expected


def test_readout_prompt_ends_before_the_answer():
    program = "a = 'x1'\nassert a == 'x1'\n"
    assert suites.readout_prompt(program, 2) == ("a = 'x1'\nassert a == ", "'x1'")
    with pytest.raises(ValueError, match="not an `assert EXPR == VALUE` readout"):
        suites.readout_prompt(program, 1)


NATURAL = '''\
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
    items = suites.natural_items(NATURAL, "doc", rng=random.Random(0), limit=100)
    assert items and all(item.task == "T3" for item in items)
    for item in items:
        at = len(item.prompt)
        assert NATURAL[:at] == item.prompt and NATURAL[at:].startswith(item.gold + item.tail)
        assert item.gold in item.candidates and len(item.candidates) >= 2
        assert NATURAL[item.anchor : item.anchor + len(item.gold)] == item.gold  # a binding of the gold
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
    [digest] = [h for name, _, h in decontam.function_hashes(NATURAL) if name == "scale"]
    items = suites.natural_items(NATURAL, "doc", rng=random.Random(0), limit=100, exclude={digest})
    assert items and not any("def scale" in item.prompt.rsplit("\ndef ", 1)[-1] for item in items)


def test_static_subset():
    import ast

    for body, inside in (("x = 1\nreturn x", True), ("global y\ny = 1", False), ("del x", False)):
        fn = ast.parse("def f(x):\n" + textwrap.indent(body, "    ")).body[0]
        assert suites.static_subset(fn, suites._own_nodes(fn)) is inside


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
        # CLONE with other names in every file: its structure is in the train split, so it gives no item.
        clone = CLONE.replace("merge", f"merge{i}").replace("result", f"out{i}")
        (tmp_path / "code" / f"m{i:02d}.py").write_text(FUNCTIONS[i % 4] + "\n\n" + clone)
    spec = tmp_path / "corpus.toml"
    spec.write_text(
        'seed = 3\nval_fraction = 0.5\n\n[[source]]\nname = "code"\nkind = "code"\nfiles = ["code/*.py"]\n'
        'url = "https://example.invalid"\nlicense = "MIT"\ndownloaded = 2026-10-01\nfilters = []\n'
    )
    c = corpus.load_corpus(spec)
    manifest = suites.build_natural(c, tmp_path / "t3", seed=1)
    val = {doc.key for doc in corpus.documents(c) if corpus.split_of(c, doc) == "val"}
    halves = {half: suites.read_items(tmp_path / "t3" / f"{half}.jsonl") for half in suites.HALVES}
    keys = {half: {item.id.split(":")[1] for item in items} for half, items in halves.items()}
    assert halves["dev"] and halves["test"] and not keys["dev"] & keys["test"]
    assert keys["dev"] | keys["test"] <= val
    assert manifest["files"]["dev"]["items"] == len(halves["dev"])
    assert len(decontam.function_hashes(CLONE)) == 1
    assert not any(item.prompt.rsplit("\ndef ", 1)[-1].startswith("merge") for v in halves.values() for item in v)
    with pytest.raises(FileExistsError):
        suites.build_natural(c, tmp_path / "t3")
