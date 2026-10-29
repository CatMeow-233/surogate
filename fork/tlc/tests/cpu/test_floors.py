"""The heuristic floors of T1/T2 (surogate_tlc.data.floors): the scope-blind reader on hand-written programs."""

import textwrap

import pytest
from surogate_tlc.data import floors

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
    found = floors.literal_heuristics(program, line)
    assert set(found) == {"recent", "line1", "same_name", "first_bound", "stale", "position"}
    assert {k: found[k] for k in expected} == expected


def test_readout_prompt_ends_before_the_answer():
    program = "a = 'x1'\nassert a == 'x1'\n"
    assert floors.readout_prompt(program, 2) == ("a = 'x1'\nassert a == ", "'x1'")
    with pytest.raises(ValueError, match="not an `assert EXPR == VALUE` readout"):
        floors.readout_prompt(program, 1)
