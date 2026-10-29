"""Synthetic binding programs of a static Python subset: T1 value tracking and T2 formal-actual correspondence
(README: 合成绑定任务).

A program binds opaque string literals ('k3f9') to names and ends in a readout `assert EXPR == 'value'`. The value
is what Python computes: every program is executed, and the readout is then written with the value it captured.
There is no arithmetic, so the tasks test binding and not execution (review SD-01).

- T1 patterns: flat, shadow-in (EXPR is a name rebound in the innermost of `depth` nested functions),
  shadow-out (read after the shadowing calls), closure (read in nested functions from the enclosing one),
  shadow-closure (a closure reads the enclosing value after sibling functions shadowed it), call-pos and call-kw
  (the value passes through a function's parameter, positionally or by keyword). Grid: live names, hops (the
  length of the copy chain from the literal to EXPR) and overwrites (the chain rebound with a new literal).
- T2 patterns: positional, keyword, mixed, default, names-reordered (module variables named like the formals
  passed in another order) and keyword-reordered calls of one of 1 + `distractors` functions with the same formal
  names; grid: formals, distractors, actuals (literals or names).

The training stream and the eval suites are disjoint on four axes (v0.7 §9, the salience memo §2.4.1): template
family (FAMILIES; HELD_OUT_FAMILIES only in eval), scope pattern (HELD_OUT_PATTERNS only in eval: shadowing
inside a closure, reordered keyword calls), identifier pool (eval adds `novel`, unseen combinations of seen
morphemes, and `rare`, strings of letters no training identifier uses) and literal space (a hash bucket of the
literals, eval only). On top, no training program has the alpha-renamed, literal-normalised AST of an eval item
(data.decontam.ast_hash): build() rejects and redraws them, and check() verifies all axes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import math
import random
import re
import string
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from ..modeldir import dump_json, write_text_atomic
from .corpus import file_sha256
from .decontam import ast_hash
from .suites import (
    HALVES,
    MANIFEST_FILE,
    SUITE_FILES,
    SUITE_VERSION,
    Item,
    literal_heuristics,
    read_items,
    readout_prompt,
    string_literals,
    write_items,
)

SYNTH_VERSION = 1
TRAIN_FILE = "train.jsonl"
LIVE = (2, 4, 8, 16, 24, 32, 64)
HOPS = (1, 2, 3, 4)
OVERWRITES = (0, 1, 2, 4)
DEPTHS = (1, 2)
FORMALS = (2, 3, 4, 6)
DISTRACTORS = (0, 1, 2)
ACTUALS = ("literal", "name")
T1_PATTERNS = ("flat", "shadow-in", "shadow-out", "closure", "shadow-closure", "call-pos", "call-kw")
T2_PATTERNS = ("positional", "keyword", "mixed", "default", "names-reordered", "keyword-reordered")
SCOPED = {"shadow-in", "shadow-out", "closure", "shadow-closure"}
FAMILIES = {"T1": ("assign", "unpack", "annotated"), "T2": ("return", "local", "lambda")}
HELD_OUT_FAMILIES = {"annotated", "lambda"}
HELD_OUT_PATTERNS = {"shadow-closure", "keyword-reordered"}
MORPHEMES = tuple(
    "acc arg buf cell col cur dst elem flag head idx item key last left line mid name node num off part pos prev ptr "
    "res right row seen size src tail tmp top val word".split()
)
RARE_LETTERS = "jkqvwxz"
NOVEL_SHARE = 0.2  # share of the two-morpheme names held out as `novel`
EVAL_LITERAL_BUCKETS = 4  # literals whose hash is 0 mod this are eval-only
DOC_CHARS = 1600  # a training document gathers programs up to about this many characters
REDRAWS = 20
T2_UNRELATED = 3  # a T2 program also binds up to this many names it never passes


def _bucket(text: str, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{salt}\0{text}".encode()).digest()[:8], "little")


def _pools() -> dict[str, tuple[str, ...]]:
    pairs = [f"{a}_{b}" for a, b in itertools.permutations(MORPHEMES, 2)]
    novel = tuple(p for p in pairs if _bucket(p, "pool") < NOVEL_SHARE * 2**64)
    seen = MORPHEMES + tuple(p for p in pairs if p not in set(novel))
    rare = tuple("".join(t) for n in (3, 4) for t in itertools.product(RARE_LETTERS, repeat=n))
    return {"seen": seen, "novel": novel, "rare": rare}


POOLS = _pools()


def eval_literal(text: str) -> bool:
    """Whether a literal's body is in the eval-only space."""
    return _bucket(text, "literal") % EVAL_LITERAL_BUCKETS == 0


class _Draw:
    """Fresh names from one pool and fresh literals from one space, for one program or document."""

    def __init__(self, rng: random.Random, pool: str, space: str) -> None:
        self.rng, self.pool, self.space = rng, POOLS[pool], space
        self.names: set[str] = set()
        self.literals: set[str] = set()

    def name(self) -> str:
        while (x := self.rng.choice(self.pool)) in self.names:
            pass
        self.names.add(x)
        return x

    def literal(self) -> str:
        alphabet = string.ascii_lowercase + string.digits
        while True:
            body = self.rng.choice(string.ascii_lowercase) + "".join(self.rng.choices(alphabet, k=3))
            if body not in self.literals and eval_literal(body) == (self.space == "eval"):
                self.literals.add(body)
                return repr(body)


def _scatter(rng: random.Random, blocks: list[list[str]], fillers: list[str]) -> list[str]:
    """`blocks` in order with each filler line in a random gap before, between or after them."""
    gaps: list[list[str]] = [[] for _ in range(len(blocks) + 1)]
    for line in fillers:
        gaps[rng.randrange(len(gaps))].append(line)
    return [line for gap, block in zip(gaps, [*blocks, []], strict=True) for line in gap + block]


def _locals(d: _Draw, indent: str) -> list[str]:
    return [f"{indent}{d.name()} = {d.literal()}" for _ in range(d.rng.randint(0, 2))]


def t1_program(rng: random.Random, cell: dict[str, Any], d: _Draw) -> list[str]:
    """Lines of a T1 program; one line is the placeholder readout __probe__(EXPR)."""
    pattern, live, hops = cell["pattern"], cell["live"], cell["hops"]
    chain = [d.name() for _ in range(hops)]  # chain[0] is read; chain[-1] holds the literal
    fillers = [f"{d.name()} = {d.literal()}" for _ in range(live - hops)]
    blocks = [
        [f"{chain[-1]} = {d.literal()}", *(f"{chain[i - 1]} = {chain[i]}" for i in range(hops - 1, 0, -1))]
        for _ in range(cell["overwrites"] + 1)
    ]
    q, head, tail = chain[0], [], []
    if pattern in ("closure", "shadow-closure"):  # the chain lives in an outer function the closures read
        outer = d.name()
        body = ["    " + line for line in _scatter(rng, blocks, fillers)]
        return [f"def {outer}():", *body, *_scoped(pattern, cell["depth"], q, d, indent=1), f"{outer}()"]
    if pattern in ("call-pos", "call-kw"):
        fn, params, result = d.name(), [d.name() for _ in range(3)], d.name()
        k = rng.randrange(3)
        head = [f"def {fn}({', '.join(params)}):", f"    return {params[k]}"]
        args = [q if i == k else d.name() for i in range(3)]
        fillers += [f"{a} = {d.literal()}" for i, a in enumerate(args) if i != k]
        call = ", ".join(args if pattern == "call-pos" else [f"{p}={a}" for p, a in zip(params, args, strict=True)])
        tail = [f"{result} = {fn}({call})", f"__probe__({result})"]
    elif pattern == "flat":
        tail = [f"__probe__({q})"]
    else:
        tail = _scoped(pattern, cell["depth"], q, d)
    return head + _scatter(rng, blocks, fillers) + tail


def _scoped(pattern: str, depth: int, q: str, d: _Draw, indent: int = 0) -> list[str]:
    """Nested functions at `indent` after the chain; each is called right after its definition, so a read inside one
    sees the bindings its text shows."""
    lines: list[str] = []

    def nest(levels: int, indent: int, bind_q: bool, probe: bool) -> None:
        # `levels` functions nested in each other, each binding q or not; the innermost may hold the readout.
        name, pad = d.name(), "    " * indent
        lines.append(f"{pad}def {name}():")
        lines.extend(_locals(d, pad + "    "))
        if bind_q:
            lines.append(f"{pad}    {q} = {d.literal()}")
        if levels > 1:
            nest(levels - 1, indent + 1, bind_q, probe)
        elif probe:
            lines.extend(_locals(d, pad + "    "))
            lines.append(f"{pad}    __probe__({q})")
        lines.append(f"{pad}{name}()")

    if pattern in ("shadow-in", "shadow-out"):  # q rebound in each of `depth` nested functions
        nest(depth, indent, True, pattern == "shadow-in")
        if pattern == "shadow-out":
            lines.append(f"{'    ' * indent}__probe__({q})")
    elif pattern == "closure":  # `depth` nested functions read the enclosing q
        nest(depth, indent, False, True)
    else:  # shadow-closure: functions shadow q, then a sibling closure reads the enclosing q
        nest(depth, indent, True, False)
        nest(1, indent, False, True)
    return lines


def t2_program(rng: random.Random, cell: dict[str, Any], d: _Draw) -> list[str]:
    """Lines of a T2 program; the last is the placeholder readout __probe__(f(ARGS))."""
    pattern, n, family = cell["pattern"], cell["formals"], cell["family"]
    params = [d.name() for _ in range(n)]
    k = rng.randrange(n)
    defaults: dict[str, str] = {}
    if pattern == "default":
        defaults = {p: d.literal() for p in params[n - rng.randint(1, n - 1) :]}
    fns = [d.name() for _ in range(cell["distractors"] + 1)]
    defs = [_definition(family, fns[0], params, defaults, params[k], d)]
    for fn in fns[1:]:
        order = _shuffled(rng, params)
        defs.append(_definition(family, fn, order, {}, rng.choice(params), d))
    rng.shuffle(defs)
    if pattern == "names-reordered":
        bound = _shuffled(rng, params, identity=True)
        args = _shuffled(rng, params)
    elif cell["actuals"] == "name":
        args = [d.name() for _ in params]
        bound = _shuffled(rng, args, identity=True)
    else:
        args, bound = [d.literal() for _ in params], []
    bindings = [[f"{a} = {d.literal()}"] for a in bound]
    unrelated = [f"{d.name()} = {d.literal()}" for _ in range(rng.randint(0, T2_UNRELATED))]
    lines = _scatter(rng, defs + bindings, unrelated)
    if pattern in ("keyword", "keyword-reordered"):
        pairs = [f"{p}={a}" for p, a in zip(params, args, strict=True)]
        call = _shuffled(rng, pairs) if pattern == "keyword-reordered" else pairs
    elif pattern == "mixed":
        j = rng.randint(1, n - 1)
        call = args[:j] + [f"{p}={a}" for p, a in zip(params[j:], args[j:], strict=True)]
    elif pattern == "default":
        free = n - len(defaults)
        call = args[:free] + [f"{p}={a}" for p, a in zip(params[free:], args[free:], strict=True) if rng.random() < 0.5]
    else:
        call = args
    return lines + [f"__probe__({fns[0]}({', '.join(call)}))"]


def _shuffled(rng: random.Random, xs: Sequence[str], identity: bool = False) -> list[str]:
    """A random order of `xs`; not the given one unless `identity` allows it (or there is only one)."""
    while True:
        out = list(xs)
        rng.shuffle(out)
        if identity or len(out) < 2 or out != list(xs):
            return out


def _definition(family: str, fn: str, params: list[str], defaults: dict[str, str], ret: str, d: _Draw) -> list[str]:
    signature = ", ".join(f"{p}={defaults[p]}" if p in defaults else p for p in params)
    if family == "lambda":
        return [f"{fn} = lambda {signature}: {ret}"]
    if family == "local":
        local = d.name()
        return [f"def {fn}({signature}):", f"    {local} = {ret}", f"    return {local}"]
    return [f"def {fn}({signature}):", f"    return {ret}"]


ASSIGN = re.compile(r"^(\s*)(\w+) = (.+)$")
LITERAL_BINDING = re.compile(r"^(\s*)(\w+) = ('[a-z0-9]+')$")


def apply_family(lines: list[str], family: str, rng: random.Random) -> list[str]:
    """T1 surface forms: `unpack` groups adjacent literal bindings at one indent into tuple assignments; `annotated`
    writes every binding as `name: str = value`. Both keep the program's meaning."""
    if family == "annotated":
        return [ASSIGN.sub(r"\1\2: str = \3", line) for line in lines]
    if family != "unpack":
        return lines
    out: list[str] = []
    group: list[re.Match] = []

    def flush() -> None:
        while group:
            size = min(len(group), rng.randint(2, 3))
            part, group[:size] = group[:size], []
            if size == 1:
                out.append(part[0].group(0))
            else:
                names, values = ", ".join(m.group(2) for m in part), ", ".join(m.group(3) for m in part)
                out.append(f"{part[0].group(1)}{names} = {values}")

    for line in lines:
        m = LITERAL_BINDING.match(line)
        if m and (not group or group[0].group(1) == m.group(1)):
            group.append(m)
            continue
        flush()
        if m:
            group.append(m)
        else:
            out.append(line)
    flush()
    return out


def run(program: str) -> list[Any]:
    """Execute a generated program without builtins and return what its __probe__ calls received."""
    captured: list[Any] = []
    exec(compile(program, "<tlc-synth>", "exec"), {"__builtins__": {}, "__probe__": captured.append})
    return captured


def program(task: str, cell: dict[str, Any], rng: random.Random, pool: str, space: str) -> tuple[str, int]:
    """(program, readout line) of one exec-verified item."""
    d = _Draw(rng, pool, space)
    lines = t1_program(rng, cell, d) if task == "T1" else t2_program(rng, cell, d)
    if task == "T1":
        lines = apply_family(lines, cell["family"], rng)
    [readout] = [i for i, line in enumerate(lines) if line.lstrip().startswith("__probe__(")]
    values = run("\n".join(lines) + "\n")
    if len(values) != 1 or not isinstance(values[0], str):
        raise AssertionError(f"the readout ran {len(values)} times: {lines}")
    pad, expr = re.match(r"(\s*)__probe__\((.*)\)$", lines[readout]).groups()
    lines[readout] = f"{pad}assert {expr} == {values[0]!r}"
    text = "\n".join(lines) + "\n"
    run(text)  # the readout holds
    return text, readout + 1


def cells(task: str, patterns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """The difficulty grid of `task` (README: 合成绑定任务)."""
    if task == "T1":
        return [
            {"pattern": p, "live": live, "hops": hops, "overwrites": o, "depth": depth}
            for p in patterns or T1_PATTERNS
            for live, hops, o in itertools.product(LIVE, HOPS, OVERWRITES)
            if hops <= live
            for depth in (DEPTHS if p in SCOPED else (0,))
        ]
    return [
        {"pattern": p, "formals": n, "distractors": k, "actuals": a}
        for p in patterns or T2_PATTERNS
        for n, k, a in itertools.product(FORMALS, DISTRACTORS, ACTUALS)
        if not (p == "names-reordered" and a == "literal")
    ]


def make_item(task: str, cell: dict[str, Any], rng: random.Random, pool: str, space: str, item_id: str) -> Item:
    text, line = program(task, cell, rng, pool, space)
    prompt, gold = readout_prompt(text, line)
    readout = text.splitlines()[line - 1]
    query = None
    if expr := re.match(r"(\s*assert )(\w+) == ", readout):  # a name: its span in the prompt's last line
        start = len(prompt) - len(readout) + len(gold) + expr.end(1)
        query = (start, start + len(expr.group(2)))
    held = {"family": HELD_OUT_FAMILIES, "pattern": HELD_OUT_PATTERNS}
    heldout = [axis for axis, values in held.items() if cell[axis] in values]
    return Item(
        id=item_id,
        task=task,
        prompt=prompt,
        candidates=list(dict.fromkeys(string_literals(prompt))),
        gold=gold,
        tail="\n",
        heuristics=literal_heuristics(text, line),
        strata={**cell, "pool": pool, "heldout": heldout + (["pool"] if pool != "seen" else [])},
        anchor=prompt.rfind(gold),
        query=query,
        program=text,
        hash=ast_hash(ast.parse(text)),
    )


def suite(task: str, half: str, per_task: int, seed: int) -> list[Item]:
    """Eval items of one task and half: every grid cell, each item with a random family and pool."""
    grid = cells(task)
    per_cell = math.ceil(per_task / len(grid))
    items = []
    for i, cell in enumerate(c for c in grid for _ in range(per_cell)):
        rng = random.Random(f"{seed}\0{task}\0{half}\0{i}")
        cell = {**cell, "family": rng.choice(FAMILIES[task])}
        items.append(make_item(task, cell, rng, rng.choice(tuple(POOLS)), "eval", f"{task}-{half}-{i:05d}"))
    return items


def train_programs(seed: int, exclude: set[str], redrawn: Counter) -> Iterator[tuple[str, dict[str, Any]]]:
    """An endless stream of training programs: T1 and T2 alternately, cells uniform over the training patterns,
    training families, the `seen` pool and the training literal space. A program with an excluded AST is redrawn
    from its cell up to REDRAWS times, then the cell is skipped; `redrawn` counts both per (task, pattern)."""
    grids = {
        task: cells(task, [p for p in patterns if p not in HELD_OUT_PATTERNS])
        for task, patterns in (("T1", T1_PATTERNS), ("T2", T2_PATTERNS))
    }
    families = {task: [f for f in fams if f not in HELD_OUT_FAMILIES] for task, fams in FAMILIES.items()}
    for i in itertools.count():
        task = ("T1", "T2")[i % 2]
        rng = random.Random(f"{seed}\0train\0{i}")
        cell = {**rng.choice(grids[task]), "family": rng.choice(families[task])}
        for _ in range(REDRAWS):
            text, _ = program(task, cell, rng, "seen", "train")
            digest = ast_hash(ast.parse(text))
            if digest not in exclude:
                yield text, {"task": task, **cell, "hash": digest}
                break
            redrawn[task, cell["pattern"], "redrawn"] += 1
        else:
            redrawn[task, cell["pattern"], "skipped"] += 1


def training_documents(seed: int, chars: int, exclude: set[str], doc_chars: int = DOC_CHARS):
    """(documents, redrawn): JSONL records {"text", "items"} of about `chars` characters in all; a document is a
    module of consecutive programs (v0.7 §9: a generated multi-function module), executed as a whole."""
    redrawn: Counter = Counter()
    stream = train_programs(seed, exclude, redrawn)
    docs, total = [], 0
    while total < chars:
        texts, meta = [], []
        while sum(map(len, texts)) < doc_chars:
            text, info = next(stream)
            texts.append(text)
            meta.append(info)
        document = "\n".join(texts)
        run(document)
        docs.append({"text": document, "items": meta})
        total += len(document)
    return docs, redrawn


IDENTIFIER = re.compile(r"\b[A-Za-z_]\w*\b")
LITERAL = re.compile(r"'([a-z0-9]+)'")


def check(out: Path) -> dict[str, Any]:
    """Disjointness of the training stream and the eval halves in `out` (raises AssertionError on a violation)."""
    out = Path(out)
    halves = {half: read_items(out / SUITE_FILES[half]) for half in HALVES}
    evals = [item for items in halves.values() for item in items]
    held = set(POOLS["novel"]) | set(POOLS["rare"])
    train_hashes, counts = set(), Counter()
    with (out / TRAIN_FILE).open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            for info in record["items"]:
                assert info["family"] not in HELD_OUT_FAMILIES and info["pattern"] not in HELD_OUT_PATTERNS, info
                train_hashes.add(info["hash"])
                counts[info["task"]] += 1
            text = record["text"]
            names = set(IDENTIFIER.findall(LITERAL.sub("''", text)))
            assert not held & names, "a held-out identifier in the training stream"
            assert not any(eval_literal(lit) for lit in LITERAL.findall(text)), "an eval literal in training"
    shared = [item.id for item in evals if item.hash in train_hashes]
    assert not shared, f"eval items with a training AST: {shared[:5]}"
    texts = {half: {item.program for item in items} for half, items in halves.items()}
    assert not texts["dev"] & texts["test"], "a program in both halves"
    for item in evals:
        assert not item.problems(), item.problems()
        assert all(eval_literal(lit[1:-1]) for lit in item.candidates), item.id
    return {
        "train_items": dict(sorted(counts.items())),
        "train_hashes": len(train_hashes),
        "eval_items": {half: len(items) for half, items in halves.items()},
        "eval_items_with_a_training_ast": 0,
    }


def build(out: Path, *, seed: int = 0, per_task: int = 4000, train_chars: int = 2_000_000) -> dict[str, Any]:
    """Write dev.jsonl and test.jsonl (T1 and T2 items), train.jsonl (a corpus source) and the manifest."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    halves = {half: [i for task in ("T1", "T2") for i in suite(task, half, per_task, seed)] for half in HALVES}
    dev = {item.program for item in halves["dev"]}
    halves["test"] = [item for item in halves["test"] if item.program not in dev]
    files = {}
    for half, items in halves.items():
        write_items(out / SUITE_FILES[half], items)
        files[half] = {"file": SUITE_FILES[half], "items": len(items), "sha256": file_sha256(out / SUITE_FILES[half])}
    exclude = {item.hash for items in halves.values() for item in items}
    docs, redrawn = training_documents(seed, train_chars, exclude)
    write_text_atomic(out / TRAIN_FILE, "".join(json.dumps(doc) + "\n" for doc in docs))
    files["train"] = {
        "file": TRAIN_FILE,
        "documents": len(docs),
        "programs": sum(len(doc["items"]) for doc in docs),
        "characters": sum(len(doc["text"]) for doc in docs),
        "sha256": file_sha256(out / TRAIN_FILE),
    }
    manifest = {
        "tlc_synth_version": SYNTH_VERSION,
        "tlc_suite_version": SUITE_VERSION,
        "tasks": ["T1", "T2"],
        "seed": seed,
        "grid": {
            "T1": {"patterns": T1_PATTERNS, "live": LIVE, "hops": HOPS, "overwrites": OVERWRITES, "depth": DEPTHS},
            "T2": {"patterns": T2_PATTERNS, "formals": FORMALS, "distractors": DISTRACTORS, "actuals": ACTUALS},
        },
        "held_out": {
            "families": sorted(HELD_OUT_FAMILIES),
            "patterns": sorted(HELD_OUT_PATTERNS),
            "pools": {"novel": len(POOLS["novel"]), "rare": len(POOLS["rare"]), "seen (training)": len(POOLS["seen"])},
            "literal_space": f"sha256 bucket 0 of {EVAL_LITERAL_BUCKETS}",
        },
        "train": {
            "doc_chars": DOC_CHARS,
            "mixture": "T1 and T2 alternately; cells uniform over the training grid; no ramp (static mixture)",
            "eval_ast_collisions": {f"{t}/{p}/{what}": n for (t, p, what), n in sorted(redrawn.items())},
        },
        "files": files,
    }
    manifest["disjoint"] = check(out)
    (out / MANIFEST_FILE).write_text(dump_json(manifest), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tlc.data.synth", description=__doc__.split("\n")[0])
    parser.add_argument("out", type=Path, help="output directory (absent or empty)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-task", type=int, default=4000, help="eval items per task and half (at least)")
    parser.add_argument("--train-chars", type=int, default=2_000_000, help="characters of training programs")
    args = parser.parse_args(argv)
    manifest = build(args.out, seed=args.seed, per_task=args.per_task, train_chars=args.train_chars)
    print(json.dumps({"files": manifest["files"], "train": manifest["train"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
