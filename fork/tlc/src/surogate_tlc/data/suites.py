"""Binding probe items, their heuristic floors, and the natural-code suite T3 (README: 绑定探针).

An item asks for one continuation of a prompt: eval.probes scores every candidate after the prompt with the model
and counts the item correct when the gold candidate scores best (v0.7 §12.2, review SD-01).

- T1 value tracking and T2 formal-actual correspondence come from data.synth: the prompt is a program up to its
  readout `assert EXPR == `, the candidates are the string literals of the prompt, the gold is the value of EXPR.
- T3 is natural code: the prompt is a held-out document up to a load of a local variable, the candidates are the
  locals bound before it in the function, the gold is the loaded name.

Every item carries the predictions of the heuristic floors. For T1 and T2 they come from a scope-blind reader of
the program text (literal_heuristics): `recent` (the last literal of the prompt), `line1` (the first literal),
`same_name` (EXPR resolved through the latest textual binding of each name, ignoring scopes; a call's keyword
arguments bind its parameters' names), `first_bound` (the same through the earliest bindings), `stale` (the value
the reader gave EXPR before its last change) and `position` (a call's returned parameter taken by argument
position, keywords ignored). For T3: `recent` (the most recently mentioned local) and `frequent` (the most
loaded so far). eval.probes reports the best of them per stratum as the floor.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import io
import json
import math
import random
import symtable
import sys
import tokenize
import warnings
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from ..modeldir import dump_json, write_text_atomic
from .corpus import Corpus, documents, file_sha256, load_corpus, split_of, unit_hash
from .decontam import function_hashes

SUITE_VERSION = 1
TASKS = ("T1", "T2", "T3")
HALVES = ("dev", "test")
SUITE_FILES = {half: f"{half}.jsonl" for half in HALVES}
MANIFEST_FILE = "manifest.json"
T3_MAX_PROMPT_CHARS = 16000  # the prompt keeps the document's tail up to the load, from a line start
T3_PER_DOCUMENT = 8
CANDIDATE_BUCKETS = ((2, "2"), (4, "3-4"), (8, "5-8"), (16, "9-16"), (math.inf, ">16"))
REFLECTION = {"eval", "exec", "locals", "globals", "vars", "getattr", "setattr", "delattr"}


@dataclasses.dataclass
class Item:
    id: str
    task: str
    prompt: str
    candidates: list[str]
    gold: str
    tail: str  # the text after the candidate to the end of its line, the same for every candidate
    heuristics: dict[str, str | None]
    strata: dict[str, Any]
    anchor: int | None = None  # offset in prompt of the binding the gold comes from (distance buckets)
    query: tuple[int, int] | None = None  # [start, end) in prompt of the identifier asked about (piece count)
    start: int = 0  # offset of prompt in its document; 0: the prompt starts the document (framing tokens)
    program: str | None = None  # T1/T2: the whole program, its readout completed with the gold
    hash: str | None = None  # T1/T2: data.decontam.ast_hash of the program

    def problems(self) -> list[str]:
        out = []
        if self.task not in TASKS:
            out.append(f"task {self.task!r}")
        if self.gold not in self.candidates or len(set(self.candidates)) != len(self.candidates):
            out.append("the gold must be one of distinct candidates")
        if not self.tail:
            out.append("empty tail")
        return [f"{self.id}: {p}" for p in out]


def write_items(path: Path, items: Iterable[Item]) -> int:
    lines = [json.dumps(dataclasses.asdict(item), ensure_ascii=False) for item in items]
    write_text_atomic(Path(path), "".join(line + "\n" for line in lines))
    return len(lines)


def read_items(path: Path) -> list[Item]:
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("query") is not None:
                record["query"] = tuple(record["query"])
            items.append(Item(**record))
    return items


def string_literals(src: str) -> list[str]:
    """The source text of the string literals of `src`, in order; tokens up to a tokenizer error count."""
    out = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type == tokenize.STRING:
                out.append(t.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


# The scope-blind reader of T1/T2 programs.

Pos = tuple[int, int]


@dataclasses.dataclass
class _Function:
    pos: Pos
    name: str
    params: list[str]
    defaults: dict[str, ast.expr]
    returns: ast.expr | None


class _Reader:
    """Textual bindings before a readout, read as if the program had one flat namespace."""

    def __init__(self, tree: ast.Module, readout: ast.stmt) -> None:
        self.events: list[tuple[Pos, str, ast.expr]] = []
        self.functions: list[_Function] = []
        self.stop = (readout.lineno, readout.col_offset)
        self._collect(tree.body)

    def _collect(self, stmts: Sequence[ast.stmt]) -> None:
        for s in stmts:
            if (s.lineno, s.col_offset) >= self.stop:
                return
            end = (s.end_lineno, s.end_col_offset)
            if isinstance(s, ast.FunctionDef):
                returns = next((x.value for x in s.body if isinstance(x, ast.Return)), None)
                self._function((s.lineno, s.col_offset), s.name, s.args, returns)
                self._collect(s.body)
            elif isinstance(s, ast.Assign):
                for target in s.targets:
                    self._bind(target, s.value, end)
            elif isinstance(s, ast.AnnAssign) and s.value is not None:
                self._bind(s.target, s.value, end)

    def _function(self, pos: Pos, name: str, args: ast.arguments, returns: ast.expr | None) -> None:
        params = [a.arg for a in args.args]
        defaults = dict(zip(params[len(params) - len(args.defaults) :], args.defaults, strict=True))
        self.functions.append(_Function(pos, name, params, defaults, returns))
        self.events += [(pos, p, d) for p, d in defaults.items()]

    def _bind(self, target: ast.expr, value: ast.expr, end: Pos) -> None:
        if isinstance(target, ast.Name):
            if isinstance(value, ast.Lambda):
                self._function(end, target.id, value.args, value.body)
            self.events.append((end, target.id, value))
        elif isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple) and len(target.elts) == len(value.elts):
            for t, v in zip(target.elts, value.elts, strict=True):
                self._bind(t, v, end)

    def _latest(self, name: str, before: Pos, first: bool = False) -> tuple[Pos, str, ast.expr] | None:
        found = [e for e in self.events if e[1] == name and e[0] < before]
        return (found[0] if first else found[-1]) if found else None

    def function(self, name: str, before: Pos) -> _Function | None:
        return next((f for f in reversed(self.functions) if f.name == name and f.pos < before), None)

    def resolve(self, expr: ast.expr | None, before: Pos, *, first: bool = False, overlay=None, depth: int = 0):
        """The literal (source text) `expr` reads as, or None."""
        overlay = overlay or {}
        if expr is None or depth > 64:
            return None
        if isinstance(expr, ast.Constant):
            return repr(expr.value) if isinstance(expr.value, str) else None
        if isinstance(expr, ast.Name):
            if expr.id in overlay:
                return self.resolve(overlay[expr.id][1], overlay[expr.id][0], first=first, depth=depth + 1)
            event = self._latest(expr.id, before, first)
            if event is None:
                return None
            return self.resolve(event[2], event[0], first=first, overlay=overlay, depth=depth + 1)
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
            fn = self.function(expr.func.id, before)
            if fn is None:
                return None
            call = (expr.end_lineno, expr.end_col_offset)
            keywords = {kw.arg: (before, kw.value) for kw in expr.keywords if kw.arg}
            return self.resolve(fn.returns, call, first=first, overlay=keywords, depth=depth + 1)
        return None

    def position(self, expr: ast.expr) -> str | None:
        """A call's returned parameter mapped by argument position over all arguments, keywords included."""
        before = self.stop
        while isinstance(expr, ast.Name):
            event = self._latest(expr.id, before)
            if event is None:
                return None
            before, _, expr = event
        if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name)):
            return None
        fn = self.function(expr.func.id, before)
        if fn is None or not isinstance(fn.returns, ast.Name):
            return None
        call, name = (expr.end_lineno, expr.end_col_offset), fn.returns.id
        while name not in fn.params and (event := self._latest(name, call)) and isinstance(event[2], ast.Name):
            name = event[2].id
        if name not in fn.params:
            return None
        k, args = fn.params.index(name), [*expr.args, *(kw.value for kw in expr.keywords)]
        if k < len(args):
            return self.resolve(args[k], before)
        return self.resolve(fn.defaults.get(name), fn.pos)

    def history(self, expr: ast.expr) -> list[str]:
        """The values the reader gives `expr` after each binding, consecutive repeats dropped."""
        out: list[str] = []
        for pos, _, _ in self.events:
            value = self.resolve(expr, (pos[0], pos[1] + 1))
            if value is not None and (not out or out[-1] != value):
                out.append(value)
        return out


def readout_statement(tree: ast.Module, line: int) -> ast.Assert:
    found = [n for n in ast.walk(tree) if isinstance(n, ast.Assert) and n.lineno == line]
    if len(found) != 1 or not isinstance(found[0].test, ast.Compare):
        raise ValueError(f"line {line} is not an `assert EXPR == VALUE` readout")
    return found[0]


def readout_prompt(program: str, line: int) -> tuple[str, str]:
    """(prompt, answer) of the readout `assert EXPR == ANSWER` on `line` of an ASCII `program`: the prompt is the
    program up to ANSWER."""
    readout = readout_statement(ast.parse(program), line)
    answer = readout.test.comparators[0]
    start = sum(len(text) for text in program.splitlines(keepends=True)[: line - 1])
    return program[: start + answer.col_offset], program[start + answer.col_offset : start + answer.end_col_offset]


def literal_heuristics(program: str, line: int) -> dict[str, str | None]:
    """The T1/T2 heuristic predictions for the readout `assert EXPR == ...` on `line` of `program`."""
    tree = ast.parse(program)
    readout = readout_statement(tree, line)
    reader, expr = _Reader(tree, readout), readout.test.left
    literals = string_literals(readout_prompt(program, line)[0])
    history = reader.history(expr)
    return {
        "recent": literals[-1] if literals else None,
        "line1": literals[0] if literals else None,
        "same_name": reader.resolve(expr, reader.stop),
        "first_bound": reader.resolve(expr, reader.stop, first=True),
        "stale": history[-2] if len(history) >= 2 else None,
        "position": reader.position(expr),
    }


# T3: the next local identifier in natural code.


def _own_nodes(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """The nodes of fn's own scope: nested functions, lambdas and classes are not entered (their names, defaults
    and decorators are), comprehensions only through their first iterable, as Python evaluates them."""
    out: list[ast.AST] = []
    stack: list[ast.AST] = list(reversed(fn.body))
    while stack:
        node = stack.pop()
        out.append(node)
        children = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Lambda):
                out.append(child)  # its name binds in fn
                children += getattr(child, "decorator_list", [])
                if not isinstance(child, ast.ClassDef):
                    children += [d for d in child.args.defaults + child.args.kw_defaults if d is not None]
            elif isinstance(child, ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
                children.append(child.generators[0].iter)
            else:
                children.append(child)
        stack += reversed(children)
    return out


def static_subset(fn: ast.FunctionDef | ast.AsyncFunctionDef, nodes: Sequence[ast.AST]) -> bool:
    """Functions whose locals a textual reading resolves: no global/nonlocal declarations, no del of a name, no
    reflection on the namespace (eval, exec, locals, getattr, ...)."""
    for node in nodes:
        if isinstance(node, ast.Global | ast.Nonlocal):
            return False
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del):
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in REFLECTION:
            return False
    return True


def _line_offsets(text: str) -> list[int]:
    offsets, total = [0], 0
    for line in text.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def _function_tables(table: symtable.SymbolTable) -> Iterator[symtable.SymbolTable]:
    for child in table.get_children():
        if child.get_type() == "function":  # a str, or a StrEnum from Python 3.13 on
            yield child
        yield from _function_tables(child)


def natural_items(
    text: str, key: str, *, rng: random.Random, limit: int = T3_PER_DOCUMENT, exclude: set[str] = frozenset()
) -> list[Item]:
    """T3 items of one Python document: loads of a function's locals that a binding earlier in the function's
    text resolves, in functions of the static subset whose ast_hash is not in `exclude`."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree, top = ast.parse(text), symtable.symtable(text, key, "exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    tables: dict[tuple[str, int], symtable.SymbolTable] = {}
    for t in _function_tables(top):
        tables.setdefault((t.get_name(), t.get_lineno()), t)
    offsets, lines = _line_offsets(text), text.splitlines(keepends=True)

    def offset(line: int, col: int) -> int:  # ast columns count UTF-8 bytes
        return offsets[line - 1] + len(lines[line - 1].encode()[:col].decode(errors="ignore"))

    excluded = {line for _, line, digest in function_hashes(text) if digest in exclude}
    found = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef) or fn.lineno in excluded:
            continue
        line = fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno
        table = tables.get((fn.name, fn.lineno)) or tables.get((fn.name, line))
        nodes = _own_nodes(fn)
        if table is None or not static_subset(fn, nodes):
            continue
        local = {s.get_name() for s in table.get_symbols() if s.is_local() or s.is_parameter()}
        found += _function_loads(fn, nodes, local, offset)
    picked = sorted(rng.sample(found, min(limit, len(found))), key=lambda x: x[0])
    items = []
    for at, name, candidates, heuristics, anchor in picked:
        if text[at : at + len(name)] != name:
            continue  # an offset this reading got wrong
        start = text.rfind("\n", 0, max(0, at - T3_MAX_PROMPT_CHARS)) + 1 if at > T3_MAX_PROMPT_CHARS else 0
        end = at + len(name)
        tail = text[end : text.find("\n", end) + 1 if "\n" in text[end:] else len(text)] or "\n"
        items.append(
            Item(
                id=f"T3:{key}:{at}",
                task="T3",
                prompt=text[start:at],
                candidates=candidates,
                gold=name,
                tail=tail,
                heuristics=heuristics,
                strata={"candidates": candidate_bucket(len(candidates))},
                anchor=anchor - start if anchor >= start else None,
                start=start,
            )
        )
    return items


def candidate_bucket(n: int) -> str:
    return next(label for bound, label in CANDIDATE_BUCKETS if n <= bound)


def _effects(nodes: Sequence[ast.AST], offset) -> dict[int, int]:
    """id(Name node) -> the offset where its binding takes effect, when that is after the name: assignment targets
    at the end of the statement, loop targets after the iterable, with-as targets after the context expression,
    walrus targets after the value."""
    out = {}
    for node in nodes:
        if isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
            at, targets = node, node.targets if isinstance(node, ast.Assign) else [node.target]
        elif isinstance(node, ast.For | ast.AsyncFor):
            at, targets = node.iter, [node.target]
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            at, targets = node.context_expr, [node.optional_vars]
        elif isinstance(node, ast.NamedExpr):
            at, targets = node, [node.target]
        else:
            continue
        for target in targets:
            for name in ast.walk(target):
                if isinstance(name, ast.Name):
                    out[id(name)] = offset(at.end_lineno, at.end_col_offset)
    return out


def _function_loads(fn, nodes, local, offset) -> list[tuple[int, str, list[str], dict[str, str], int]]:
    """(offset, name, candidates, heuristics, anchor offset) per prefix-resolvable load of a local with at least two
    candidates. Bindings (parameters, stored names, nested def/class names, import aliases, except-as names) count
    from where Python performs them; the anchor is the binding's name in the text."""
    events = []  # (effect offset, order, name, kind, text offset); a load and a binding at one offset: load first
    args = fn.args
    for a in args.posonlyargs + args.args + args.kwonlyargs + [x for x in (args.vararg, args.kwarg) if x]:
        at = offset(a.lineno, a.col_offset)
        events.append((at, 1, a.arg, "bind", at))
    effects = _effects(nodes, offset)
    for node in nodes:
        if isinstance(node, ast.Name):
            at = offset(node.lineno, node.col_offset)
            if isinstance(node.ctx, ast.Load):
                events.append((at, 0, node.id, "load", at))
            else:
                events.append((effects.get(id(node), at), 1, node.id, "bind", at))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            at = offset(node.lineno, node.col_offset)
            events.append((offset(node.end_lineno, node.end_col_offset), 1, node.name, "bind", at))
        elif isinstance(node, ast.alias) and hasattr(node, "lineno"):
            at, name = offset(node.lineno, node.col_offset), (node.asname or node.name).split(".")[0]
            events.append((offset(node.end_lineno, node.end_col_offset), 1, name, "bind", at))
        elif isinstance(node, ast.ExceptHandler) and node.name:
            at = offset(node.lineno, node.col_offset)
            events.append((offset(node.body[0].lineno, node.body[0].col_offset), 1, node.name, "bind", at))
    events.sort(key=lambda e: (e[0], e[1]))
    mentioned: dict[str, int] = {}  # local -> order of its latest mention
    bound_at: dict[str, int] = {}  # local -> text offset of its latest binding
    loads: Counter = Counter()
    out = []
    for order, (at, _, name, kind, text_at) in enumerate(events):
        if name not in local:
            continue
        if kind == "load" and name in bound_at and len(mentioned) >= 2:
            candidates = sorted(mentioned)
            heuristics = {
                "recent": max(candidates, key=lambda c: mentioned[c]),
                "frequent": max(candidates, key=lambda c: (loads[c], mentioned[c])),
            }
            out.append((at, name, candidates, heuristics, bound_at[name]))
        if kind == "bind":
            bound_at[name] = text_at
        if kind == "load" and name not in bound_at:
            continue  # resolved by a later binding only: not a candidate yet
        mentioned[name] = order
        loads[name] += kind == "load"
    return out


def train_function_hashes(corpus: Corpus) -> set[str]:
    """ast_hash of every function in the train split's code documents."""
    return {
        digest
        for doc in documents(corpus)
        if doc.source.kind == "code" and split_of(corpus, doc) == "train"
        for _, _, digest in function_hashes(doc.text)
    }


def build_natural(corpus: Corpus, out: Path, *, limit: int = 4000, seed: int = 0) -> dict[str, Any]:
    """T3 dev and test halves from the val split's code documents (README: 绑定探针)."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    exclude = train_function_hashes(corpus)
    halves: dict[str, list[Item]] = {half: [] for half in HALVES}
    for doc in documents(corpus):
        if doc.source.kind != "code" or split_of(corpus, doc) != "val":
            continue
        rng = random.Random(f"{seed}\0{doc.key}")
        half = HALVES[unit_hash(seed, "half\0" + doc.group) % 2]
        halves[half] += natural_items(doc.text, doc.key, rng=rng, exclude=exclude)
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for half, items in halves.items():
        items = random.Random(f"{seed}\0{half}").sample(items, min(limit, len(items)))
        write_items(out / SUITE_FILES[half], items)
        files[half] = {"file": SUITE_FILES[half], "items": len(items), "sha256": file_sha256(out / SUITE_FILES[half])}
    manifest = {
        "tlc_suite_version": SUITE_VERSION,
        "tasks": ["T3"],
        "corpus": {"spec_sha256": file_sha256(corpus.path), "split": "val", "seed": corpus.seed},
        "seed": seed,
        "rules": [
            "documents: code sources of the corpus's val split; dev or test by sha256(seed, group)",
            "functions of the static subset (no global/nonlocal, del of a name, eval/exec/locals/getattr...)",
            "functions whose alpha-renamed AST equals a train-split function's are left out",
            f"at most {T3_PER_DOCUMENT} loads per document: loads of a local bound earlier in the function's text, "
            "with at least two candidates (the locals mentioned so far)",
        ],
        "files": files,
    }
    (out / MANIFEST_FILE).write_text(dump_json(manifest), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tlc.data.suites", description="write the T3 suite")
    parser.add_argument("corpus", type=Path, help="corpus spec (TOML)")
    parser.add_argument("out", type=Path, help="output directory (absent or empty)")
    parser.add_argument("--limit", type=int, default=4000, help="items per half")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    manifest = build_natural(load_corpus(args.corpus), args.out, limit=args.limit, seed=args.seed)
    print(json.dumps(manifest["files"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
