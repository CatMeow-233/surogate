"""T3, the natural-code binding probe: the next local identifier in held-out Python (docs/training.md: 绑定探针).

The prompt is a val-split code document up to a load of a function's local, the candidates are the locals mentioned
before it in the function, and the gold is the loaded name. Only functions whose locals a textual reading resolves
are used (the static subset: no global/nonlocal, no del of a name, no reflection on the namespace), and no function
whose alpha-renamed AST also occurs in the train split. Sources marked `generated` in the corpus spec (the synthetic
binding programs of data.synth) give no items: T3 measures natural code, and the synthetic suites have their own
T1/T2. The heuristic floors are `recent` (the most recently mentioned local) and `frequent` (the most loaded so far).
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import symtable
import sys
import warnings
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from ..files import MANIFEST_FILE, dump_json, file_sha256, write_text_atomic
from .corpus import Corpus, documents, load_corpus, split_of, unit_hash
from .decontam import function_hashes
from .items import HALVES, SUITE_FILES, SUITE_VERSION, Item, write_items

T3_MAX_PROMPT_CHARS = 16000  # the prompt keeps the document's tail up to the load, from a line start
T3_PER_DOCUMENT = 8
CANDIDATE_BUCKETS = ((2, "2"), (4, "3-4"), (8, "5-8"), (16, "9-16"), (math.inf, ">16"))
REFLECTION = {"eval", "exec", "locals", "globals", "vars", "getattr", "setattr", "delattr"}


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
    """T3 dev and test halves from the val split's natural code documents (sources not marked `generated`)."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    exclude = train_function_hashes(corpus)
    halves: dict[str, list[Item]] = {half: [] for half in HALVES}
    for doc in documents(corpus):
        if doc.source.kind != "code" or doc.source.generated or split_of(corpus, doc) != "val":
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
        "corpus": {
            "spec_sha256": file_sha256(corpus.path),
            "split": "val",
            "seed": corpus.seed,
            "sources": [s.name for s in corpus.sources if s.kind == "code" and not s.generated],
        },
        "seed": seed,
        "rules": [
            "documents: code sources not marked generated, val split; dev or test by sha256(seed, group)",
            "functions of the static subset (no global/nonlocal, del of a name, eval/exec/locals/getattr...)",
            "functions whose alpha-renamed AST equals a train-split function's are left out",
            f"at most {T3_PER_DOCUMENT} loads per document: loads of a local bound earlier in the function's text, "
            "with at least two candidates (the locals mentioned so far)",
        ],
        "files": files,
    }
    write_text_atomic(out / MANIFEST_FILE, dump_json(manifest))
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tlc.data.natural", description=__doc__.split("\n")[0])
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
