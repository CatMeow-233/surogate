"""Decontamination of the corpus against HumanEval, MBPP and EvalPlus (docs/data.md: 去污染).

Every document is scanned with four candidate rules. Only high-precision evidence removes a document; the rest is
flagged and goes to the sampled manual audit, because a false removal costs training data silently while a flag
costs one look.

- `code13`: coverage of a benchmark solution's informative 13-grams of the identifier-abstracted Python token
  stream (identifiers become ID, string and number literals STR and NUM, comments and docstrings are dropped). A
  13-gram is informative when at least four of its tokens are words (keywords, builtins, ID), so lists of literals
  do not match everything. Coverage = the item's 13-grams found in the document / all of the item's 13-grams;
  items with fewer than MIN_CODE_GRAMS are not indexed.
- `text8`: coverage of a benchmark description's lowercased word 8-grams (MBPP's text, HumanEval's docstring) by
  the document's words, for items with at least MIN_TEXT_GRAMS.
- `ast`: a function of the document whose alpha-renamed, literal-normalised AST equals a benchmark function's
  (ast_hash: every bound name, parameter, except-as and import-as name, and the keyword arguments of calls to
  functions defined in the same tree, is renamed in visiting order; functions of fewer than MIN_AST_NODES nodes are
  ignored, they match everything).
- `name`: a function named like a benchmark's distinctive entry point.

A text document is scanned as text, and its markdown code blocks (data.corpus.fences, as the shards frame them) as
code. The report (scan_corpus) lists every removed and flagged document with the sha256 of its text and the hits; a
corpus spec that names the report under `decontam` drops the removed documents (surogate_tlc.data.corpus).
calibrate() plants alpha-renamed benchmark solutions into corpus documents and measures recall and the
false-positive rate (gate G6).
"""

from __future__ import annotations

import argparse
import ast
import builtins
import gzip
import hashlib
import io
import json
import keyword
import multiprocessing
import random
import re
import sys
import tokenize
import warnings
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..files import dump_json, file_sha256, text_sha256, write_text_atomic
from .corpus import DECONTAM_VERSION, Corpus, documents, fences, load_corpus

CODE_N, TEXT_N = 13, 8
MIN_CODE_GRAMS, MIN_TEXT_GRAMS, MIN_AST_NODES = 15, 5, 40
MIN_WORDS_PER_GRAM = 4
FLAG_COVERAGE = 0.5
# Removal thresholds: high precision on the calibration recorded in docs/measurements.md.
REMOVE = {"code13": 0.8, "text8": 0.8}
GENERIC_NAMES = {"main", "solve", "solution", "func", "test", "check", "helper", "compute", "process", "run"}
AUDIT_SAMPLE = 20
BUILTINS = frozenset(dir(builtins))
WORD = re.compile(r"[a-z0-9]+")
IDENTIFIER = re.compile(r"(?<![.\w])[A-Za-z_]\w*\b")  # not an attribute name


@dataclass(frozen=True)
class BenchmarkItem:
    id: str  # "<file stem>/<task_id>"
    code: str
    text: str
    name: str | None


def _records(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        if path.name.endswith(".json"):
            data = json.load(f)
            return data if isinstance(data, list) else list(data.values())
        return [json.loads(line) for line in f if line.strip()]


def _docstrings(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ""
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)]
    return "\n".join(doc for n in nodes if (doc := ast.get_docstring(n)))


def _item(stem: str, record: dict[str, Any]) -> BenchmarkItem:
    """HumanEval(+) records hold a signature-and-docstring prompt plus a body; MBPP and MBPP+ hold whole code and a
    description."""
    task = record.get("task_id")
    if task is None:
        raise ValueError(f"{stem}: a record without task_id")
    prompt = record.get("prompt") or ""
    if "canonical_solution" in record and re.search(r"^\s*(?:async\s+)?def\s+\w+\s*\(", prompt, re.M):
        code, text = prompt + record["canonical_solution"], _docstrings(prompt) or prompt
    elif "canonical_solution" in record or "code" in record:
        code = record.get("code") or record["canonical_solution"]
        text = record.get("text") or prompt
    else:
        raise ValueError(f"{stem}/{task}: neither canonical_solution nor code")
    name = record.get("entry_point") or next(iter(re.findall(r"^\s*def\s+(\w+)", code, re.M)), None)
    return BenchmarkItem(f"{stem}/{task}", code, text, name)


def load_benchmarks(directory: Path) -> tuple[list[BenchmarkItem], list[dict[str, Any]]]:
    """Items of every *.jsonl, *.jsonl.gz and *.json file in `directory`, and per file its name, sha256 and size."""
    paths = sorted(p for p in Path(directory).iterdir() if p.name.endswith((".jsonl", ".jsonl.gz", ".json")))
    if not paths:
        raise ValueError(f"{directory}: no benchmark files (*.jsonl, *.jsonl.gz, *.json)")
    items, files = [], []
    for path in paths:
        stem = path.name.removesuffix(".gz").removesuffix(".jsonl").removesuffix(".json")
        mine = [_item(stem, record) for record in _records(path)]
        items += mine
        files.append({"file": path.name, "sha256": file_sha256(path), "items": len(mine)})
    return items, files


def id_tokens(src: str) -> list[str]:
    """The identifier-abstracted token stream; tokens up to a tokenizer error are kept."""
    out: list[str] = []
    line_start = True
    try:
        for t in tokenize.generate_tokens(io.StringIO(src).readline):
            if t.type in (tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                line_start = True
                continue
            if t.type in (tokenize.NL, tokenize.COMMENT, tokenize.ENDMARKER):
                continue
            first, line_start = line_start, False
            if t.type == tokenize.STRING:
                if not first:  # a string that starts a logical line is a docstring or a bare expression
                    out.append("STR")
            elif t.type == tokenize.NUMBER:
                out.append("NUM")
            elif t.type == tokenize.NAME and not keyword.iskeyword(t.string) and t.string not in BUILTINS:
                out.append("ID")
            else:
                out.append(t.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def _informative(gram: Sequence[str]) -> bool:
    return sum(t not in ("STR", "NUM") and (t[:1].isalpha() or t[:1] == "_") for t in gram) >= MIN_WORDS_PER_GRAM


def code_grams(src: str) -> set[tuple[str, ...]]:
    toks = id_tokens(src)
    grams = (tuple(toks[i : i + CODE_N]) for i in range(len(toks) - CODE_N + 1))
    return {g for g in grams if _informative(g)}


def text_grams(text: str) -> set[tuple[str, ...]]:
    words = WORD.findall(text.lower())
    return {tuple(words[i : i + TEXT_N]) for i in range(len(words) - TEXT_N + 1)}


class _Alpha(ast.NodeTransformer):
    """Bound names become v0, v1, ... in visiting order: names, parameters, except-as, import-as and global/nonlocal
    names, and the keyword arguments of calls to a function, named lambda or class (its __init__) defined in the tree,
    which name one of its parameters. Any other keyword (sorted's key=, print's end=) is an API name and stays, so
    renaming locals alone never changes the hash. Constants become their type name, functions F; docstrings,
    decorators, annotations and return annotations go, and an annotated assignment becomes a plain one. Attribute
    names and imported module names stay. Visit a Module (ast_hash parses a copy into one)."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}
        self.params: dict[str, set[str]] = {}  # callee name -> its parameters, filled by visit_Module

    def _name(self, name: str) -> str:
        return self.names.setdefault(name, f"v{len(self.names)}")

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=self._name(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        return ast.copy_location(ast.arg(arg=self._name(node.arg)), node)

    def visit_Module(self, node: ast.Module) -> ast.AST:
        for n in ast.walk(node):
            if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef):
                callees = [(n.name, n.args)]
            elif isinstance(n, ast.Assign) and isinstance(n.value, ast.Lambda):
                callees = [(t.id, n.value.args) for t in n.targets if isinstance(t, ast.Name)]
            elif isinstance(n, ast.ClassDef):
                init = (f for f in n.body if isinstance(f, ast.FunctionDef | ast.AsyncFunctionDef))
                callees = [(n.name, f.args) for f in init if f.name == "__init__"]
            else:
                continue
            for name, args in callees:
                self.params.setdefault(name, set()).update(
                    a.arg for a in args.posonlyargs + args.args + args.kwonlyargs
                )
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        params = self.params.get(node.func.id, set()) if isinstance(node.func, ast.Name) else set()
        func, args = self.visit(node.func), [self.visit(arg) for arg in node.args]
        keywords = []
        for k in node.keywords:
            value = self.visit(k.value)
            arg = self._name(k.arg) if k.arg in params else k.arg  # None (**kwargs) is in no parameter set
            keywords.append(ast.copy_location(ast.keyword(arg=arg, value=value), k))
        return ast.copy_location(ast.Call(func=func, args=args, keywords=keywords), node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        node.type = None if node.type is None else self.visit(node.type)
        node.name = None if node.name is None else self._name(node.name)
        node.body = [self.visit(stmt) for stmt in node.body]
        return node

    def visit_alias(self, node: ast.alias) -> ast.AST:
        return ast.copy_location(ast.alias(name=node.name, asname=node.asname and self._name(node.asname)), node)

    def _declared(self, node: ast.Global | ast.Nonlocal) -> ast.AST:
        node.names = [self._name(name) for name in node.names]
        return node

    visit_Global = visit_Nonlocal = _declared

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        target = self.visit(node.target)
        if node.value is None:
            return ast.copy_location(ast.Pass(), node)
        return ast.copy_location(ast.Assign(targets=[target], value=self.visit(node.value)), node)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        node.name, node.returns = "F", None
        node.decorator_list = []
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            node.body = body[1:] or [ast.Pass()]
        self.generic_visit(node)
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = _function


def ast_hash(node: ast.AST) -> str:
    """sha256 (16 hex digits) of the alpha-renamed, literal-normalised dump of a copy of `node`."""
    copy = ast.parse(ast.unparse(node))
    return hashlib.sha256(ast.dump(_Alpha().visit(copy)).encode()).hexdigest()[:16]


def function_hashes(src: str, min_nodes: int = MIN_AST_NODES) -> list[tuple[str, int, str]]:
    """(name, line, ast_hash) of every function in `src` with at least `min_nodes` nodes."""
    try:
        with warnings.catch_warnings():  # invalid escape sequences in old code
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and sum(1 for _ in ast.walk(node)) >= min_nodes:
            try:
                out.append((node.name, node.lineno, ast_hash(node)))
            except (RecursionError, ValueError):
                continue
    return out


def code_blocks(text: str) -> str:
    """The markdown code blocks of a text document, joined (data.corpus.fences, the shards' framing)."""
    return "\n".join(chunk for in_code, chunk in fences(text) if in_code)


class Index:
    """The benchmark side of the four rules."""

    def __init__(self, items: Sequence[BenchmarkItem]) -> None:
        self.ids = [item.id for item in items]
        self.code: dict[tuple[str, ...], list[int]] = defaultdict(list)
        self.text: dict[tuple[str, ...], list[int]] = defaultdict(list)
        self.code_n: dict[int, int] = {}
        self.text_n: dict[int, int] = {}
        self.ast: dict[str, list[int]] = defaultdict(list)
        self.names: dict[str, list[int]] = defaultdict(list)
        for i, item in enumerate(items):
            for index, sizes, grams, least in (
                (self.code, self.code_n, code_grams(item.code), MIN_CODE_GRAMS),
                (self.text, self.text_n, text_grams(item.text), MIN_TEXT_GRAMS),
            ):
                if len(grams) >= least:
                    sizes[i] = len(grams)
                    for gram in grams:
                        index[gram].append(i)
            for _, _, digest in function_hashes(item.code):
                self.ast[digest].append(i)
            if item.name and distinctive(item.name):
                self.names[item.name].append(i)

    def scan(self, text: str, kind: str) -> list[dict[str, Any]]:
        """The hits of one document, `kind` "code" or "text"."""
        code = text if kind == "code" else code_blocks(text)
        hits = []
        for rule, index, sizes, grams in (
            ("code13", self.code, self.code_n, code_grams(code) if code else set()),
            ("text8", self.text, self.text_n, text_grams(text)),
        ):
            found = Counter(i for gram in grams for i in index.get(gram, ()))
            for i, n in sorted(found.items()):
                if n / sizes[i] >= FLAG_COVERAGE:
                    hits.append({"rule": rule, "item": self.ids[i], "coverage": round(n / sizes[i], 3)})
        for name, line, digest in function_hashes(code) if code else ():
            hits += [{"rule": "ast", "item": self.ids[i], "function": name, "line": line} for i in self.ast[digest]]
        for name, line in _defs(code):
            hits += [{"rule": "name", "item": self.ids[i], "function": name, "line": line} for i in self.names[name]]
        return hits


def distinctive(name: str) -> bool:
    return len(name) >= 10 and "_" in name and name not in GENERIC_NAMES


def _defs(code: str) -> Iterator[tuple[str, int]]:
    for line_no, line in enumerate(code.splitlines(), 1):
        if match := re.match(r"\s*(?:async\s+)?def\s+(\w+)", line):
            yield match.group(1), line_no


def removing(hits: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The hits that remove a document: any ast hit, or a coverage at the rule's REMOVE threshold."""
    return [h for h in hits if h["rule"] == "ast" or h.get("coverage", 0) >= REMOVE.get(h["rule"], 2)]


_INDEX: Index | None = None


def _scan_one(doc: tuple[str, str, str, str]) -> tuple[str, str, str, list[dict[str, Any]]]:
    key, source, kind, text = doc
    assert _INDEX is not None
    return key, source, text_sha256(text), _INDEX.scan(text, kind)


def _set_index(index: Index) -> None:
    global _INDEX
    _INDEX = index


def scan(index: Index, docs: Iterable[tuple[str, str, str, str]], jobs: int = 1) -> Iterator[tuple]:
    """(key, source, text sha256, hits) per (key, source, kind, text); jobs > 1 forks worker processes."""
    _set_index(index)
    if jobs <= 1:
        yield from map(_scan_one, docs)
        return
    with multiprocessing.get_context("fork").Pool(jobs) as pool:
        yield from pool.imap(_scan_one, docs, chunksize=32)


def rename(code: str) -> str:
    """Every identifier that is not a keyword, builtin or attribute name, strings included, becomes zz_<reversed>."""
    return IDENTIFIER.sub(
        lambda m: m.group(0) if keyword.iskeyword(m.group(0)) or m.group(0) in BUILTINS else "zz_" + m.group(0)[::-1],
        code,
    )


def calibrate(index: Index, items: Sequence[BenchmarkItem], hosts: Sequence[str], *, plants: int = 40, seed: int = 0):
    """Recall on `plants` alpha-renamed benchmark solutions appended to random host code documents, and the share
    of the unchanged hosts that would be removed (false positives) or flagged."""
    rng = random.Random(seed)
    chosen = rng.sample([item for item in items if code_grams(item.code)], min(plants, len(items)))
    clean = [index.scan(text, "code") for text in hosts]
    by_rule: Counter = Counter()
    recalled = flagged = 0
    for item, host in zip(chosen, rng.choices(range(len(hosts)), k=len(chosen)), strict=True):
        mine = [h for h in index.scan(hosts[host] + "\n\n" + rename(item.code) + "\n", "code") if h["item"] == item.id]
        by_rule.update({h["rule"] for h in removing(mine)})
        recalled += bool(removing(mine))
        flagged += bool(mine)
    return {
        "plants": len(chosen),
        "hosts": len(hosts),
        "seed": seed,
        "recall": recalled / len(chosen),
        "recall_flagged": flagged / len(chosen),
        "recall_by_rule": dict(sorted(by_rule.items())),
        "false_positive_rate": sum(bool(removing(h)) for h in clean) / len(hosts),
        "flag_rate": sum(bool(h) for h in clean) / len(hosts),
    }


def scan_corpus(
    corpus: Corpus, benchmark_dir: Path, *, jobs: int = 1, plants: int = 0, hosts: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """The decontamination report of every document of `corpus` (docs/data.md: 去污染)."""
    items, files = load_benchmarks(benchmark_dir)
    index = Index(items)
    docs = ((doc.key, doc.source.name, doc.source.kind, doc.text) for doc in documents(corpus))
    removed, flagged, scanned = [], [], Counter()
    calibration = None
    for key, source, digest, hits in scan(index, docs, jobs):
        scanned[source] += 1
        if hits:
            entry = {"key": key, "source": source, "sha256": digest, "hits": hits}
            (removed if removing(hits) else flagged).append(entry)
    if plants:
        # Natural code only: generated programs (synth) are short and never match a benchmark, so as hosts they would
        # lower the recorded false-positive and flag rates and always parse for the ast rule.
        natural = (doc for doc in documents(corpus) if doc.source.kind == "code" and not doc.source.generated)
        code = [(doc.source.name, doc.text) for doc in natural]
        chosen = random.Random(seed).sample(code, min(hosts, len(code)))
        if chosen:
            calibration = calibrate(index, items, [text for _, text in chosen], plants=plants, seed=seed)
            calibration["host_sources"] = dict(sorted(Counter(name for name, _ in chosen).items()))
    audit = [e["key"] for e in removed + flagged]
    return {
        "tlc_decontam_version": DECONTAM_VERSION,
        "corpus_spec_sha256": file_sha256(corpus.path),
        "benchmarks": files,
        "rules": {
            "code13": f"informative identifier-abstracted {CODE_N}-gram coverage >= {FLAG_COVERAGE} flags, "
            f">= {REMOVE['code13']} removes (items with >= {MIN_CODE_GRAMS} grams)",
            "text8": f"word {TEXT_N}-gram coverage >= {FLAG_COVERAGE} flags, >= {REMOVE['text8']} removes "
            f"(items with >= {MIN_TEXT_GRAMS} grams)",
            "ast": f"equal alpha-renamed, literal-normalised function AST (>= {MIN_AST_NODES} nodes) removes",
            "name": "a function named like a distinctive benchmark entry point flags",
        },
        "documents": dict(sorted(scanned.items())),
        "removed": removed,
        "flagged": flagged,
        "audit_sample": sorted(random.Random(seed).sample(audit, min(AUDIT_SAMPLE, len(audit)))),
        "calibration": calibration,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tlc.data.decontam", description=__doc__.split("\n")[0])
    parser.add_argument("corpus", type=Path, help="corpus spec (TOML); its own decontam key is ignored")
    parser.add_argument("benchmarks", type=Path, help="directory of HumanEval/MBPP/EvalPlus *.jsonl[.gz] and *.json")
    parser.add_argument("out", type=Path, help="report to write (JSON); name it under `decontam` in the spec")
    parser.add_argument("--jobs", type=int, default=1, help="worker processes")
    parser.add_argument("--plant", type=int, default=0, help="planted solutions for the calibration (G6: 40)")
    parser.add_argument("--hosts", type=int, default=2000, help="code documents the calibration plants into")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")
    report = scan_corpus(
        load_corpus(args.corpus, decontam=False),
        args.benchmarks,
        jobs=args.jobs,
        plants=args.plant,
        hosts=args.hosts,
        seed=args.seed,
    )
    write_text_atomic(args.out, dump_json(report))
    summary = {k: report[k] for k in ("documents", "benchmarks", "calibration")}
    print(json.dumps({**summary, "removed": len(report["removed"]), "flagged": len(report["flagged"])}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
