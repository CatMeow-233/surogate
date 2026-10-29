"""The heuristic floors of the T1/T2 probes: what a scope-blind reader of a program's text answers.

A probe result only means something above what text-level shortcuts already get right, so every T1/T2 item carries
the predictions of a reader that sees the program as one flat namespace (literal_heuristics):

- `recent`: the last literal of the prompt; `line1`: the first one;
- `same_name`: the readout expression resolved through the latest textual binding of each name, ignoring scopes; a
  call's keyword arguments bind its parameters' names;
- `first_bound`: the same through the earliest bindings;
- `stale`: the value the reader gave the expression before its last change;
- `position`: a call's returned parameter taken by argument position, keywords included in written order.

Where one of them is right on every item of a grid cell it is the cell's algorithm (same_name where nothing
shadows), and eval.probes reports it as the floor rather than as a shortcut.
"""

from __future__ import annotations

import ast
import dataclasses
import io
import tokenize
from collections.abc import Sequence


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
