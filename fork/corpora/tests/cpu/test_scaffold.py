"""Package boundaries and repository wiring of fork/corpora (implementation step S0).

Importing `surogate_corpora`, listing its commands and importing any of its modules load no pyarrow, torch or
`surogate*` module; its source imports only the stdlib and pyarrow, pyarrow only inside functions, and never a project
or the study package; the package ships in no wheel of the fork; and fork/dev, CI, the sync checks, the environment
table and the ignore rules know the suite.
"""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

CORPORA = Path(__file__).resolve().parents[2]
REPO = CORPORA.parents[1]
SRC = CORPORA / "src"
PACKAGE = SRC / "surogate_corpora"

PLANNED_COMMANDS = ["sources", "plan", "fetch", "verify", "verify-samples", "compose"]
FORBIDDEN_LOADED = ("pyarrow", "torch", "numpy", "surogate", "surogate_tlc", "surogate_bilingual", "surogate_tokstudy")


def _pyproject(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


# --- package import ------------------------------------------------------------------------------------------------

IMPORT_CHECK = """
import importlib, pkgutil, sys
sys.path.insert(0, sys.argv[1])
import surogate_corpora
from surogate_corpora import __main__ as cli
cli.usage()
modules = sorted(m.name for m in pkgutil.walk_packages(surogate_corpora.__path__, "surogate_corpora."))
for name in modules:
    importlib.import_module(name)
bad = tuple(sys.argv[2].split(","))
loaded = sorted(m for m in sys.modules if m.split(".")[0] in bad)
print(surogate_corpora.__file__)
sys.exit(f"importing surogate_corpora and {modules} loaded {loaded}" if loaded else 0)
"""


def test_imports_load_neither_pyarrow_nor_torch_nor_any_project(tmp_path):
    """A fresh isolated interpreter (no PYTHONPATH, no user site, cwd off sys.path) whose site-packages may hold
    pyarrow and torch: importing the package, listing its commands and importing every module loads none of them."""
    command = [sys.executable, "-I", "-B", "-c", IMPORT_CHECK, str(SRC), ",".join(FORBIDDEN_LOADED)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(result.stdout.strip()).resolve() == PACKAGE / "__init__.py"


def test_tests_import_the_checkout():
    import surogate_corpora

    assert Path(surogate_corpora.__file__).resolve().parent == PACKAGE
    assert sys.path[0] == str(SRC)


# --- packaging -----------------------------------------------------------------------------------------------------


def test_pyproject_declares_an_unpublished_src_layout_package():
    config = _pyproject(CORPORA / "pyproject.toml")
    project = config["project"]
    assert project["name"] == "surogate-corpora"
    assert project["dependencies"] == []
    lower = re.fullmatch(r">=\s*3\.(\d+)", project["requires-python"])
    assert lower and int(lower.group(1)) >= 12, project["requires-python"]
    find = config["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["src"] and find["include"] == ["surogate_corpora*"]
    packages = {p.parent.relative_to(SRC).as_posix() for p in SRC.rglob("__init__.py")}
    assert packages == {"surogate_corpora"}, packages
    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests/cpu"]


def test_requirements_pin_pyarrow():
    lines = [
        line for line in (CORPORA / "requirements.txt").read_text("utf-8").splitlines() if not line.startswith("#")
    ]
    pins = dict(line.split("==") for line in lines if line.strip())
    assert set(pins) == {"pyarrow", "pytest", "pytest-timeout"} and all(
        re.fullmatch(r"[\d.]+", v) for v in pins.values()
    )


def _shipped_roots() -> list[Path]:
    """The directories the native and the experimental wheel take their packages from, read from their configs."""
    native = _pyproject(REPO / "pyproject.toml")["tool"]["scikit-build"]["wheel"]["packages"]
    find = _pyproject(REPO / "fork" / "experimental" / "pyproject.toml")["tool"]["setuptools"]["packages"]["find"]
    return [REPO / package for package in native] + [REPO / "fork" / "experimental" / where for where in find["where"]]


def test_corpora_tree_is_outside_every_shipped_wheel_root():
    roots = _shipped_roots()
    assert roots and all(root.is_dir() for root in roots), roots
    assert not [root for root in roots if CORPORA.is_relative_to(root) or root.is_relative_to(CORPORA)]


# --- import boundary of src/ ---------------------------------------------------------------------------------------

LAZY = {"pyarrow"}  # imported only inside functions, so that importing a module never loads it
PATH_LOADERS = {"spec_from_file_location", "SourceFileLoader", "run_path"}


def boundary_problems(source: str, name: str) -> list[str]:
    """Imports in one source file that cross the package's import boundary: the stdlib, pyarrow (inside functions
    only) and the package itself; nothing that loads code by path."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def in_function(node: ast.AST) -> bool:
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                return True
        return False

    problems = []
    for node in ast.walk(tree):
        modules = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
            first = node.args[0] if node.args else None
            if called in ("import_module", "__import__") and isinstance(first, ast.Constant):
                if isinstance(first.value, str) and not first.value.startswith("."):
                    modules = [first.value]
            if called in PATH_LOADERS:
                problems.append(f"{name}:{node.lineno}: loads code by path ({called})")
        for module in modules:
            top = module.split(".")[0]
            if top in LAZY:
                if not in_function(node):
                    problems.append(f"{name}:{node.lineno}: imports {module} at module level (import it in functions)")
            elif top != "surogate_corpora" and top not in sys.stdlib_module_names:
                problems.append(f"{name}:{node.lineno}: imports {module}, which is outside the package's boundary")
    return problems


def test_src_imports_stay_inside_the_boundary():
    found = []
    for path in sorted(SRC.rglob("*.py")):
        found += boundary_problems(path.read_text(encoding="utf-8"), str(path.relative_to(CORPORA)))
    assert not found, "\n".join(found)


def test_the_boundary_check_catches_each_kind_of_crossing():
    planted = """
import pyarrow
import torch
import surogate
from surogate_tlc.data import corpus
import surogate_bilingual.data.manifest
import importlib
importlib.import_module("surogate_tokstudy")
spec = importlib.util.spec_from_file_location("m", "x.py")
import numpy as np
def f():
    import pyarrow.parquet as pq
"""
    problems = boundary_problems(planted, "planted.py")
    assert sorted(int(p.split(":")[1]) for p in problems) == [2, 3, 4, 5, 6, 8, 9, 10], problems
    allowed = """
from __future__ import annotations
import json, os.path, urllib.request
from . import canonical
from surogate_corpora import sources
def read():
    import pyarrow.parquet as pq
    from pyarrow import compute
"""
    assert boundary_problems(allowed, "allowed.py") == []


# --- command table -------------------------------------------------------------------------------------------------


def test_command_table_lists_the_planned_commands():
    from surogate_corpora.__main__ import COMMANDS

    assert list(COMMANDS) == PLANNED_COMMANDS
    for name, command in COMMANDS.items():
        assert re.fullmatch(r"[a-z_]+(\.[a-z_]+)*", command.module), name
        assert command.step in {"S1", "S2", "S3", "S4", "S5"}, name


def test_implemented_commands_resolve_to_a_function():
    import importlib

    from surogate_corpora.__main__ import COMMANDS, is_implemented, module_name

    for name in filter(is_implemented, COMMANDS):
        assert callable(getattr(importlib.import_module(module_name(name)), COMMANDS[name].function)), name


def test_command_line_lists_refuses_and_names_the_step(capsys, monkeypatch):
    from surogate_corpora.__main__ import COMMANDS, Command, main

    assert main([]) == 0 and main(["--help"]) == 0
    listing = capsys.readouterr().out
    assert all(re.search(rf"^  {re.escape(name)} ", listing, re.MULTILINE) for name in PLANNED_COMMANDS), listing
    assert main(["no-such-command"]) == 2
    assert "unknown command 'no-such-command'" in capsys.readouterr().err
    monkeypatch.setitem(COMMANDS, "future", Command("no_such_module", "a later command", "S9"))
    assert main(["future"]) == 2
    assert "'future' is not implemented yet; it lands in step S9" in capsys.readouterr().err


# --- repository wiring ---------------------------------------------------------------------------------------------


def _run(*command: str) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=120)


def test_fork_dev_help_lists_the_suite():
    result = _run("fork/dev", "help")
    assert result.returncode == 0, result.stderr
    assert re.search(r"^\s+test corpora \[pytest args\]", result.stdout, re.MULTILINE), result.stdout


def test_an_unknown_suite_exits_with_a_usage_that_names_the_suite():
    result = _run("fork/dev", "test", "no-such-suite")
    assert result.returncode == 2
    usage = re.search(r"usage: fork/dev test (\S+)", result.stderr)
    assert usage and "corpora" in usage.group(1).split("|"), result.stderr


@pytest.mark.parametrize(
    "path",
    [
        "fork/.venv-corpora",  # before the environment exists, too
        "fork/.venv-corpora/bin/python",
        "fork/corpora/src/surogate_corpora.egg-info/PKG-INFO",
        "fork/corpora/build/lib/surogate_corpora/__init__.py",
        "fork/corpora/dist/surogate_corpora-0.1.0.dev0-py3-none-any.whl",
    ],
)
def test_environment_and_build_outputs_are_ignored(path):
    assert _run("git", "check-ignore", "-q", "--no-index", path).returncode == 0, path


def test_environment_table_lists_the_corpora_venv():
    """AGENTS.md rule 4 points to this table for the environments; the corpora one is listed there."""
    readme = (REPO / "fork" / "README.md").read_text(encoding="utf-8")
    section = readme.split("\n## 首次准备环境\n", 1)[1].split("\n## ", 1)[0]
    assert re.search(r"^\| `fork/\.venv-corpora` \|", section, re.MULTILINE), section


def test_ci_and_the_sync_checks_run_the_suite():
    workflow = (REPO / ".github" / "workflows" / "fork-ci.yml").read_text(encoding="utf-8")
    assert "run: FORK_PY=python fork/dev test corpora\n" in workflow
    source = (REPO / "fork" / "tools" / "sync_upstream.py").read_text(encoding="utf-8")
    checks = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "checks")
    assert '"corpora"' in ast.get_source_segment(source, checks)
