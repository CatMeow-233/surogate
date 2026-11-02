"""Package boundaries and repository wiring of fork/bilingual (docs/spec.md, step I1).

Importing `surogate_bilingual` loads neither torch nor any `surogate` module; the package ships in no wheel of the
fork; its source imports only the stdlib, torch, numpy, tokenizers, safetensors and (guarded) scipy, and never the
native package or the owner's v1.0 files in reference/; those files still match the sha256 pinned in
reference/MANIFEST.json; and fork/dev, CI, the sync checks and the ignore rules know the suite.
"""

import ast
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

BILINGUAL = Path(__file__).resolve().parents[2]
REPO = BILINGUAL.parents[1]
SRC = BILINGUAL / "src"
REFERENCE = BILINGUAL / "reference"

# The owner's v1.0 files kept in reference/ (docs/spec.md, "参照代码"). test_compare.py stays out: a fork test_*.py
# outside the test trees of fork/tools/gen_maps.py fails `fork/dev check`; its assertions are ported in step I3.
REFERENCE_FILES = {
    "compare.py",
    "eval_cases.example.jsonl",
    "evaluate.py",
    "moe_pilot.py",
    "smoke_results.json",
    "verify_merge_math.py",
}

# The command-line surface of docs/spec.md, in its order.
PLANNED_COMMANDS = [
    "env",
    "tokenizer",
    "manifest",
    "split",
    "decontam",
    "shard",
    "synth",
    "train",
    "assemble",
    "verify",
    "eval-tasks",
    "eval-nll",
    "switch",
    "audit",
    "stats",
    "bench",
    "matrix",
    "registry",
]


def _pyproject(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


# --- package import ------------------------------------------------------------------------------------------------

IMPORT_CHECK = """
import sys
sys.path.insert(0, sys.argv[1])
import surogate_bilingual
from surogate_bilingual import __main__ as cli
cli.usage()
loaded = sorted(m for m in sys.modules if m in ("torch", "surogate") or m.startswith(("torch.", "surogate.")))
print(surogate_bilingual.__file__)
sys.exit(f"importing surogate_bilingual loaded {loaded}" if loaded else 0)
"""


def test_package_import_loads_neither_torch_nor_surogate(tmp_path):
    """A fresh isolated interpreter (no PYTHONPATH, no user site, cwd off sys.path) whose site-packages may hold
    torch: importing the package and listing its commands imports no torch and no `surogate` module."""
    command = [sys.executable, "-I", "-B", "-c", IMPORT_CHECK, str(SRC)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(result.stdout.strip()).resolve() == SRC / "surogate_bilingual" / "__init__.py"


def test_tests_import_the_checkout():
    import surogate_bilingual

    assert Path(surogate_bilingual.__file__).resolve().parent == SRC / "surogate_bilingual"
    assert sys.path[0] == str(SRC)


# --- packaging -----------------------------------------------------------------------------------------------------


def test_pyproject_declares_an_unpublished_src_layout_package():
    config = _pyproject(BILINGUAL / "pyproject.toml")
    project = config["project"]
    assert project["name"] == "surogate-bilingual"
    assert project["dependencies"] == []
    lower = re.fullmatch(r">=\s*3\.(\d+)", project["requires-python"])
    assert lower and int(lower.group(1)) >= 12, project["requires-python"]
    find = config["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["src"] and find["include"] == ["surogate_bilingual*"]
    packages = {p.parent.relative_to(SRC).as_posix() for p in SRC.rglob("__init__.py")}
    assert packages and all(p.split("/")[0] == "surogate_bilingual" for p in packages), packages
    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests/cpu"]


def test_ruff_keeps_the_repository_settings_and_leaves_reference_alone():
    """`fork/dev fmt` runs ruff over fork/; without the exclusion it would rewrite the verbatim reference files."""
    ruff = _pyproject(BILINGUAL / "pyproject.toml")["tool"]["ruff"]
    assert (BILINGUAL / ruff["extend"]).resolve() == REPO / "pyproject.toml"
    assert ruff["extend-exclude"] == ["reference"]


def _shipped_roots() -> list[Path]:
    """The directories the native and the experimental wheel take their packages from, read from their configs."""
    native = _pyproject(REPO / "pyproject.toml")["tool"]["scikit-build"]["wheel"]["packages"]
    find = _pyproject(REPO / "fork" / "experimental" / "pyproject.toml")["tool"]["setuptools"]["packages"]["find"]
    return [REPO / package for package in native] + [REPO / "fork" / "experimental" / where for where in find["where"]]


def test_bilingual_tree_is_outside_every_shipped_wheel_root():
    roots = _shipped_roots()
    assert roots and all(root.is_dir() for root in roots), roots
    overlapping = [root for root in roots if BILINGUAL.is_relative_to(root) or root.is_relative_to(BILINGUAL)]
    assert not overlapping, overlapping


# --- import boundary of src/ ---------------------------------------------------------------------------------------

DEPENDENCIES = {"torch", "numpy", "tokenizers", "safetensors"}
OPTIONAL = {"scipy"}  # imported only inside `try: ... except ImportError`
PATH_LOADERS = {"spec_from_file_location", "SourceFileLoader", "run_path"}  # how code would reach reference/


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(n, ast.Name) and n.id in ("ImportError", "ModuleNotFoundError") for n in names)


def boundary_problems(source: str, name: str) -> list[str]:
    """Imports in one source file that cross the package's import boundary (docs/spec.md, "导入边界")."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def guarded(node: ast.AST) -> bool:
        while node in parents:
            parent = parents[node]
            if isinstance(parent, ast.Try) and node in parent.body and any(map(_catches_import_error, parent.handlers)):
                return True
            node = parent
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
            if top == "surogate":
                problems.append(
                    f"{name}:{node.lineno}: imports {module} (the native package or fork code shipped in a wheel)"
                )
            elif top == "reference":
                problems.append(f"{name}:{node.lineno}: imports {module} (the owner's v1.0 files)")
            elif top in OPTIONAL:
                if not guarded(node):
                    problems.append(f"{name}:{node.lineno}: imports {module} outside try/except ImportError")
            elif top not in DEPENDENCIES | {"surogate_bilingual"} and top not in sys.stdlib_module_names:
                problems.append(f"{name}:{node.lineno}: imports {module}, which is not a dependency of the package")
    return problems


def test_src_imports_stay_inside_the_boundary():
    found = []
    for path in sorted(SRC.rglob("*.py")):
        found += boundary_problems(path.read_text(encoding="utf-8"), str(path.relative_to(BILINGUAL)))
    assert not found, "\n".join(found)


def test_the_boundary_check_catches_each_kind_of_crossing():
    """The scan above passes trivially while src/ is small; here it must flag every planted crossing."""
    planted = """
import surogate
import surogate.experimental.train
from surogate._fork import grpo
import reference.moe_pilot
import importlib
importlib.import_module("surogate.dsl")
__import__("reference")
import scipy.optimize
import transformers
spec = importlib.util.spec_from_file_location("m", "fork/bilingual/reference/moe_pilot.py")
"""
    problems = boundary_problems(planted, "planted.py")
    assert sorted(int(p.split(":")[1]) for p in problems) == [2, 3, 4, 5, 7, 8, 9, 10, 11], problems
    allowed = """
from __future__ import annotations
import json, os.path
import numpy as np
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from . import config
from surogate_bilingual.ops import norm
import importlib
importlib.import_module(".data", "surogate_bilingual")
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None
try:
    import scipy
except (ModuleNotFoundError, OSError):
    scipy = None
"""
    assert boundary_problems(allowed, "allowed.py") == []


# --- reference/ ----------------------------------------------------------------------------------------------------


# --- intra-package layering (CR1 review; docs/spec.md, 模块边界) --------------------------------------------------
#
# Groups of modules (longest prefix wins) and the groups each may import from, top-level and function-level imports
# alike. The direction is one way: evaluation never imports training or the audit reports, data never imports a
# model, the counting audit (params, flops, memory) is below training, and the audit reports (budget, bench,
# resources) sit on top of training and import nothing from eval.

LAYER_GROUPS = {
    "canonical": "core", "config": "core", "devices": "core",
    "ops": "ops", "routing": "routing", "model": "model", "data": "data", "registry": "registry",
    "assembly": "assembly", "audit.params": "counting", "audit.flops": "counting", "audit.memory": "counting",
    "train": "train", "eval": "eval", "audit.budget": "reports", "audit.bench": "reports", "audit.resources": "reports",
}  # fmt: skip
LAYER_ALLOWED = {
    "core": {"core"},
    "ops": {"core", "ops"},
    "routing": {"core", "routing"},
    "model": {"core", "ops", "routing", "model"},
    "data": {"core", "data"},
    "registry": {"core", "registry"},
    "assembly": {"core", "model", "routing", "data", "registry", "assembly"},
    "counting": {"core", "ops", "routing", "counting"},
    "train": {"core", "model", "routing", "data", "registry", "assembly", "counting", "train"},
    "eval": {"core", "ops", "model", "routing", "data", "registry", "assembly", "eval"},
    "reports": {"core", "ops", "model", "routing", "registry", "counting", "train", "reports"},
}


def layer_of(module: str) -> str | None:
    """The group of a module path relative to the package (e.g. "audit.budget"); None for the package root."""
    parts = module.split(".")
    for n in range(len(parts), 0, -1):
        if ".".join(parts[:n]) in LAYER_GROUPS:
            return LAYER_GROUPS[".".join(parts[:n])]
    return None


def layering_problems(source: str, module: str) -> list[str]:
    """Relative imports of `module` (package-relative dotted path) that cross the layer table."""
    tree = ast.parse(source)
    package = module.split(".")[:-1]  # a module's package; for "x.__init__" that is x itself
    here = layer_of(module)
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        base = package[: len(package) - node.level + 1]
        targets = [".".join([*base, node.module])] if node.module else [".".join([*base, a.name]) for a in node.names]
        for target in targets:
            there = layer_of(target)
            if here is None or there is None:
                problems.append(f"{module}: import of {target or '(package root)'} outside the layer table")
            elif there not in LAYER_ALLOWED[here]:
                problems.append(f"{module} ({here}) imports {target} ({there})")
    return problems


def test_src_modules_respect_the_layer_table():
    problems = []
    for path in sorted((SRC / "surogate_bilingual").rglob("*.py")):
        rel = path.relative_to(SRC / "surogate_bilingual").with_suffix("")
        if rel.parts in (("__init__",), ("__main__",)):
            continue
        module = ".".join(rel.parts)
        problems += layering_problems(path.read_text("utf-8"), module)
    assert not problems, problems
    assert set(LAYER_GROUPS.values()) == set(LAYER_ALLOWED)


def test_the_layer_check_catches_a_crossing_at_any_depth():
    lazy = "def main():\n    from ..eval.report import results_markdown\n"
    assert layering_problems(lazy, "audit.budget") == ["audit.budget (reports) imports eval.report (eval)"]
    assert layering_problems("from ..train import loop\n", "eval.score") == ["eval.score (eval) imports train (train)"]
    assert layering_problems("from ..model.lm import RoutedLM\n", "data.shards") == [
        "data.shards (data) imports model.lm (model)"
    ]
    assert not layering_problems("from ..train.recipes import ExpertRecipe\n", "audit.budget")


def test_reference_files_match_their_manifest():
    manifest = json.loads((REFERENCE / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "bilingual-reference/1"
    files = manifest["files"]
    assert set(files) == REFERENCE_FILES
    present = {p.name for p in REFERENCE.iterdir() if p.is_file()} - {"MANIFEST.json", "README.md"}
    assert present == REFERENCE_FILES, present ^ REFERENCE_FILES
    for name, entry in files.items():
        data = (REFERENCE / name).read_bytes()
        assert (hashlib.sha256(data).hexdigest(), len(data)) == (entry["sha256"], entry["bytes"]), name


def test_reference_holds_no_test_files():
    assert not [p for p in REFERENCE.rglob("test_*.py")]


# --- command table -------------------------------------------------------------------------------------------------


def test_command_table_lists_the_planned_commands():
    from surogate_bilingual.__main__ import COMMANDS

    assert list(COMMANDS) == PLANNED_COMMANDS
    for name, command in COMMANDS.items():
        assert re.fullmatch(r"[a-z_]+(\.[a-z_]+)*", command.module), name
        assert command.step in {"I2", "I3", "I4", "I5", "I6"}, name


def test_implemented_commands_resolve_to_a_function():
    import importlib

    from surogate_bilingual.__main__ import COMMANDS, is_implemented, module_name

    for name in filter(is_implemented, COMMANDS):
        assert callable(getattr(importlib.import_module(module_name(name)), COMMANDS[name].function)), name


def test_command_line_lists_refuses_and_names_the_step(capsys, monkeypatch):
    from surogate_bilingual.__main__ import COMMANDS, Command, is_implemented, main

    assert main([]) == 0 and main(["--help"]) == 0
    listing = capsys.readouterr().out
    assert all(re.search(rf"^  {re.escape(name)} ", listing, re.MULTILINE) for name in PLANNED_COMMANDS), listing
    assert "not implemented yet" not in listing and all(is_implemented(name) for name in COMMANDS)  # since I6
    assert main(["no-such-command"]) == 2
    assert "unknown command 'no-such-command'" in capsys.readouterr().err
    # a planned command whose module has not landed still exits 2 and names its step
    monkeypatch.setitem(COMMANDS, "future", Command("eval.no_such_module", "a later command", "I7"))
    assert main(["future", "--dry-run"]) == 2
    assert "'future' is not implemented yet; it lands in step I7" in capsys.readouterr().err


# --- repository wiring ---------------------------------------------------------------------------------------------


def _run(*command: str) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=120)


def test_fork_dev_help_lists_the_suite():
    result = _run("fork/dev", "help")
    assert result.returncode == 0, result.stderr
    assert re.search(r"^\s+test bilingual \[pytest args\]", result.stdout, re.MULTILINE), result.stdout


def test_an_unknown_suite_exits_with_a_usage_that_names_the_suite():
    result = _run("fork/dev", "test", "no-such-suite")
    assert result.returncode == 2
    usage = re.search(r"usage: fork/dev test (\S+)", result.stderr)
    assert usage and "bilingual" in usage.group(1).split("|"), result.stderr


@pytest.mark.parametrize(
    "path",
    [
        "fork/.venv-bilingual",  # before the environment exists, too
        "fork/.venv-bilingual/bin/python",
        "fork/bilingual/src/surogate_bilingual.egg-info/PKG-INFO",
        "fork/bilingual/build/lib/surogate_bilingual/__init__.py",
        "fork/bilingual/dist/surogate_bilingual-0.1.0.dev0-py3-none-any.whl",
    ],
)
def test_environment_and_build_outputs_are_ignored(path):
    """Untracked, they would make every run record's code_dirty true."""
    assert _run("git", "check-ignore", "-q", "--no-index", path).returncode == 0, path


def test_environment_table_lists_the_bilingual_venv():
    """AGENTS.md rule 4 points to this table for the environments; the bilingual one is listed there."""
    readme = (REPO / "fork" / "README.md").read_text(encoding="utf-8")
    section = readme.split("\n## 首次准备环境\n", 1)[1].split("\n## ", 1)[0]
    assert re.search(r"^\| `fork/\.venv-bilingual` \|", section, re.MULTILINE), section


def test_ci_and_the_sync_checks_run_the_suite():
    workflow = (REPO / ".github" / "workflows" / "fork-ci.yml").read_text(encoding="utf-8")
    assert "run: FORK_PY=python fork/dev test bilingual\n" in workflow
    source = (REPO / "fork" / "tools" / "sync_upstream.py").read_text(encoding="utf-8")
    checks = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "checks")
    assert '"bilingual"' in ast.get_source_segment(source, checks)


def test_the_cpu_suite_hides_cuda_on_every_host():
    """CR1 review: on the GPU host the suite trained on CUDA in bf16 (auto device), failed I5-T13 and ran the full G0.5
    bench grid. conftest hides CUDA before torch is used, for subprocesses too."""
    import os
    import subprocess
    import sys

    import torch

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "" and os.environ["BILINGUAL_CPU_SUITE"] == "1"
    assert not torch.cuda.is_available()
    code = "import os, torch; print(os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_available())"
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "True"
