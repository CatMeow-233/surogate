"""Package boundaries and repository wiring of fork/tokstudy (implementation step S0).

Importing `surogate_tokstudy` and listing its commands load neither torch nor any `surogate*` module; its source
imports only the stdlib, numpy, torch, tokenizers and safetensors (mamba_ssm only in kernels.py, guarded), plus the
few public project modules the import table allows per module, and never the native package or the corpus adapters;
the package ships in no wheel of the fork; and fork/dev, CI, the sync checks, the environment table and the ignore
rules know the suite.
"""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

TOKSTUDY = Path(__file__).resolve().parents[2]
REPO = TOKSTUDY.parents[1]
SRC = TOKSTUDY / "src"
PACKAGE = SRC / "surogate_tokstudy"

PLANNED_COMMANDS = ["arms", "intrinsic", "align", "train", "branch", "eval", "probe", "verdict", "flops", "bench"]
STEPS = {f"S{n}" for n in range(8, 18)}


def _pyproject(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


# --- package import ------------------------------------------------------------------------------------------------

IMPORT_CHECK = """
import sys
sys.path.insert(0, sys.argv[1])
import surogate_tokstudy
from surogate_tokstudy import __main__ as cli
cli.usage()
bad = ("torch", "numpy", "surogate", "surogate_tlc", "surogate_bilingual", "surogate_corpora", "mamba_ssm")
loaded = sorted(m for m in sys.modules if m.split(".")[0] in bad)
print(surogate_tokstudy.__file__)
sys.exit(f"importing surogate_tokstudy loaded {loaded}" if loaded else 0)
"""


def test_package_import_loads_neither_torch_nor_any_project(tmp_path):
    command = [sys.executable, "-I", "-B", "-c", IMPORT_CHECK, str(SRC)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(result.stdout.strip()).resolve() == PACKAGE / "__init__.py"


def test_tests_import_the_checkout():
    import surogate_tokstudy

    assert Path(surogate_tokstudy.__file__).resolve().parent == PACKAGE
    assert sys.path[0] == str(SRC)


# --- packaging -----------------------------------------------------------------------------------------------------


def test_pyproject_declares_an_unpublished_src_layout_package():
    config = _pyproject(TOKSTUDY / "pyproject.toml")
    project = config["project"]
    assert project["name"] == "surogate-tokstudy"
    assert project["dependencies"] == []
    lower = re.fullmatch(r">=\s*3\.(\d+)", project["requires-python"])
    assert lower and int(lower.group(1)) >= 12, project["requires-python"]
    find = config["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["src"] and find["include"] == ["surogate_tokstudy*"]
    packages = {p.parent.relative_to(SRC).as_posix() for p in SRC.rglob("__init__.py")}
    assert packages and all(p.split("/")[0] == "surogate_tokstudy" for p in packages), packages
    assert config["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests/cpu"]


def _shipped_roots() -> list[Path]:
    native = _pyproject(REPO / "pyproject.toml")["tool"]["scikit-build"]["wheel"]["packages"]
    find = _pyproject(REPO / "fork" / "experimental" / "pyproject.toml")["tool"]["setuptools"]["packages"]["find"]
    return [REPO / package for package in native] + [REPO / "fork" / "experimental" / where for where in find["where"]]


def test_tokstudy_tree_is_outside_every_shipped_wheel_root():
    roots = _shipped_roots()
    assert roots and all(root.is_dir() for root in roots), roots
    assert not [root for root in roots if TOKSTUDY.is_relative_to(root) or root.is_relative_to(TOKSTUDY)]


# --- import boundary of src/ ---------------------------------------------------------------------------------------
#
# Every module may import the stdlib, the package itself and DEPENDENCIES. Project modules are allowed per module
# (PROJECT_ALLOWED: module path relative to the package -> importable modules); mamba_ssm only in kernels.py, inside
# try/except ImportError. Loading code by path is for tests only (prototypes/ref_mamba2.py).

DEPENDENCIES = {"numpy", "torch", "tokenizers", "safetensors"}
PROJECT_ALLOWED = {
    "docs": {"surogate_tlc.data.corpus", "surogate_bilingual.data.docs"},
    "models.trunk_bilingual": {"surogate_bilingual.model.lm", "surogate_bilingual.config"},
}
GUARDED = {"kernels": {"mamba_ssm"}}
PATH_LOADERS = {"spec_from_file_location", "SourceFileLoader", "run_path"}


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(n, ast.Name) and n.id in ("ImportError", "ModuleNotFoundError") for n in names)


def boundary_problems(source: str, module: str) -> list[str]:
    """Imports of one module (dotted path relative to the package) that cross the import table."""
    tree = ast.parse(source)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def guarded(node: ast.AST) -> bool:
        while node in parents:
            parent = parents[node]
            if isinstance(parent, ast.Try) and node in parent.body and any(map(_catches_import_error, parent.handlers)):
                return True
            node = parent
        return False

    allowed_projects = PROJECT_ALLOWED.get(module, set())
    problems = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            # `from surogate_tlc.data import corpus` imports surogate_tlc.data.corpus: judge each imported name
            project = node.module.split(".")[0] in ("surogate_tlc", "surogate_bilingual")
            names = [f"{node.module}.{alias.name}" for alias in node.names] if project else [node.module]
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
            first = node.args[0] if node.args else None
            if called in ("import_module", "__import__") and isinstance(first, ast.Constant):
                if isinstance(first.value, str) and not first.value.startswith("."):
                    names = [first.value]
            if called in PATH_LOADERS:
                problems.append(f"{module}:{node.lineno}: loads code by path ({called})")
        for name in names:
            top = name.split(".")[0]
            if top in ("surogate_tlc", "surogate_bilingual"):
                if not any(name == a or name.startswith(a + ".") for a in allowed_projects):
                    problems.append(f"{module}:{node.lineno}: imports {name}, not in its import table")
            elif top in GUARDED.get(module, set()):
                if not guarded(node):
                    problems.append(f"{module}:{node.lineno}: imports {name} outside try/except ImportError")
            elif top not in DEPENDENCIES | {"surogate_tokstudy"} and top not in sys.stdlib_module_names:
                problems.append(f"{module}:{node.lineno}: imports {name}, which is outside the package's boundary")
    return problems


def test_src_imports_stay_inside_the_boundary():
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        module = ".".join(path.relative_to(PACKAGE).with_suffix("").parts)
        found += boundary_problems(path.read_text(encoding="utf-8"), module)
    assert not found, "\n".join(found)


def test_the_boundary_check_catches_each_kind_of_crossing():
    planted = """
import surogate
from surogate._fork import grpo
import surogate_corpora.sources
from surogate_tlc.data import corpus
import surogate_bilingual.data.docs
import mamba_ssm
import transformers
import importlib
importlib.import_module("surogate_tlc.data.shards")
spec = importlib.util.spec_from_file_location("m", "fork/tlc/prototypes/ref_mamba2.py")
"""
    lines = lambda problems: sorted(int(p.split(":")[1]) for p in problems)  # noqa: E731
    assert lines(boundary_problems(planted, "stream")) == [2, 3, 4, 5, 6, 7, 8, 10, 11]
    # docs may read the two corpus modules, trunk_bilingual the bilingual model and config, kernels a guarded mamba_ssm
    assert lines(boundary_problems(planted, "docs")) == [2, 3, 4, 7, 8, 10, 11]
    trunk = "import surogate_bilingual.model.lm\nfrom surogate_bilingual.config import load_toml\n"
    assert boundary_problems(trunk, "models.trunk_bilingual") == []
    assert lines(boundary_problems(trunk, "models.hnet")) == [1, 2]
    kernels = "try:\n    from mamba_ssm.ops.triton import ssd_combined\nexcept ImportError:\n    ssd_combined = None\n"
    assert boundary_problems(kernels, "kernels") == []
    assert lines(boundary_problems("import mamba_ssm\n", "kernels")) == [1]
    allowed = "from __future__ import annotations\nimport json\nimport numpy as np\nimport torch\nfrom . import arms\n"
    assert boundary_problems(allowed, "train") == []


# --- command table -------------------------------------------------------------------------------------------------


def test_command_table_lists_the_planned_commands():
    from surogate_tokstudy.__main__ import COMMANDS

    assert list(COMMANDS) == PLANNED_COMMANDS
    for name, command in COMMANDS.items():
        assert re.fullmatch(r"[a-z_]+(\.[a-z_]+)*", command.module), name
        assert command.step in STEPS, name


def test_command_line_lists_refuses_and_names_the_step(capsys):
    from surogate_tokstudy.__main__ import COMMANDS, is_implemented, main

    assert main([]) == 0 and main(["--help"]) == 0
    listing = capsys.readouterr().out
    assert all(re.search(rf"^  {re.escape(name)} ", listing, re.MULTILINE) for name in PLANNED_COMMANDS), listing
    assert main(["no-such-command"]) == 2
    assert "unknown command 'no-such-command'" in capsys.readouterr().err
    pending = [name for name in COMMANDS if not is_implemented(name)]
    for name in pending:
        assert main([name]) == 2
        assert f"{name!r} is not implemented yet; it lands in step {COMMANDS[name].step}" in capsys.readouterr().err


# --- repository wiring ---------------------------------------------------------------------------------------------


def _run(*command: str) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=120)


def test_fork_dev_help_lists_the_suite():
    result = _run("fork/dev", "help")
    assert result.returncode == 0, result.stderr
    assert re.search(r"^\s+test tokstudy \[pytest args\]", result.stdout, re.MULTILINE), result.stdout


def test_an_unknown_suite_exits_with_a_usage_that_names_the_suite():
    result = _run("fork/dev", "test", "no-such-suite")
    assert result.returncode == 2
    usage = re.search(r"usage: fork/dev test (\S+)", result.stderr)
    assert usage and "tokstudy" in usage.group(1).split("|"), result.stderr


@pytest.mark.parametrize(
    "path",
    [
        "fork/.venv-tokstudy",
        "fork/.venv-tokstudy/bin/python",
        "fork/tokstudy/src/surogate_tokstudy.egg-info/PKG-INFO",
        "fork/tokstudy/build/lib/surogate_tokstudy/__init__.py",
        "fork/tokstudy/dist/surogate_tokstudy-0.1.0.dev0-py3-none-any.whl",
    ],
)
def test_environment_and_build_outputs_are_ignored(path):
    assert _run("git", "check-ignore", "-q", "--no-index", path).returncode == 0, path


def test_environment_table_lists_the_tokstudy_venv():
    readme = (REPO / "fork" / "README.md").read_text(encoding="utf-8")
    section = readme.split("\n## 首次准备环境\n", 1)[1].split("\n## ", 1)[0]
    assert re.search(r"^\| `fork/\.venv-tokstudy` \|", section, re.MULTILINE), section


def test_ci_and_the_sync_checks_run_the_suite():
    workflow = (REPO / ".github" / "workflows" / "fork-ci.yml").read_text(encoding="utf-8")
    assert "run: FORK_PY=python fork/dev test tokstudy\n" in workflow
    source = (REPO / "fork" / "tools" / "sync_upstream.py").read_text(encoding="utf-8")
    checks = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == "checks")
    assert '"tokstudy"' in ast.get_source_segment(source, checks)


def test_the_cpu_suite_hides_cuda():
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
