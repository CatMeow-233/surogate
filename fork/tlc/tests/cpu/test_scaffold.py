"""Package boundaries of fork/tlc: importing `surogate_tlc` loads no `surogate` module (so it works
without the native extension), and no TLC code lands in a wheel other than the package's own (the
native wheel ships `surogate/` whole, the experimental wheel `surogate.experimental*`).
"""

import ast
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
TLC = REPO / "fork" / "tlc"


def _pyproject(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def test_package_import_does_not_load_surogate():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import surogate_tlc; "
        "native = sorted(m for m in sys.modules if m.startswith('surogate.') or m == 'surogate'); "
        "sys.exit(f'imported {native}' if native else 0)"
    )
    result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(TLC / "src")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_checkout_dsl_imports_on_cpu():
    from surogate.dsl import py_compiler

    assert Path(py_compiler.__file__).resolve().is_relative_to(REPO / "surogate")


def test_tlc_wheel_holds_only_surogate_tlc():
    find = _pyproject(TLC / "pyproject.toml")["tool"]["setuptools"]["packages"]["find"]
    assert find["where"] == ["src"] and find["include"] == ["surogate_tlc*"]
    packages = {p.parent.relative_to(TLC / "src").as_posix() for p in (TLC / "src").rglob("__init__.py")}
    assert packages and all(p.split("/")[0] == "surogate_tlc" for p in packages), packages


def _shipped_roots() -> list[Path]:
    """The directories the native and the experimental wheel take their packages from, read from their configs."""
    native = _pyproject(REPO / "pyproject.toml")["tool"]["scikit-build"]["wheel"]["packages"]
    find = _pyproject(REPO / "fork" / "experimental" / "pyproject.toml")["tool"]["setuptools"]["packages"]["find"]
    return [REPO / package for package in native] + [REPO / "fork" / "experimental" / where for where in find["where"]]


def _markers() -> set[str]:
    """Names only TLC code uses: the package, its config keys and its classes (TLCModel, TLCMambaFFNBlock, ...)."""
    names = {"surogate_tlc", "tlc_layer_pattern", "tlc_split_first_layer"}
    for path in (TLC / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names |= {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name.startswith("TLC")}
    return names


def test_tlc_code_is_not_in_the_native_or_experimental_wheel():
    """Three checks, since a move or a copy under another name passes a check of file names: the TLC tree lies outside
    every shipped root, TLC code imports no fork code that ships in a wheel (moving a module there means importing
    it from there), and no shipped file holds a TLC name (a copy). A copy with every TLC name stripped still passes:
    review is the boundary, these checks back it up."""
    roots = _shipped_roots()
    assert len(roots) == 2 and not any(TLC.is_relative_to(root) or root.is_relative_to(TLC) for root in roots)
    shipped_fork = ("surogate._fork", "surogate.experimental")
    for path in TLC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            modules += [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else []
            assert not [m for m in modules if m.startswith(shipped_fork)], (path, modules)
    markers = _markers()
    assert {"TLCModel", "TLCMambaFFNBlock", "TLCConfig"} <= markers
    found = [
        f"{path.relative_to(REPO)}: {sorted(name for name in markers if name in text)}"
        for root in roots
        for path in root.rglob("*.py")
        if any(name in (text := path.read_text(encoding="utf-8", errors="replace")) for name in markers)
    ]
    assert not found, found


def test_gpu_tests_do_not_skip_in_the_runbook(tmp_path):
    """Without CUDA the GPU tests skip and pytest exits 0, which the runbook (TLC_RUNS set) would read as a passed
    gate; there the session stops with a usage error instead. Without TLC_RUNS (a CPU host) they still skip."""
    import os

    import pytest

    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("checks the CPU-only behaviour")
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c", str(TLC / "pyproject.toml")]
    command.append(str(TLC / "tests" / "gpu" / "test_gpu_environment.py"))
    env = {k: v for k, v in os.environ.items() if k != "TLC_RUNS"}
    skipped = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert skipped.returncode == 0 and "2 skipped" in skipped.stdout, skipped.stdout + skipped.stderr
    stopped = subprocess.run(command, cwd=REPO, env={**env, "TLC_RUNS": str(tmp_path)}, capture_output=True, text=True)
    assert stopped.returncode == pytest.ExitCode.USAGE_ERROR, stopped.stdout + stopped.stderr
    assert "TLC_RUNS is set but torch.cuda.is_available() is False" in stopped.stdout + stopped.stderr


def test_build_outputs_of_the_package_are_ignored():
    """A non-editable install or a wheel build of fork/tlc writes build/ (and dist/) into the package directory;
    untracked, they would make every run record's git_dirty true."""
    paths = [f"fork/tlc/{name}/lib/surogate_tlc/__init__.py" for name in ("build", "dist")]
    result = subprocess.run(["git", "check-ignore", "--no-index", *paths], cwd=REPO, capture_output=True, text=True)
    assert result.stdout.split() == paths, result.stdout + result.stderr


# Section numbers and finding ids of the planning documents, which are not in the repository: a maintainer could not
# look them up, and they change with every revision. State the reason, or cite a document under fork/tlc/docs.
EXTERNAL_CITATION = re.compile(r"\u00a7|\bv0\.\d|\b(?:IMPL|ARC|SD|EV)-[A-Z0-9]+\b|\bO-\d+\b|plan test \d")


def test_no_citations_of_documents_outside_the_repository():
    found = []
    for path in sorted(TLC.rglob("*")):
        if path.suffix in {".py", ".md", ".yaml", ".toml"} and "__pycache__" not in path.parts:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if EXTERNAL_CITATION.search(line):
                    found.append(f"{path.relative_to(TLC)}:{number}: {line.strip()[:80]}")
    assert not found, "\n".join(found)
