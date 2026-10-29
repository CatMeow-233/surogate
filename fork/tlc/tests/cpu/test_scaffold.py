"""Package boundaries of fork/tlc: importing `surogate_tlc` loads no `surogate` module (so it works
without the native extension), and no TLC code lands in a wheel other than the package's own (the
native wheel ships `surogate/` whole, the experimental wheel `surogate.experimental*`).
"""

import fnmatch
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


def test_tlc_code_is_not_in_the_native_or_experimental_wheel():
    native = _pyproject(REPO / "pyproject.toml")["tool"]["scikit-build"]["wheel"]["packages"]
    for package in native:
        assert not any((REPO / package).rglob("*tlc*")), f"TLC code under {package}/ would ship in the native wheel"
    experimental = _pyproject(REPO / "fork" / "experimental" / "pyproject.toml")
    include = experimental["tool"]["setuptools"]["packages"]["find"]["include"]
    assert not any(fnmatch.fnmatch("surogate_tlc", pattern) for pattern in include)


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
