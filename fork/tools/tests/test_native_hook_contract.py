"""The configure-time hook contract in fork/native/openmp-contract (hook_contract.cmake) fails when
the csrc/CMakeLists.txt hook no longer gives surogate-common OpenMP and the fork include root
(verification finding TOOL-2). Each case configures a copy of csrc/CMakeLists.txt and fork/native
with one line removed; nothing is compiled."""

import re
import shutil
import subprocess

import pytest
from toolrepo import REPO

pytestmark = pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which("c++") is None, reason="needs cmake and a C++ compiler"
)

CALL = "surogate_fork_configure_common(surogate-common)\n"
INCLUDE = "include(${CMAKE_CURRENT_SOURCE_DIR}/../fork/native/cmake/fork_native.cmake)\n"
OPENMP_CALL = re.compile(r"^\s*surogate_enable_training_openmp\(\$\{target\}\)\n", re.M)
INCLUDE_ROOT = re.compile(r"^\s*target_include_directories\(\$\{target\} PRIVATE [^\n]*\n", re.M)


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "csrc").mkdir()
    shutil.copy2(REPO / "csrc" / "CMakeLists.txt", tmp_path / "csrc" / "CMakeLists.txt")
    shutil.copytree(REPO / "fork" / "native", tmp_path / "fork" / "native")
    return tmp_path


def edit(path, pattern, replacement=""):
    """Replace the first match of pattern (a compiled regex or a plain string) in the file."""
    text = path.read_text(encoding="utf-8")
    if isinstance(pattern, re.Pattern):
        new = pattern.sub(replacement, text, count=1)
    else:
        new = text.replace(pattern, replacement, 1)
    assert new != text, f"{pattern!r} not found in {path}"
    path.write_text(new, encoding="utf-8")


def configure(tree):
    result = subprocess.run(
        ["cmake", "-S", str(tree / "fork/native/openmp-contract"), "-B", str(tree / "build")],
        capture_output=True,
        text=True,
    )
    return result.returncode, " ".join((result.stdout + result.stderr).split())


def test_the_hook_as_committed_passes(tree):
    rc, out = configure(tree)
    assert rc == 0, out
    assert "fork hook contract: csrc/CMakeLists.txt gives surogate-common OpenMP::OpenMP_CXX" in out


def test_a_lost_hook_call_fails(tree):
    edit(tree / "csrc/CMakeLists.txt", CALL)
    rc, out = configure(tree)
    assert rc != 0
    assert "has no surogate_fork_configure_common(<target>) call" in out


def test_a_lost_openmp_call_in_fork_native_fails(tree):
    edit(tree / "fork/native/cmake/fork_native.cmake", OPENMP_CALL)
    rc, out = configure(tree)
    assert rc != 0
    assert "surogate-common does not link OpenMP::OpenMP_CXX" in out
    assert "If upstream now links OpenMP into surogate-common itself, remove this check" in out


def test_the_target_comes_from_the_hook_call(tree):
    """The T5 playbook's rename case: upstream renames surogate-common and the hook argument follows;
    the contract needs no edit (review of the TOOL-2 fix)."""
    path = tree / "csrc/CMakeLists.txt"
    path.write_text(path.read_text(encoding="utf-8").replace("surogate-common", "surogate-core"), encoding="utf-8")
    rc, out = configure(tree)
    assert rc == 0, out
    assert "fork hook contract: csrc/CMakeLists.txt gives surogate-core OpenMP::OpenMP_CXX" in out


def test_a_hook_argument_that_names_no_earlier_target_fails(tree):
    edit(tree / "csrc/CMakeLists.txt", CALL, "surogate_fork_configure_common(surogate-core)\n")
    rc, out = configure(tree)
    assert rc != 0
    assert "must come after add_library(surogate-core ...)" in out


def test_a_lost_include_root_fails(tree):
    edit(tree / "fork/native/cmake/fork_native.cmake", INCLUDE_ROOT)
    rc, out = configure(tree)
    assert rc != 0
    assert "surogate-common does not have the include root" in out


def test_a_lost_hook_fails(tree):
    edit(tree / "csrc/CMakeLists.txt", INCLUDE + CALL)
    rc, out = configure(tree)
    assert rc != 0
    assert "has no fork hook line" in out


def test_a_hook_before_the_target_fails(tree):
    path = tree / "csrc/CMakeLists.txt"
    edit(path, INCLUDE + CALL)
    edit(path, "add_library(surogate-common", INCLUDE + CALL + "add_library(surogate-common")
    rc, out = configure(tree)
    assert rc != 0
    assert "must come after add_library(surogate-common ...)" in out
