"""check_divergence.py: the [[exact_diff]] contract for the csrc/CMakeLists.txt hook (verification
finding TOOL-2) and the [sync].last_synced ancestry check (TOOL-5)."""

import pytest
from toolrepo import commit, copy_tools, git, init_repo, run_tool, write

UPSTREAM_CMAKE = """add_library(surogate-common SHARED)
target_link_libraries(surogate-common PUBLIC ${MULTI_GPU_DEPS})
target_compile_definitions(surogate-common PUBLIC ${MULTI_GPU_DEFS})
"""
INCLUDE = "include(${CMAKE_CURRENT_SOURCE_DIR}/../fork/native/cmake/fork_native.cmake)"
CALL = "surogate_fork_configure_common(surogate-common)"

LEDGER = """[sync]
last_synced = "{last_synced}"

[owned]
prefixes = ["fork/"]
files = []

[[exact_diff]]
path = "csrc/CMakeLists.txt"
topic = "{exact_topic}"
lines = [
    "+{include}",
    "+{call}",
]

[[topic]]
id = "T5"
title = "OpenMP"
files = ["csrc/CMakeLists.txt"]
fork_files = []
tests = []

[[topic]]
id = "T9"
title = "other"
files = ["pyproject.toml"]
fork_files = []
tests = []
"""


def hooked(*lines):
    """UPSTREAM_CMAKE with lines after its second line, where csrc/CMakeLists.txt has the hook."""
    first, second, rest = UPSTREAM_CMAKE.split("\n", 2)
    return "\n".join([first, second, *lines, rest])


class Repo:
    def __init__(self, tmp_path):
        self.root = init_repo(tmp_path / "repo")
        self.u0 = commit(
            self.root,
            "upstream 0",
            {".gitignore": "__pycache__/\n", "csrc/CMakeLists.txt": UPSTREAM_CMAKE, "pyproject.toml": "a\n"},
        )
        git(self.root, "update-ref", "refs/remotes/upstream/main", self.u0)
        copy_tools(self.root, "check_divergence.py")
        self.ledger()
        self.fork = commit(
            self.root,
            "fork",
            {"csrc/CMakeLists.txt": hooked(INCLUDE, CALL), "pyproject.toml": "b\n"},
        )

    def ledger(self, last_synced=None, exact_topic="T5"):
        text = LEDGER.format(last_synced=last_synced or self.u0, exact_topic=exact_topic, include=INCLUDE, call=CALL)
        write(self.root, {"fork/patches/ledger.toml": text})

    def cmake(self, *lines):
        write(self.root, {"csrc/CMakeLists.txt": hooked(*lines)})

    def check(self):
        result = run_tool(self.root, "check_divergence", "--strict")
        return result.returncode, result.stdout + result.stderr


@pytest.fixture
def repo(tmp_path):
    return Repo(tmp_path)


def test_the_exact_hook_passes(repo):
    rc, out = repo.check()
    assert rc == 0, out
    assert "2 M ledgered" in out


def test_a_lost_hook_call_fails_although_the_file_stays_modified(repo):
    """The finding's scenario: conflict resolution keeps the include line and drops the call."""
    repo.cmake(INCLUDE)
    rc, out = repo.check()
    assert rc == 1, out
    assert "2 M ledgered" in out  # the per-file check alone still passes
    assert f"missing  +{CALL}" in out


def test_an_extra_line_fails(repo):
    repo.cmake(INCLUDE, CALL, "target_link_libraries(surogate-common PRIVATE extra)")
    rc, out = repo.check()
    assert rc == 1
    assert "extra    +target_link_libraries(surogate-common PRIVATE extra)" in out


def test_an_edited_upstream_line_fails(repo):
    write(repo.root, {"csrc/CMakeLists.txt": hooked(INCLUDE, CALL).replace("SHARED", "STATIC")})
    rc, out = repo.check()
    assert rc == 1
    assert "extra    -add_library(surogate-common SHARED)" in out


def test_the_hook_lines_in_another_order_fail(repo):
    repo.cmake(CALL, INCLUDE)
    rc, out = repo.check()
    assert rc == 1
    assert "same lines, different order" in out


def test_an_entry_must_name_a_file_of_its_topic(repo):
    repo.ledger(exact_topic="T9")
    rc, out = repo.check()
    assert rc == 1
    assert "[[exact_diff]] csrc/CMakeLists.txt: topic T9 does not list it in files" in out


def test_last_synced_must_be_an_upstream_commit(repo):
    repo.ledger(last_synced=repo.fork)
    rc, out = repo.check()
    assert rc == 1
    assert f"[sync].last_synced {repo.fork} is not an upstream commit" in out


def test_last_synced_that_is_not_a_commit_fails(repo):
    repo.ledger(last_synced="0" * 40)
    rc, out = repo.check()
    assert rc == 1
    assert "is not an upstream commit" in out


def test_an_older_upstream_last_synced_only_warns(repo):
    git(repo.root, "switch", "-q", "-c", "up", repo.u0)
    u1 = commit(repo.root, "upstream 1", {"upstream.txt": "1\n"})
    git(repo.root, "switch", "-q", "main")
    git(repo.root, "merge", "-q", "--no-ff", "-m", "sync", u1)
    git(repo.root, "update-ref", "refs/remotes/upstream/main", u1)
    rc, out = repo.check()  # the ledger still names u0
    assert rc == 0, out
    assert f"warning: [sync].last_synced is {repo.u0}, merge base is {u1}" in out
