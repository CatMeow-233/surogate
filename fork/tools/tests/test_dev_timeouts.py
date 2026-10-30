"""fork/dev's pytest suites and fork/tools/test_cpu.sh stop a blocking test after a per-test limit
and name it, and fork/dev passes extra arguments to pytest (verification finding TOOL-6)."""

import os
import shutil
import signal
import subprocess
import sys
import time

import pytest
from toolrepo import TOOLS, write

FAKE_PY = """#!/bin/sh
# Stands in for FORK_PY: `-c` probes succeed unless FAKE_NO_TIMEOUT is set; other calls are logged.
if [ "$1" = -c ]; then
    [ -z "$FAKE_NO_TIMEOUT" ]
    exit
fi
echo "$*" >> "$FAKE_PY_LOG"
"""


@pytest.fixture
def fake_tree(tmp_path):
    """A copy of fork/dev and fork/tools/test_cpu.sh, run with a fake FORK_PY that logs its calls."""
    (tmp_path / "fork" / "tools").mkdir(parents=True)
    shutil.copy2(TOOLS.parent / "dev", tmp_path / "fork" / "dev")
    shutil.copy2(TOOLS / "test_cpu.sh", tmp_path / "fork" / "tools" / "test_cpu.sh")
    write(tmp_path, {"fork/tools/known_failures.txt": ""})
    fake = tmp_path / "fake-python"
    fake.write_text(FAKE_PY, encoding="utf-8")
    fake.chmod(0o755)
    log = tmp_path / "calls.log"
    env = {k: v for k, v in os.environ.items() if k != "FORK_TEST_TIMEOUT"}
    env.update(FORK_PY=str(fake), FAKE_PY_LOG=str(log))
    return tmp_path, env, log


def run(tree, env, *args, **extra):
    return subprocess.run([*args], cwd=tree, env={**env, **extra}, capture_output=True, text=True)


@pytest.mark.parametrize(
    ("suite", "path"),
    [("patches", "fork/patches/tests"), ("tlc", "fork/tlc/tests/cpu"), ("tools", "fork/tools/tests")],
)
def test_suites_pass_the_limit_and_extra_arguments(fake_tree, suite, path):
    tree, env, log = fake_tree
    result = run(tree, env, "fork/dev", "test", suite, "-k", "handoff", "-x")
    assert result.returncode == 0, result.stderr
    assert log.read_text().split() == [
        "-m", "pytest", "-q", "-p", "no:cacheprovider", "--timeout=120", path, "-k", "handoff", "-x"
    ]  # fmt: skip


def test_the_limit_can_be_changed(fake_tree):
    tree, env, log = fake_tree
    run(tree, env, "fork/dev", "test", "patches", FORK_TEST_TIMEOUT="15")
    assert "--timeout=15" in log.read_text().split()


def test_a_python_without_pytest_timeout_is_named(fake_tree):
    tree, env, log = fake_tree
    result = run(tree, env, "fork/dev", "test", "patches", FAKE_NO_TIMEOUT="1")
    assert result.returncode == 2
    assert "cannot import pytest and pytest_timeout" in result.stderr
    assert not log.exists()


def test_cpu_lane_passes_the_limit_to_every_partition(fake_tree):
    tree, env, log = fake_tree
    result = run(tree, env, "fork/dev", "test", "cpu", FORK_TEST_TIMEOUT="45")
    assert result.returncode == 0, result.stdout + result.stderr
    calls = log.read_text().splitlines()
    assert len(calls) == 7  # one pytest process per partition
    assert all("--timeout=45" in call.split() for call in calls)


def test_cpu_lane_names_a_python_without_pytest_timeout(fake_tree):
    tree, env, log = fake_tree
    result = run(tree, env, "fork/tools/test_cpu.sh", FAKE_NO_TIMEOUT="1")
    assert result.returncode == 1
    assert "cannot import pytest and pytest_timeout" in result.stderr


def test_a_blocking_test_fails_by_name_within_the_limit(tmp_path):
    """End to end with the real pytest-timeout: a test that blocks, as the GRPO tests do when the
    in-process handoff is lost, is reported by name instead of hanging the run."""
    (tmp_path / "fork").mkdir()
    shutil.copy2(TOOLS.parent / "dev", tmp_path / "fork" / "dev")
    write(
        tmp_path,
        {
            "fork/patches/tests/test_blocks.py": (
                "import threading\n\n\n"
                "def test_waits_for_a_batch_that_never_arrives():\n"
                "    threading.Event().wait(300)  # bounded: a pytest left behind by a failure still ends\n\n\n"
                "def test_still_runs():\n"
                "    assert True\n"
            )
        },
    )
    env = {k: v for k, v in os.environ.items() if k != "FORK_TEST_TIMEOUT"}
    start = time.monotonic()
    # fork/dev runs pytest as a child of its own: when this test fails because fork/dev lost --timeout,
    # killing fork/dev alone would leave that pytest blocked. So run it in a session of its own and kill
    # the whole session afterwards, pass or fail.
    proc = subprocess.Popen(
        ["fork/dev", "test", "patches"],
        cwd=tmp_path,
        env={**env, "FORK_PY": sys.executable, "FORK_TEST_TIMEOUT": "2"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=60)
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
    assert time.monotonic() - start < 50
    assert proc.returncode == 1, stdout + stderr
    assert "FAILED fork/patches/tests/test_blocks.py::test_waits_for_a_batch_that_never_arrives" in stdout
    assert "Timeout" in stdout
    assert "1 failed, 1 passed" in stdout
