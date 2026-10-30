"""fork/tools/test_cpu.sh compares upstream failures with known_failures.txt: an unlisted failure fails
the run, a listed entry that passes is reported as stale, and an entry marked "flaky:" may pass or
fail without either (final verification finding on tests/grpo/test_teacher_logprobs.py)."""

import os
import shutil
import subprocess

import pytest
from toolrepo import TOOLS, write

FLAKY = "tests/grpo/test_teacher_logprobs.py::test_k0_request_and_single_candidate"
LISTED = "tests/grpo/test_config.py"

# Stands in for FORK_PY: the pytest-timeout probe succeeds, and each pytest run writes a JUnit report
# (xunit1, as test_cpu.sh asks) in which partition 3 (tests/grpo) fails the test named in FAKE_FAILING.
FAKE_PY = """#!/bin/sh
[ "$1" = -c ] && exit 0
for arg; do case $arg in --junitxml=*) report=${arg#--junitxml=} ;; esac; done
case $report in
    */3.xml) [ -n "$FAKE_FAILING" ] || { echo '<testsuite/>' >"$report"; exit 0; }
        file=${FAKE_FAILING%%::*} name=${FAKE_FAILING#*::}
        module=$(echo "${file%.py}" | tr / .)
        printf '<testsuite><testcase file="%s" classname="%s" name="%s"><failure/></testcase></testsuite>\\n' \\
            "$file" "$module" "$name" >"$report"
        exit 1 ;;
    *) echo '<testsuite/>' >"$report" ;;
esac
"""


@pytest.fixture
def lane(tmp_path):
    (tmp_path / "fork" / "tools").mkdir(parents=True)
    shutil.copy2(TOOLS / "test_cpu.sh", tmp_path / "fork" / "tools" / "test_cpu.sh")
    write(
        tmp_path,
        {
            "fork/tools/known_failures.txt": (
                "# header\n"
                f"{LISTED}  # collection error; first seen at upstream 282f9901\n"
                f"{FLAKY}  # flaky: request order; first seen at upstream 282f9901\n"
            )
        },
    )
    fake = tmp_path / "fake-python"
    fake.write_text(FAKE_PY, encoding="utf-8")
    fake.chmod(0o755)

    def run(failing):
        env = {**os.environ, "FORK_PY": str(fake), "FAKE_FAILING": failing}
        result = subprocess.run(
            ["fork/tools/test_cpu.sh"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        return result.returncode, result.stdout

    return run


def test_a_flaky_entry_that_fails_is_known(lane):
    status, out = lane(FLAKY)
    assert status == 0, out
    assert "not in known_failures.txt" not in out
    assert "1 failing node ids, 1 known" in out


def test_a_flaky_entry_that_passes_is_not_stale(lane):
    status, out = lane("")
    assert status == 0, out
    stale = out.split("stale known_failures.txt entries", 1)[1]
    assert LISTED in stale and FLAKY not in stale


def test_an_unlisted_failure_fails_the_run(lane):
    status, out = lane("tests/grpo/test_teacher_logprobs.py::test_vllm_actual_first_candidates_give_the_same_scores")
    assert status == 1
    assert "upstream failures not in known_failures.txt" in out
