#!/usr/bin/env bash
# CPU test lane: the fork's patch tests (GRPO, and the native patches' source pins) plus the
# upstream pytest suites, without a GPU or a built native extension. Upstream failures are
# compared with known_failures.txt.
#
# Usage: FORK_PY=<python with requirements-cpu-lane.txt and torch> fork/tools/test_cpu.sh
#   FORK_PY defaults to .venv/bin/python. JUnit reports go to build/fork-test-cpu/<n>.xml.
#   A test that runs longer than FORK_TEST_TIMEOUT seconds (default 120; the slowest upstream CPU
#   test took 17 s on 4 CPUs) fails by name (pytest-timeout) instead of hanging the job.
#
# Each partition is its own pytest process: tests/grpo/test_failure_exit_status.py leaves a
# MagicMock in sys.modules["surogate._surogate"], which changes every module collected after
# it in the same process. --no-gpu is passed only to partitions under tests/ or
# csrc/src/testing/serve, because tests/conftest.py defines it.
#
# Exit status 1 when a fork patch test fails, when an upstream test fails or errors (including
# collection errors) and is not listed in known_failures.txt, or when a partition does not run.
# Listed entries that no longer fail are reported as stale and do not fail the run.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

PY=${FORK_PY:-.venv/bin/python}
OUT=build/fork-test-cpu
TIMEOUT=${FORK_TEST_TIMEOUT:-120}
if ! "$PY" -c 'import pytest, pytest_timeout' 2>/dev/null; then
    echo "test_cpu: $PY cannot import pytest and pytest_timeout; install fork/tools/requirements-cpu-lane.txt" >&2
    exit 1
fi
partitions=(
    "fork/patches/tests"
    "tests/serve tests/test_serve_contract.py csrc/src/testing/serve --no-gpu"
    "tests/grpo --ignore=tests/grpo/test_failure_exit_status.py --no-gpu"
    "tests/grpo/test_failure_exit_status.py --no-gpu"
    "tests/train --no-gpu"
    "tests/distill tests/dpo tests/debug tests/examples --no-gpu"
    "tests --ignore=tests/serve --ignore=tests/test_serve_contract.py --ignore=tests/grpo --ignore=tests/train
     --ignore=tests/distill --ignore=tests/dpo --ignore=tests/debug --ignore=tests/examples --no-gpu"
)

rm -rf "$OUT"
mkdir -p "$OUT"
status=0
for i in "${!partitions[@]}"; do
    n=$((i + 1))
    # xunit1 keeps the file attribute, which the comparison below needs to rebuild node ids.
    cmd=("$PY" -m pytest -p no:cacheprovider -q --continue-on-collection-errors --timeout="$TIMEOUT"
        -o junit_family=xunit1 --junitxml="$OUT/$n.xml" ${partitions[$i]})
    echo "+ ${cmd[*]}" >&2
    "${cmd[@]}"
    rc=$?
    # pytest exits 1 when tests failed; anything else means the partition itself did not run.
    if [ "$rc" -gt 1 ]; then
        echo "test_cpu: partition $n exited $rc" >&2
        status=1
    fi
done

python3 -I - "$OUT" fork/tools/known_failures.txt <<'EOF' || status=1
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

out, known_file = Path(sys.argv[1]), Path(sys.argv[2])


def node_id(case):
    """Rebuild the pytest node id from a JUnit testcase (inverse of pytest's name mangling)."""
    path, classname, name = case.get("file", ""), case.get("classname", ""), case.get("name", "")
    if not classname:  # collection error: the name is the dotted module path
        return path
    module = path.removesuffix(".py").replace("/", ".")
    classes = classname[len(module):].strip(".").split(".") if classname.startswith(module) else []
    return "::".join([path, *filter(None, classes), name])


failed = {}  # node id -> partition number
for report in sorted(out.glob("*.xml"), key=lambda p: int(p.stem)):
    for case in ET.parse(report).iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            failed.setdefault(node_id(case), int(report.stem))

known = {}
for line in known_file.read_text(encoding="utf-8").splitlines():
    entry = re.split(r"(?:^|\s)#", line, maxsplit=1)[0].strip()
    if entry:
        known[entry] = line

fork_failures = sorted(n for n, part in failed.items() if part == 1)
new_failures = sorted(n for n, part in failed.items() if part != 1 and n not in known)
stale = sorted(n for n in known if n not in failed)

for title, items in [
    ("fork patch tests failed", fork_failures),
    ("upstream failures not in known_failures.txt", new_failures),
    ("stale known_failures.txt entries (pass now or no longer exist)", stale),
]:
    if items:
        print(f"test_cpu: {title}:")
        print("".join(f"  {n}\n" for n in items), end="")
print(f"test_cpu: {len(failed)} failing node ids, {len(failed) - len(new_failures) - len(fork_failures)} known")
sys.exit(1 if fork_failures or new_failures else 0)
EOF
exit "$status"
