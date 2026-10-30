"""`fork/dev sync --finish` pushes to main only the commit whose required checks passed and that the
PR shows (verification finding TOOL-1).

A throwaway repository with a bare `origin` runs the copied sync_upstream.py; a fake `gh` answers
`gh pr view` with the PR head and `gh api .../commits/<sha>/check-runs` from a JSON state file.
"""

import json
import os
import sys

import pytest
from toolrepo import commit, copy_tools, git, init_repo, run_tool

BRANCH = "sync/20261030-0123abcd"
PASSING = ["check", "experimental", "fork-tests", "openmp-contract"]

FAKE_GH = """#!{python}
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_GH_LOG"], "a", encoding="utf-8") as log:
    log.write(" ".join(args) + "\\n")
with open(os.environ["FAKE_GH_STATE"], encoding="utf-8") as f:
    state = json.load(f)
if args[:2] == ["pr", "view"]:
    print(state["pr_head"])
elif args[:1] == ["api"] and "/check-runs" in args[1]:
    sha = args[1].split("/commits/")[1].split("/")[0]
    runs = state["checks"].get(sha, [])
    print(json.dumps({{"total_count": len(runs), "check_runs": runs}}))
else:
    sys.exit("fake gh: unexpected call " + " ".join(args))
"""


def runs(names=PASSING, conclusion="success", first_id=1):
    return [
        {"id": first_id + i, "name": name, "status": "completed", "conclusion": conclusion}
        for i, name in enumerate(names)
    ]


class Sim:
    """origin (bare), work (the maintainer's clone on the sync branch) and a fake gh."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.origin = tmp_path / "origin.git"
        git(tmp_path, "init", "-q", "--bare", "-b", "main", str(self.origin))
        self.work = init_repo(tmp_path / "work")
        copy_tools(self.work, "sync_upstream.py", "check_doc_links.py")
        self.base = commit(
            self.work,
            "base",
            {"fork/patches/ledger.toml": '[sync]\nlast_synced = ""\n', "README.md": "fork\n"},
        )
        git(self.work, "remote", "add", "origin", str(self.origin))
        git(self.work, "push", "-q", "origin", "main")
        git(self.work, "switch", "-q", "-c", BRANCH)
        self.checked = commit(self.work, "sync: merge upstream", {"upstream.txt": "1\n"})
        git(self.work, "push", "-q", "-u", "origin", BRANCH)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        gh = bin_dir / "gh"
        gh.write_text(FAKE_GH.format(python=sys.executable), encoding="utf-8")
        gh.chmod(0o755)
        self.log = tmp_path / "gh.log"
        self.state_file = tmp_path / "gh.json"
        self.env = dict(
            os.environ,
            PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            FAKE_GH_LOG=str(self.log),
            FAKE_GH_STATE=str(self.state_file),
        )
        self.state(pr_head=self.checked, checks={self.checked: runs()})

    def state(self, **state):
        self.state_file.write_text(json.dumps(state), encoding="utf-8")

    def finish(self, *extra):
        return run_tool(self.work, "sync_upstream", "--finish", *extra, env=self.env)

    def origin_main(self):
        return git(self.origin, "rev-parse", "refs/heads/main")

    def second_clone(self):
        other = self.tmp / "other"
        git(self.tmp, "clone", "-q", str(self.origin), str(other))
        return other


@pytest.fixture
def sim(tmp_path):
    return Sim(tmp_path)


def test_finish_pushes_the_checked_pr_head(sim):
    result = sim.finish()
    assert result.returncode == 0, result.stdout + result.stderr
    assert sim.origin_main() == sim.checked
    assert git(sim.work, "branch", "--show-current") == "main"
    assert f"commits/{sim.checked}/check-runs" in sim.log.read_text()


def test_finish_refuses_a_local_commit_that_ci_never_checked(sim):
    """The finding's scenario: a fix committed after the last push, while the PR shows a green run."""
    unchecked = commit(sim.work, "local: unreviewed edit", {"surogate.txt": "edit\n"})
    result = sim.finish()
    assert result.returncode == 4, result.stdout + result.stderr
    assert sim.origin_main() == sim.base
    assert f"HEAD {unchecked[:12]} is not origin/{BRANCH}" in result.stdout
    assert f"the PR's head is {sim.checked[:12]}" in result.stdout
    assert git(sim.work, "branch", "--show-current") == BRANCH


def test_finish_refuses_a_local_branch_behind_the_pr_head(sim):
    other = sim.second_clone()
    git(other, "switch", "-q", BRANCH)
    newer = commit(other, "fix pushed from elsewhere", {"fix.txt": "1\n"})
    git(other, "push", "-q", "origin", BRANCH)
    sim.state(pr_head=newer, checks={sim.checked: runs(), newer: runs(["check"], "failure")})
    result = sim.finish()
    assert result.returncode == 4, result.stdout + result.stderr
    assert sim.origin_main() == sim.base
    assert f"the PR's head is {newer[:12]}" in result.stdout


def test_finish_requires_the_checks_of_the_head_commit_itself(sim):
    earlier = sim.base
    failing = runs(PASSING[:3]) + runs(["openmp-contract"], "failure", first_id=10)
    sim.state(pr_head=sim.checked, checks={earlier: runs(), sim.checked: failing})
    result = sim.finish()
    assert result.returncode == 4, result.stdout + result.stderr
    assert "openmp-contract: failure" in result.stdout
    assert sim.origin_main() == sim.base


def test_finish_reports_checks_that_have_not_run(sim):
    sim.state(pr_head=sim.checked, checks={sim.checked: runs(PASSING[:2])})
    result = sim.finish()
    assert result.returncode == 4
    assert "fork-tests: not reported" in result.stdout
    assert "openmp-contract: not reported" in result.stdout


def test_finish_counts_the_newest_run_of_a_rerun_check(sim):
    rerun = runs(["check"], "failure", first_id=1) + runs(PASSING, first_id=5)
    sim.state(pr_head=sim.checked, checks={sim.checked: rerun})
    result = sim.finish()
    assert result.returncode == 0, result.stdout + result.stderr
    assert sim.origin_main() == sim.checked


def test_finish_refuses_when_main_moved(sim):
    other = sim.second_clone()
    commit(other, "fork: change merged into main meanwhile", {"other.txt": "1\n"})
    git(other, "push", "-q", "origin", "main")
    moved = sim.origin_main()
    result = sim.finish()
    assert result.returncode == 4, result.stdout + result.stderr
    assert "origin/main has commits that are not on this branch" in result.stdout
    assert sim.origin_main() == moved


def test_finish_local_fast_forwards_without_github(sim):
    local = commit(sim.work, "rehearsal commit", {"x.txt": "1\n"})
    result = sim.finish("--local")
    assert result.returncode == 0, result.stdout + result.stderr
    assert git(sim.work, "rev-parse", "main") == local
    assert sim.origin_main() == sim.base
    assert not sim.log.exists()
