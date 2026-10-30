"""`fork/dev sync --continue` finds the sync merge from the upstream commit the branch is named
after, not the newest merge on the branch (verification finding TOOL-5).

History used below (upstream line U*, fork line F*):

    U0 -- U1 -- U2                      refs/remotes/upstream/main = U2 (U1 for most tests)
      \\     \\
       F1 ---- M  (sync/<date>-<U1>)     M = git merge --no-ff U1
"""

import sys

import pytest
from toolrepo import commit, git, init_repo, load_tool, write

LEDGER = '[sync]\nlast_synced = ""\n'


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = init_repo(tmp_path / "repo")
    u0 = commit(root, "upstream 0", {"a.txt": "0\n"})
    git(root, "switch", "-q", "-c", "upstream")
    u1 = commit(root, "upstream 1", {"b.txt": "1\n"})
    git(root, "update-ref", "refs/remotes/upstream/main", u1)
    git(root, "switch", "-q", "main")
    commit(root, "fork 1", {"fork/patches/ledger.toml": LEDGER, ".gitignore": "__pycache__/\n"})
    branch = f"sync/20261030-{u1[:8]}"
    git(root, "switch", "-q", "-c", branch)
    git(root, "merge", "-q", "--no-ff", "-m", "sync: merge upstream/main", u1)
    sync = load_tool("sync_upstream", root, monkeypatch)
    return {"root": root, "sync": sync, "u0": u0, "u1": u1, "branch": branch, "merge": git(root, "rev-parse", "HEAD")}


def test_finds_the_merge_of_the_named_upstream_commit(repo):
    assert repo["sync"].sync_merge(repo["branch"]) == (repo["merge"], repo["u1"])


def test_a_merge_of_main_into_the_sync_branch_is_not_the_sync_merge(repo):
    """The finding's scenario: main moved during the sync and was merged into the sync branch."""
    root = repo["root"]
    git(root, "switch", "-q", "main")
    fork_commit = commit(root, "fork: docs tweak on main", {"docs.txt": "tweak\n"})
    git(root, "switch", "-q", repo["branch"])
    git(root, "merge", "-q", "--no-ff", "-m", "Merge main", "main")
    newest = git(root, "rev-list", "--first-parent", "--merges", "-n", "1", "HEAD")
    assert git(root, "rev-parse", f"{newest}^2") == fork_commit  # what --continue used to record
    assert repo["sync"].sync_merge(repo["branch"]) == (repo["merge"], repo["u1"])


def test_an_aborted_merge_does_not_pick_an_earlier_sync_merge_from_main(repo):
    root, sync = repo["root"], repo["sync"]
    git(root, "switch", "-q", "main")
    git(root, "merge", "-q", "--ff-only", repo["branch"])  # the earlier sync, finished
    git(root, "switch", "-q", "upstream")
    u2 = commit(root, "upstream 2", {"c.txt": "2\n"})
    git(root, "update-ref", "refs/remotes/upstream/main", u2)
    branch = f"sync/20261031-{u2[:8]}"
    git(root, "switch", "-q", "-c", branch, "main")  # the merge was aborted: no merge of U2 here
    with pytest.raises(SystemExit, match=f"no merge of upstream {u2[:8]}"):
        sync.sync_merge(branch)


def test_a_merge_of_a_commit_outside_upstream_is_refused(repo):
    root = repo["root"]
    git(root, "switch", "-q", "-c", "side", "main")
    side = commit(root, "fork side branch", {"side.txt": "1\n"})
    branch = f"sync/20261030-{side[:8]}"
    git(root, "switch", "-q", "-c", branch, "main")
    git(root, "merge", "-q", "--no-ff", "-m", "not upstream", side)
    with pytest.raises(SystemExit, match="not an ancestor of upstream/main"):
        repo["sync"].sync_merge(branch)


def test_a_branch_name_without_the_upstream_commit_is_refused(repo):
    with pytest.raises(SystemExit, match="not a sync branch name"):
        repo["sync"].sync_merge("sync/manual-fix")


def test_continue_hands_the_named_merge_to_the_remaining_steps(repo, monkeypatch):
    root, sync = repo["root"], repo["sync"]
    git(root, "switch", "-q", "main")
    commit(root, "fork: change on main", {"docs.txt": "tweak\n"})
    git(root, "switch", "-q", repo["branch"])
    git(root, "merge", "-q", "--no-ff", "-m", "Merge main", "main")
    seen = []
    monkeypatch.setattr(sync, "after_merge", lambda *args: seen.append(args[:3]) or 0)
    monkeypatch.setattr(sys, "argv", ["sync_upstream.py", "--continue", "--fast", "--local"])
    assert sync.main() == 0
    assert seen == [(repo["branch"], repo["merge"], repo["u1"])]


def test_continue_needs_a_clean_tree(repo, monkeypatch):
    write(repo["root"], {"stray.txt": "x\n"})
    monkeypatch.setattr(sys, "argv", ["sync_upstream.py", "--continue"])
    with pytest.raises(SystemExit, match="clean tree"):
        repo["sync"].main()
