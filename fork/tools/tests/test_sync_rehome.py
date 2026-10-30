"""Re-homing a new root note never breaks an upstream doc link, and the sync refresh does not put a
link broken by a fork move into the doc-link baseline (verification finding TOOL-3)."""

import subprocess

import pytest
from toolrepo import commit, copy_tools, git, init_repo, load_tool

LEDGER = """[sync]
last_synced = ""   # unset

[root_rehome]
dest = "docs/upstream-notes/"
patterns = ["TODO*.md", "RELEASE_*.md"]
never = ["README*"]
tooling_refs = ["Makefile", ".github/**"]
{relocations}"""

TODO_RELOCATION = """
[[relocation]]
from = "TODO.md"
to = "docs/upstream-notes/TODO.md"
"""


def ledger_text(relocations=""):
    return LEDGER.format(relocations=relocations)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = init_repo(tmp_path / "repo")
    commit(
        root,
        "fork",
        {
            ".gitignore": "__pycache__/\n",
            "fork/patches/ledger.toml": ledger_text(),
            "README.md": "# Project\n",
            "Makefile": "all:\n\techo build\n",
        },
    )
    sync = load_tool("sync_upstream", root, monkeypatch)
    return root, sync


@pytest.mark.parametrize(
    ("doc", "line"),
    [
        ("README.md", "See the [v9 notes](RELEASE_v9.md)."),
        ("README.md", "See the [v9 notes](./RELEASE_v9.md#highlights)."),
        ("docs/guide.md", "Details: [notes](../RELEASE_v9.md)"),
        ("docs/guide.md", "[notes]: /RELEASE_v9.md"),
        ("docs/index.mdx", '<a href="../RELEASE_v9.md">notes</a>'),
    ],
)
def test_a_linked_note_stays_at_the_root(repo, doc, line):
    root, sync = repo
    commit(root, "upstream adds a linked note", {"RELEASE_v9.md": "# v9\n", doc: f"{line}\n"})
    reason = sync.keep_reason("RELEASE_v9.md", "HEAD", sync.ledger())
    assert reason == f"linked from {doc}"


@pytest.mark.parametrize(
    "text",
    [
        "The v9 handoff is in RELEASE_v9.md.\n",  # bare name: code and prose cite notes this way
        "Read `RELEASE_v9.md`.\n",
        "```\n[notes](RELEASE_v9.md)\n```\n",  # fenced code is not a link
        "[elsewhere](docs/RELEASE_v9.md)\n",  # a different path with the same name
        "[web](https://example.invalid/RELEASE_v9.md)\n",
    ],
)
def test_a_note_that_nothing_links_to_is_rehomed(repo, text):
    root, sync = repo
    commit(root, "upstream adds a note", {"RELEASE_v9.md": "# v9\n", "README.md": text})
    assert sync.keep_reason("RELEASE_v9.md", "HEAD", sync.ledger()) is None


def test_the_report_reads_links_at_the_merged_commit(repo):
    root, sync = repo
    plain = commit(root, "note only", {"RELEASE_v9.md": "# v9\n"})
    linked = commit(root, "link it", {"README.md": "[v9](RELEASE_v9.md)\n"})
    git(root, "switch", "-q", "-c", "elsewhere", plain)
    assert sync.keep_reason("RELEASE_v9.md", plain, sync.ledger()) is None
    assert sync.keep_reason("RELEASE_v9.md", linked, sync.ledger()) == "linked from README.md"


def test_rehome_keeps_the_linked_note_and_moves_the_other(repo):
    """The finding's simulation: upstream adds RELEASE_v1.6.0.md and a README link to it."""
    root, sync = repo
    base = git(root, "rev-parse", "HEAD")
    git(root, "switch", "-q", "-c", "upstream")
    commit(
        root,
        "upstream: release notes",
        {
            "RELEASE_v1.6.0.md": "# v1.6.0\n",
            "TODO-cache.md": "# todo\n",
            "README.md": "# Project\n\nSee the [v1.6.0 release notes](RELEASE_v1.6.0.md).\n",
        },
    )
    git(root, "switch", "-q", "-c", "sync/20261030-00000000", base)
    git(root, "merge", "-q", "--no-ff", "-m", "sync", "upstream")
    sync.rehome(git(root, "rev-parse", "HEAD"), sync.ledger())
    files = set(git(root, "ls-files").splitlines())
    assert "RELEASE_v1.6.0.md" in files
    assert "docs/upstream-notes/TODO-cache.md" in files and "TODO-cache.md" not in files
    relocations = [r["from"] for r in sync.ledger().get("relocation", [])]
    assert relocations == ["TODO-cache.md"]
    assert git(root, "log", "-1", "--format=%s") == "sync: re-home upstream root notes"


def test_broken_by_fork_names_links_to_moved_or_deleted_paths(repo):
    _, sync = repo
    book = {
        "relocation": [{"from": "TODO.md", "to": "docs/upstream-notes/TODO.md"}],
        "deletion": [{"path": "RELEASE_v1.4.0.md"}],
    }
    entries = {
        "README.md\tTODO.md",
        "docs/guide.md\t../TODO.md#plan",
        "docs/guide.md\t../RELEASE_v1.4.0.md",
        "docs/guide.md\t../index.mdx",  # upstream's own breakage
    }
    assert sync.broken_by_fork(entries, book) == [
        "README.md -> TODO.md (TODO.md moved to docs/upstream-notes/TODO.md)",
        "docs/guide.md -> ../RELEASE_v1.4.0.md (RELEASE_v1.4.0.md deleted by the fork)",
        "docs/guide.md -> ../TODO.md#plan (TODO.md moved to docs/upstream-notes/TODO.md)",
    ]


@pytest.fixture
def refresh_repo(tmp_path, monkeypatch):
    """A repository where refresh() runs: the real check_doc_links.py and a no-op gen_maps.py."""
    root = init_repo(tmp_path / "refresh")
    copy_tools(root, "check_doc_links.py")
    commit(
        root,
        "fork",
        {
            ".gitignore": "__pycache__/\n",
            "fork/patches/ledger.toml": ledger_text(TODO_RELOCATION),
            "fork/patches/README.md": "ledger\n",
            "fork/docs/maps/docs.md": "maps\n",
            "fork/tools/gen_maps.py": "",
            "fork/tools/doc_links_baseline.txt": "",
            "docs/upstream-notes/TODO.md": "# todo\n",
            "README.md": "# Project\n",
        },
    )
    sync = load_tool("sync_upstream", root, monkeypatch)
    return root, sync


def test_refresh_refuses_to_baseline_a_link_the_fork_broke(refresh_repo, capsys):
    root, sync = refresh_repo
    target = commit(root, "upstream links the moved note", {"README.md": "# Project\n\n[todo](TODO.md)\n"})
    with pytest.raises(SystemExit) as stop:
        sync.refresh(target)
    assert stop.value.code == 4
    out = capsys.readouterr().out
    assert "README.md -> TODO.md (TODO.md moved to docs/upstream-notes/TODO.md)" in out
    assert git(root, "rev-parse", "HEAD") == target  # nothing committed
    assert git(root, "status", "--porcelain") == ""  # ledger and baseline restored
    assert (root / "fork/tools/doc_links_baseline.txt").read_text() == ""


def test_refresh_still_baselines_upstream_breakage(refresh_repo):
    root, sync = refresh_repo
    target = commit(root, "upstream adds a dead link", {"README.md": "# Project\n\n[gone](missing.md)\n"})
    sync.refresh(target)
    assert (root / "fork/tools/doc_links_baseline.txt").read_text() == "README.md\tmissing.md\n"
    assert git(root, "log", "-1", "--format=%s") == "sync: refresh ledger and maps"
    assert "README.md -> missing.md" in git(root, "log", "-1", "--format=%b")
    assert f'last_synced = "{target}"' in (root / "fork/patches/ledger.toml").read_text()


def test_refresh_accepts_a_fork_broken_link_the_maintainer_baselined(refresh_repo):
    root, sync = refresh_repo
    target = commit(
        root,
        "maintainer accepts the broken link",
        {"README.md": "# Project\n\n[todo](TODO.md)\n", "fork/tools/doc_links_baseline.txt": "README.md\tTODO.md\n"},
    )
    sync.refresh(target)
    assert git(root, "log", "-1", "--format=%s") == "sync: refresh ledger and maps"


def test_check_doc_links_imports_without_reading_the_repository(tmp_path):
    """sync_upstream.py imports check_doc_links for its link syntax; the import itself runs no git."""
    code = "import sys; sys.path.insert(0, sys.argv[1]); import check_doc_links as c; print(len(c.TRACKED))"
    copy_tools(tmp_path, "check_doc_links.py")  # tmp_path is not a git repository
    out = subprocess.run(
        ["python3", "-I", "-c", code, str(tmp_path / "fork/tools")], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "0"
