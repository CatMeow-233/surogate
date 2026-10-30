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


def test_a_note_is_not_linked_from_itself(repo):
    """A note that names itself (a title) and has a table of contents only links into itself."""
    root, sync = repo
    note = "# RELEASE_v9.md\n\n- [Highlights](#highlights)\n- [again](RELEASE_v9.md#highlights)\n\n## Highlights\n"
    commit(root, "upstream adds a note", {"RELEASE_v9.md": note})
    assert sync.linking_docs("RELEASE_v9.md", "HEAD") == []
    assert sync.keep_reason("RELEASE_v9.md", "HEAD", sync.ledger()) is None


@pytest.mark.parametrize(
    ("line", "target"),
    [
        ("See [main](src/main.py).", "src/main.py"),  # breaks: docs/upstream-notes/src/main.py
        ("See [serve](src/serve/).", "src/serve/"),  # a directory
        ("See [project](README.md).", "README.md"),  # would silently reach docs/upstream-notes/README.md
        ("[main]: src/main.py", "src/main.py"),  # reference link
        ('<img src="src/diagram.png">', "src/diagram.png"),
        ("[up](../outside.md)", "../outside.md"),
    ],
)
def test_a_note_whose_own_links_the_move_would_retarget_stays_at_the_root(repo, line, target):
    """The note's outgoing relative links resolve from its new directory after a move, so they would
    break or reach a different file (review of the TOOL-3 fix)."""
    root, sync = repo
    commit(root, "upstream adds a note with links", {"RELEASE_v9.md": f"# v9\n\n{line}\n"})
    reason = sync.keep_reason("RELEASE_v9.md", "HEAD", sync.ledger())
    assert reason == f"its links would reach other paths from docs/upstream-notes/: {target}"


@pytest.mark.parametrize(
    "line",
    [
        "[Highlights](#highlights)",
        "[self](RELEASE_v9.md#highlights)",
        "[todo](TODO.md)",  # a note the fork moved: the link reaches it from docs/upstream-notes/
        "[project](/README.md)",  # root-relative: the same path from any directory
        "[web](https://example.invalid/src/main.py)",
        "[mail](mailto:dev@example.invalid)",
        "```\n[main](src/main.py)\n```",  # fenced code is not a link
        "`src/main.py`",
    ],
)
def test_links_the_move_keeps_do_not_keep_a_note(tmp_path, monkeypatch, line):
    root = init_repo(tmp_path / "repo")
    commit(
        root,
        "fork",
        {
            "fork/patches/ledger.toml": ledger_text(TODO_RELOCATION),
            "docs/upstream-notes/TODO.md": "# todo\n",
            "README.md": "# Project\n",
        },
    )
    sync = load_tool("sync_upstream", root, monkeypatch)
    commit(root, "upstream adds a note", {"RELEASE_v9.md": f"# v9\n\n{line}\n\n## Highlights\n"})
    assert sync.keep_reason("RELEASE_v9.md", "HEAD", sync.ledger()) is None


def test_the_reason_lists_three_links_and_counts_the_rest(repo):
    root, sync = repo
    links = "".join(f"[f{i}](src/f{i}.py)\n" for i in range(5))
    commit(root, "upstream adds a note with links", {"RELEASE_v9.md": links})
    assert sync.keep_reason("RELEASE_v9.md", "HEAD", sync.ledger()) == (
        "its links would reach other paths from docs/upstream-notes/: src/f0.py, src/f1.py, src/f2.py and 2 more"
    )


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
    assert sync.broken_by_fork(entries, book, {"docs", "docs/upstream-notes", "docs/upstream-notes/TODO.md"}) == [
        "README.md -> TODO.md (TODO.md moved to docs/upstream-notes/TODO.md)",
        "docs/guide.md -> ../RELEASE_v1.4.0.md (RELEASE_v1.4.0.md deleted by the fork)",
        "docs/guide.md -> ../TODO.md#plan (TODO.md moved to docs/upstream-notes/TODO.md)",
    ]


def test_broken_by_fork_names_links_in_a_moved_note_that_worked_at_its_upstream_path(repo):
    """Backstop for the links of a note the fork moved: a new broken link there is the fork's when it
    reaches an existing path from the note's upstream location."""
    _, sync = repo
    book = {
        "relocation": [
            {"from": "TODO.md", "to": "docs/upstream-notes/TODO.md"},
            {"from": "REFACTOR.md", "to": "docs/upstream-notes/REFACTOR.md"},
        ],
        "deletion": [{"path": "RELEASE_v1.4.0.md"}],
    }
    tree = {"src", "src/main.py", "src/serve", "README.md", "docs", "docs/upstream-notes"}
    tree |= {"docs/upstream-notes/TODO.md", "docs/upstream-notes/REFACTOR.md"}
    entries = {
        "docs/upstream-notes/REFACTOR.md\tsrc/main.py",
        "docs/upstream-notes/REFACTOR.md\tsrc/serve/",
        "docs/upstream-notes/REFACTOR.md\tRELEASE_v1.4.0.md",
        "docs/upstream-notes/REFACTOR.md\tgone.py",  # broken at the upstream path too
        "docs/upstream-notes/REFACTOR.md\t#no-such-heading",  # moves with the note
        "docs/upstream-notes/TODO.md\tTODO.md#no-such-heading",  # reaches the note itself
    }
    assert sync.broken_by_fork(entries, book, tree) == [
        "docs/upstream-notes/REFACTOR.md -> RELEASE_v1.4.0.md (RELEASE_v1.4.0.md deleted by the fork)",
        "docs/upstream-notes/REFACTOR.md -> src/main.py (moved from REFACTOR.md, where it reaches src/main.py)",
        "docs/upstream-notes/REFACTOR.md -> src/serve/ (moved from REFACTOR.md, where it reaches src/serve)",
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


def test_a_sync_keeps_a_note_with_links_of_its_own_and_baselines_nothing(refresh_repo):
    """The reviewers' end-to-end case: upstream adds a root note whose own relative links break, or
    reach the fork's docs/upstream-notes/README.md, from docs/upstream-notes/. Before the fix,
    rehome() moved it and refresh() baselined `docs/upstream-notes/RELEASE_v2.md -> src/main.py`."""
    root, sync = refresh_repo
    commit(root, "fork index of the notes", {"docs/upstream-notes/README.md": "# Notes\n"})
    base = git(root, "rev-parse", "HEAD")
    git(root, "switch", "-q", "-c", "upstream")
    note = "# v2\n\n[main.py](src/main.py) and the [project README](README.md).\n"
    upstream = commit(root, "upstream: v2 notes", {"RELEASE_v2.md": note, "src/main.py": "print()\n"})
    git(root, "switch", "-q", "-c", "sync/20261030-00000000", base)
    git(root, "merge", "-q", "--no-ff", "-m", "sync", "upstream")
    sync.rehome(git(root, "rev-parse", "HEAD"), sync.ledger())
    sync.refresh(upstream)
    files = set(git(root, "ls-files").splitlines())
    assert "RELEASE_v2.md" in files and "docs/upstream-notes/RELEASE_v2.md" not in files
    assert [r["from"] for r in sync.ledger()["relocation"]] == ["TODO.md"]
    assert (root / "fork/tools/doc_links_baseline.txt").read_text() == ""
    assert git(root, "log", "-1", "--format=%s") == "sync: refresh ledger and maps"


def test_refresh_refuses_a_link_that_a_moved_note_gains_upstream(refresh_repo, capsys):
    """An upstream edit to an already moved note merges into docs/upstream-notes/; its new relative
    link resolves from there."""
    root, sync = refresh_repo
    commit(root, "upstream file", {"src/main.py": "print()\n"})
    target = commit(root, "upstream links from the note", {"docs/upstream-notes/TODO.md": "# todo\n[m](src/main.py)\n"})
    with pytest.raises(SystemExit) as stop:
        sync.refresh(target)
    assert stop.value.code == 4
    out = capsys.readouterr().out
    assert "docs/upstream-notes/TODO.md -> src/main.py (moved from TODO.md, where it reaches src/main.py)" in out
    assert git(root, "status", "--porcelain") == ""


def test_check_doc_links_imports_without_reading_the_repository(tmp_path):
    """sync_upstream.py imports check_doc_links for its link syntax; the import itself runs no git."""
    code = "import sys; sys.path.insert(0, sys.argv[1]); import check_doc_links as c; print(len(c.TRACKED))"
    copy_tools(tmp_path, "check_doc_links.py")  # tmp_path is not a git repository
    out = subprocess.run(
        ["python3", "-I", "-c", code, str(tmp_path / "fork/tools")], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "0"
