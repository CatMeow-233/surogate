#!/usr/bin/env python3
"""Merge upstream/main into the fork on a sync branch; fork/docs/sync.md describes the procedure.

A sync runs these steps and stops at the first one that needs the owner:
  1. preflight: clean tree on main, equal to origin/main; upstream push URL disabled; rerere on;
     merge.directoryRenames unset; check_divergence --strict passes; no v* tags on origin; the
     upstream release workflows are disabled on the fork
  2. fetch upstream main
  3. report: new commits, conflicts mapped to ledger topics, upstream commits touching ledgered
     files, new root notes, new top-level directories, new workflows (--dry-run stops here)
  4. branch sync/<yyyymmdd>-<sha> and merge with --no-ff (never rebase)
  5. resolve the one known conflict (upstream deleted a relocated note); stop on any other, and
     when an upstream directory rename would leave a fork-owned subdirectory behind
  6. commit "sync: re-home upstream root notes" (see [root_rehome] in the ledger); a note that a
     markdown doc links to, or whose own relative links would reach other paths once moved, stays
     at the root
  7. commit "sync: refresh ledger and maps" ([sync].last_synced, gen_maps, doc-link baseline); stop
     when the fork broke a new broken link: it points at a path the fork moved or deleted, or sits
     in a note the fork moved (broken_by_fork)
  8. fork/dev check, then the patch, experimental, TLC, tooling and CPU test suites
  9. push the branch and open a PR (or update the open one); --finish fast-forwards main to the
     branch's commit once that commit is the PR head and its REQUIRED_CHECKS passed

Usage: fork/dev sync [--dry-run | --continue | --finish] [--to REV] [--local] [--fast]
  --dry-run   steps 1-3 only; changes nothing (main and origin are not required)
  --continue  after committing a conflict resolution or a fix on the sync branch: steps 6-9; the
              sync merge is the merge of the upstream commit the branch is named after
  --finish    on the sync branch, after the PR's required checks pass on its head: fast-forward
              main to that commit and push it; refuses when HEAD is not the pushed PR head
  --to REV    merge REV, an ancestor of the fetched upstream/main, instead of fetching upstream main
  --local     no origin checks, no push, no PR (rehearsals)
  --fast      skip the test suites in step 8 (rehearsals)
Exit status: 0 done, 2 manual conflict, 3 preflight failure, 4 check failure, 1 other errors.
"""

import argparse
import datetime
import json
import re
import subprocess
import sys
import tomllib
from fnmatch import fnmatchcase
from pathlib import Path

# The link syntax of check_doc_links.py, imported from this directory (-I leaves it off sys.path)
# without writing bytecode into the checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
import check_doc_links  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LEDGER = "fork/patches/ledger.toml"
BASELINE = "fork/tools/doc_links_baseline.txt"
GENERATED = ["fork/patches/README.md", "fork/docs/maps"]
REPO = "CatMeow-233/surogate"
PUSH_DISABLED = "DISABLED-see-AGENTS.md"
# Upstream workflows that publish releases or images (jackalope.yml on any push to main that
# touches jackalope/**); the owner disables them on the fork.
UPSTREAM_WORKFLOWS = (".github/workflows/wheel.yml", ".github/workflows/docker.yml", ".github/workflows/jackalope.yml")
# fork-ci jobs that must pass on the sync branch's head commit before --finish; the same list as
# main's branch protection in fork/docs/sync.md, checked here so that --finish also works without it.
REQUIRED_CHECKS = ("check", "experimental", "fork-tests", "openmp-contract")
# sync/<yyyymmdd>-<first 8 hex digits of the merged upstream commit>
BRANCH = re.compile(r"sync/\d{8}-([0-9a-f]{8})")
NOTES_PLAYBOOK = 'fork/docs/sync.md, section "移动过的上游笔记与删除的文件"'


def git(*args, ok=(0,)):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    if result.returncode not in ok:
        sys.exit(f"git {' '.join(args)} failed:\n{result.stderr.strip()}")
    return result.stdout.strip()


def run(*cmd, capture=False):
    """Print a command and run it from the repository root; return its exit status, or the
    CompletedProcess with text output when capture is set."""
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT, capture_output=capture, text=True)
    return result if capture else result.returncode


def is_ancestor(commit, rev):
    return subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, rev]).returncode == 0


def ledger():
    return tomllib.loads((ROOT / LEDGER).read_text(encoding="utf-8"))


def files_at(rev):
    return set(git("ls-tree", "-r", "--name-only", "-z", rev).split("\0")) - {""}


def paths_at(rev):
    """Files and directories at rev: the paths a link can reach."""
    return set(git("ls-tree", "-r", "-t", "--name-only", "-z", rev).split("\0")) - {""}


def label(path, book):
    """The ledger entries a path belongs to, for conflict reports."""
    ids = [t["id"] for t in book.get("topic", []) if path in t.get("files", []) + t.get("fork_files", [])]
    if ids:
        return ", ".join(ids)
    if any(path in (r["from"], r["to"]) for r in book.get("relocation", [])):
        return "[[relocation]]"
    if any(path == d["path"] for d in book.get("deletion", [])):
        return "[[deletion]]"
    return "not in the ledger"


def is_note(path, book):
    return label(path, book) in ("[[relocation]]", "[[deletion]]")


def unmerged(lines):
    """Map each conflicted path to its index stages, from `<mode> <oid> <stage>\\t<path>` lines."""
    stages = {}
    for line in lines:
        info, _, path = line.partition("\t")
        stages.setdefault(path, set()).add(int(info.split()[2]))
    return stages


def upstream_deleted_note(path, stages, book):
    """True for a relocated note that upstream deleted: base and our stage, no upstream stage."""
    return stages == {1, 2} and any(path == r["to"] for r in book.get("relocation", []))


def keep_reason(name, rev, book):
    """Why a new root note stays at the root, or None when it is re-homed."""
    rule = book.get("root_rehome", {})
    if any(fnmatchcase(name, p) for p in rule.get("never", [])):
        return "on the [root_rehome] never list"
    if not any(fnmatchcase(name, p) for p in rule.get("patterns", [])):
        return "matches no [root_rehome] pattern; decide by hand"
    # `**/x` also matches x at the root, as in .gitignore.
    tooling = [
        f
        for f in sorted(files_at(rev))
        if any(fnmatchcase(f, p) or (p.startswith("**/") and fnmatchcase(f, p[3:])) for p in rule["tooling_refs"])
    ]
    refs = git("grep", "-l", "-F", "-e", name, rev, "--", *tooling, ok=(0, 1)).splitlines() if tooling else []
    if refs:
        return "named in " + ", ".join(r.removeprefix(f"{rev}:") for r in refs)
    # Moving a note that a doc links to would break the link (upstream docs are not edited).
    docs = linking_docs(name, rev)
    if docs:
        return "linked from " + ", ".join(docs)
    # So would moving a note whose own relative links then resolve from dest.
    links = retargeted_links(name, rule["dest"] + name, rev, book)
    if links:
        more = f" and {len(links) - 3} more" if len(links) > 3 else ""
        return f"its links would reach other paths from {rule['dest']}: " + ", ".join(links[:3]) + more
    if rule["dest"] + name in files_at("HEAD"):
        return f"{rule['dest']}{name} already exists"
    return None


def linking_docs(name, rev):
    """Markdown files at rev, other than name itself, with a local link to the path name, as
    check_doc_links reads links. A note's links into itself (#anchors) move with it."""
    found = []
    for entry in git("grep", "-l", "-F", "-e", name, rev, "--", "*.md", "*.mdx", ok=(0, 1)).splitlines():
        doc = entry.removeprefix(f"{rev}:")
        if doc == name:
            continue
        text = git("show", f"{rev}:{doc}")
        if any(check_doc_links.resolve(doc, target) == name for _, target in check_doc_links.local_links(text)):
            found.append(doc)
    return found


def retargeted_links(name, dest, rev, book):
    """Link targets in the note name at rev that would reach a different path from dest.

    Paths are compared in the fork's layout, with name already at dest: a link to a note the fork
    moved, or into the note itself, reaches the same file from either place. Any other relative
    link would break, or reach another file (README.md would reach the fork's index of the notes),
    and the fork does not edit upstream notes to repair it.
    """
    moved = {r["from"]: r["to"] for r in book.get("relocation", [])} | {name: dest}
    found = []
    for _, target in check_doc_links.local_links(git("show", f"{rev}:{name}")):
        meant = check_doc_links.resolve(name, target)
        if moved.get(meant, meant) != check_doc_links.resolve(dest, target) and target not in found:
            found.append(target)
    return found


def stranded(base, target, book):
    """Fork-owned paths whose enclosing upstream directory is gone at target.

    git applies an upstream directory rename to files the fork added directly in that directory,
    but not to a subdirectory that only the fork has (surogate/_fork/, docs/upstream-notes/):
    such a path would silently stay at the old location, so it is reported instead.
    """
    old, new = (set(git("ls-tree", "-r", "-d", "--name-only", rev).splitlines()) for rev in (base, target))
    owned = book.get("owned", {})
    found = []
    for path in owned.get("prefixes", []) + owned.get("files", []):
        parts = path.rstrip("/").split("/")[:-1]
        parent = next((d for d in ("/".join(parts[:i]) for i in range(len(parts), 0, -1)) if d in old), None)
        if parent and parent not in new:
            found.append(f"{path}: upstream no longer has {parent}/; move it to the new location by hand")
    return found


def new_root_notes(old, new):
    added = git("diff", "--no-renames", "--name-only", "--diff-filter=A", old, new).splitlines()
    return [f for f in added if "/" not in f and f.endswith(".md")]


def report(target, book):
    """Lines describing the merge of target into HEAD, and the merge's conflicted paths."""
    base = git("merge-base", "HEAD", target)
    if base == target:
        return [f"upstream {target[:12]}: up to date, 0 conflicts"], {}
    count = git("rev-list", "--count", f"HEAD..{target}")
    out = git("merge-tree", "--write-tree", "--messages", "HEAD", target, ok=(0, 1))
    info, _, messages = out.partition("\n\n")
    conflicts = unmerged(info.splitlines()[1:])
    manual = [p for p, s in conflicts.items() if not upstream_deleted_note(p, s, book)]
    lost = stranded(base, target, book)
    lines = [
        f"upstream {target[:12]}: {count} new commits, {len(conflicts)} conflicted paths, "
        f"{len(manual) + len(lost)} to resolve by hand"
    ]
    for path in sorted(conflicts):
        how = "by hand" if path in manual else "automatic: upstream deleted the note"
        lines.append(f"  {path}: {label(path, book)}, {how}")
    lines += [f"  {m}" for m in messages.splitlines() if m.startswith("CONFLICT")]
    lines += [f"  {entry}" for entry in lost]

    groups = [(f"{t['id']} {t['title']}", t.get("files", [])) for t in book.get("topic", [])]
    notes = [r["from"] for r in book.get("relocation", [])] + [d["path"] for d in book.get("deletion", [])]
    groups.append(("relocated or deleted upstream notes", notes))
    touched = []
    for name, paths in groups:
        log = git("log", "--format=%h %as %s", f"HEAD..{target}", "--", *paths) if paths else ""
        if log:
            touched += [f"  {name}:"] + [f"    {entry}" for entry in log.splitlines()]
    lines += ["upstream commits that touch ledgered files:"] + (touched or ["  none"])

    notes = new_root_notes(base, target)
    lines.append("new root notes:" + ("" if notes else " none"))
    for name in notes:
        reason = keep_reason(name, target, book)
        lines.append(f"  {name}: " + (f"kept at the root, {reason}" if reason else "re-homed"))

    for prefix, table in (("csrc/src/", "CSRC_ENGINES"), ("surogate/", "PY_ROLES")):
        old, new = (
            set(git("ls-tree", "-d", "--name-only", rev, "--", prefix).splitlines()) for rev in ("HEAD", target)
        )
        for path in sorted(new - old):
            lines.append(f"new directory {path}: classify it in {table} in fork/tools/gen_maps.py")
    workflows = git("diff", "--no-renames", "--name-only", "--diff-filter=A", base, target, "--", ".github/workflows")
    for path in workflows.splitlines():
        lines.append(f"new upstream workflow {path}: check its triggers; disable it on the fork if it publishes")
    if git("diff", "--name-only", base, target, "--", ".github/workflows/wheel.yml"):
        lines.append(
            "upstream changed .github/workflows/wheel.yml: port the change into fork/native/release/build-wheel.sh"
        )
    return lines, conflicts


def edit_ledger(edit, *args):
    """Apply edit(text, *args) to the ledger's text and check that the result still parses."""
    path = ROOT / LEDGER
    text = edit(path.read_text(encoding="utf-8"), *args)
    tomllib.loads(text)
    path.write_text(text, encoding="utf-8")


def drop_relocation(text, to):
    entry = re.compile(r'\[\[relocation\]\]\nfrom = "[^"\n]*"\nto = "' + re.escape(to) + r'"\n\n?')
    text, n = entry.subn("", text)
    if n != 1:
        sys.exit(f"cannot find the [[relocation]] entry for {to} in {LEDGER}; edit it by hand")
    return text


def add_relocation(text, old, new):
    entry = f'[[relocation]]\nfrom = "{old}"\nto = "{new}"\n'
    entries = list(re.finditer(r'\[\[relocation\]\]\nfrom = "[^"\n]*"\nto = "[^"\n]*"\n', text))
    if not entries:
        return text.rstrip("\n") + "\n\n" + entry
    end = entries[-1].end()
    return text[:end] + "\n" + entry + text[end:]


def set_last_synced(text, line):
    text, n = re.subn(r"^last_synced = .*$", lambda _: line, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"cannot find [sync].last_synced in {LEDGER}")
    return text


def topic_lines(paths, book):
    lines = []
    for t in book.get("topic", []):
        if set(paths) & set(t.get("files", []) + t.get("fork_files", [])):
            lines.append(f"  {t['id']} {t['title']} (switch: {t.get('switch', '')})")
            lines.append(f"     files: {', '.join(t.get('files', []))}")
            lines.append(f"     tests: {', '.join(t.get('tests', []))}")
            lines.append(f"     last conflict: {(t.get('conflict_history') or ['none'])[-1]}")
    return lines


def merge(target, book):
    base = git("merge-base", "HEAD", target)
    count = git("rev-list", "--count", f"HEAD..{target}")
    status = run("git", "merge", "--no-ff", "-m", f"sync: merge upstream/main {target[:12]} ({count} commits)", target)
    conflicts = unmerged(git("ls-files", "-u").splitlines())
    if status != 0 and not conflicts:
        sys.exit("git merge failed; see its output above")
    manual = []
    for path, stages in sorted(conflicts.items()):
        if upstream_deleted_note(path, stages, book):
            run("git", "rm", "-q", "--", path)
            edit_ledger(drop_relocation, path)
            run("git", "add", "--", LEDGER)
        else:
            manual.append(path)
    lost = stranded(base, target, book)
    if manual or lost:
        print("\nTo resolve by hand:")
        print("".join(f"  {path}: {label(path, book)}\n" for path in manual), end="")
        print("".join(f"  {entry}\n" for entry in lost), end="")
        topics = topic_lines(manual, book)
        if topics:
            print("Ledger topics of these files (fork/docs/sync.md has a playbook per topic):")
            print("\n".join(topics))
        if any(is_note(path, book) for path in manual):
            print(f"Relocated or deleted upstream notes: see {NOTES_PLAYBOOK}.")
        print(
            "\nResolve them (rerere may have replayed an earlier resolution; review it), `git add` them and\n"
            "commit (`git commit --no-edit` finishes a conflicted merge), then run `fork/dev sync --continue`."
        )
        sys.exit(2)
    if conflicts:
        run("git", "commit", "-q", "--no-edit")


def rehome(merge_commit, book):
    rule = book.get("root_rehome", {})
    moved, lines = [], []
    root = files_at("HEAD")
    for name in new_root_notes(f"{merge_commit}^1", merge_commit):
        if name not in root:
            continue  # re-homed by an earlier run
        reason = keep_reason(name, "HEAD", book)
        if reason:
            lines.append(f"  {name}: kept at the root, {reason}")
            continue
        dest = rule["dest"] + name
        (ROOT / rule["dest"]).mkdir(parents=True, exist_ok=True)
        if run("git", "mv", "--", name, dest) != 0:
            sys.exit(f"git mv {name} {dest} failed")
        edit_ledger(add_relocation, name, dest)
        moved.append(f"{name} -> {dest}")
    if moved:
        run("git", "add", "--", LEDGER)
        body = "".join(f"{m}\n" for m in moved)
        run("git", "commit", "-q", "-m", "sync: re-home upstream root notes", "-m", body.rstrip("\n"))
        lines = [f"  {m}" for m in moved] + lines
    print("new root notes:\n" + "\n".join(lines) if lines else "new root notes: none")


def broken_by_fork(entries, book, tree):
    """Baseline entries (doc<TAB>target) that the fork broke, so they are not upstream breakage to
    accept. In upstream's layout (a moved doc at its `from` path) the link reaches an upstream path
    that the fork deleted, or one the fork has (at its new path, if it moved it) and the link no
    longer reaches: a link to a moved note, or a relative link in a moved note. tree is the set of
    files and directories in the fork (paths_at)."""
    moved = {r["from"]: r["to"] for r in book.get("relocation", [])}
    origin = {to: old for old, to in moved.items()}
    deleted = {d["path"] for d in book.get("deletion", [])}
    found = []
    for entry in sorted(entries):
        doc, _, target = entry.partition("\t")
        reached = check_doc_links.resolve(doc, target)
        meant = check_doc_links.resolve(origin.get(doc, doc), target)
        if meant in deleted:
            found.append(f"{doc} -> {target} ({meant} deleted by the fork)")
        elif moved.get(meant, meant) != reached and moved.get(meant, meant) in tree:
            if doc in origin:
                found.append(f"{doc} -> {target} (moved from {origin[doc]}, where it reaches {meant})")
            else:
                found.append(f"{doc} -> {target} ({meant} moved to {moved[meant]})")
    return found


def refresh(target):
    date = git("log", "-1", "--format=%cs", target)
    edit_ledger(set_last_synced, f'last_synced = "{target}"   # {date}')
    run("python3", "-I", "fork/tools/gen_maps.py", "--write")  # unclassified paths fail the checks below
    before = set((ROOT / BASELINE).read_text(encoding="utf-8").splitlines())
    run("python3", "-I", "fork/tools/check_doc_links.py", "--update-baseline")
    after = set((ROOT / BASELINE).read_text(encoding="utf-8").splitlines())
    ours = broken_by_fork(after - before, ledger(), paths_at("HEAD"))  # as rehome() left them
    if ours:
        run("git", "checkout", "HEAD", "--", LEDGER, BASELINE, *GENERATED)
        print("\nThe fork's moves or deletions broke these new broken links in upstream docs; the refresh does")
        print("not add them to the doc-link baseline:")
        print("".join(f"  {entry}\n" for entry in ours), end="")
        print(
            "Move the note back to the root (git mv, and delete its [[relocation]] from the ledger), or, to\n"
            f"accept the broken link, add the line to {BASELINE} yourself. Commit, then run\n"
            f"`fork/dev sync --continue`. See {NOTES_PLAYBOOK}."
        )
        sys.exit(4)
    run("git", "add", "--", LEDGER, BASELINE, *GENERATED)
    if git("diff", "--cached", "--name-only") == "":
        return
    body = [f"[sync].last_synced is now {target} ({date}); maps regenerated."]
    for title, entries in (
        ("New broken links in upstream docs, added to the baseline:", after - before),
        ("Links fixed upstream, removed from the baseline:", before - after),
    ):
        if entries:
            body += ["", title] + [f"  {e.replace(chr(9), ' -> ')}" for e in sorted(entries)]
    run("git", "commit", "-q", "-m", "sync: refresh ledger and maps", "-m", "\n".join(body))


def checks(fast):
    suites = [["fork/dev", "check"]]
    if not fast:
        suites += [["fork/dev", "test", suite] for suite in ("patches", "experimental", "tlc", "tools", "cpu")]
    failed = [" ".join(cmd) for cmd in suites if run(*cmd) != 0]
    if failed:
        print(f"\nFailed: {', '.join(failed)}.")
        print("Fix the cause on this branch and commit. An upstream test failure that also fails at")
        print("upstream/main goes into fork/tools/known_failures.txt. Then run `fork/dev sync --continue`.")
        sys.exit(4)


def report_file(name="fork-sync-report.md"):
    return ROOT / git("rev-parse", "--git-path", name)


def sync_merge(branch):
    """The sync merge on HEAD and the upstream commit it merged.

    The branch name records the upstream commit (BRANCH). The sync merge is the newest first-parent
    merge of HEAD whose second parent has that prefix and is an ancestor of upstream/main, so a
    merge of main into the sync branch, or an earlier sync merge in main's history, is never taken
    for it.
    """
    name = BRANCH.fullmatch(branch)
    if not name:
        sys.exit(f"{branch}: not a sync branch name (sync/<yyyymmdd>-<8 hex digits of the upstream commit>)")
    for line in git("log", "--first-parent", "--merges", "--format=%H %P", "HEAD").splitlines():
        commit, *parents = line.split()
        if len(parents) == 2 and parents[1].startswith(name.group(1)):
            if not is_ancestor(parents[1], "refs/remotes/upstream/main"):
                sys.exit(
                    f"{commit[:12]} merges {parents[1][:12]}, which is not an ancestor of upstream/main; "
                    "fetch upstream (git fetch --no-tags upstream main:refs/remotes/upstream/main) and retry"
                )
            return commit, parents[1]
    sys.exit(
        f"no merge of upstream {name.group(1)} on {branch}: commit the merge (`git commit --no-edit` "
        "after resolving), or start the sync again from main"
    )


def after_merge(branch, merge_commit, target, args, book):
    rehome(merge_commit, book)
    refresh(target)
    checks(args.fast)
    if args.local:
        print(f"{branch}: checks passed; not pushed (--local). Next: fork/dev sync --finish --local")
        return 0
    if run("git", "push", "-u", "origin", branch) != 0:
        sys.exit("push failed")
    state = run("gh", "pr", "view", branch, "-R", REPO, "--json", "state", "--jq", ".state", capture=True)
    if state.stdout.strip() == "OPEN":
        print(f"The PR for {branch} is already open; the push above updated it.")
    else:
        # The PR body is the step-3 report plus the branch's commits, rebuilt on every run.
        step3 = report_file()
        commits = git("log", "--first-parent", "--format=- %h %s", f"{merge_commit}^1..HEAD")
        body = report_file("fork-sync-pr.md")
        body.write_text(
            (step3.read_text(encoding="utf-8") if step3.exists() else "") + "\nCommits:\n" + commits + "\n",
            encoding="utf-8",
        )
        create = ["gh", "pr", "create", "-R", REPO, "--base", "main", "--head", branch]
        if run(*create, "--title", f"sync: upstream {target[:12]}", "--body-file", str(body)):
            sys.exit("gh pr create failed; open the PR by hand")
    print(f"When the PR's checks {', '.join(REQUIRED_CHECKS)} pass: fork/dev sync --finish")
    return 0


def preflight(args):
    problems = []
    if git("status", "--porcelain"):
        problems.append("the working tree is not clean")
    if not args.dry_run and git("branch", "--show-current") != "main":
        problems.append("the current branch is not main")
    if git("remote", "get-url", "--push", "upstream", ok=(0, 2)) != PUSH_DISABLED:
        problems.append(f"the upstream push URL is not {PUSH_DISABLED}; run fork/dev setup")
    if git("config", "--type=bool", "--get", "rerere.enabled", ok=(0, 1)) != "true":
        problems.append("rerere.enabled is not true; run fork/dev setup")
    if git("config", "--get", "merge.directoryRenames", ok=(0, 1)) not in ("", "conflict"):
        problems.append("merge.directoryRenames is set; unset it so directory renames stay conflicts")
    if not (args.dry_run or args.local):
        run("git", "fetch", "--no-tags", "origin", "main")
        if git("rev-parse", "HEAD") != git("rev-parse", "refs/remotes/origin/main"):
            problems.append("main differs from origin/main; push or reset it first")
        if git("ls-remote", "--tags", "origin", "v*", "jackalope-v*"):
            problems.append("origin has v* or jackalope-v* tags, which start upstream release workflows; delete them")
        jq = '.workflows[] | "\\(.path)\\t\\(.state)"'
        listing = run("gh", "api", f"repos/{REPO}/actions/workflows", "--jq", jq, capture=True)
        states = dict(line.split("\t", 1) for line in listing.stdout.splitlines() if "\t" in line)
        active = [path for path in UPSTREAM_WORKFLOWS if states.get(path) == "active"]
        if listing.returncode != 0:
            problems.append(f"cannot read the workflow states of {REPO} with gh: {listing.stderr.strip()}")
        elif active:
            problems.append(
                f"upstream workflows still enabled on {REPO}: {', '.join(active)}; "
                "run the owner actions in fork/docs/sync.md"
            )
    if run("python3", "-I", "fork/tools/check_divergence.py", "--strict") != 0:
        problems.append("check_divergence --strict fails")
    if problems:
        print("Preflight failed:\n" + "".join(f"  {p}\n" for p in problems), end="")
        sys.exit(3)


def checks_passed(sha):
    """True when every job in REQUIRED_CHECKS has a successful check run on commit sha; works
    without branch protection. A re-run adds a check run, so the newest one of each name counts."""
    result = run("gh", "api", f"repos/{REPO}/commits/{sha}/check-runs?per_page=100", capture=True)
    try:
        runs = json.loads(result.stdout)["check_runs"]
        newest = {}
        for check in sorted(runs, key=lambda check: check["id"]):
            newest[check["name"]] = check["conclusion"] if check["status"] == "completed" else check["status"]
    except (ValueError, KeyError, TypeError):
        print((result.stderr or result.stdout).strip() or "gh api returned no check runs")
        return False
    pending = [
        f"{name}: {newest.get(name, 'not reported')}" for name in REQUIRED_CHECKS if newest.get(name) != "success"
    ]
    print("".join(f"  {line}\n" for line in pending), end="")
    return not pending


def checked_head(branch, head):
    """Problems that stop --finish from pushing head to main: head must be what CI checked and the
    PR shows, that is origin/<branch> and the PR's head commit, and must contain origin/main."""
    fetch = ["git", "fetch", "--no-tags", "origin", "+refs/heads/main:refs/remotes/origin/main"]
    if run(*fetch, f"+refs/heads/{branch}:refs/remotes/origin/{branch}") != 0:
        return [f"cannot fetch main and {branch} from origin; push the branch with `fork/dev sync --continue`"]
    problems = []
    remote = git("rev-parse", f"refs/remotes/origin/{branch}")
    if remote != head:
        problems.append(
            f"HEAD {head[:12]} is not origin/{branch} {remote[:12]}: CI checks only pushed commits. Push "
            "local commits with `fork/dev sync --continue`, or pull the remote ones, and wait for the checks"
        )
    pr = run("gh", "pr", "view", branch, "-R", REPO, "--json", "headRefOid", "--jq", ".headRefOid", capture=True)
    if pr.returncode != 0:
        problems.append(f"cannot read the PR for {branch}: {pr.stderr.strip()}")
    elif pr.stdout.strip() != head:
        problems.append(f"the PR's head is {pr.stdout.strip()[:12] or 'unknown'}, not HEAD {head[:12]}")
    if not is_ancestor("refs/remotes/origin/main", head):
        problems.append(
            "origin/main has commits that are not on this branch: `git merge origin/main`, then "
            "`fork/dev sync --continue`, and wait for the checks"
        )
    return problems


def finish(local):
    branch = git("branch", "--show-current")
    if not branch.startswith("sync/") or git("status", "--porcelain"):
        sys.exit("--finish runs on a clean sync/* branch")
    head = git("rev-parse", "HEAD")
    if not local:
        problems = checked_head(branch, head)
        if problems:
            print("--finish: main is not updated:\n" + "".join(f"  {p}\n" for p in problems), end="")
            return 4
        if not checks_passed(head):
            print(f"The required checks have not all passed on {head[:12]}.")
            return 4
    # The commit named, not the branch: the branch may move after the checks above.
    for cmd in [["git", "switch", "main"], ["git", "merge", "--ff-only", head]] + (
        [] if local else [["git", "push", "origin", f"{head}:refs/heads/main"]]
    ):
        if run(*cmd) != 0:
            sys.exit(f"{' '.join(cmd)} failed")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="report only")
    mode.add_argument("--continue", dest="resume", action="store_true", help="resume after a commit on the sync branch")
    mode.add_argument("--finish", action="store_true", help="fast-forward main to the sync branch")
    parser.add_argument("--to", metavar="REV", help="merge this fetched upstream commit instead of upstream main")
    parser.add_argument("--local", action="store_true", help="no origin checks, push or PR")
    parser.add_argument("--fast", action="store_true", help="skip the test suites")
    args = parser.parse_args()
    book = ledger()

    if args.finish:
        return finish(args.local)
    if args.resume:
        branch = git("branch", "--show-current")
        if not branch.startswith("sync/") or git("status", "--porcelain") or git("ls-files", "-u"):
            sys.exit("--continue runs on a sync/* branch with the merge committed and a clean tree")
        return after_merge(branch, *sync_merge(branch), args, book)

    preflight(args)
    if args.to:
        target = git("rev-parse", "--verify", f"{args.to}^{{commit}}")
        if not is_ancestor(target, "refs/remotes/upstream/main"):
            print(f"--to {args.to}: not an ancestor of upstream/main; name an already fetched upstream commit")
            return 3
    else:
        if run("git", "fetch", "--no-tags", "upstream", "main:refs/remotes/upstream/main") != 0:
            sys.exit("fetching upstream main failed")
        target = git("rev-parse", "refs/remotes/upstream/main")
    lines, _ = report(target, book)
    print("\n".join(lines))
    if args.dry_run or lines[0].endswith("up to date, 0 conflicts"):
        return 0
    report_file().write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    branch = f"sync/{datetime.date.today():%Y%m%d}-{target[:8]}"
    if run("git", "switch", "-c", branch) != 0:
        sys.exit(f"cannot create {branch}")
    merge(target, book)
    return after_merge(branch, *sync_merge(branch), args, book)


if __name__ == "__main__":
    sys.exit(main())
