#!/usr/bin/env python3
"""Check that every difference from upstream is recorded in the ledger.

Compares `git diff -M --name-status <base>`, where <base> is
`git merge-base HEAD upstream/main`, with fork/patches/ledger.toml. The diff is
taken against the working tree, so staged and unstaged edits to tracked files
count before they are committed; untracked files are not seen until `git add`.

  A  must be under an [owned] prefix or listed in [owned].files
  M  must be listed in the files of some [[topic]]
  D  must be listed in [[deletion]]
  R  must match a [[relocation]] from/to pair at 100% similarity

Any other change fails. The check is per file, so a file whose whole diff is
fixed (a hook of a few lines) also has an [[exact_diff]] entry: its
`git diff -U0 <base>` changed lines must equal the entry's lines, in order, so
losing or adding a line fails although the file stays M. [sync].last_synced
must be an ancestor of upstream/main (an upstream commit); when it differs
from <base> a warning is printed.

--strict also fails on stale ledger entries: a topic file that no longer
differs from upstream, an owned path, topic fork file or test that is not
tracked, and a relocation or deletion that is not in the diff.

Usage: python3 -I fork/tools/check_divergence.py [--strict]
Exit status: 0 when everything is recorded, 1 otherwise.
"""

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "fork" / "patches" / "ledger.toml"


def git(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def is_ancestor(commit, rev):
    result = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, rev], capture_output=True)
    return result.returncode == 0


def changed_lines(base, path):
    """The +/- lines of `git diff -U0 base -- path` (base against the working tree), in order."""
    lines, in_hunk = [], False
    for line in git("diff", "-U0", "--no-color", "--no-ext-diff", "--no-renames", base, "--", path).split("\n"):
        if line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line[:1] in ("+", "-"):
            lines.append(line)
    return lines


def exact_diff_errors(entries, topics, base):
    """Problems with the [[exact_diff]] entries: each names a file of its topic, and the file's
    changed lines must be exactly the entry's lines."""
    errors = []
    for entry in entries:
        path, expected = entry["path"], entry["lines"]
        topic = next((t for t in topics if t["id"] == entry.get("topic")), None)
        if topic is None or path not in topic.get("files", []):
            errors.append(f"[[exact_diff]] {path}: topic {entry.get('topic')} does not list it in files")
        actual = changed_lines(base, path)
        if actual == expected:
            continue
        missing = [line for line in expected if line not in actual]
        extra = [line for line in actual if line not in expected]
        detail = "".join(f"\n      missing  {line}" for line in missing)
        detail += "".join(f"\n      extra    {line}" for line in extra)
        errors.append(
            f"M {path}: the diff from upstream is not exactly the [[exact_diff]] lines of {entry.get('topic')}"
            + (detail or " (same lines, different order)")
        )
    return errors


def diff_entries(base):
    """Yield (status, path, new_path) from the -z name-status diff of base against the working tree."""
    fields = git("diff", "-M", "--name-status", "-z", base).split("\0")[:-1]
    i = 0
    while i < len(fields):
        status = fields[i]
        if status[0] in "RC":
            yield status, fields[i + 1], fields[i + 2]
            i += 3
        else:
            yield status, fields[i + 1], None
            i += 2


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true", help="also fail on stale ledger entries")
    args = parser.parse_args()

    ledger = tomllib.loads(LEDGER.read_text(encoding="utf-8"))
    owned = ledger.get("owned", {})
    prefixes = tuple(owned.get("prefixes", []))
    owned_files = set(owned.get("files", []))
    topics = ledger.get("topic", [])
    topic_files = {path for topic in topics for path in topic.get("files", [])}
    relocations = {(r["from"], r["to"]) for r in ledger.get("relocation", [])}
    deletions = {d["path"] for d in ledger.get("deletion", [])}

    base = git("merge-base", "HEAD", "upstream/main").strip()
    errors = []
    added, modified, deleted, renamed = [], set(), set(), set()
    for status, path, new_path in diff_entries(base):
        kind = status[0]
        if kind == "A" and (path in owned_files or path.startswith(prefixes)):
            added.append(path)
        elif kind == "A":
            errors.append(f"A {path}: not owned; put it under fork/ or list it in [owned]")
        elif kind == "M" and path in topic_files:
            modified.add(path)
        elif kind == "M":
            errors.append(f"M {path}: upstream file edited outside the ledger; add it to a [[topic]] or revert it")
        elif kind == "D" and path in deletions:
            deleted.add(path)
        elif kind == "D":
            errors.append(f"D {path}: upstream file deleted but not listed in [[deletion]]")
        elif status == "R100" and (path, new_path) in relocations:
            renamed.add((path, new_path))
        elif kind == "R":
            errors.append(f"{status} {path} -> {new_path}: not a recorded 100% [[relocation]]")
        else:
            errors.append(f"{status} {path}: unsupported change")
    errors += exact_diff_errors(ledger.get("exact_diff", []), topics, base)

    if args.strict:
        tracked = set(git("ls-files", "-z").split("\0"))
        for topic in topics:
            for path in topic.get("files", []):
                if path not in modified:
                    errors.append(f"stale: {topic['id']} lists {path}, which is not modified (M) in the diff")
            for path in dict.fromkeys(topic.get("fork_files", []) + topic.get("tests", [])):
                if path not in tracked:
                    errors.append(f"stale: {topic['id']} lists {path}, which does not exist")
        for prefix in prefixes:
            if not any(path.startswith(prefix) for path in tracked):
                errors.append(f"stale: [owned] prefix {prefix} matches no file")
        for path in sorted(owned_files - tracked):
            errors.append(f"stale: [owned] file {path} does not exist")
        for old, new in sorted(relocations - renamed):
            errors.append(f"stale: relocation {old} -> {new} is not an R100 rename in the diff")
        for path in sorted(deletions - deleted):
            errors.append(f"stale: deletion {path} is not a deletion in the diff")

    last_synced = ledger.get("sync", {}).get("last_synced", "")
    if last_synced and not is_ancestor(last_synced, "upstream/main"):
        errors.append(f"[sync].last_synced {last_synced} is not an upstream commit (not an ancestor of upstream/main)")
    elif last_synced != base:
        print(f"warning: [sync].last_synced is {last_synced or 'unset'}, merge base is {base}")

    print(
        f"check_divergence: base {base[:8]}: {len(modified)} M ledgered, {len(added)} A owned, "
        f"{len(renamed)} R relocated, {len(deleted)} D deleted"
    )
    for error in errors:
        print(f"  {error}")
    if errors:
        print(f"FAIL: {len(errors)} problem(s); see fork/patches/ledger.toml")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
