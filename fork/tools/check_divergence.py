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

Any other change fails. --strict also fails on stale ledger entries: a topic
file that no longer differs from upstream, an owned path, topic fork file or
test that is not tracked, and a relocation or deletion that is not in the
diff.

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
    if last_synced != base:
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
