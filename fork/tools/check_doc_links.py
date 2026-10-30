#!/usr/bin/env python3
"""Check local links in the repository's markdown files.

Every tracked .md/.mdx file outside third_party is read. Markdown, reference and HTML links
outside fenced code must point at a tracked file or directory, and an #anchor on a .md target
must match one of its headings.

Fork-owned docs (fork/**, AGENTS.md, docs/experimental/**, docs/upstream-notes/README.md) are
checked strictly: every broken link fails, and so does a repository path that does not exist,
whether it is written in backticks or used in a fenced code block. A path counts as a
repository path when its first component is a tracked top-level entry or it resolves next to
the doc; git-ignored paths (build outputs, venvs) are allowed. In `path:N` only the path is
checked: line numbers drift with every upstream edit, so fork docs name symbols instead, and a
doc that keeps line numbers states the upstream commit they refer to.

Upstream docs fail only on broken links that are not listed in fork/tools/doc_links_baseline.txt.
A sync refresh commit runs --update-baseline and lists the new entries in its message; it refuses
an entry that the fork's moves or deletions broke (broken_by_fork in fork/tools/sync_upstream.py).
local_links() and resolve() are the link syntax that sync_upstream.py also uses.

Usage: python3 -I fork/tools/check_doc_links.py [--update-baseline]
Exit status: 0 when nothing new is broken, 1 otherwise.
"""

import argparse
import fnmatch
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "fork" / "tools" / "doc_links_baseline.txt"
STRICT = ("fork/*", "AGENTS.md", "docs/experimental/*", "docs/upstream-notes/README.md")

LINK = re.compile(r"!?\[(?:[^\]\[]|\[[^\]]*\])*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
REF_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)")
HTML_LINK = re.compile(r"(?:href|src)=\"([^\"]+)\"")
TICK = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"^\s*(```|~~~)")
FENCED_PATH = re.compile(
    r"(?<![\w./$-])((?:\.\./|\./)?[A-Za-z0-9_][A-Za-z0-9_.-]*(?:/[A-Za-z0-9_.-]+)+"
    r"\.(?:py|sh|yaml|yml|json|jsonl|toml|cu|cpp|h|md|txt|cmake))\b"
)
FENCED_MODULE = re.compile(r"python3?\s+(?:-\w+\s+)*-m\s+(surogate[\w.]*)")
LINE_REF = re.compile(r":(\d+)(?:-(\d+))?(?:,\d+)*$")


def git_lines(*args):
    out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout
    return [line for line in out.split("\n") if line]


# The tracked tree, filled in by load_tree() so that importing this module runs nothing.
TRACKED, DIRS, TOP = set(), set(), set()
_anchors = {}


def load_tree():
    TRACKED.update(git_lines("ls-files"))
    DIRS.update("/".join(p.split("/")[:i]) for p in TRACKED for i in range(1, p.count("/") + 1))
    TOP.update(p.split("/")[0] for p in TRACKED)


def resolve(doc, target):
    """The repository path a local link target in doc points at (the doc itself for `#anchor`)."""
    path = target.partition("#")[0]
    if not path:
        return doc
    base = "" if path.startswith("/") else os.path.dirname(doc)
    return os.path.normpath(os.path.join(base, path.lstrip("/")))


def local_links(text):
    """Yield (line number, target) for each markdown, reference or HTML link outside fenced code
    whose target has no URL scheme (http:, mailto:, ...)."""
    fenced = False
    for number, line in enumerate(text.split("\n"), 1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            continue
        for regex in (LINK, REF_LINK, HTML_LINK):
            for m in regex.finditer(line):
                if not re.match(r"^[a-z][a-z0-9+.-]*:", m.group(1), re.I):
                    yield number, m.group(1)


def exists(path):
    return path in TRACKED or path in DIRS or path == "."


def slug(heading):
    text = re.sub(r"<[^>]+>", "", heading.strip().lower()).replace("`", "")
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def anchors(path):
    if path not in _anchors:
        found, seen, fenced = set(), Counter(), False
        text = (ROOT / path).read_text(encoding="utf-8", errors="replace")
        for line in text.split("\n"):
            if FENCE.match(line):
                fenced = not fenced
            elif not fenced and (m := re.match(r"^#{1,6}\s+(.*?)\s*#*\s*$", line)):
                s = slug(m.group(1))
                found.add(s if seen[s] == 0 else f"{s}-{seen[s]}")
                seen[s] += 1
        found.update(re.findall(r"<a\s+(?:name|id)=\"([^\"]+)\"", text))
        _anchors[path] = found
    return _anchors[path]


def check_link(doc, target):
    """Return a problem description for a local link target, or None."""
    anchor = target.partition("#")[2]
    resolved = resolve(doc, target)
    if not exists(resolved):
        return "missing target"
    if (
        anchor
        and resolved.endswith((".md", ".mdx"))
        and resolved in TRACKED
        and anchor.lower() not in anchors(resolved)
    ):
        return "missing anchor"
    return None


def expand(token):
    """Expand one level of {a,b} alternatives."""
    m = re.search(r"\{([^{}]*,[^{}]*)\}", token)
    if not m:
        return [token]
    return [token[: m.start()] + alt + token[m.end() :] for alt in m.group(1).split(",")]


def repo_path_problem(doc, token, doc_relative=True):
    """Problem with a path-like token in a strict doc, or None when it resolves or is not a repo path.

    Paths resolve from the repository root and, when doc_relative, from the doc's directory.
    """
    token = LINE_REF.sub("", re.sub(r"::.*$", "", token))  # pytest node ids, `path:N`
    for path in expand(token.removeprefix("./")):
        candidates = [os.path.normpath(path)]
        if doc_relative:
            candidates.append(os.path.normpath(os.path.join(os.path.dirname(doc), path)))
        hits = [c for c in candidates if exists(c) or ("*" in c and any(fnmatch.fnmatch(t, c) for t in TRACKED))]
        if not hits and (path.split("/")[0] in TOP or (doc_relative and path.startswith(".."))):
            return f"missing {path}"
        # Otherwise not a repository path, e.g. a path inside a build output or an archive.
    return None


def module_exists(name):
    """True when `python -m name` finds a module in the package or the experimental or TLC checkout."""
    path = name.replace(".", "/")
    return any(
        exists(root + path + suffix)
        for root in ("", "fork/experimental/src/", "fork/tlc/src/")
        for suffix in (".py", "/__init__.py", "/__main__.py")
    )


def scan(doc):
    """Yield (line number, kind, text, problem or None) for each link and path checked in one doc."""
    strict = any(fnmatch.fnmatch(doc, pattern) for pattern in STRICT)
    text = (ROOT / doc).read_text(encoding="utf-8", errors="replace")
    links = {}
    for number, target in local_links(text):
        links.setdefault(number, []).append(target)
    fenced = False
    for number, line in enumerate(text.split("\n"), 1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            # Commands run from the repository root. The experimental project's docs ship with its
            # release, and their commands use the layout of the extracted examples archive.
            if strict and not doc.startswith("fork/experimental/"):
                for m in FENCED_PATH.finditer(line):
                    yield number, "code path", m.group(1), repo_path_problem(doc, m.group(1), False)
            if strict:
                for m in FENCED_MODULE.finditer(line):
                    if not module_exists(m.group(1)):
                        yield number, "code module", m.group(1), "no such module"
            continue
        for target in links.get(number, []):
            yield number, "link", target, check_link(doc, target)
        if strict:
            for m in TICK.finditer(line):
                token = m.group(1).strip()
                path_like = "/" in token or LINE_REF.search(token)  # also `Makefile:275`
                if path_like and not re.search(r"\s|://|^[-$<~/]|[<>|=()\[\]'\"]", token):
                    yield number, "path", token, repo_path_problem(doc, token)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--update-baseline", action="store_true", help="rewrite the baseline of upstream breakage")
    args = parser.parse_args()
    load_tree()

    docs = sorted(p for p in TRACKED if p.endswith((".md", ".mdx")) and "third_party/" not in p)
    links = 0
    strict_problems, upstream_broken = [], set()
    for doc in docs:
        strict = any(fnmatch.fnmatch(doc, pattern) for pattern in STRICT)
        for number, kind, text, problem in scan(doc):
            links += kind == "link"
            if problem is None:
                continue
            if strict:
                strict_problems.append((doc, number, kind, text, problem))
            else:
                upstream_broken.add(f"{doc}\t{text}")

    # Paths git ignores are build outputs or local environments, not broken references.
    # A trailing slash lets directory patterns such as `dist/` match.
    def missing_path(problem):
        doc, number, kind, text, description = problem
        return (
            description[len("missing ") :].rstrip("/")
            if kind != "link" and description.startswith("missing ")
            else None
        )

    missing = sorted({missing_path(p) for p in strict_problems} - {None})
    ignored = subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "--no-index", "--stdin"],
        input="".join(f"{path}\n{path}/\n" for path in missing),
        capture_output=True,
        text=True,
    ).stdout.split("\n")
    ignored = {path.rstrip("/") for path in ignored if path}
    strict_problems = [p for p in strict_problems if missing_path(p) not in ignored]

    if args.update_baseline:
        BASELINE.write_text("".join(f"{entry}\n" for entry in sorted(upstream_broken)), encoding="utf-8")
        print(f"check_doc_links: wrote {len(upstream_broken)} entries to {BASELINE.relative_to(ROOT)}")
    baseline = set(BASELINE.read_text(encoding="utf-8").splitlines()) if BASELINE.exists() else set()
    new = sorted(upstream_broken - baseline)
    fixed = sorted(baseline - upstream_broken)

    print(
        f"check_doc_links: {len(docs)} docs, {links} local links, {len(upstream_broken)} broken in upstream docs "
        f"({len(upstream_broken) - len(new)} in the baseline), {len(strict_problems)} problems in fork docs"
    )
    for doc, number, kind, text, problem in strict_problems:
        print(f"  {doc}:{number}: {kind} {text}: {problem}")
    for entry in new:
        doc, target = entry.split("\t")
        print(f"  new broken link in {doc}: {target}")
    for entry in fixed:
        print(f"  baseline entry no longer broken (run --update-baseline): {entry.replace(chr(9), ' -> ')}")
    return 1 if strict_problems or new else 0


if __name__ == "__main__":
    sys.exit(main())
