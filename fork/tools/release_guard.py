#!/usr/bin/env python3
"""Publish a fork release once, and only with the files this run built.

Both release workflows (native-release.yml, experimental-release.yml) upload their files with
softprops/action-gh-release and ``draft: true``, between two calls of this script:

  release_guard.py absent --repository R --tag T
      Before the upload: exit 0 only when no release exists for T, published or draft. A published
      release's files are never replaced, and it is never deleted and published again under the same
      version (a rebuild has new SHA-256 values: the builds carry timestamps): publish a new version.
      The one exception is a public release that is missing files (made by hand, or by a workflow from
      before drafts); the owner may complete it by hand with the original run's artefact, checked
      against its SHA256SUMS (fork/docs/build-and-release.md, section 一个版本只发布一次). A draft
      left by a failed run is deleted by hand first: it was never public, so a new run may publish
      the same version.
  release_guard.py publish --repository R --tag T --release-id ID --artifacts DIR
      After the upload: release ID must be a draft for T whose assets are exactly the files in DIR,
      uploaded completely and byte-identical (SHA-256) to them. Then publish it (draft false; it stays
      a prerelease and is not marked latest). Anything else fails and leaves the draft unpublished.
      When the publish request reports an error, GitHub may still have applied it (a 5xx or a dropped
      connection after the server committed), so the release is read again and its real state is
      reported: still a draft (exit 1), published (exit 0, with a warning) or unknown (exit 1).

Publishing the draft creates the tag at the release's target commit when the tag does not exist yet
(native-v*); an experimental-v* tag was pushed before the run and stays where it is.

Needs the GitHub CLI (gh) with GH_TOKEN set to a token that can push (contents: write): only such a
token sees draft releases. Makes GET requests and, in publish, one PATCH of the release; never
uploads, replaces or deletes anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PAGE = 100


class GhError(RuntimeError):
    pass


def gh_api(*arguments: str) -> bytes:
    result = subprocess.run(["gh", "api", *arguments], capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip() or f"exit status {result.returncode}"
        raise GhError(f"gh api {' '.join(arguments)}: {message}")
    return result.stdout


def releases_for(repository: str, tag: str) -> list[dict]:
    """Every release of the repository whose tag is ``tag``, drafts included (listing releases is
    the only endpoint that returns drafts; releases/tags/<tag> returns only a published one)."""
    found, page = [], 1
    while True:
        batch = json.loads(gh_api(f"repos/{repository}/releases?per_page={PAGE}&page={page}"))
        found += [release for release in batch if release.get("tag_name") == tag]
        if len(batch) < PAGE:
            return found
        page += 1


def sha256_of_asset(repository: str, asset_id: int) -> str:
    with tempfile.TemporaryFile() as handle:
        command = [
            "gh",
            "api",
            "-H",
            "Accept: application/octet-stream",
            f"repos/{repository}/releases/assets/{asset_id}",
        ]
        result = subprocess.run(command, stdout=handle, stderr=subprocess.PIPE, check=False)
        if result.returncode != 0:
            message = result.stderr.decode(errors="replace").strip() or f"exit status {result.returncode}"
            raise GhError(f"downloading asset {asset_id}: {message}")
        handle.seek(0)
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha256_of_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def draft_problems(repository: str, tag: str, release: dict, artifacts: Path) -> list[str]:
    """Why ``release`` is not exactly this run's files as a draft for ``tag``; empty when it is."""
    if release.get("tag_name") != tag:
        return [f"release {release.get('id')} is for tag {release.get('tag_name')!r}, not {tag}"]
    if release.get("draft") is not True:
        return [f"release {release.get('id')} is not a draft; a published release is never changed"]
    built = {path.name: path for path in sorted(artifacts.iterdir()) if path.is_file()}
    if not built:
        return [f"{artifacts} holds no files"]
    problems, seen = [], set()
    for asset in sorted(release.get("assets", []), key=lambda asset: asset["name"]):
        # GitHub may rename an uploaded file; softprops then puts the original name in the label.
        name = asset["name"] if asset["name"] in built else asset.get("label") or asset["name"]
        if name not in built:
            problems.append(f"{asset['name']}: on the draft, but not built by this run")
        elif name in seen:
            problems.append(f"{name}: uploaded twice")
        elif asset.get("state") != "uploaded":
            problems.append(f"{name}: upload incomplete (state {asset.get('state')!r})")
        elif asset.get("size") != built[name].stat().st_size:
            problems.append(f"{name}: {asset.get('size')} bytes on the draft, {built[name].stat().st_size} built")
        elif sha256_of_asset(repository, asset["id"]) != sha256_of_file(built[name]):
            problems.append(f"{name}: the draft's bytes differ from this run's build")
        seen.add(name)
    problems += [f"{name}: built by this run, but not on the draft" for name in sorted(set(built) - seen)]
    return problems


def absent(arguments: argparse.Namespace) -> int:
    existing = releases_for(arguments.repository, arguments.tag)
    if not existing:
        print(f"No release exists for {arguments.tag}.")
        return 0
    print(f"::error::a release already exists for {arguments.tag}; a version is published once.", file=sys.stderr)
    for release in existing:
        kind = "draft" if release.get("draft") else "published"
        print(f"  release {release.get('id')} ({kind}): {release.get('html_url', '')}", file=sys.stderr)
    print(
        "A published release's files are never replaced, and it is never deleted and published again\n"
        "under the same version: publish a new version. The one exception is a published release that is\n"
        "missing files (made by hand, or by a workflow from before drafts): complete it by hand with the\n"
        "original run's artefact, checked against its SHA256SUMS (fork/docs/build-and-release.md, section\n"
        "一个版本只发布一次). A draft is what a failed run leaves; it was never public: delete it\n"
        "(gh release delete <tag> -R <repository>, without --cleanup-tag) and run the release again.",
        file=sys.stderr,
    )
    return 1


def publish(arguments: argparse.Namespace) -> int:
    path = f"repos/{arguments.repository}/releases/{arguments.release_id}"
    release = json.loads(gh_api(path))
    problems = draft_problems(arguments.repository, arguments.tag, release, arguments.artifacts)
    if problems:
        print(
            f"::error::draft release {arguments.release_id} for {arguments.tag} is not this run's build; not publishing.",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print("The draft was never public: delete it and run the release again.", file=sys.stderr)
        return 1
    count = len(release.get("assets", []))
    try:
        published = json.loads(gh_api("-X", "PATCH", path, "-F", "draft=false", "-f", "make_latest=false"))
    except GhError as error:
        print(f"::error::the request to publish draft release {arguments.release_id} failed: {error}", file=sys.stderr)
        return after_failed_publish(arguments, path, count)
    if published.get("draft") is not False or published.get("tag_name") != arguments.tag:
        print(
            f"::error::release {arguments.release_id} did not become a published release for {arguments.tag}",
            file=sys.stderr,
        )
        return 1
    print(f"Published {arguments.tag} with {count} verified files: {published.get('html_url', '')}")
    return 0


def after_failed_publish(arguments: argparse.Namespace, path: str, count: int) -> int:
    """The publish request reported an error, but GitHub may have applied it anyway (a 5xx or a
    dropped connection after the server committed). Read the release again and report what it is."""
    check = f"gh release view {arguments.tag} -R {arguments.repository} --json isDraft --jq .isDraft"
    try:
        current = json.loads(gh_api(path))
    except (GhError, json.JSONDecodeError) as error:
        print(
            f"::error::cannot read release {arguments.release_id} again ({error}); it may already be public.\n"
            f"Delete it only after `{check}` prints true. See fork/docs/build-and-release.md.",
            file=sys.stderr,
        )
        return 1
    if current.get("draft") is True:
        print(
            f"Release {arguments.release_id} is still a draft and was never public; its files are verified.\n"
            "See fork/docs/build-and-release.md.",
            file=sys.stderr,
        )
        return 1
    if current.get("draft") is False and current.get("tag_name") == arguments.tag:
        print(
            f"::warning::the publish request reported an error, but GitHub published release "
            f"{arguments.release_id}: {arguments.tag} is public. Do not delete it."
        )
        print(f"Published {arguments.tag} with {count} verified files: {current.get('html_url', '')}")
        return 0
    print(
        f"::error::release {arguments.release_id} is now draft={current.get('draft')!r}, "
        f"tag {current.get('tag_name')!r}; it may be public. Do not delete it before `{check}` prints true.",
        file=sys.stderr,
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("absent", "exit 0 only when no release, published or draft, exists for the tag"),
        ("publish", "verify a draft's files against this run's build, then publish it"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--repository", required=True, help="owner/name")
        command.add_argument("--tag", required=True)
        if name == "publish":
            command.add_argument("--release-id", required=True, type=int, help="the draft softprops created")
            command.add_argument("--artifacts", required=True, type=Path, help="directory of this run's files")
    arguments = parser.parse_args(argv)
    try:
        return absent(arguments) if arguments.command == "absent" else publish(arguments)
    except (GhError, json.JSONDecodeError, KeyError, OSError) as error:
        print(f"::error::cannot check the releases of {arguments.tag}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
