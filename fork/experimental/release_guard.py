#!/usr/bin/env python3
"""Refuse to publish experimental assets over a release that holds different bytes.

The release job of experimental-release.yml runs this before uploading. A
published experimental release is never replaced: the builds carry timestamps,
so a rebuild of the same commit has new SHA-256 values and would break every
hash a user recorded. When no release exists for the tag, publishing proceeds.
When one exists, each asset it already holds must be byte-identical to this
run's artifact of the same name; the upload step (``overwrite_files: false``)
then adds only the missing assets. That completes a release whose upload
stopped part-way when the release job of the same run is re-run, because that
job reuses the run's artifacts. Any other difference fails the job.

Needs the GitHub CLI (``gh``) with ``GH_TOKEN`` set; uses only REST ``gh api``
GET requests. Reads only; never uploads or deletes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _gh_api(path: str, *, accept: str | None = None) -> bytes:
    command = ["gh", "api", *(["-H", f"Accept: {accept}"] if accept else []), path]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        message = result.stderr.decode(errors="replace").strip() or f"exit status {result.returncode}"
        raise LookupError(message) if "(HTTP 404)" in message else RuntimeError(f"gh api {path}: {message}")
    return result.stdout


def published_assets(repository: str, tag: str) -> list[dict] | None:
    """The assets of the release for ``tag``, or None when no release exists."""
    try:
        release = json.loads(_gh_api(f"repos/{repository}/releases/tags/{tag}"))
    except LookupError:
        return None
    return sorted(release["assets"], key=lambda asset: asset["name"])


def differences(repository: str, assets: list[dict], artifacts: Path) -> list[str]:
    problems = []
    for asset in assets:
        name, built = asset["name"], artifacts / asset["name"]
        if not built.is_file():
            problems.append(f"{name}: published, but not built by this run")
            continue
        published = _gh_api(f"repos/{repository}/releases/assets/{asset['id']}", accept="application/octet-stream")
        if published != built.read_bytes():
            problems.append(f"{name}: published bytes differ from this run's build")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repository", required=True, help="owner/name")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--artifacts", required=True, type=Path, help="directory of this run's verified artifacts")
    arguments = parser.parse_args(argv)
    try:
        assets = published_assets(arguments.repository, arguments.tag)
        if assets is None:
            print(f"No release exists for {arguments.tag}; publishing a new one.")
            return 0
        problems = differences(arguments.repository, assets, arguments.artifacts)
    except (LookupError, RuntimeError) as error:
        print(f"::error::cannot compare the published assets of {arguments.tag}: {error}", file=sys.stderr)
        return 1
    if problems:
        print(
            f"::error::release {arguments.tag} already exists with other assets; "
            "a published experimental release is never replaced.",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "Publish a new version instead. If this release is broken, inspect it and delete it by hand "
            "before running the release again.",
            file=sys.stderr,
        )
        return 1
    print(f"Release {arguments.tag} exists; its {len(assets)} assets match this build. Uploading only missing assets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
