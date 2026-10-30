"""The release workflow publishes only on a tag push and never replaces published bytes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

EXPERIMENTAL = Path(__file__).resolve().parents[1]
GUARD = EXPERIMENTAL / "release_guard.py"
WORKFLOW = EXPERIMENTAL.parents[1] / ".github" / "workflows" / "experimental-release.yml"
TAG = "experimental-v9.9.9"

# Stands in for the GitHub CLI's REST calls: answers the release query from
# FAKE_RELEASE ("404", "500" or a JSON file) and serves asset i as the bytes of
# the i-th file (sorted) in FAKE_PUBLISHED.
FAKE_GH = f"""#!{sys.executable}
import json, os, sys
from pathlib import Path
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps(sys.argv[1:]) + "\\n")
release = os.environ["FAKE_RELEASE"]
published = sorted(Path(os.environ["FAKE_PUBLISHED"]).iterdir())
if sys.argv[1:] == ["api", "repos/owner/name/releases/tags/{TAG}"]:
    if release in ("404", "500"):
        message = "Not Found" if release == "404" else "Server Error"
        sys.exit(print(f"gh: {{message}} (HTTP {{release}})", file=sys.stderr) or 1)
    print(Path(release).read_text())
elif sys.argv[1:4] == ["api", "-H", "Accept: application/octet-stream"]:
    asset = int(sys.argv[4].removeprefix("repos/owner/name/releases/assets/"))
    sys.stdout.buffer.write(published[asset].read_bytes())
else:
    sys.exit(f"unexpected gh call: {{sys.argv[1:]}}")
"""


@pytest.fixture
def release(tmp_path):
    binaries = tmp_path / "bin"
    binaries.mkdir()
    gh = binaries / "gh"
    gh.write_text(FAKE_GH)
    gh.chmod(0o755)
    artifacts, published = tmp_path / "artifacts", tmp_path / "published"
    artifacts.mkdir()
    published.mkdir()
    for name, content in {"package.whl": b"wheel", "SHA256SUMS": b"sums", "README.md": b"readme"}.items():
        (artifacts / name).write_bytes(content)
    log = tmp_path / "gh.log"

    def run(release_answer, published_files=None):
        for name, content in (published_files or {}).items():
            (published / name).write_bytes(content)
        if published_files is not None:
            answer = tmp_path / "release.json"
            names = sorted(published_files)
            assets = [{"id": index, "name": name} for index, name in enumerate(names)]
            answer.write_text(json.dumps({"tag_name": TAG, "assets": assets}))
            release_answer = str(answer)
        environment = {
            **os.environ,
            "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
            "FAKE_RELEASE": release_answer,
            "FAKE_PUBLISHED": str(published),
            "FAKE_LOG": str(log),
        }
        result = subprocess.run(
            [sys.executable, str(GUARD), "--repository", "owner/name", "--tag", TAG, "--artifacts", str(artifacts)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls

    return run


QUERY = ["api", f"repos/owner/name/releases/tags/{TAG}"]


def download(asset_id):
    return ["api", "-H", "Accept: application/octet-stream", f"repos/owner/name/releases/assets/{asset_id}"]


def test_a_new_tag_publishes(release):
    result, calls = release("404")
    assert result.returncode == 0, result.stderr
    assert "No release exists" in result.stdout
    assert calls == [QUERY]


def test_a_failed_query_does_not_publish(release):
    result, _ = release("500")
    assert result.returncode == 1
    assert "HTTP 500" in result.stderr and "Traceback" not in result.stderr


def test_a_partial_release_with_identical_bytes_is_completed(release):
    result, calls = release(None, {"package.whl": b"wheel", "SHA256SUMS": b"sums"})
    assert result.returncode == 0, result.stderr
    assert "only missing assets" in result.stdout
    assert calls == [QUERY, download(0), download(1)]


def test_an_existing_release_without_assets_is_filled(release):
    result, calls = release(None, {})
    assert result.returncode == 0, result.stderr
    assert calls == [QUERY]


@pytest.mark.parametrize(
    "published_files,problem",
    [
        ({"package.whl": b"rebuilt wheel", "SHA256SUMS": b"sums"}, "package.whl: published bytes differ"),
        ({"package.whl": b"wheel", "old-examples.zip": b"zip"}, "old-examples.zip: published, but not built"),
    ],
)
def test_published_bytes_are_never_replaced(release, published_files, problem):
    result, calls = release(None, published_files)
    assert result.returncode == 1
    assert "never replaced" in result.stderr and problem in result.stderr
    # The guard only reads: GET requests, no upload or delete.
    assert calls[0] == QUERY and all(call[:3] == download(0)[:3] for call in calls[1:])


def test_only_a_tag_push_publishes_and_the_upload_never_overwrites():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    release = workflow["jobs"]["release"]
    assert "github.event_name == 'push'" in release["if"]
    assert "startsWith(github.ref, 'refs/tags/experimental-v')" in release["if"]
    steps = release["steps"]
    guard = next(index for index, step in enumerate(steps) if "release_guard.py" in step.get("run", ""))
    publish = next(index for index, step in enumerate(steps) if "action-gh-release" in step.get("uses", ""))
    assert guard < publish
    assert steps[publish]["with"]["overwrite_files"] is False
    assert workflow["concurrency"]["cancel-in-progress"] is False
