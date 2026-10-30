"""fork/tools/release_guard.py: a tag gets one release, and a draft is published only when it holds
exactly the files the run built (final verification findings on experimental-release.yml and the
native recovery steps)."""

import hashlib
import json
import os
import subprocess
import sys

import pytest
from toolrepo import TOOLS

GUARD = TOOLS / "release_guard.py"
TAG = "experimental-v9.9.9"
FILES = {"package.whl": b"wheel bytes", "SHA256SUMS": b"sums", "README.md": b"readme"}

# Stands in for the GitHub CLI. State is a JSON file: {"releases": [...], "fail": {...}, "patched": ...};
# asset i serves the bytes of file assets/<i>. Every call is logged. fail keys: list, download, patch
# (refused, nothing changes), patch_applied (published, then an error) and reread (reading the release
# fails once it was published).
FAKE_GH = f"""#!{sys.executable}
import json, os, re, sys
from pathlib import Path
state_file = Path(os.environ["FAKE_STATE"])
state = json.loads(state_file.read_text())
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
def fail(code):
    sys.exit(print(f"gh: failed (HTTP {{code}})", file=sys.stderr) or 1)
if args[:1] != ["api"]:
    sys.exit(f"unexpected gh call: {{args}}")
args = args[1:]
if m := re.fullmatch(r"repos/owner/name/releases\\?per_page=100&page=(\\d+)", args[0]) if len(args) == 1 else None:
    if "list" in state["fail"]:
        fail(state["fail"]["list"])
    page = int(m.group(1))
    print(json.dumps(state["releases"][(page - 1) * 100 : page * 100]))
elif len(args) == 1 and (m := re.fullmatch(r"repos/owner/name/releases/(\\d+)", args[0])):
    release = next((r for r in state["releases"] if r["id"] == int(m.group(1))), None)
    if release is None:
        fail(404)
    if "reread" in state["fail"] and state.get("patched"):
        fail(state["fail"]["reread"])
    print(json.dumps(release))
elif args[:2] == ["-H", "Accept: application/octet-stream"]:
    asset = args[2].removeprefix("repos/owner/name/releases/assets/")
    if "download" in state["fail"]:
        fail(state["fail"]["download"])
    sys.stdout.buffer.write((Path(os.environ["FAKE_ASSETS"]) / asset).read_bytes())
elif args[:2] == ["-X", "PATCH"] and args[3:] == ["-F", "draft=false", "-f", "make_latest=false"]:
    if "patch" in state["fail"]:
        fail(state["fail"]["patch"])
    release_id = int(args[2].removeprefix("repos/owner/name/releases/"))
    release = next(r for r in state["releases"] if r["id"] == release_id)
    release["draft"] = False
    state["patched"] = True
    state_file.write_text(json.dumps(state))
    if "patch_applied" in state["fail"]:  # the server published it, then the response was lost
        fail(state["fail"]["patch_applied"])
    print(json.dumps(release))
else:
    sys.exit(f"unexpected gh api call: {{args}}")
"""


@pytest.fixture
def github(tmp_path):
    """Runs release_guard.py against a fake GitHub: github(command, releases, fail) -> (result, calls)."""
    binaries, assets, artifacts = tmp_path / "bin", tmp_path / "assets", tmp_path / "artifacts"
    for directory in (binaries, assets, artifacts):
        directory.mkdir()
    (binaries / "gh").write_text(FAKE_GH)
    (binaries / "gh").chmod(0o755)
    for name, content in FILES.items():
        (artifacts / name).write_bytes(content)
    state, log = tmp_path / "state.json", tmp_path / "gh.log"

    def run(arguments, releases, fail=None):
        state.write_text(json.dumps({"releases": releases, "fail": fail or {}}))
        log.unlink(missing_ok=True)
        environment = {
            **os.environ,
            "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}",
            "FAKE_STATE": str(state),
            "FAKE_ASSETS": str(assets),
            "FAKE_LOG": str(log),
        }
        command = [sys.executable, str(GUARD), *arguments, "--repository", "owner/name", "--tag", TAG]
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls, json.loads(state.read_text())["releases"]

    run.assets, run.artifacts = assets, artifacts
    return run


def release(release_id, tag=TAG, draft=True, assets=()):
    return {
        "id": release_id,
        "tag_name": tag,
        "draft": draft,
        "html_url": f"https://r/{release_id}",
        "assets": list(assets),
    }


def uploaded(github, files=FILES, start=0, **overrides):
    """Asset records for ``files`` as GitHub lists them, with their bytes served by the fake."""
    records = []
    for offset, (name, content) in enumerate(files.items()):
        asset_id = start + offset
        (github.assets / str(asset_id)).write_bytes(content)
        record = {"id": asset_id, "name": name, "label": "", "state": "uploaded", "size": len(content)}
        records.append({**record, **overrides.get(name, {})})
    return records


def publish_arguments(github, release_id=7):
    return ["publish", "--release-id", str(release_id), "--artifacts", str(github.artifacts)]


LIST = ["api", "repos/owner/name/releases?per_page=100&page=1"]


# absent


def test_a_tag_without_a_release_may_be_published(github):
    result, calls, _ = github(["absent"], [release(1, tag="experimental-v0.1.0a3", draft=False)])
    assert result.returncode == 0, result.stderr
    assert "No release exists" in result.stdout
    assert calls == [LIST]


@pytest.mark.parametrize("draft", [False, True])
def test_an_existing_release_or_draft_is_refused(github, draft):
    result, _, _ = github(["absent"], [release(5, draft=draft, assets=uploaded(github))])
    assert result.returncode == 1
    assert "a release already exists" in result.stderr
    assert ("(draft)" if draft else "(published)") in result.stderr


def test_every_page_of_releases_is_searched(github):
    releases = [release(index, tag=f"other-{index}", draft=False) for index in range(100)] + [release(100)]
    result, calls, _ = github(["absent"], releases)
    assert result.returncode == 1
    assert calls == [LIST, ["api", "repos/owner/name/releases?per_page=100&page=2"]]


def test_a_failed_listing_publishes_nothing(github):
    result, _, _ = github(["absent"], [], fail={"list": 502})
    assert result.returncode == 1
    assert "HTTP 502" in result.stderr and "Traceback" not in result.stderr


# publish


def test_a_draft_with_exactly_the_built_files_is_published(github):
    result, calls, releases = github(publish_arguments(github), [release(7, assets=uploaded(github))])
    assert result.returncode == 0, result.stderr
    assert "Published experimental-v9.9.9 with 3 verified files" in result.stdout
    assert releases[0]["draft"] is False
    # Reads the draft, downloads each file once, then one PATCH: nothing is uploaded or deleted.
    assert calls[0] == ["api", "repos/owner/name/releases/7"]
    assert sorted(call[3] for call in calls[1:4]) == [f"repos/owner/name/releases/assets/{i}" for i in range(3)]
    assert calls[4:] == [
        ["api", "-X", "PATCH", "repos/owner/name/releases/7", "-F", "draft=false", "-f", "make_latest=false"]
    ]


def test_a_file_github_renamed_is_found_by_its_label(github):
    files = {"surogate-1.5.12.post1+cu130-cp312-abi3-manylinux_2_39_x86_64.whl": b"wheel", "SHA256SUMS": b"s"}
    for name in FILES:
        (github.artifacts / name).unlink()
    for name, content in files.items():
        (github.artifacts / name).write_bytes(content)
    renamed = {next(iter(files)): {"name": "surogate-1.5.12.post1.cu130-cp312-abi3-manylinux_2_39_x86_64.whl"}}
    assets = uploaded(github, files, **renamed)
    assets[0]["label"] = next(iter(files))
    result, _, releases = github(publish_arguments(github), [release(7, assets=assets)])
    assert result.returncode == 0, result.stderr
    assert releases[0]["draft"] is False


def changed_bytes(github, assets):
    (github.assets / "0").write_bytes(b"other bytes")
    return assets


@pytest.mark.parametrize(
    ("make_assets", "problem"),
    [
        (lambda github: changed_bytes(github, uploaded(github)), "the draft's bytes differ from this run's build"),
        (lambda github: uploaded(github, {"README.md": b"readme", "SHA256SUMS": b"sums"}), "package.whl: built by this run, but not on the draft"),
        (lambda github: uploaded(github, {**FILES, "old-examples.zip": b"zip"}), "old-examples.zip: on the draft, but not built by this run"),
        (lambda github: uploaded(github, **{"package.whl": {"state": "starter"}}), "package.whl: upload incomplete"),
        (lambda github: uploaded(github, **{"package.whl": {"size": 3}}), "package.whl: 3 bytes on the draft, 11 built"),
    ],
    ids=["bytes", "missing", "extra", "incomplete", "size"],
)  # fmt: skip
def test_a_draft_that_is_not_this_build_stays_a_draft(github, make_assets, problem):
    result, calls, releases = github(publish_arguments(github), [release(7, assets=make_assets(github))])
    assert result.returncode == 1
    assert problem in result.stderr and "not publishing" in result.stderr
    assert releases[0]["draft"] is True
    assert not any("PATCH" in call for call in calls)


@pytest.mark.parametrize(
    ("record", "problem"),
    [({"draft": False}, "is not a draft"), ({"tag_name": "experimental-v9.9.8"}, "is for tag 'experimental-v9.9.8'")],
)
def test_only_a_draft_for_this_tag_is_published(github, record, problem):
    result, calls, _ = github(publish_arguments(github), [{**release(7, assets=uploaded(github)), **record}])
    assert result.returncode == 1
    assert problem in result.stderr
    assert not any("PATCH" in call for call in calls)


@pytest.mark.parametrize("fail", [{"download": 500}, {"patch": 422}], ids=["download", "patch"])
def test_api_failures_leave_the_draft_unpublished(github, fail):
    result, _, releases = github(publish_arguments(github), [release(7, assets=uploaded(github))], fail=fail)
    assert result.returncode == 1
    assert f"HTTP {next(iter(fail.values()))}" in result.stderr and "Traceback" not in result.stderr
    assert releases[0]["draft"] is True


def test_a_refused_publish_request_is_checked_against_the_release(github):
    draft = release(7, assets=uploaded(github))
    result, calls, releases = github(publish_arguments(github), [draft], fail={"patch": 422})
    assert result.returncode == 1
    assert "is still a draft and was never public" in result.stderr
    assert releases[0]["draft"] is True
    assert calls[-1] == ["api", "repos/owner/name/releases/7"]  # read again after the failed PATCH


def test_a_publish_request_github_applied_despite_an_error_is_reported_as_published(github):
    """A 5xx after the server committed: the release is public, and the owner must not delete it."""
    draft = release(7, assets=uploaded(github))
    result, calls, releases = github(publish_arguments(github), [draft], fail={"patch_applied": 502})
    assert result.returncode == 0, result.stderr
    assert "HTTP 502" in result.stderr
    assert "experimental-v9.9.9 is public. Do not delete it." in result.stdout
    assert "Published experimental-v9.9.9 with 3 verified files" in result.stdout
    assert "still a draft" not in result.stdout + result.stderr
    assert releases[0]["draft"] is False
    assert calls[-1] == ["api", "repos/owner/name/releases/7"]


def test_an_unknown_state_after_a_failed_publish_request_is_not_called_a_draft(github):
    fail = {"patch_applied": 502, "reread": 503}
    result, _, releases = github(publish_arguments(github), [release(7, assets=uploaded(github))], fail=fail)
    assert result.returncode == 1
    assert "it may already be public" in result.stderr and "HTTP 503" in result.stderr
    assert "gh release view experimental-v9.9.9 -R owner/name --json isDraft --jq .isDraft" in result.stderr
    assert "still a draft" not in result.stderr and "Traceback" not in result.stderr
    assert releases[0]["draft"] is False


def test_the_digest_of_the_downloaded_bytes_is_compared(github):
    """The guard hashes what GitHub serves, not what the upload step reported."""
    assets = uploaded(github)
    assets[0]["digest"] = "sha256:" + hashlib.sha256(FILES["package.whl"]).hexdigest()  # GitHub's own field
    changed_bytes(github, assets)
    result, _, _ = github(publish_arguments(github), [release(7, assets=assets)])
    assert result.returncode == 1
