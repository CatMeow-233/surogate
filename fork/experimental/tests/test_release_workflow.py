"""experimental-release.yml publishes only from a tag push, uploads to a draft and publishes it only
after fork/tools/release_guard.py has refused an existing release and verified the uploaded files
(its behaviour is tested in fork/tools/tests/test_release_guard.py)."""

from __future__ import annotations

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "experimental-release.yml"


def load():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def step_index(steps, text):
    return next(index for index, step in enumerate(steps) if text in step.get("run", "") + step.get("uses", ""))


def test_only_a_tag_push_publishes():
    release = load()["jobs"]["release"]
    assert "github.event_name == 'push'" in release["if"]


def test_the_release_job_uploads_to_a_draft_and_publishes_it_after_checking():
    workflow = load()
    steps = workflow["jobs"]["release"]["steps"]
    absent = step_index(steps, "fork/tools/release_guard.py absent")
    upload = step_index(steps, "softprops/action-gh-release@")
    publish = step_index(steps, "fork/tools/release_guard.py publish")
    assert absent < upload < publish
    upload_inputs = steps[upload]["with"]
    assert upload_inputs["draft"] is True and upload_inputs["overwrite_files"] is False
    assert upload_inputs["prerelease"] is True and upload_inputs["make_latest"] is False
    assert steps[publish]["env"]["RELEASE_ID"] == f"${{{{ steps.{steps[upload]['id']}.outputs.id }}}}"
    assert "--artifacts experimental-artifacts" in steps[publish]["run"]
    assert workflow["concurrency"]["cancel-in-progress"] is False
