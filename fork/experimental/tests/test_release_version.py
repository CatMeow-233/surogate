"""The independent release channel must not change the native package version."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


@pytest.mark.parametrize("native_tag", [None, "v1.5.0"])
def test_experimental_tag_does_not_change_native_scm_version(tmp_path: Path, native_tag: str | None) -> None:
    pytest.importorskip("setuptools_scm")
    if shutil.which("git") is None:
        pytest.skip("git is required for the release-version fixture")
    root = Path(__file__).resolve().parents[3]
    options = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["setuptools_scm"]
    configuration = "[project]\nname = 'release-version-fixture'\ndynamic = ['version']\n[tool.setuptools_scm]\n"
    configuration += "".join(f"{key} = {json.dumps(value)}\n" for key, value in options.items())
    (tmp_path / "pyproject.toml").write_text(configuration)

    def git(*arguments: str) -> None:
        subprocess.run(["git", *arguments], cwd=tmp_path, check=True, capture_output=True, text=True)

    def version() -> str:
        return subprocess.run(
            [sys.executable, "-m", "setuptools_scm"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.name", "Release version fixture")
    git("config", "user.email", "fixture@example.invalid")
    (tmp_path / "model.txt").write_text("initial")
    git("add", ".")
    git("commit", "-qm", "initial native version")
    if native_tag is not None:
        git("tag", native_tag)
    (tmp_path / "model.txt").write_text("next")
    git("add", "model.txt")
    git("commit", "-qm", "next change")
    before = version()
    git("tag", "experimental-v0.1.0a1")
    assert before
    if native_tag is not None:
        assert before == "1.5.1.dev1"
    assert version() == before
