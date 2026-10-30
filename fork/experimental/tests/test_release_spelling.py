"""build_release.py builds and tags only the canonical PEP 440 spelling of the package version, the
spelling fork/tools/release_gate.sh requires of native and experimental releases."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest
from packaging.version import InvalidVersion, Version

EXPERIMENTAL = Path(__file__).resolve().parents[1]
GATE = EXPERIMENTAL.parent / "tools" / "release_gate.sh"

spec = importlib.util.spec_from_file_location("build_release", EXPERIMENTAL / "build_release.py")
build_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_release)

CANONICAL = ["0", "1.0", "0.1.0a4", "0.1.0a4.dev0", "0.1.0b2", "0.1.0rc10", "1.5.12.post1", "1.5.12rc1.post2.dev3"]
OTHER = [
    "0.1.0.a4",
    "0.1.0-alpha4",
    "0.1.0alpha4",
    "0.1.0a04",
    "0.1.0A4",
    "0.1.0c4",
    "0.1.0a4.dev",
    "0.1.0a4-dev0",
    "1.5.12.post01",
    "1.5.12post1",
    "01.5.12",
    "v0.1.0a4",
    "1!0.1.0a4",
    "0.1.0a4+local",
    "0.1.0a4 ",
    "",
]


def canonical(version: str) -> bool:
    """What packaging says: the version parses, has no epoch or local part, and prints as written."""
    try:
        parsed = Version(version)
    except InvalidVersion:
        return False
    return parsed.epoch == 0 and parsed.local is None and str(parsed) == version


@pytest.mark.parametrize("version", CANONICAL + OTHER)
def test_the_pattern_is_packagings_canonical_spelling(version):
    assert bool(build_release.CANONICAL_VERSION.fullmatch(version)) == canonical(version)
    assert canonical(version) == (version in CANONICAL)


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash runs fork/tools/release_gate.sh")
def test_the_release_gate_accepts_the_same_spellings():
    script = '. "$1"; shift; for v; do if release_version_ok "$v"; then echo "yes $v"; else echo "no $v"; fi; done'
    result = subprocess.run(
        ["bash", "-c", script, "_", str(GATE), *CANONICAL, *OTHER], capture_output=True, text=True, check=True
    )
    gate = [line.split(" ", 1)[0] == "yes" for line in result.stdout.split("\n")[:-1]]
    assert gate == [bool(build_release.CANONICAL_VERSION.fullmatch(v)) for v in CANONICAL + OTHER]


@pytest.mark.parametrize("version", ["0.1.0.a4", "0.1.0a04", "1.5.12.post01", "0.1.0a4+local"])
def test_a_non_canonical_version_is_refused_with_or_without_a_tag(version):
    for tag in (None, f"experimental-v{version}"):
        with pytest.raises(ValueError, match="canonical spelling"):
            build_release.check_version(version, tag)


def test_a_release_tag_names_the_version():
    assert build_release.check_version("0.1.0a4", "experimental-v0.1.0a4") == "experimental-v0.1.0a4"
    assert build_release.check_version("0.1.0a4.dev0", None) == "experimental-v0.1.0a4.dev0"
    with pytest.raises(ValueError, match="must match the package version: experimental-v0.1.0a4"):
        build_release.check_version("0.1.0a4", "experimental-v0.1.0a5")


def test_a_development_version_is_never_tagged():
    with pytest.raises(ValueError, match="development version"):
        build_release.check_version("0.1.0a4.dev0", "experimental-v0.1.0a4.dev0")
