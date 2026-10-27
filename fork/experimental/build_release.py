#!/usr/bin/env python3
"""Build the standalone experimental distribution without invoking CUDA builds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METADATA = Path(__file__).resolve().parent


def check_archives(wheel: Path, sdist: Path, version: str) -> None:
    """Reject accidentally publishing the native distribution or namespace root."""
    required = "surogate/experimental/train/__main__.py"
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if required not in names or "surogate/__init__.py" in names:
            raise RuntimeError("wheel must contain only the experimental namespace")
        if any(name.endswith((".so", ".pyd", ".dylib", ".cu")) for name in names):
            raise RuntimeError("experimental wheel unexpectedly contains native artifacts")
        wheel_metadata = next(name for name in names if name.endswith(".dist-info/WHEEL"))
        if "Root-Is-Purelib: true" not in archive.read(wheel_metadata).decode():
            raise RuntimeError("experimental wheel is not marked pure Python")
        metadata_path = next(name for name in names if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
        if metadata["Name"] != "surogate-experimental" or metadata["Version"] != version:
            raise RuntimeError("unexpected experimental distribution name or version")
    with tarfile.open(sdist, "r:gz") as archive:
        names = archive.getnames()
        if not any(name.endswith("/" + required) for name in names):
            raise RuntimeError("sdist is missing the experimental CLI")
        if any(name.endswith("/surogate/__init__.py") for name in names):
            raise RuntimeError("sdist must not install the stable namespace root")
        project_file = next(name for name in names if name.endswith("/pyproject.toml"))
        project = archive.extractfile(project_file)
        if project is None:
            raise RuntimeError("sdist is missing pyproject.toml")
        if tomllib.loads(project.read().decode())["build-system"]["build-backend"] != "setuptools.build_meta":
            raise RuntimeError("sdist unexpectedly uses the native build backend")


def examples_archive(destination: Path) -> None:
    examples = METADATA / "examples"
    guide = METADATA / "docs" / "training.md"
    example_files = [*sorted(examples.glob("*.yaml")), *sorted(examples.glob("*.jsonl"))]
    required = {examples / "smoke.yaml", examples / "smoke.tokens.jsonl"}
    if not required.issubset(example_files) or not all(path.is_file() for path in [guide, *example_files]):
        raise RuntimeError("experimental guide and example files must exist before building")
    # Archive names keep the published layout of earlier releases, not the checkout layout.
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(METADATA / "README.md", "README.md")
        archive.write(guide, "docs/experimental/training.md")
        for path in example_files:
            archive.write(path, f"examples/experimental/{path.name}")


def source_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def source_dirty() -> bool | None:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build(output: Path, tag: str | None) -> list[Path]:
    metadata = tomllib.loads((METADATA / "pyproject.toml").read_text())
    version = metadata["project"]["version"]
    expected_tag = f"experimental-v{version}"
    if tag is not None and tag != expected_tag:
        raise ValueError(f"release tag must match the package version: {expected_tag}")
    if shutil.which("uv") is None:
        raise RuntimeError("uv is required to build the experimental release")
    source = METADATA / "src" / "surogate" / "experimental"
    if not (source / "train" / "__main__.py").is_file():
        raise RuntimeError("experimental trainer CLI is missing")

    environment = os.environ.copy()
    environment.setdefault("UV_CACHE_DIR", str(ROOT / ".cache" / "experimental-release" / "uv"))
    with tempfile.TemporaryDirectory(prefix="surogate-experimental-build-") as temporary:
        temporary_root = Path(temporary)
        staging = temporary_root / "project"
        staging.mkdir()
        shutil.copytree(
            source,
            staging / "src" / "surogate" / "experimental",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
        )
        for filename in ("pyproject.toml", "README.md"):
            shutil.copy2(METADATA / filename, staging / filename)
        shutil.copy2(ROOT / "LICENSE", staging / "LICENSE")
        distributions = temporary_root / "dist"
        subprocess.run(
            ["uv", "--no-config", "build", str(staging), "--wheel", "--sdist", "--out-dir", str(distributions)],
            cwd=staging,
            env=environment,
            check=True,
        )
        wheel = distributions / f"surogate_experimental-{version}-py3-none-any.whl"
        sdist = distributions / f"surogate_experimental-{version}.tar.gz"
        check_archives(wheel, sdist, version)
        bundle = distributions / "experimental-examples.zip"
        examples_archive(bundle)
        shutil.copy2(METADATA / "README.md", distributions / "README.md")
        (distributions / "build-info.json").write_text(
            json.dumps(
                {
                    "distribution": metadata["project"]["name"],
                    "version": version,
                    "tag": expected_tag,
                    "source_commit": source_commit(),
                    "source_dirty": source_dirty(),
                    "native_cuda_artifacts": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        files = [wheel, sdist, bundle, distributions / "README.md", distributions / "build-info.json"]
        (distributions / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in files))
        files.append(distributions / "SHA256SUMS")
        output.mkdir(parents=True, exist_ok=True)
        return [Path(shutil.copy2(path, output / path.name)) for path in files]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "experimental-release" / "artifacts")
    parser.add_argument("--tag", help="require an exact experimental-v<package-version> tag")
    arguments = parser.parse_args()
    paths = build(arguments.output_dir.resolve(), arguments.tag)
    print(json.dumps({"artifacts": [str(path) for path in paths]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
