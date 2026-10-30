"""Throwaway repositories for the tests of the fork's maintenance tools.

The tools find the repository from their own path (ROOT = parents[2]), so a test either copies a
tool into a throwaway repository under tmp_path and runs it there, or loads the module and points
its ROOT at the throwaway repository. conftest.py gives git a private global config (no signing,
no hooks or settings from the developer's config), so the tests behave the same on every machine.
"""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
REPO = TOOLS.parents[1]


def git(cwd, *args, check=True):
    result = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed:\n{result.stdout}{result.stderr}")
    return result.stdout.strip()


def write(root, files):
    for path, text in files.items():
        target = Path(root) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def commit(root, message, files=None):
    """Write files (path -> text), stage everything and commit; return the new sha."""
    write(root, files or {})
    git(root, "add", "-A")
    git(root, "commit", "-q", "--allow-empty", "-m", message)
    return git(root, "rev-parse", "HEAD")


def init_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    return path


def copy_tools(root, *names):
    """Copy fork/tools/<name> into the throwaway repository at the same path."""
    for name in names:
        target = Path(root) / "fork" / "tools" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(TOOLS / name, target)


def load_tool(name, root, monkeypatch):
    """Import fork/tools/<name>.py from the checkout with its ROOT pointed at root."""
    spec = importlib.util.spec_from_file_location(f"fork_tool_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", Path(root))
    return module


def run_tool(root, name, *args, env=None):
    """Run the copy of fork/tools/<name>.py inside root, as fork/dev runs the tools."""
    return subprocess.run(
        [sys.executable, "-I", f"fork/tools/{name}.py", *args],
        cwd=root,
        capture_output=True,
        text=True,
        env=env,
    )
