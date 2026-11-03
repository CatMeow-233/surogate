"""`python -m surogate_corpora COMMAND [args]`: the one command table of the package (README: 命令行).

COMMANDS maps each command to the submodule and function that implement it, and to the step of the study's
implementation spec that adds it. The function takes the remaining arguments, `(argv: list[str]) -> int`; it exits 0
on success, 1 with a reason on every refusal, and 2 on a usage error. A command whose module does not exist yet exits
2 and names its step. Listing the commands imports no command module (and so no pyarrow).
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

PACKAGE = "surogate_corpora"


class Command(NamedTuple):
    module: str  # submodule of the package
    summary: str
    step: str  # the step of the implementation spec that adds the command
    function: str = "main"  # (argv: list[str]) -> int in that module


COMMANDS: dict[str, Command] = {
    "sources": Command("sources", "list the pinned source registry and the profiles", "S1"),
    "plan": Command("plan", "write a corpora-plan/1: row groups chosen by rank key up to a byte budget", "S2"),
    "fetch": Command("run", "read the plan's row groups, filter and export (resumable)", "S5"),
    "verify": Command("verify", "re-check an export directory against its provenance", "S4"),
    "verify-samples": Command("verify", "check datasets-server sample rows against the registry", "S5", "samples_main"),
    # Not in the spec's command table: the per-project wiring of the exports (the workflow's S5).
    "compose": Command("compose", "one project corpus spec from several export directories", "S5"),
}


def module_name(command: str) -> str:
    return f"{PACKAGE}.{COMMANDS[command].module}"


def is_implemented(command: str) -> bool:
    """True when the command's module file exists in this package; looked up on disk, so listing imports nothing."""
    path = Path(__file__).resolve().parent.joinpath(*COMMANDS[command].module.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = [f"usage: python -m {PACKAGE} COMMAND [args]   (COMMAND --help for its arguments)", "", "commands:"]
    for name, command in COMMANDS.items():
        status = "" if is_implemented(name) else f"   [not implemented yet: step {command.step}]"
        lines.append(f"  {name:<{width}}  {command.summary}{status}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help", "help"):
        print(usage())
        return 0
    name, rest = args[0], args[1:]
    if name not in COMMANDS:
        print(f"{PACKAGE}: unknown command {name!r}\n\n{usage()}", file=sys.stderr)
        return 2
    if not is_implemented(name):
        print(f"{PACKAGE}: {name!r} is not implemented yet; it lands in step {COMMANDS[name].step}", file=sys.stderr)
        return 2
    return int(getattr(importlib.import_module(module_name(name)), COMMANDS[name].function)(rest))


if __name__ == "__main__":
    sys.exit(main())
