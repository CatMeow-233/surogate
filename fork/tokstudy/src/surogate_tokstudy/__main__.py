"""`python -m surogate_tokstudy COMMAND [args]`: the one command table of the package (README: 命令行).

COMMANDS maps each command to the submodule and function that implement it, and to the step of the study's
implementation spec that adds that module. The function takes the remaining arguments, `(argv: list[str]) -> int`.
A command whose module does not exist yet exits 2 and names its step, so the table lists the whole planned surface
before the modules land. Listing the commands imports neither torch nor any command module.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

PACKAGE = "surogate_tokstudy"


class Command(NamedTuple):
    module: str  # submodule of the package
    summary: str
    step: str  # the step of the implementation spec that adds the module
    function: str = "main"  # (argv: list[str]) -> int in that module


COMMANDS: dict[str, Command] = {
    "arms": Command("arms", "load and check an arm registry (sha256 of every tokenizer)", "S8"),
    "intrinsic": Command("intrinsic", "intrinsic metrics per arm and domain (CPU screen)", "S10"),
    "align": Command("align", "student/teacher token alignment rates (CPU screen)", "S10"),
    "stream": Command("stream", "a table's training stream: document order, byte mixture, parts", "S9"),
    "train": Command("train", "one run of one arm at the table's F_target; resumes", "S14"),
    "branch": Command("train", "decay branch from a stable checkpoint", "S14", "branch_main"),
    "eval": Command("evaluate", "per-document NLL and BPB on the common byte prefix", "S15"),
    "probe": Command("probes", "candidate byte NLLs of the downstream probes", "S15"),
    "verdict": Command("stats", "paired deltas and the preregistered decision of one table", "S16", "verdict_main"),
    "flops": Command("flops", "training FLOPs per arm; H-Net configuration matching", "S13"),
    # The spec's file tree names no module for the G0.5-ts grid; it gets its own when S17 lands.
    "bench": Command("bench", "G0.5-ts throughput and memory grid (GPU host)", "S17"),
}


def module_name(command: str) -> str:
    return f"{PACKAGE}.{COMMANDS[command].module}"


def is_implemented(command: str) -> bool:
    """True when the command's module file exists in this package. Looked up on disk, so that listing the commands
    imports no subpackage (whose __init__ could import torch)."""
    path = Path(__file__).resolve().parent.joinpath(*COMMANDS[command].module.split("."))
    return path.with_suffix(".py").is_file() or (path / "__init__.py").is_file()


def usage() -> str:
    width = max(map(len, COMMANDS))
    lines = [f"usage: python -m {PACKAGE} COMMAND [args]", "", "commands:"]
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
