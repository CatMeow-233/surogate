"""`python -m surogate_bilingual COMMAND [args]`: the one command table of the package (README: 命令行).

COMMANDS maps each command to the submodule and function that implement it, and to the step of docs/spec.md that
adds that module. The function takes the remaining arguments, `(argv: list[str]) -> int`, and accepts `--dry-run`.
A command whose module does not exist yet exits with status 2 and names its step, so the table lists the whole
planned surface before the modules land. Listing the commands imports neither torch nor any command module.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

PACKAGE = "surogate_bilingual"


class Command(NamedTuple):
    module: str  # submodule of the package
    summary: str
    step: str  # the step of docs/spec.md that adds the module
    function: str = "main"  # (argv: list[str]) -> int in that module


COMMANDS: dict[str, Command] = {
    "env": Command("registry.env", "write environment.json", "I5"),
    "tokenizer": Command("data.tokenizer", "train or check the tokenizer", "I4"),
    "manifest": Command("data.manifest", "build the corpus manifest", "I4"),
    "split": Command("data.split", "assign document-level splits", "I4"),
    "decontam": Command("data.decontam", "overlap report and filter", "I4"),
    "shard": Command("data.shards", "write shards", "I4"),
    "synth": Command("data.synth.build", "build Track S", "I4"),
    "train": Command("train.loop", "one run from a row config; resumes when a state exists", "I5"),
    "assemble": Command("assembly.assemble", "write an assembly manifest", "I2"),
    "verify": Command("assembly.assemble", "fidelity report", "I2", "verify_main"),
    "eval-tasks": Command("eval.score", "candidate ranking and generation scoring", "I6"),
    "eval-nll": Command("eval.nll", "NLL, BPB, 2x2 matrix", "I6"),
    "switch": Command("eval.switch", "switch-cost decomposition", "I6"),
    "audit": Command("audit.budget", "budget and parameter report", "I2"),
    "stats": Command("eval.stats", "confirmatory tests from results", "I6"),
    "bench": Command("audit.bench", "G0.5 throughput and memory grid", "I6"),
    "matrix": Command("registry.matrix", "status of the experiment matrix rows", "I6"),
    "registry": Command("registry.run", "list or show runs", "I5"),
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
    lines = [f"usage: python -m {PACKAGE} COMMAND [args]   (every command accepts --dry-run)", "", "commands:"]
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
        step = COMMANDS[name].step
        print(f"{PACKAGE}: {name!r} is not implemented yet; it lands in step {step} (docs/spec.md)", file=sys.stderr)
        return 2
    return int(getattr(importlib.import_module(module_name(name)), COMMANDS[name].function)(rest))


if __name__ == "__main__":
    sys.exit(main())
