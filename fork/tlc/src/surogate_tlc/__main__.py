"""`python -m surogate_tlc COMMAND ...`: the TLC commands, and every `surogate` command with the TLC models registered.

TLC commands: init, train-tokenizer, build-data, train, eval, export (README: 命令行). Any other command goes to
upstream's CLI after register(): cli_main runs its command module in this process, so the module sees the TLC models.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

# Commands that take the arguments of an existing module entry point.
DELEGATED = {
    "train-tokenizer": "train a tokenizer (python -m surogate_tlc.data.tokenizer train)",
    "build-data": "write the token shards (python -m surogate_tlc.data.shards)",
}


def _parser() -> argparse.ArgumentParser:
    from .hf import ROWS

    parser = argparse.ArgumentParser(
        prog="python -m surogate_tlc",
        description="TLC commands. Any other command runs `surogate COMMAND ...` with the TLC models registered.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    init = commands.add_parser("init", help="write a model directory of a P0 row, with its initial weights")
    init.add_argument("row", choices=list(ROWS))
    init.add_argument("out", type=Path, help="model directory to create (absent or empty)")
    init.add_argument("--tokenizer", type=Path, required=True, help="tokenizer directory from train-tokenizer")
    init.add_argument("--seed", type=int, required=True, help="init_seed, recorded in tlc_meta.json")
    for name, text in DELEGATED.items():
        commands.add_parser(name, help=text, add_help=False)
    train = commands.add_parser("train", help="train one run configuration (fork/tlc/examples)")
    train.add_argument("config", type=Path, help="run configuration (YAML)")
    train.add_argument("--stage", help="stage of the configuration, e.g. smoke, overfit, pilot")
    train.add_argument("--microbatch", type=int, help="microbatch size; the accumulation keeps tokens per update")
    evaluate = commands.add_parser("eval", help="teacher-forced loss of a model directory on a shard split")
    evaluate.add_argument("model", type=Path, help="model directory")
    evaluate.add_argument("data", type=Path, help="shard directory from build-data")
    evaluate.add_argument("--split", default="val", choices=["train", "val"])
    evaluate.add_argument("--batch", type=int, default=32, help="rows per native forward")
    evaluate.add_argument("--max-rows", type=int)
    evaluate.add_argument("--out", type=Path, help="also write the result to this JSON file")
    export = commands.add_parser("export", help="model directory from a checkpoint step directory")
    export.add_argument("checkpoint", type=Path, help="step_<8 digits> directory of a run")
    export.add_argument("source", type=Path, help="the model directory the run started from")
    export.add_argument("out", type=Path, help="model directory to create (absent or empty)")
    return parser


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _passthrough(argv: list[str]) -> int:
    from . import register

    register()
    sys.argv = ["surogate", *argv]  # surogate.cli.main reads sys.argv on import
    from surogate.cli.main import cli_main

    cli_main()
    return 0


def _init(args: argparse.Namespace) -> int:
    from . import hf, init, modeldir

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty")
    modeldir.write_model_dir(args.out, hf.row_config(args.row), args.tokenizer, init_seed=args.seed)
    _print({"model_dir": str(args.out), "weights": str(init.init_model_dir(args.out))})
    return 0


def _train(args: argparse.Namespace) -> int:
    from .train import config, driver

    try:
        record = driver.train(config.load(args.config, stage=args.stage, microbatch=args.microbatch))
    except driver.RunError as error:
        print(error, file=sys.stderr)
        return 2
    except driver.RunFailed as error:
        print(error, file=sys.stderr)
        return 1
    summary = record["attempts"][-1].get("summary")
    _print({"run_dir": record["sft"]["output_dir"], "downshifts": record["downshifts"], "summary": summary})
    return 0


def _eval(args: argparse.Namespace) -> int:
    from .eval.loss import evaluate

    result = evaluate(args.model, args.data, split=args.split, batch=args.batch, max_rows=args.max_rows)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _print(result)
    return 0


def _export(args: argparse.Namespace) -> int:
    from .checkpoint import export

    _print({"model_dir": str(export(args.checkpoint, args.source, args.out))})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "train-tokenizer":
        from .data import tokenizer

        return tokenizer.main(["train", *argv[1:]])
    if argv and argv[0] == "build-data":
        from .data import shards

        return shards.main(argv[1:])
    if argv and not argv[0].startswith("-") and argv[0] not in {"init", "train", "eval", "export"}:
        return _passthrough(argv)
    args = _parser().parse_args(argv)
    return {"init": _init, "train": _train, "eval": _eval, "export": _export}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
