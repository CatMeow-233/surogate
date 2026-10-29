"""`python -m surogate_tlc COMMAND ...`: the commands that work on model directories (README: 命令行).

init writes a model directory, train runs a configuration, eval, probe and sample measure a model directory, export
turns a checkpoint into one. The data tools are modules of their own: python -m surogate_tlc.data.<tool>.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _write(path: Path | None, result: object) -> None:
    if path:
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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
    _print({"run_dir": record["sft"]["output_dir"], "summary": record["attempts"][-1].get("summary")})
    return 0


def _eval(args: argparse.Namespace) -> int:
    from .eval.loss import evaluate

    result = evaluate(args.model, args.data, split=args.split, batch=args.batch, max_rows=args.max_rows)
    _write(args.out, result)
    _print(result)
    return 0


def _probe(args: argparse.Namespace) -> int:
    from .eval.probes import evaluate_suite

    result = evaluate_suite(args.model, args.data, args.suite, batch=args.batch, max_items=args.max_items)
    _write(args.out, result)
    _print({k: v for k, v in result.items() if k != "results"})  # the per-item results go to --out only
    return 0


def _sample(args: argparse.Namespace) -> int:
    from .eval.samples import sample

    result = sample(args.model, args.data, max_new=args.max_new, batch=args.batch)
    _write(args.out, result)
    _print(result)
    return 0


def _export(args: argparse.Namespace) -> int:
    from .checkpoint import export

    _print({"model_dir": str(export(args.checkpoint, args.source, args.out))})
    return 0


COMMANDS: dict[str, tuple[Callable[[argparse.Namespace], int], str]] = {
    "init": (_init, "write a model directory of a P0 row, with its initial weights"),
    "train": (_train, "train one run configuration (fork/tlc/examples/p0.yaml)"),
    "eval": (_eval, "teacher-forced loss on a shard split, against the unigram baseline"),
    "probe": (_probe, "binding probes: candidate scoring on a suite file"),
    "sample": (_sample, "greedy samples at fixed prompts"),
    "export": (_export, "model directory from a checkpoint step directory"),
}


def _parser() -> argparse.ArgumentParser:
    from .hf import ROWS

    parser = argparse.ArgumentParser(prog="python -m surogate_tlc", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    sub = {name: commands.add_parser(name, help=text) for name, (_, text) in COMMANDS.items()}
    sub["init"].add_argument("row", choices=list(ROWS))
    sub["init"].add_argument("out", type=Path, help="model directory to create (absent or empty)")
    sub["init"].add_argument("--tokenizer", type=Path, required=True, help="tokenizer directory (data.tokenizer)")
    sub["init"].add_argument("--seed", type=int, required=True, help="init_seed, recorded in tlc_meta.json")
    sub["train"].add_argument("config", type=Path, help="run configuration (YAML)")
    sub["train"].add_argument("--stage", help="stage of the configuration, e.g. smoke, overfit, pilot")
    sub["train"].add_argument(
        "--microbatch", type=int, help="microbatch size; the accumulation keeps tokens per update"
    )
    for name in ("eval", "probe", "sample"):
        sub[name].add_argument("model", type=Path, help="model directory")
        sub[name].add_argument("data", type=Path, help="shard directory (data.shards): its rows set the length")
        sub[name].add_argument("--batch", type=int, default=32 if name != "sample" else 8, help="rows per forward")
        sub[name].add_argument("--out", type=Path, help="also write the full result to this JSON file")
    sub["eval"].add_argument("--split", default="val", choices=["train", "val"])
    sub["eval"].add_argument("--max-rows", type=int, help="rows of the split to use")
    sub["probe"].add_argument(
        "suite", type=Path, help="suite file (dev.jsonl or test.jsonl of data.synth or data.natural)"
    )
    sub["probe"].add_argument("--max-items", type=int, help="items of the suite to use")
    sub["sample"].add_argument("--max-new", type=int, default=64, help="tokens per sample at most")
    sub["export"].add_argument("checkpoint", type=Path, help="step_<8 digits> directory of a run")
    sub["export"].add_argument("source", type=Path, help="the model directory the run started from")
    sub["export"].add_argument("out", type=Path, help="model directory to create (absent or empty)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return COMMANDS[args.command][0](args)


if __name__ == "__main__":
    sys.exit(main())
