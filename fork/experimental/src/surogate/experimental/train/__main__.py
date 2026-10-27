"""Explicit CLI; this entry point never imports the native trainer."""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent experimental PyTorch training for Surogate")
    parser.add_argument("config", help="experimental YAML or JSON configuration")
    parser.add_argument("--resume", help="completed-step experimental checkpoint file")
    parser.add_argument("--validate-only", action="store_true", help="validate configuration without loading a model")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.validate_only:
            print(json.dumps({"status": "valid", "backend": "experimental-pytorch", "config": config.to_dict()}))
            return 0
        from .engine import run_training

        print(json.dumps(run_training(config, resume_from=args.resume), default=str))
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"experimental training: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
