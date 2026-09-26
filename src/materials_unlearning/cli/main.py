"""Unified command-line entry point."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence


COMMANDS = {
    "cache-features": "materials_unlearning.cache_features",
    "collect-neural": "materials_unlearning.collect_neural",
    "controlled": "materials_unlearning.controlled",
    "neural-floor": "materials_unlearning.neural_floor",
    "neural-geometry": "materials_unlearning.neural_geometry",
    "neural-shard": "materials_unlearning.neural_shard",
    "operating-points": "materials_unlearning.operating_points",
    "paired": "materials_unlearning.paired",
    "prepare-data": "materials_unlearning.prepare_data",
    "ridge": "materials_unlearning.ridge",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="materials-unlearning",
        description="Run materials-unlearning analyses and data utilities.",
    )
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    module = importlib.import_module(COMMANDS[args.command])
    previous_argv = sys.argv
    sys.argv = [f"materials-unlearning {args.command}", *args.arguments]
    try:
        result = module.main()
    finally:
        sys.argv = previous_argv
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
