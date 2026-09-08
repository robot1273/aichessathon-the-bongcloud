"""Benchmark search speed across representative positions."""

from __future__ import annotations

import argparse
import sys

from src.scripts.benchmark import print_summary, run_benchmark
from src.scripts.positions import get_positions
from src.search import Bot


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark engine NPS.")
    parser.add_argument("--depth", "-d", type=int, help="Fixed search depth")
    parser.add_argument("--time", "-t", type=float, help="Fixed time in seconds per position")
    parser.add_argument("--time-ms", type=int, help="Fixed time in milliseconds per position")
    parser.add_argument("--position", "-p", default="all", help="Position index, name, or 'all'")
    parser.add_argument("--fen", help="Custom FEN string")
    parser.add_argument("--quiet", "-q", action="store_true", help="Hide iteration output")
    args = parser.parse_args()

    movetime_ms = args.time_ms
    if movetime_ms is None and args.time is not None:
        movetime_ms = int(args.time * 1000)
    depth = args.depth if args.depth is not None or movetime_ms is not None else 5

    try:
        positions = get_positions(position_arg=args.position, fen_arg=args.fen)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    mode = f"Depth {depth}" if movetime_ms is None else f"Time {movetime_ms / 1000:.2f}s"
    print("=" * 86)
    print(f"{'ENGINE NPS BENCHMARK':^86}")
    print(f"Mode: {mode} | Positions: {len(positions)}")
    print("=" * 86)
    print_summary(
        run_benchmark(Bot(), positions, depth=depth, movetime_ms=movetime_ms, quiet=args.quiet)
    )


if __name__ == "__main__":
    main()
