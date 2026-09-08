"""Profile codebase bottlenecks and function execution time using cProfile."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from typing import Any

import chess

from src.scripts.positions import get_positions
from src.scripts.profile_nps import format_score
from src.search import Bot


def run_benchmark(
    bot: Bot,
    positions: list[tuple[str, str]],
    depth: int | None = None,
    movetime_ms: int | None = None,
    quiet: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for idx, (name, fen) in enumerate(positions, 1):
        board = chess.Board(fen)

        print(f"\n[{idx}/{len(positions)}] {name}")
        print(f"FEN: {fen}")
        print("-" * 86)

        def iter_callback(info: dict[str, Any]) -> None:
            if quiet:
                return
            d = info["depth"]
            sd = info["sel_depth"]
            score = info["score_str"]
            nodes = info["nodes"]
            nps = info["nps"]
            t_ms = info["time_ms"]
            pv = info["pv"].uci() if info.get("pv") else "-"
            print(
                f"  depth {d:>2}/{sd:<2} "
                f"| score {score:>9} "
                f"| nodes {nodes:>10,} "
                f"| nps {nps:>10,} "
                f"| time {t_ms:>6}ms "
                f"| pv {pv}"
            )

        start_time = time.perf_counter()
        best_move = bot.get_best_move(
            board,
            depth=depth,
            movetime_ms=movetime_ms,
            verbose=False,
            callback=iter_callback,
        )
        elapsed = max(time.perf_counter() - start_time, 0.0001)

        completed_d = bot.completed_depth
        sel_d = bot.sel_depth
        total_nodes = bot.nodes
        nps = int(total_nodes / elapsed)
        score_str = format_score(bot.best_score)

        results.append(
            {
                "idx": idx,
                "name": name,
                "depth": completed_d,
                "sel_depth": sel_d,
                "score": score_str,
                "move": best_move.uci() if best_move else "-",
                "nodes": total_nodes,
                "time": elapsed,
                "nps": nps,
            }
        )

        print(
            f"=> Result: Depth {completed_d}/{sel_d} "
            f"| Score: {score_str} "
            f"| Move: {best_move.uci() if best_move else '-'} "
            f"| Nodes: {total_nodes:,} "
            f"| Time: {elapsed:.3f}s "
            f"| NPS: {nps:,}"
        )

    return results


def profile_codebase(
    positions: list[tuple[str, str]],
    depth: int | None = None,
    movetime_ms: int | None = None,
    top_n: int = 20,
    sort_by: str = "cumtime",
    quiet: bool = False,
) -> None:
    bot = Bot()

    if depth is None and movetime_ms is None:
        depth = 4

    if depth is not None and movetime_ms is not None:
        mode_str = f"Depth {depth} (Time Limit: {movetime_ms / 1000.0:.2f}s per position)"
    elif depth is not None:
        mode_str = f"Fixed Depth (Depth {depth})"
    else:
        assert movetime_ms is not None
        mode_str = f"Fixed Time ({movetime_ms / 1000.0:.2f}s per position)"

    print("=" * 86)
    print(f"{'ENGINE FUNCTION PROFILER (cProfile)':^86}")
    print(
        f"Mode: {mode_str} | Positions: {len(positions)} | Top: {top_n} | Sort: {sort_by}"
    )
    print("=" * 86)

    profiler = cProfile.Profile()
    profiler.enable()

    results = run_benchmark(
        bot=bot,
        positions=positions,
        depth=depth,
        movetime_ms=movetime_ms,
        quiet=quiet,
    )

    profiler.disable()

    total_nodes_all = sum(r["nodes"] for r in results)
    total_time_all = sum(r["time"] for r in results)
    overall_nps = int(total_nodes_all / total_time_all) if total_time_all > 0 else 0
    avg_depth = sum(r["depth"] for r in results) / len(results) if results else 0

    print("\n" + "=" * 86)
    print(f"{'SEARCH EXECUTION SUMMARY':^86}")
    print("=" * 86)
    print(f" Positions Profiled:  {len(results)}")
    print(f" Average Depth:       {avg_depth:.1f}")
    print(f" Total Search Time:   {total_time_all:.3f} s")
    print(f" Total Nodes Visited: {total_nodes_all:,}")
    print(f" Overall Search NPS:  {overall_nps:,}")
    print("=" * 86)

    print("\n" + "=" * 86)
    print(f"TOP {top_n} FUNCTION BOTTLENECKS (Sorted by {sort_by})".center(86))
    print("=" * 86)

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats(sort_by)
    stats.print_stats(top_n)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile engine bottlenecks with cProfile."
    )
    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        default=None,
        help="Search depth limit (default: 4 if no time specified)",
    )
    parser.add_argument(
        "--time",
        "-t",
        type=float,
        default=None,
        help="Time limit per position in seconds (e.g. 1.0, 2.5)",
    )
    parser.add_argument(
        "--time-ms",
        type=int,
        default=None,
        help="Time limit per position in milliseconds (e.g. 1000)",
    )
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=20,
        help="Number of top functions to show in profiler report (default: 20)",
    )
    parser.add_argument(
        "--sort",
        "-s",
        type=str,
        default="cumtime",
        choices=["cumtime", "tottime", "ncalls", "pcalls"],
        help="Sort metric for profiler report (default: cumtime)",
    )
    parser.add_argument(
        "--position",
        "-p",
        type=str,
        default="all",
        help="Filter positions by 1-based index, name (e.g. 'kiwipete', 'endgame'), or 'all'",
    )
    parser.add_argument(
        "--fen",
        type=str,
        default=None,
        help="Custom FEN string to profile",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress per-iteration progression output",
    )
    args = parser.parse_args()

    movetime_ms = args.time_ms
    if movetime_ms is None and args.time is not None:
        movetime_ms = int(args.time * 1000)

    try:
        positions = get_positions(position_arg=args.position, fen_arg=args.fen)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    profile_codebase(
        positions=positions,
        depth=args.depth,
        movetime_ms=movetime_ms,
        top_n=args.top,
        sort_by=args.sort,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
