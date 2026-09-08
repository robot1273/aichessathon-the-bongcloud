"""Profile engine search performance and NPS (Nodes Per Second)."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import chess

from src.scripts.positions import get_positions
from src.search import Bot


def format_score(score: int) -> str:
    """Format evaluation score into cp or mate."""
    from src.search import MATE_SCORE, MATE_THRESHOLD

    if abs(score) > MATE_THRESHOLD:
        plies = MATE_SCORE - abs(score)
        mate_in = (plies + 1) // 2
        return f"mate {mate_in}" if score > 0 else f"mate -{mate_in}"
    return f"cp {score:+d}" if score != 0 else "cp 0"


def profile_benchmark(
    positions: list[tuple[str, str]],
    depth: int | None = None,
    movetime_ms: int | None = None,
    quiet: bool = False,
) -> None:
    bot = Bot()

    if depth is None and movetime_ms is None:
        depth = 5

    # Determine mode string
    if depth is not None and movetime_ms is not None:
        mode_str = f"Depth {depth} (Time Limit: {movetime_ms / 1000.0:.2f}s per position)"
    elif depth is not None:
        mode_str = f"Fixed Depth (Depth {depth})"
    else:
        assert movetime_ms is not None
        mode_str = f"Fixed Time ({movetime_ms / 1000.0:.2f}s per position)"

    print("=" * 86)
    print(f"{'ENGINE NPS BENCHMARK':^86}")
    print(f"Mode: {mode_str} | Positions: {len(positions)}")
    print("=" * 86)

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
        score = bot.best_score
        score_str = format_score(score)

        results.append(
            {
                "idx": idx,
                "name": name,
                "depth": f"{completed_d}/{sel_d}",
                "completed_depth": completed_d,
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

    # Summary Table
    total_nodes_all = sum(r["nodes"] for r in results)
    total_time_all = sum(r["time"] for r in results)
    overall_nps = int(total_nodes_all / total_time_all) if total_time_all > 0 else 0
    avg_depth = (
        sum(r["completed_depth"] for r in results) / len(results) if results else 0
    )

    print("\n" + "=" * 90)
    print(f"{'BENCHMARK SUMMARY':^90}")
    print("=" * 90)
    print(
        f" {'Pos':<3} | {'Position Name':<20} | {'Depth':<8} | {'Score':<10} | {'Move':<6} "
        f"| {'Nodes':<12} | {'Time (s)':<9} | {'NPS':<11}"
    )
    print("-" * 90)

    for r in results:
        pos_part = f" {r['idx']:<3} | {r['name']:<20} | {r['depth']:<8}"
        eval_part = f" | {r['score']:<10} | {r['move']:<6}"
        stats_part = f" | {r['nodes']:>12,} | {r['time']:>9.3f} | {r['nps']:>11,}"
        print(pos_part + eval_part + stats_part)

    print("-" * 90)
    total_label = f" {'TOTAL / OVERALL':<26} | {avg_depth:>4.1f} avg | {'':<10} | {'':<6}"
    total_stats = f" | {total_nodes_all:>12,} | {total_time_all:>9.3f} | {overall_nps:>11,}"
    print(total_label + total_stats)
    print("=" * 90 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile engine search performance and NPS across benchmark positions."
    )
    parser.add_argument(
        "--depth",
        "-d",
        type=int,
        default=None,
        help="Search depth limit (default: 5 if no time specified)",
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
    parser.add_argument(
        "--agent",
        type=str,
        default=".",
        help="Unused parameter for Makefile compatibility",
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

    profile_benchmark(
        positions=positions,
        depth=args.depth,
        movetime_ms=movetime_ms,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
