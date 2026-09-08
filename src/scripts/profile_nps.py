import argparse
import time

import chess
from positions import BENCHMARK_POSITIONS

from ..search import Bot


def profile_bot(depth: int) -> None:
    bot = Bot()

    total_nodes = 0
    total_time = 0.0

    print(f"\n================ PERFORMANCE PROFILE (Depth {depth}) ================")
    print(f"{'Pos':<5} | {'Name':<15} | {'Time (s)':<10} | {'Nodes':<12} | {'NPS':<12}")
    print("-" * 65)

    for idx, (name, fen) in enumerate(BENCHMARK_POSITIONS, 1):
        board = chess.Board(fen)

        bot.nodes_visited = 0

        start_time = time.perf_counter()
        bot.get_best_move(board, depth=depth)
        elapsed = time.perf_counter() - start_time

        nodes = bot.nodes_visited
        nps = nodes / elapsed if elapsed > 0 else 0

        total_nodes += nodes
        total_time += elapsed

        print(f"{idx:<5} | {name:<15} | {elapsed:<10.3f} | {nodes:<12,} | {nps:<12,.0f}")

    avg_nps = total_nodes / total_time if total_time > 0 else 0

    print("=" * 65)
    print(f"TOTAL TIME:   {total_time:.3f} s")
    print(f"TOTAL NODES:  {total_nodes:,}")
    print(f"AVERAGE NPS:  {avg_nps:,.0f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Bot search performance in NPS.")
    parser.add_argument("--depth", type=int, default=5, help="Search depth limit")
    parser.add_argument(
        "--agent", type=str, default=".", help="Unused parameter for Makefile compatibility"
    )
    args = parser.parse_args()

    profile_bot(depth=args.depth)
