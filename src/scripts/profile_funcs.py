"""profile performance on function calls"""

import argparse
import cProfile
import pstats

import chess

from ..search import Bot
from .positions import BENCHMARK_POSITIONS


def run_benchmark(depth: int):
    bot = Bot()
    for _, fen in BENCHMARK_POSITIONS:
        board = chess.Board(fen)
        bot.nodes_visited = 0
        _ = bot.get_best_move(board, depth=depth)


def profile_codebase(depth: int, top_n: int):
    profiler = cProfile.Profile()

    print(f"\n================ CODEBASE PROFILER (Depth {depth}) ================")
    profiler.enable()
    run_benchmark(depth)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats("cumtime")
    stats.print_stats(top_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile engine bottlenecks.")
    parser.add_argument("--depth", type=int, default=3, help="Search depth limit")
    parser.add_argument("--top", type=int, default=20, help="Number of top functions to show")
    args = parser.parse_args()

    profile_codebase(depth=args.depth, top_n=args.top)
