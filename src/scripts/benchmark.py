from __future__ import annotations

import time
from dataclasses import dataclass

from src.board import Board, move_to_uci
from src.constants import MATE_SCORE, MATE_THRESHOLD
from src.search import Bot, SearchInfo, SearchStats


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    index: int
    name: str
    depth: int
    score: int
    move: int
    nodes: int
    elapsed: float
    stats: SearchStats | None

    @property
    def nps(self) -> int:
        return int(self.nodes / self.elapsed)


def format_score(score: int) -> str:
    if abs(score) > MATE_THRESHOLD:
        plies = MATE_SCORE - abs(score)
        mate_in = (plies + 1) // 2
        return f"mate {mate_in}" if score > 0 else f"mate -{mate_in}"
    return f"cp {score:+d}" if score else "cp 0"


def run_benchmark(
    bot: Bot,
    positions: list[tuple[str, str]],
    depth: int | None,
    movetime_ms: int | None,
    quiet: bool,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    for index, (name, fen) in enumerate(positions, 1):
        board = Board.from_fen(fen)
        print(f"\n[{index}/{len(positions)}] {name}")
        print(f"FEN: {fen}")
        print("-" * 86)

        def show_iteration(info: SearchInfo) -> None:
            if quiet:
                return
            move = info["pv"]
            assert isinstance(move, int)
            print(
                f"  depth {info['depth']:>2} "
                f"| score {info['score_str']:>9} "
                f"| nodes {info['nodes']:>10,} "
                f"| nps {info['nps']:>10,} "
                f"| time {info['time_ms']:>6}ms "
                f"| pv {move_to_uci(move)}"
            )

        start = time.perf_counter()
        move = bot.get_best_move(
            board,
            time_left_ms=100_000,
            depth=depth,
            movetime_ms=movetime_ms,
            verbose=False,
            callback=show_iteration,
        )
        elapsed = max(time.perf_counter() - start, 0.0001)
        result = BenchmarkResult(
            index=index,
            name=name,
            depth=bot.completed_depth,
            score=bot.best_score,
            move=move,
            nodes=bot.nodes,
            elapsed=elapsed,
            stats=bot.search_stats,
        )
        results.append(result)
        print(
            f"=> Result: Depth {result.depth} "
            f"| Score: {format_score(result.score)} "
            f"| Move: {move_to_uci(result.move)} "
            f"| Nodes: {result.nodes:,} "
            f"| Time: {result.elapsed:.3f}s "
            f"| NPS: {result.nps:,}"
        )
        if result.stats is not None:
            print_stats(result.stats)

    return results


def print_stats(stats: SearchStats) -> None:
    qshare = 100 * stats.qnodes / stats.nodes if stats.nodes else 0.0
    ttrate = 100 * stats.tt_hits / stats.tt_probes if stats.tt_probes else 0.0
    print(
        f"   Stats: qnodes {stats.qnodes:,} ({qshare:.1f}%) "
        f"| TT {stats.tt_hits:,}/{stats.tt_probes:,} ({ttrate:.1f}%), cutoffs {stats.tt_cutoffs:,} "
        f"| beta {stats.beta_cutoffs:,} "
        f"| prunes rfp {stats.rfp_prunes:,}, null {stats.null_prunes:,} "
        f"futility {stats.futility_prunes:,} "
        f"| LMR {stats.lmr_reductions:,}"
    )


def print_summary(results: list[BenchmarkResult]) -> None:
    total_nodes = sum(result.nodes for result in results)
    total_time = sum(result.elapsed for result in results)
    overall_nps = int(total_nodes / total_time) if total_time else 0
    average_depth = sum(result.depth for result in results) / len(results) if results else 0

    print("\n" + "=" * 86)
    print(f"{'BENCHMARK SUMMARY':^86}")
    print("=" * 86)
    print(
        f"Positions: {len(results)} | "
        f"Total Nodes: {total_nodes:,} | "
        f"Total Time: {total_time:.3f}s | "
        f"Average Depth: {average_depth:.1f}"
    )
    print(f"Overall NPS: {overall_nps:,}")
    print("=" * 86)


BENCHMARK_POSITIONS = [
    ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("Kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("Silver Suite Pos 2", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("Middlegame Pos 3", "r1bqk2r/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQK2R w KQkq - 2 7"),
    ("Endgame Pos 4", "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1"),
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run engine benchmark suite.")
    parser.add_argument("--depth", type=int, default=6, help="Search depth per position.")
    parser.add_argument("--movetime", type=int, default=None, help="Move time in ms per position.")
    parser.add_argument(
        "--position",
        type=str,
        default="all",
        choices=["all", "startpos", "kiwipete"],
        help="Position to benchmark.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-iteration logging.")
    args = parser.parse_args()

    positions = BENCHMARK_POSITIONS
    if args.position == "startpos":
        positions = [BENCHMARK_POSITIONS[0]]
    elif args.position == "kiwipete":
        positions = [BENCHMARK_POSITIONS[1]]

    bot = Bot()
    results = run_benchmark(
        bot, positions, depth=args.depth, movetime_ms=args.movetime, quiet=args.quiet
    )
    print_summary(results)
