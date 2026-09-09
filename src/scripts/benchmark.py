from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import chess

from src.search import MATE_SCORE, MATE_THRESHOLD, Bot, SearchStats


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    index: int
    name: str
    depth: int
    sel_depth: int
    score: int
    move: chess.Move
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
        board = chess.Board(fen)
        print(f"\n[{index}/{len(positions)}] {name}")
        print(f"FEN: {fen}")
        print("-" * 86)

        def show_iteration(info: dict[str, Any]) -> None:
            if quiet:
                return
            move = info["pv"]
            assert isinstance(move, chess.Move)
            print(
                f"  depth {info['depth']:>2}/{info['sel_depth']:<2} "
                f"| score {info['score_str']:>9} "
                f"| nodes {info['nodes']:>10,} "
                f"| nps {info['nps']:>10,} "
                f"| time {info['time_ms']:>6}ms "
                f"| pv {move.uci()}"
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
            sel_depth=bot.sel_depth,
            score=bot.best_score,
            move=move,
            nodes=bot.nodes,
            elapsed=elapsed,
            stats=bot.search_stats,
        )
        results.append(result)
        print(
            f"=> Result: Depth {result.depth}/{result.sel_depth} "
            f"| Score: {format_score(result.score)} "
            f"| Move: {result.move.uci()} "
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
        f"| LMR {stats.lmr_reductions:,} | movegen {stats.move_generations:,}"
    )


def print_summary(results: list[BenchmarkResult]) -> None:
    total_nodes = sum(result.nodes for result in results)
    total_time = sum(result.elapsed for result in results)
    overall_nps = int(total_nodes / total_time) if total_time else 0
    average_depth = sum(result.depth for result in results) / len(results) if results else 0

    print("\n" + "=" * 90)
    print(f"{'BENCHMARK SUMMARY':^90}")
    print("=" * 90)
    print(
        f" {'Pos':<3} | {'Position Name':<20} | {'Depth':<8} | {'Score':<10} | {'Move':<6} "
        f"| {'Nodes':<12} | {'Time (s)':<9} | {'NPS':<11}"
    )
    print("-" * 90)
    for result in results:
        print(
            f" {result.index:<3} | {result.name:<20} | {f'{result.depth}/{result.sel_depth}':<8} "
            f"| {format_score(result.score):<10} | {result.move.uci():<6} "
            f"| {result.nodes:>12,} | {result.elapsed:>9.3f} | {result.nps:>11,}"
        )
    print("-" * 90)
    print(
        f" {'TOTAL / OVERALL':<26} | {average_depth:>4.1f} avg | {'':<10} | {'':<6} "
        f"| {total_nodes:>12,} | {total_time:>9.3f} | {overall_nps:>11,}"
    )
    print("=" * 90 + "\n")
