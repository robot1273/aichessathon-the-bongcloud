"""Bot wrapper for Numba-jitted search."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.board import Board, move_to_uci
from src.constants import INF, MATE_SCORE, MATE_THRESHOLD, MAX_PLY, NO_MOVE, STATE_SIZE
from src.move_ordering import order_moves
from src.search_numba import (
    board_to_state,
    search_root,
)
from src.tablebase import RootTablebase, initialize_tablebase, tb_probe_root
from src.time_manager import DEFAULT_TIME_CONFIG, TimeConfig, TimeManager
from src.tt import create_tt_arrays, probe_tt

# Libc clock for microsecond time tracking
libc = ctypes.CDLL(None)
clock = libc.clock
clock.restype = ctypes.c_long
clock.argtypes = []


STATS_NODES = 0
STATS_ABORTED = 1
STATS_T0 = 2
STATS_DEADLINE = 3
STATS_QNODES = 4
STATS_TT_PROBES = 5
STATS_TT_HITS = 6
STATS_TT_CUTOFFS = 7
STATS_BETA_CUTOFFS = 8
STATS_RFP_PRUNES = 9
STATS_NULL_PRUNES = 10
STATS_FUTILITY_PRUNES = 11
STATS_LMR = 12
STATS_SIZE = 13


@dataclass
class SearchStats:
    nodes: int = 0
    qnodes: int = 0
    move_generations: int = 0
    tt_probes: int = 0
    tt_hits: int = 0
    tt_cutoffs: int = 0
    beta_cutoffs: int = 0
    rfp_prunes: int = 0
    futility_prunes: int = 0
    null_prunes: int = 0
    lmr_reductions: int = 0


INCREMENT_S = 0.5


@dataclass(frozen=True)
class MoveTiming:
    clock_before_s: float
    soft_limit_s: float
    hard_limit_s: float
    elapsed_s: float
    completed_depth: int
    aborted_depth: int | None
    predicted_next_s: float
    aspiration_used: bool
    aspiration_failed: bool
    aspiration_reserve_skip: bool
    root_gap: int | None
    band: str
    policy: str
    stop_reason: str

    def format(self) -> str:
        aborted = "-" if self.aborted_depth is None else str(self.aborted_depth)
        gap = "-" if self.root_gap is None else str(self.root_gap)
        return (
            "timing "
            f"clock={self.clock_before_s:.3f}s soft={self.soft_limit_s:.3f}s "
            f"hard={self.hard_limit_s:.3f}s elapsed={self.elapsed_s:.3f}s "
            f"depth={self.completed_depth} aborted={aborted} "
            f"prediction={self.predicted_next_s:.3f}s "
            f"aspiration={int(self.aspiration_used)}/{int(self.aspiration_failed)} "
            f"reserve_skip={int(self.aspiration_reserve_skip)} root_gap={gap} "
            f"band={self.band} policy={self.policy} stop={self.stop_reason}"
        )


def is_root_ambiguous(root_gap: int | None, config: TimeConfig) -> bool:
    if root_gap is None or root_gap > config.root_uncertainty_gap:
        return False
    return not config.require_positive_root_gap or root_gap > 0


class Bot:
    def __init__(
        self,
        tt_exp_size: int = 20,
        collect_stats: bool = False,
        increment_s: float = INCREMENT_S,
        time_config: TimeConfig = DEFAULT_TIME_CONFIG,
        trace_timing: bool = False,
        use_tb: bool = True,
    ) -> None:
        self.tt_exp_size = tt_exp_size
        self.collect_stats = collect_stats
        self.increment_s = increment_s
        self.time_config = time_config
        self.trace_timing = trace_timing
        self.use_tb = use_tb
        self.current_age = 0
        self.tt_arrays = create_tt_arrays(tt_exp_size)
        self.time_mgr = TimeManager(increment_s=increment_s, config=time_config)

        # Pre-allocate reusable search arrays across moves
        self.undo_stack = np.zeros((MAX_PLY, STATE_SIZE), dtype=np.uint64)
        self.moves_stack = np.zeros((MAX_PLY, 256), dtype=np.int32)
        self.scores_stack = np.zeros((MAX_PLY, 256), dtype=np.int32)
        self.killers = np.zeros((MAX_PLY, 2), dtype=np.int32)
        self.history = np.zeros((2, 64, 64), dtype=np.int32)
        self.stats = np.zeros(STATS_SIZE, dtype=np.int64)

        self.completed_depth = 0
        self.sel_depth = 0
        self.best_score = 0
        self.nodes = 0
        self.best_move = NO_MOVE
        self.runner_up_score = -INF
        self.root_score_gap: int | None = None
        self.search_stats: SearchStats | None = None
        self.last_timing: MoveTiming | None = None
        self._game_hashes: list[int] = []
        self._pending_position: Board | None = None

        if self.use_tb:
            initialize_tablebase()
        self._warmup_jit()

    def _warmup_jit(self) -> None:
        """Warm up the JIT compiler with a dummy search."""
        board = Board.from_fen()
        # Warm the Python-board attack path (bishop/rook magic lookups with
        # scalar signatures) so the first real move pays ~20us, not ~40-150ms
        # of lazy Numba compilation on clock. See B1 baseline.
        board.generate_moves()
        board.is_in_check()
        board.evaluate()
        ep_board = Board.from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq e3 0 1"
        )
        ep_board.generate_moves()
        state = board_to_state(board)
        hash_history = np.zeros(MAX_PLY, dtype=np.uint64)
        # S5: depth 1 suffices — Numba compiles whole reachable call graph on
        # first call regardless of branch depth (NMP/futility/LMR need no
        # execution for compilation). Depth 3 only burned search time.
        search_root(
            state,
            self.undo_stack,
            self.moves_stack,
            self.scores_stack,
            -100,
            100,
            1,
            self.killers,
            self.history,
            self.stats,
            hash_history,
            0,
            0,
            *self.tt_arrays,
        )
        for array in self.tt_arrays:
            array.fill(0)
        self.killers.fill(0)
        self.history.fill(0)
        self.stats.fill(0)

    def get_best_move(
        self,
        board: Board,
        time_left_ms: int,
        depth: int | None = None,
        movetime_ms: int | None = None,
        verbose: bool = False,
        callback: Any | None = None,
        started_at: float | None = None,
    ) -> int:
        self.current_age += 1
        # Include all per-move Python work in the budget enforced by the runner.
        self.time_mgr.start(time_left_ms, board, movetime_ms, started_at)
        self._record_position(board)

        # Check for immediate return on forced moves or no moves
        legal_moves = board.generate_moves()
        if not legal_moves:
            self.best_move = NO_MOVE
            self.nodes = 0
            self.completed_depth = 0
            self.best_score = 0
            self.runner_up_score = -INF
            self.root_score_gap = None
            self._record_timing(0, None, 0.0, False, False, False, None, "no-legal-move")
            return NO_MOVE

        if len(legal_moves) == 1:
            self.best_move = legal_moves[0]
            self.nodes = 1
            self.completed_depth = 1
            self.best_score = 0
            self.runner_up_score = -INF
            self.root_score_gap = None
            self._record_selected_move(board, legal_moves[0])
            self._record_timing(1, None, 0.0, False, False, False, None, "forced-move")
            return legal_moves[0]

        # Prepare an exact root policy before search. Fixed-depth callers are
        # benchmarks/tests of the search itself; the platform never sets depth.
        tb_root: RootTablebase | None = None
        if depth is None and self.use_tb:
            tb_root = tb_probe_root(board, legal_moves, self._game_hashes)
            if tb_root is not None and tb_root.wdl == -2:
                self.best_move = tb_root.fallback_move
                self.nodes = 1
                self.completed_depth = 1
                self.best_score = 0
                self.runner_up_score = -INF
                self.root_score_gap = None
                self._record_selected_move(board, tb_root.fallback_move)
                self._record_timing(
                    1, None, 0.0, False, False, False, None, "tablebase"
                )
                return tb_root.fallback_move

        default_move = legal_moves[0]
        if (depth is None or movetime_ms is not None) and self.time_mgr.is_time_up():
            self.best_move = default_move
            self.nodes = 0
            self.completed_depth = 0
            self.best_score = 0
            self.runner_up_score = -INF
            self.root_score_gap = None
            self._record_selected_move(board, default_move)
            self._record_timing(0, None, 0.0, False, False, False, None, "request-deadline")
            return default_move

        # Prioritize known TT move or best-ordered move for safe emergency fallback.
        found, tt_move_val, _, _, _, _ = probe_tt(
            np.uint64(board.hash),
            *self.tt_arrays,
        )
        if found and tt_move_val in legal_moves:
            default_move = tt_move_val
        else:
            ordered = order_moves(board, legal_moves, tt_move=tt_move_val if found else 0)
            default_move = ordered[0]

        state = board_to_state(board)
        start_t = self.time_mgr.elapsed()

        # Build game history for repetition checking. The current root is last.
        hash_history = np.zeros(MAX_PLY, dtype=np.uint64)
        hist_len = min(len(self._game_hashes), MAX_PLY)
        for i in range(hist_len):
            hash_history[i] = np.uint64(self._game_hashes[-hist_len + i])

        # The search measures CPU time, while the runner measures wall time.
        # Convert the remaining complete-request budget at every deadline update.
        if depth is not None:
            # Fixed depth: no deadline unless movetime_ms specified
            deadline_ticks = 0
            if movetime_ms is not None:
                deadline_ticks = clock() + int(
                    max(self.time_mgr.hard_limit - self.time_mgr.elapsed(), 0.0) * 1_000_000
                )
        else:
            deadline_ticks = clock() + int(
                max(self.time_mgr.hard_limit - self.time_mgr.elapsed(), 0.0) * 1_000_000
            )

        self.stats[0] = 0  # nodes
        self.stats[1] = 0  # aborted flag
        self.stats[2] = clock()
        self.stats[3] = deadline_ticks
        self.stats[4:] = 0  # telemetry counters (S7); nodes/abort above

        # Clear killer heuristics per search; age history
        self.killers.fill(0)
        self.history //= 2

        best_move = default_move
        best_score = 0
        prev_score: int | None = None
        prev_move = NO_MOVE
        stable_iterations = 0
        iteration_times: list[float] = []
        root_move_count = len(legal_moves)
        root_in_check = board.is_in_check()
        last_root_ambiguous = False
        last_aspiration_failed = False
        runner_up_score = -INF
        root_score_gap: int | None = None
        completed_runner_up_score = -INF
        completed_root_score_gap: int | None = None
        completed_depth = 0
        aborted_depth: int | None = None
        last_prediction = 0.0
        aspiration_used_any = False
        aspiration_failed_any = False
        aspiration_reserve_skip = False
        stop_reason = "max-depth"
        max_d = depth if depth is not None else 64

        for d in range(1, max_d + 1):
            if d > 1 and depth is None and self.time_mgr.should_stop_iterating():
                stop_reason = "soft-limit"
                break

            if (
                d > self.time_config.panic_max_depth
                and depth is None
                and self.time_mgr.is_panic_clock()
            ):
                stop_reason = "panic-depth"
                break

            predicted_time = 0.0
            prediction_safety = self.time_config.iteration_safety_factor
            if len(iteration_times) >= 2:
                previous_time = max(iteration_times[-2], 0.0001)
                growth = iteration_times[-1] / previous_time
                minimum_growth = self.time_config.normal_min_growth
                if root_move_count >= self.time_config.busy_root_moves or root_in_check:
                    minimum_growth = self.time_config.busy_min_growth
                    prediction_safety = self.time_config.busy_prediction_safety
                if last_root_ambiguous or last_aspiration_failed:
                    minimum_growth = max(
                        minimum_growth,
                        self.time_config.uncertain_min_growth,
                    )
                    prediction_safety = max(
                        prediction_safety,
                        self.time_config.uncertain_prediction_safety,
                    )
                predicted_time = iteration_times[-1] * min(
                    max(growth, minimum_growth),
                    self.time_config.max_growth,
                )
            last_prediction = predicted_time

            if (
                d >= self.time_config.prediction_min_depth
                and depth is None
                and predicted_time
                and self.time_mgr.elapsed() + predicted_time * prediction_safety
                >= self.time_mgr.hard_limit
            ):
                stop_reason = "prediction"
                break

            iteration_start = self.time_mgr.elapsed()
            aspiration_failed = False
            use_aspiration = (
                d >= self.time_config.aspiration_min_depth
                and abs(best_score) < MATE_THRESHOLD
            )
            if use_aspiration and (self.time_mgr.is_low_clock() or not predicted_time):
                use_aspiration = False
            elif use_aspiration:
                retry_reserve = predicted_time * prediction_safety
                if (
                    self.time_mgr.elapsed()
                    + self.time_config.aspiration_retry_reserve * retry_reserve
                    >= self.time_mgr.hard_limit
                ):
                    use_aspiration = False
                    aspiration_reserve_skip = True
            aspiration_used_any = aspiration_used_any or use_aspiration

            # Aspiration window for configured depths
            if use_aspiration:
                delta = self.time_config.aspiration_delta
                alpha = max(-INF, best_score - delta)
                beta = min(INF, best_score + delta)
                score, move, runner_up_score, aborted = search_root(
                    state,
                    self.undo_stack,
                    self.moves_stack,
                    self.scores_stack,
                    alpha,
                    beta,
                    d,
                    self.killers,
                    self.history,
                    self.stats,
                    hash_history,
                    hist_len,
                    self.current_age,
                    *self.tt_arrays,
                )

                if not aborted and score <= alpha:
                    # Fail low: widen window to -INF
                    aspiration_failed = True
                    retry_time = self.time_mgr.elapsed() - iteration_start
                    if (
                        self.time_mgr.elapsed() + retry_time * prediction_safety
                        >= self.time_mgr.hard_limit
                    ):
                        aborted = True
                    else:
                        score, move, runner_up_score, aborted = search_root(
                            state,
                            self.undo_stack,
                            self.moves_stack,
                            self.scores_stack,
                            -INF,
                            beta,
                            d,
                            self.killers,
                            self.history,
                            self.stats,
                            hash_history,
                            hist_len,
                            self.current_age,
                            *self.tt_arrays,
                        )
                elif not aborted and score >= beta:
                    # Fail high: widen window to INF
                    aspiration_failed = True
                    retry_time = self.time_mgr.elapsed() - iteration_start
                    if (
                        self.time_mgr.elapsed() + retry_time * prediction_safety
                        >= self.time_mgr.hard_limit
                    ):
                        aborted = True
                    else:
                        score, move, runner_up_score, aborted = search_root(
                            state,
                            self.undo_stack,
                            self.moves_stack,
                            self.scores_stack,
                            alpha,
                            INF,
                            d,
                            self.killers,
                            self.history,
                            self.stats,
                            hash_history,
                            hist_len,
                            self.current_age,
                            *self.tt_arrays,
                        )
            else:
                score, move, runner_up_score, aborted = search_root(
                    state,
                    self.undo_stack,
                    self.moves_stack,
                    self.scores_stack,
                    -INF,
                    INF,
                    d,
                    self.killers,
                    self.history,
                    self.stats,
                    hash_history,
                    hist_len,
                    self.current_age,
                    *self.tt_arrays,
                )

            if aborted:
                aborted_depth = d
                stop_reason = "hard-limit"
                break

            if move != NO_MOVE:
                best_move = move
                best_score = score
            completed_depth = d
            iteration_times.append(self.time_mgr.elapsed() - iteration_start)

            self.time_mgr.extend_if_unstable(prev_score, score)
            root_score_gap = score - runner_up_score if runner_up_score != -INF else None
            last_root_ambiguous = is_root_ambiguous(root_score_gap, self.time_config)
            completed_runner_up_score = runner_up_score
            completed_root_score_gap = root_score_gap
            if last_root_ambiguous:
                self.time_mgr.extend_for_root_uncertainty()
            if (
                move == prev_move
                and prev_score is not None
                and abs(score - prev_score) <= self.time_config.stable_score_delta
            ):
                stable_iterations += 1
            else:
                stable_iterations = 1

            if (
                best_score >= self.time_config.stable_win_score
                and stable_iterations >= self.time_config.stable_win_iterations
                and not last_root_ambiguous
            ):
                self.time_mgr.shorten_for_stable_win()

            if depth is None or movetime_ms is not None:
                self.stats[3] = clock() + int(
                    max(self.time_mgr.hard_limit - self.time_mgr.elapsed(), 0.0) * 1_000_000
                )
            prev_score = score
            prev_move = move
            last_aspiration_failed = aspiration_failed
            aspiration_failed_any = aspiration_failed_any or aspiration_failed

            nodes = int(self.stats[0])
            elapsed = max(self.time_mgr.elapsed() - start_t, 0.0001)
            nps = int(nodes / elapsed)

            if verbose or callback:
                if abs(best_score) > MATE_THRESHOLD:
                    mate_dist = (MATE_SCORE - abs(best_score) + 1) // 2
                    score_str = f"mate {mate_dist if best_score > 0 else -mate_dist}"
                else:
                    score_str = f"cp {best_score}"

                info = {
                    "depth": d,
                    "sel_depth": d,
                    "score": best_score,
                    "score_str": score_str,
                    "nodes": nodes,
                    "nps": nps,
                    "time": elapsed,
                    "time_ms": int(elapsed * 1000),
                    "pv": best_move,
                    "root_score_gap": root_score_gap,
                }
                if callback:
                    callback(info)
                if verbose:
                    print(
                        f"info depth {d} score {score_str} nodes {nodes} "
                        f"nps {nps} time {int(elapsed * 1000)} pv {move_to_uci(best_move)}"
                    )

            if abs(best_score) > MATE_THRESHOLD:
                stop_reason = "mate"
                break

        # Search supplies technique and swindling chances, but never gets to
        # discard a tablebase win/draw. Probing already happened before the
        # deadline, so this fallback is constant-time even after an abort.
        if tb_root is not None and best_move not in tb_root.search_moves:
            best_move = tb_root.fallback_move
            stop_reason = "tablebase"

        self.best_move = best_move
        self.best_score = best_score
        self.runner_up_score = completed_runner_up_score
        self.root_score_gap = completed_root_score_gap
        self.completed_depth = completed_depth
        self.sel_depth = completed_depth
        self.nodes = int(self.stats[0])
        if self.collect_stats:
            st = self.stats
            self.search_stats = SearchStats(
                nodes=int(st[STATS_NODES]),
                qnodes=int(st[STATS_QNODES]),
                move_generations=int(st[STATS_NODES]),
                tt_probes=int(st[STATS_TT_PROBES]),
                tt_hits=int(st[STATS_TT_HITS]),
                tt_cutoffs=int(st[STATS_TT_CUTOFFS]),
                beta_cutoffs=int(st[STATS_BETA_CUTOFFS]),
                rfp_prunes=int(st[STATS_RFP_PRUNES]),
                futility_prunes=int(st[STATS_FUTILITY_PRUNES]),
                null_prunes=int(st[STATS_NULL_PRUNES]),
                lmr_reductions=int(st[STATS_LMR]),
            )
        else:
            self.search_stats = None
        self._record_selected_move(board, best_move)
        self._record_timing(
            completed_depth,
            aborted_depth,
            last_prediction,
            aspiration_used_any,
            aspiration_failed_any,
            aspiration_reserve_skip,
            completed_root_score_gap,
            stop_reason,
        )

        return best_move

    def _record_timing(
        self,
        completed_depth: int,
        aborted_depth: int | None,
        predicted_next_s: float,
        aspiration_used: bool,
        aspiration_failed: bool,
        aspiration_reserve_skip: bool,
        root_gap: int | None,
        stop_reason: str,
    ) -> None:
        self.last_timing = MoveTiming(
            clock_before_s=self.time_mgr.clock_left_s,
            soft_limit_s=self.time_mgr.soft_limit,
            hard_limit_s=self.time_mgr.hard_limit,
            elapsed_s=self.time_mgr.elapsed(),
            completed_depth=completed_depth,
            aborted_depth=aborted_depth,
            predicted_next_s=predicted_next_s,
            aspiration_used=aspiration_used,
            aspiration_failed=aspiration_failed,
            aspiration_reserve_skip=aspiration_reserve_skip,
            root_gap=root_gap,
            band=self.time_mgr.clock_band,
            policy=self.time_mgr.policy,
            stop_reason=stop_reason,
        )
        if self.trace_timing:
            print(self.last_timing.format())

    def _record_position(self, board: Board) -> None:
        # Hash-only repetition tracking (B2): the old candidate.copy() +
        # generate_moves() + make/unmake loop cost ~200-300us per move to
        # rediscover the opponent reply we already know is legal (referee
        # enforces legality). Board.hash is repetition-safe, so O(1) append
        # is equivalent and never triggers lazy JIT on clock.
        if not self._game_hashes or self._game_hashes[-1] != board.hash:
            self._game_hashes.append(board.hash)

    def _record_selected_move(self, board: Board, move: int) -> None:
        if move == NO_MOVE:
            return
        self._pending_position = board.copy()
        self._pending_position.make_move(move)
        self._game_hashes.append(self._pending_position.hash)
