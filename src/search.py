"""Bot wrapper for Numba-jitted search."""

from __future__ import annotations

import ctypes
import time
from typing import Any

import numpy as np

from src.board import Board, move_to_uci
from src.constants import INF, MATE_SCORE, MATE_THRESHOLD, MAX_PLY, NO_MOVE, STATE_SIZE
from src.search_numba import (
    board_to_state,
    search_root,
)
from src.time_manager import TimeManager
from src.tt import create_tt_arrays

# Libc clock for microsecond time tracking
libc = ctypes.CDLL(None)
clock = libc.clock
clock.restype = ctypes.c_long
clock.argtypes = []


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


class Bot:
    def __init__(
        self,
        tt_exp_size: int = 22,
        collect_stats: bool = False,
        increment_s: float = INCREMENT_S,
    ) -> None:
        self.tt_exp_size = tt_exp_size
        self.collect_stats = collect_stats
        self.increment_s = increment_s
        self.current_age = 0
        self.tt_arrays = create_tt_arrays(tt_exp_size)
        self.time_mgr = TimeManager(increment_s=increment_s)

        # Pre-allocate reusable search arrays across moves
        self.undo_stack = np.zeros((MAX_PLY, STATE_SIZE), dtype=np.uint64)
        self.moves_stack = np.zeros((MAX_PLY, 256), dtype=np.int32)
        self.scores_stack = np.zeros((MAX_PLY, 256), dtype=np.int32)
        self.killers = np.zeros((MAX_PLY, 2), dtype=np.int32)
        self.history = np.zeros((2, 64, 64), dtype=np.int32)
        self.stats = np.zeros(4, dtype=np.int64)

        self.completed_depth = 0
        self.sel_depth = 0
        self.best_score = 0
        self.nodes = 0
        self.best_move = NO_MOVE
        self.search_stats: SearchStats | None = None
        self._game_hashes: list[int] = []
        self._pending_position: Board | None = None

        self._warmup_jit()

    def _warmup_jit(self) -> None:
        """Warm up the JIT compiler with a dummy search."""
        board = Board.from_fen()
        state = board_to_state(board)
        hash_history = np.zeros(MAX_PLY, dtype=np.uint64)
        search_root(
            state,
            self.undo_stack,
            self.moves_stack,
            self.scores_stack,
            -100,
            100,
            3,
            self.killers,
            self.history,
            self.stats,
            hash_history,
            0,
            0,
            *self.tt_arrays,
        )

    def get_best_move(
        self,
        board: Board,
        time_left_ms: int,
        depth: int | None = None,
        movetime_ms: int | None = None,
        verbose: bool = False,
        callback: Any | None = None,
    ) -> int:
        self.current_age += 1
        self._record_position(board)

        # Check for immediate return on forced moves or no moves
        legal_moves = board.generate_moves()
        if not legal_moves:
            self.best_move = NO_MOVE
            self.nodes = 0
            self.completed_depth = 0
            self.best_score = 0
            return NO_MOVE

        if len(legal_moves) == 1:
            self.best_move = legal_moves[0]
            self.nodes = 1
            self.completed_depth = 1
            self.best_score = 0
            self._record_selected_move(board, legal_moves[0])
            return legal_moves[0]

        # Configure time management
        self.time_mgr.start(time_left_ms, board, movetime_ms)
        hard_limit_s = self.time_mgr._hard_limit

        state = board_to_state(board)
        start_t = time.perf_counter()

        # Build game history for repetition checking. The current root is last.
        hash_history = np.zeros(MAX_PLY, dtype=np.uint64)
        hist_len = min(len(self._game_hashes), MAX_PLY)
        for i in range(hist_len):
            hash_history[i] = np.uint64(self._game_hashes[-hist_len + i])

        # Setup deadline in clock ticks
        start_ticks = clock()
        if depth is not None:
            # Fixed depth: no deadline unless movetime_ms specified
            deadline_ticks = (
                0 if movetime_ms is None else start_ticks + int(hard_limit_s * 1_000_000)
            )
        else:
            deadline_ticks = start_ticks + int(hard_limit_s * 1_000_000)

        self.stats[0] = 0  # nodes
        self.stats[1] = 0  # aborted flag
        self.stats[2] = start_ticks
        self.stats[3] = deadline_ticks

        # Clear killer heuristics per search; age history
        self.killers.fill(0)
        self.history //= 2

        best_move = legal_moves[0]
        best_score = 0
        prev_score = 0
        completed_depth = 0
        max_d = depth if depth is not None else 64

        for d in range(1, max_d + 1):
            if d > 1 and depth is None and self.time_mgr.should_stop_iterating():
                break

            # Aspiration window for depth >= 5
            if d >= 5 and abs(best_score) < MATE_THRESHOLD:
                delta = 30
                alpha = max(-INF, best_score - delta)
                beta = min(INF, best_score + delta)
                score, move, aborted = search_root(
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
                    score, move, aborted = search_root(
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
                    score, move, aborted = search_root(
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
                score, move, aborted = search_root(
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
                break

            if move != NO_MOVE:
                best_move = move
                best_score = score
            completed_depth = d

            self.time_mgr.extend_if_unstable(prev_score, score)
            if depth is None:
                self.stats[3] = start_ticks + int(self.time_mgr._hard_limit * 1_000_000)
            prev_score = score

            nodes = int(self.stats[0])
            elapsed = max(time.perf_counter() - start_t, 0.0001)
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
                }
                if callback:
                    callback(info)
                if verbose:
                    print(
                        f"info depth {d} score {score_str} nodes {nodes} "
                        f"nps {nps} time {int(elapsed * 1000)} pv {move_to_uci(best_move)}"
                    )

            if abs(best_score) > MATE_THRESHOLD:
                break

        self.best_move = best_move
        self.best_score = best_score
        self.completed_depth = completed_depth
        self.sel_depth = completed_depth
        self.nodes = int(self.stats[0])
        self._record_selected_move(board, best_move)

        return best_move

    def _record_position(self, board: Board) -> None:
        if self._pending_position is None:
            self._game_hashes = [board.hash]
            return

        candidate = self._pending_position.copy()
        for move in candidate.generate_moves():
            candidate.make_move(move)
            if candidate.hash == board.hash:
                self._game_hashes.append(board.hash)
                return
            candidate.unmake_move()

        self._game_hashes = [board.hash]
        self._pending_position = None

    def _record_selected_move(self, board: Board, move: int) -> None:
        if move == NO_MOVE:
            return
        self._pending_position = board.copy()
        self._pending_position.make_move(move)
        self._game_hashes.append(self._pending_position.hash)
