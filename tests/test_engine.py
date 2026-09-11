import random
import unittest
from dataclasses import replace
from unittest.mock import patch

import chess.polyglot
import numpy as np

from src import board_primitives
from src.board import Board, move_to_uci
from src.board import make_move as encode_move
from src.constants import HASH, INF, MAX_PLY, NO_MOVE, STATE_SIZE
from src.evaluation import evaluate_with_phase
from src.move_ordering import pick_fallback_move
from src.search import STATS_SIZE, Bot, is_root_ambiguous
from src.search_numba import alpha_beta, board_to_state, is_draw, quiescence, score_move
from src.time_manager import DEFAULT_TIME_CONFIG
from src.tt import create_tt_arrays, probe_tt
from src.zobrist import calculate_hash, has_legal_en_passant


class SearchTests(unittest.TestCase):
    def test_countermove_orders_quiet_response_after_killers(self) -> None:
        board = Board.from_fen()
        move = next(move for move in board.generate_moves() if move_to_uci(move) == "e2e4")
        state = board_to_state(board)
        killers = np.zeros((MAX_PLY, 2), dtype=np.int32)
        history = np.zeros((2, 64, 64), dtype=np.int32)

        self.assertEqual(
            score_move(state, move, NO_MOVE, move, 0, killers, history),
            600_000_000,
        )
        killers[0, 1] = move
        self.assertEqual(score_move(state, move, NO_MOVE, move, 0, killers, history), 700_000_000)

    def test_fallback_prefers_queen_promotion_or_valid_tt_move(self) -> None:
        board = Board.from_fen("k7/4P3/8/8/8/8/8/K7 w - - 0 1")
        moves = board.generate_moves()

        self.assertEqual(move_to_uci(pick_fallback_move(board, moves)), "e7e8q")

        knight_promotion = next(move for move in moves if move_to_uci(move) == "e7e8n")
        self.assertEqual(pick_fallback_move(board, moves, knight_promotion), knight_promotion)

    def test_forced_move_resets_public_state(self) -> None:
        bot = Bot(collect_stats=True)
        bot.get_best_move(Board.from_fen(), time_left_ms=100_000, depth=1)
        board = Board.from_fen("7k/8/8/8/8/8/8/K5Q1 b - - 0 1")

        move = bot.get_best_move(board, time_left_ms=100_000, depth=2)

        self.assertEqual(move_to_uci(move), "h8h7")
        self.assertEqual(bot.completed_depth, 1)
        self.assertEqual(bot.nodes, 1)
        self.assertIsNone(bot.search_stats)

    def test_repetition_requires_three_occurrences(self) -> None:
        board = Board.from_fen()
        state = board_to_state(board)
        undo_stack = np.zeros((MAX_PLY, STATE_SIZE), dtype=np.uint64)
        history = np.array([board.hash], dtype=np.uint64)
        moves = ("g1f3", "g8f6", "f3g1", "f6g8")

        for ply in range(8):
            move = next(
                move for move in board.generate_moves() if move_to_uci(move) == moves[ply % 4]
            )
            board_primitives.make_move(state, undo_stack, ply, move)
            board.make_move(move)
            expected_draw = ply == 7
            self.assertEqual(
                is_draw(state, undo_stack, ply + 1, history, len(history)), expected_draw
            )

    def test_zero_pvs_bound_is_excluded_from_root_uncertainty_by_default(self) -> None:
        baseline = DEFAULT_TIME_CONFIG
        allow_zero = replace(baseline, require_positive_root_gap=False)

        self.assertFalse(is_root_ambiguous(0, baseline))
        self.assertTrue(is_root_ambiguous(0, allow_zero))
        self.assertTrue(is_root_ambiguous(1, baseline))
        self.assertFalse(is_root_ambiguous(51, baseline))

    def test_completed_search_reports_root_score_gap(self) -> None:
        bot = Bot()
        board = Board.from_fen()

        bot.get_best_move(board, time_left_ms=100_000, depth=2)

        self.assertGreater(bot.runner_up_score, -INF)
        self.assertEqual(bot.root_score_gap, bot.best_score - bot.runner_up_score)
        self.assertIsNotNone(bot.last_timing)
        assert bot.last_timing is not None
        self.assertEqual(bot.last_timing.completed_depth, 2)
        self.assertEqual(bot.last_timing.root_gap, bot.root_score_gap)

    def test_panic_clock_uses_allocated_search_time(self) -> None:
        config = replace(
            DEFAULT_TIME_CONFIG,
            panic_soft_increment=0.05,
            panic_hard_increment=0.08,
            panic_soft_floor_s=0.025,
            panic_hard_floor_s=0.04,
        )
        bot = Bot(time_config=config, use_book=False, use_tb=False)

        bot.get_best_move(Board.from_fen(), time_left_ms=3_000)

        self.assertGreater(bot.completed_depth, 2)
        self.assertIsNotNone(bot.last_timing)
        assert bot.last_timing is not None
        self.assertNotEqual(bot.last_timing.stop_reason, "panic-depth")

    def test_opening_move_requires_stable_search_confirmation(self) -> None:
        bot = Bot(use_tb=False)
        board = Board.from_fen()
        opening_move = next(
            move for move in board.generate_moves() if move_to_uci(move) == "e2e4"
        )

        with (
            patch("src.search.book_move", return_value=opening_move),
            patch.object(Bot, "_search_root", return_value=(20, opening_move, -100, False)),
        ):
            move = bot.get_best_move(board, time_left_ms=100_000)

        self.assertEqual(move, opening_move)
        self.assertEqual(bot.completed_depth, 8)
        self.assertIsNotNone(bot.last_timing)
        assert bot.last_timing is not None
        self.assertEqual(bot.last_timing.stop_reason, "opening-book")

    def test_checkmate_takes_precedence_over_halfmove_draw(self) -> None:
        bot = Bot()
        board = Board.from_fen("7k/8/6QK/8/8/8/8/8 w - - 99 1")

        move = bot.get_best_move(board, time_left_ms=100_000, depth=1)

        self.assertEqual(move_to_uci(move), "g6g7")
        self.assertGreater(bot.best_score, 900_000)

    def test_quiescence_scores_sparse_stalemate_as_draw(self) -> None:
        board = Board.from_fen("7k/5Q2/7K/8/8/8/8/8 b - - 0 1")
        state = board_to_state(board)
        undo_stack = np.zeros((MAX_PLY, STATE_SIZE), dtype=np.uint64)
        moves_stack = np.zeros((MAX_PLY, 256), dtype=np.int32)
        scores_stack = np.zeros((MAX_PLY, 256), dtype=np.int32)
        stats = np.zeros(STATS_SIZE, dtype=np.int64)

        score = quiescence(
            state, undo_stack, moves_stack, scores_stack, -INF, INF, 0, stats
        )

        self.assertEqual(score, 0)

    def test_quiescence_scores_dense_stalemate_as_draw(self) -> None:
        board = Board.from_fen("7k/5Q2/7K/8/8/8/PPPP4/8 b - - 0 1")
        state = board_to_state(board)
        undo_stack = np.zeros((MAX_PLY, STATE_SIZE), dtype=np.uint64)
        moves_stack = np.zeros((MAX_PLY, 256), dtype=np.int32)
        scores_stack = np.zeros((MAX_PLY, 256), dtype=np.int32)
        stats = np.zeros(STATS_SIZE, dtype=np.int64)

        score = quiescence(
            state, undo_stack, moves_stack, scores_stack, -INF, INF, 0, stats
        )

        self.assertEqual(score, 0)

    def test_null_search_does_not_claim_repetition(self) -> None:
        board = Board.from_fen("6nk/p7/8/8/8/8/8/K7 w - - 20 1")
        state = board_to_state(board)
        undo_stack = np.zeros((MAX_PLY, STATE_SIZE), dtype=np.uint64)
        history = np.array([board.hash], dtype=np.uint64)
        knight_out = encode_move(62, 45)
        knight_back = encode_move(45, 62)

        for base_ply in (0, 4):
            board_primitives.make_null_move(state, undo_stack, base_ply)
            board_primitives.make_move(state, undo_stack, base_ply + 1, knight_out)
            board_primitives.make_null_move(state, undo_stack, base_ply + 2)
            board_primitives.make_move(state, undo_stack, base_ply + 3, knight_back)

        self.assertEqual(int(state[HASH]), board.hash)
        self.assertFalse(is_draw(state, undo_stack, 8, history, len(history)))

    def test_null_search_quiescence_does_not_claim_halfmove_draw(self) -> None:
        board = Board.from_fen("4k2r/8/8/8/8/8/8/N3K3 w - - 99 1")
        state = board_to_state(board)
        undo_stack = np.zeros((MAX_PLY, STATE_SIZE), dtype=np.uint64)
        moves_stack = np.zeros((MAX_PLY, 256), dtype=np.int32)
        scores_stack = np.zeros((MAX_PLY, 256), dtype=np.int32)
        stats = np.zeros(STATS_SIZE, dtype=np.int64)

        board_primitives.make_null_move(state, undo_stack, 0)
        board_primitives.make_move(state, undo_stack, 1, encode_move(63, 55))
        score = quiescence(
            state, undo_stack, moves_stack, scores_stack, -INF, INF, 2, stats
        )

        self.assertNotEqual(score, 0)

    def test_insufficient_material_is_drawn_inside_search(self) -> None:
        board = Board.from_fen("8/8/8/8/8/8/6N1/K6k w - - 0 1")
        state = board_to_state(board)
        undo_stack = np.zeros((MAX_PLY, STATE_SIZE), dtype=np.uint64)
        history = np.array([board.hash], dtype=np.uint64)

        self.assertTrue(is_draw(state, undo_stack, 0, history, len(history)))

    def test_tt_does_not_bypass_next_check_extension(self) -> None:
        board = Board.from_fen("4k3/8/8/8/8/8/8/4R1K1 b - - 0 1")
        state = board_to_state(board)
        undo_stack = np.zeros((MAX_PLY, STATE_SIZE), dtype=np.uint64)
        moves_stack = np.zeros((MAX_PLY, 256), dtype=np.int32)
        scores_stack = np.zeros((MAX_PLY, 256), dtype=np.int32)
        killers = np.zeros((MAX_PLY, 2), dtype=np.int32)
        history = np.zeros((2, 64, 64), dtype=np.int32)
        countermoves = np.zeros((64, 64), dtype=np.int32)
        hash_history = np.zeros(MAX_PLY, dtype=np.uint64)
        tt_arrays = create_tt_arrays(10)
        stats = np.zeros(STATS_SIZE, dtype=np.int64)

        first_score = alpha_beta(
            state,
            undo_stack,
            moves_stack,
            scores_stack,
            -INF,
            INF,
            1,
            1,
            True,
            NO_MOVE,
            killers,
            history,
            countermoves,
            stats,
            hash_history,
            0,
            1,
            *tt_arrays,
        )
        found, _, _, stored_depth, _, _ = probe_tt(board.hash, *tt_arrays)
        self.assertTrue(found)
        self.assertEqual(stored_depth, 2)

        stats.fill(0)
        alpha_beta(
            state,
            undo_stack,
            moves_stack,
            scores_stack,
            first_score - 1,
            first_score,
            2,
            1,
            True,
            NO_MOVE,
            killers,
            history,
            countermoves,
            stats,
            hash_history,
            0,
            1,
            *tt_arrays,
        )

        self.assertGreater(stats[0], 1)


class EvaluationTests(unittest.TestCase):
    def test_initial_position_is_balanced_and_full_phase(self) -> None:
        board = Board.from_fen()
        score, phase = evaluate_with_phase(board)

        self.assertEqual(score, 0)
        self.assertEqual(phase, 1.0)


class ZobristTests(unittest.TestCase):
    def test_incremental_hash_matches_polyglot_over_random_game(self) -> None:
        rng = random.Random(42)
        board = Board.from_fen()

        for _ in range(500):
            moves = board.generate_moves()
            if not moves or board.is_halfmove_draw():
                board = Board.from_fen()
                continue

            m = rng.choice(moves)
            board.make_move(m)

            self.assertEqual(board.hash, calculate_hash(board))

            py_board = chess.Board(board.fen())
            py_hash = (
                chess.polyglot.zobrist_hash(py_board)
                if board.ep_square == -1 or has_legal_en_passant(board)
                else board.hash
            )
            self.assertEqual(
                board.hash,
                py_hash,
                f"Zobrist mismatch after move {move_to_uci(m)} at {board.fen()}",
            )


if __name__ == "__main__":
    unittest.main()
