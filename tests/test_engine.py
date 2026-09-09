import random
import unittest

import chess.polyglot
import numpy as np

from src import board_primitives
from src.board import Board, move_to_uci
from src.constants import MAX_PLY, STATE_SIZE
from src.evaluation import evaluate_with_phase
from src.search import Bot
from src.search_numba import board_to_state, is_draw
from src.zobrist import calculate_hash, has_legal_en_passant

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"


class SearchTests(unittest.TestCase):
    def test_forced_move_resets_public_state(self) -> None:
        bot = Bot()
        board = Board.from_fen("7k/8/8/8/8/8/8/K5Q1 b - - 0 1")

        move = bot.get_best_move(board, time_left_ms=100_000, depth=2)

        self.assertEqual(move_to_uci(move), "h8h7")

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
