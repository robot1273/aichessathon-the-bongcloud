import random
import unittest
from unittest.mock import patch

import chess

from src.search import INF, Bot
from src.zobrist import calculate_hash, push_hash


class SearchTests(unittest.TestCase):
    def test_quiescence_scores_stalemate_as_draw(self) -> None:
        board = chess.Board("7k/8/5KQ1/8/8/8/8/8 b - - 0 1")

        score = Bot(tt_exp_size=10)._quiescence(board, -INF, INF, 0)

        self.assertEqual(score, 0)

    def test_timeout_restores_board(self) -> None:
        board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
        original_fen = board.fen()
        bot = Bot(tt_exp_size=10)

        with (
            patch("src.search.NODE_CHECK_INTERVAL", 1),
            patch.object(bot.time_mgr, "is_time_up", return_value=True),
        ):
            bot.get_best_move(board, movetime_ms=100)

        self.assertEqual(board.fen(), original_fen)
        self.assertFalse(board.move_stack)

    def test_forced_move_resets_public_state(self) -> None:
        bot = Bot(tt_exp_size=10)
        bot.nodes = 123
        bot.best_score = 456
        board = chess.Board("7k/8/8/8/8/8/8/K5Q1 b - - 0 1")

        move = bot.get_best_move(board, depth=2)

        self.assertEqual(move, chess.Move.from_uci("h8h7"))
        self.assertEqual(bot.nodes, 0)
        self.assertEqual(bot.best_score, 0)
        self.assertEqual(bot.best_move, move)


class ZobristTests(unittest.TestCase):
    def test_incremental_hash_matches_python_chess(self) -> None:
        rng = random.Random(0)
        board = chess.Board()
        hash_value = calculate_hash(board)

        for _ in range(500):
            moves = list(board.legal_moves)
            if not moves:
                board.reset()
                hash_value = calculate_hash(board)
                continue

            hash_value = push_hash(board, rng.choice(moves), hash_value)
            self.assertEqual(hash_value, calculate_hash(board))


if __name__ == "__main__":
    unittest.main()
