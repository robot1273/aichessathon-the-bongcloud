import random
import unittest
from unittest.mock import patch

import chess

from src.evaluation import evaluate, evaluate_with_phase
from src.search import INF, Bot
from src.tt import TT, Bound
from src.zobrist import calculate_hash, push_hash


class SearchTests(unittest.TestCase):
    def test_quiescence_scores_stalemate_as_draw(self) -> None:
        board = chess.Board("7k/8/5KQ1/8/8/8/8/8 b - - 0 1")

        score = Bot(tt_exp_size=10)._quiescence(board, -INF, INF, 0, calculate_hash(board))

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

    def test_quiescence_reuses_transposition(self) -> None:
        board = chess.Board("3q3k/8/8/8/8/8/8/K2Q4 w - - 0 1")
        original_fen = board.fen()
        hash_value = calculate_hash(board)
        bot = Bot(tt_exp_size=10)

        first_score = bot._quiescence(board, -INF, INF, 0, hash_value)
        first_nodes = bot.nodes
        bot.nodes = 0
        second_score = bot._quiescence(board, -INF, INF, 0, hash_value)

        self.assertEqual(second_score, first_score)
        self.assertLess(bot.nodes, first_nodes)
        self.assertEqual(board.fen(), original_fen)


class EvaluationTests(unittest.TestCase):
    def test_evaluation_is_symmetric(self) -> None:
        board = chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
        score = evaluate(board)

        board.turn = not board.turn
        self.assertEqual(evaluate(board), -score)
        self.assertEqual(evaluate(board.mirror()), -score)

    def test_initial_position_is_balanced_and_full_phase(self) -> None:
        score, phase = evaluate_with_phase(chess.Board())

        self.assertEqual(score, 0)
        self.assertEqual(phase, 1.0)


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


class TranspositionTableTests(unittest.TestCase):
    def test_shallow_exact_entry_does_not_replace_deeper_result(self) -> None:
        table = TT(exp_size=4)
        hash_value = 123
        move = chess.Move.from_uci("e2e4")
        table.store(hash_value, move, 20, 5, Bound.LOWER)

        table.store(hash_value, move, 10, 0, Bound.EXACT)

        entry = table.probe(hash_value)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.depth, 5)
        self.assertEqual(entry.score, 20)


if __name__ == "__main__":
    unittest.main()
