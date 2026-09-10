import unittest
from unittest.mock import patch

import chess
import chess.polyglot

from src.board import Board, move_to_uci
from src.opening_book import book_move


class _FakeReader:
    def __init__(self, entries: list[chess.polyglot.Entry]) -> None:
        self.entries = entries
        self.requested_key: int | None = None

    def find_all(self, key: int) -> list[chess.polyglot.Entry]:
        self.requested_key = key
        return self.entries


def _entry(uci: str, weight: int) -> chess.polyglot.Entry:
    move = chess.Move.from_uci(uci)
    return chess.polyglot.Entry(0, 0, weight, 0, move)


class OpeningBookTests(unittest.TestCase):
    def test_uses_custom_hash_and_engine_preference_among_credible_moves(self) -> None:
        board = Board.from_fen()
        reader = _FakeReader(
            [
                _entry("a1a8", 100),
                _entry("g1f3", 20),
                _entry("e2e4", 30),
                _entry("h2h4", 1),
            ]
        )

        with patch("src.opening_book._get_reader", return_value=reader):
            move = book_move(board, board.generate_moves())

        self.assertIsNotNone(move)
        assert move is not None
        self.assertEqual(move_to_uci(move), "g1f3")
        self.assertEqual(reader.requested_key, board.hash)

    def test_normalizes_polyglot_castling(self) -> None:
        board = Board.from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 10")
        reader = _FakeReader([_entry("e1h1", 1)])

        with patch("src.opening_book._get_reader", return_value=reader):
            move = book_move(board, board.generate_moves())

        self.assertIsNotNone(move)
        assert move is not None
        self.assertEqual(move_to_uci(move), "e1g1")

    def test_matches_promotion_piece(self) -> None:
        board = Board.from_fen("k7/4P3/8/8/8/8/8/K7 w - - 0 20")
        reader = _FakeReader([_entry("e7e8n", 1), _entry("e7e8q", 2)])

        with patch("src.opening_book._get_reader", return_value=reader):
            move = book_move(board, board.generate_moves())

        self.assertIsNotNone(move)
        assert move is not None
        self.assertEqual(move_to_uci(move), "e7e8q")

    def test_never_probes_after_fullmove_twenty(self) -> None:
        board = Board.from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 21"
        )
        reader = _FakeReader([_entry("e2e4", 1)])

        with patch("src.opening_book._get_reader", return_value=reader):
            move = book_move(board, board.generate_moves())

        self.assertIsNone(move)
        self.assertIsNone(reader.requested_key)


if __name__ == "__main__":
    unittest.main()
