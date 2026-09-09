"""Comprehensive test suite for custom bitboard board.

Tests:
- Perft correctness against standard chess engine reference values
- Move generation equivalence with python-chess
- Zobrist hash equivalence with chess.polyglot
- Pre-move gives_check accuracy
- Make/unmake move idempotence and eval consistency
"""

from __future__ import annotations

import random
import unittest

import chess
import chess.polyglot
import numpy as np

from src import board_primitives
from src.board import (
    Board,
    move_to_uci,
    parse_uci_to_move,
)
from src.constants import HASH, STATE_SIZE
from src.search_numba import board_to_state

TEST_FENS = [
    # Startpos
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # Kiwipete (dense tactics, pins, checks, castling, en passant)
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    # Position 3 (pawns, promotions)
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    # Position 4 (castling and checks)
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    # Position 5 (pinned rook and ep)
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    # Position 6 (endgame)
    "8/8/4k3/8/8/8/4P3/4K3 w - - 0 1",
    # En passant check
    "8/8/8/8/k1pP4/8/8/4K3 b - d3 0 1",
    # Discovered check position
    "rnbqk2r/ppp2ppp/3b1n2/3p4/3P4/3B1N2/PPP2PPP/RNBQK2R w KQkq - 4 6",
]


def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1
    moves = board.generate_moves()
    if depth == 1:
        return len(moves)
    nodes = 0
    for move in moves:
        board.make_move(move)
        nodes += perft(board, depth - 1)
        board.unmake_move()
    return nodes


class TestBoard(unittest.TestCase):
    def assert_numba_moves_match(self, py_board: chess.Board) -> None:
        board = Board.from_fen(py_board.fen())
        state = board_to_state(board)
        moves = np.zeros(256, dtype=np.int32)
        count = board_primitives.generate_moves(state, moves)
        numba_moves = moves[:count]

        self.assertEqual(
            {move_to_uci(int(move)) for move in numba_moves},
            {move.uci() for move in py_board.legal_moves},
            py_board.fen(),
        )

        undo_stack = np.zeros((1, STATE_SIZE), dtype=np.uint64)
        original_state = state.copy()
        for move in numba_moves:
            move_int = int(move)
            self.assertEqual(
                bool(board_primitives.gives_check(state, move_int)),
                py_board.gives_check(chess.Move.from_uci(move_to_uci(move_int))),
                f"gives_check mismatch for {move_to_uci(move_int)} at {py_board.fen()}",
            )
            board_primitives.make_move(state, undo_stack, 0, move_int)
            expected_board = board.copy()
            expected_board.make_move(move_int)
            self.assertEqual(int(state[HASH]), expected_board.hash)
            board_primitives.unmake_move(state, undo_stack, 0)
            self.assertTrue(np.array_equal(state, original_state))

    def test_numba_movegen_matches_python_chess(self) -> None:
        for fen in TEST_FENS:
            self.assert_numba_moves_match(chess.Board(fen))

        rng = random.Random(0)
        py_board = chess.Board()
        for _ in range(256):
            self.assert_numba_moves_match(py_board)
            legal_moves = list(py_board.legal_moves)
            if legal_moves:
                py_board.push(rng.choice(legal_moves))
            else:
                py_board.reset()

    def test_invalid_castling_rights_do_not_create_moves(self) -> None:
        self.assert_numba_moves_match(chess.Board("4k3/8/8/8/8/8/8/4K3 w K - 0 1"))

    def test_perft_startpos(self) -> None:
        board = Board.from_fen()
        self.assertEqual(perft(board, 1), 20)
        self.assertEqual(perft(board, 2), 400)
        self.assertEqual(perft(board, 3), 8902)
        self.assertEqual(perft(board, 4), 197281)

    def test_perft_kiwipete(self) -> None:
        board = Board.from_fen(TEST_FENS[1])
        self.assertEqual(perft(board, 1), 48)
        self.assertEqual(perft(board, 2), 2039)
        self.assertEqual(perft(board, 3), 97862)

    def test_perft_position3(self) -> None:
        board = Board.from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1")
        self.assertEqual(perft(board, 1), 14)
        self.assertEqual(perft(board, 2), 191)
        self.assertEqual(perft(board, 3), 2812)

    def test_perft_position4(self) -> None:
        board = Board.from_fen("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
        self.assertEqual(perft(board, 1), 6)
        self.assertEqual(perft(board, 2), 264)
        self.assertEqual(perft(board, 3), 9467)

    def test_movegen_matches_python_chess(self) -> None:
        """Every generated legal move must match python-chess exactly."""
        for fen in TEST_FENS:
            our_board = Board.from_fen(fen)
            py_board = chess.Board(fen)

            our_moves = {move_to_uci(m) for m in our_board.generate_moves()}
            py_moves = {m.uci() for m in py_board.legal_moves}

            self.assertEqual(
                our_moves,
                py_moves,
                (
                    f"Movegen mismatch for FEN: {fen}\n"
                    f"Our: {sorted(our_moves)}\n"
                    f"Py:  {sorted(py_moves)}"
                ),
            )

    def test_zobrist_matches_polyglot(self) -> None:
        """Board hash matches Polyglot when en passant is legal."""
        for fen in TEST_FENS:
            our_board = Board.from_fen(fen)
            py_board = chess.Board(fen)

            our_hash = our_board.hash
            py_hash = chess.polyglot.zobrist_hash(py_board)

            self.assertEqual(
                our_hash,
                py_hash,
                f"Hash mismatch for FEN: {fen}\nOur: {hex(our_hash)}\nPy:  {hex(py_hash)}",
            )

    def test_zobrist_ignores_pinned_en_passant(self) -> None:
        with_ep = Board.from_fen("8/8/8/8/2kpP2R/8/8/4K3 b - e3 0 1")
        without_ep = Board.from_fen("8/8/8/8/2kpP2R/8/8/4K3 b - - 0 1")
        self.assertEqual(with_ep.hash, without_ep.hash)

    def test_pawn_move_refreshes_both_evaluation_bonuses(self) -> None:
        board = Board.from_fen("4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1")
        board.make_move(parse_uci_to_move(board, "e2e4"))
        self.assertEqual(board.evaluate(), Board.from_fen(board.fen()).evaluate())

    def test_numba_evaluation_matches_board_after_moves(self) -> None:
        undo_stack = np.zeros((1, STATE_SIZE), dtype=np.uint64)
        for fen in TEST_FENS:
            board = Board.from_fen(fen)
            state = board_to_state(board)
            self.assertEqual(board_primitives.evaluate(state), board.evaluate())

            for move in board.generate_moves():
                board_primitives.make_move(state, undo_stack, 0, move)
                board.make_move(move)
                self.assertEqual(
                    board_primitives.evaluate(state),
                    board.evaluate(),
                    f"evaluation mismatch after {move_to_uci(move)} at {fen}",
                )
                board.unmake_move()
                board_primitives.unmake_move(state, undo_stack, 0)

    def test_repetition_scans_same_side_to_move(self) -> None:
        board = Board.from_fen()
        cycle = ("g1f3", "g8f6", "f3g1", "f6g8")

        for uci in cycle:
            board.make_move(parse_uci_to_move(board, uci))
        self.assertTrue(board.is_repetition(2))
        self.assertFalse(board.is_repetition(3))

        for uci in cycle:
            board.make_move(parse_uci_to_move(board, uci))
        self.assertTrue(board.is_repetition(3))

    def test_insufficient_material_matches_python_chess(self) -> None:
        fens = (
            "8/8/8/8/8/8/8/K6k w - - 0 1",
            "8/8/8/8/8/8/6N1/K6k w - - 0 1",
            "5b1k/8/8/8/8/8/8/K1B5 w - - 0 1",
            "6bk/8/8/8/8/8/8/K1B5 w - - 0 1",
            "8/8/8/8/8/8/5NN1/K6k w - - 0 1",
        )
        for fen in fens:
            self.assertEqual(
                Board.from_fen(fen).is_insufficient_material(),
                chess.Board(fen).is_insufficient_material(),
                fen,
            )

    def test_gives_check(self) -> None:
        """Pre-move gives_check must match whether target position is in check."""
        for fen in TEST_FENS:
            board = Board.from_fen(fen)
            for move in board.generate_moves():
                predicted_check = board.gives_check(move)
                board.make_move(move)
                actual_check = board.is_in_check()
                board.unmake_move()
                self.assertEqual(
                    predicted_check,
                    actual_check,
                    f"gives_check mismatch for move {move_to_uci(move)} at FEN {fen}",
                )

    def test_make_unmake_idempotence(self) -> None:
        """Making and unmaking moves must restore state identically."""
        for fen in TEST_FENS:
            board = Board.from_fen(fen)
            orig_fen = board.fen()
            orig_hash = board.hash
            orig_eval = board.evaluate()

            for move in board.generate_moves():
                board.make_move(move)
                board.unmake_move()

                self.assertEqual(board.fen(), orig_fen)
                self.assertEqual(board.hash, orig_hash)
                self.assertEqual(board.evaluate(), orig_eval)

    def test_fen_roundtrip(self) -> None:
        """board.fen() must match the original FEN."""
        for fen in TEST_FENS:
            board = Board.from_fen(fen)
            self.assertEqual(board.fen(), fen)

    def test_parse_uci(self) -> None:
        """parse_uci_to_move correctly identifies legal moves from string."""
        board = Board.from_fen()
        m = parse_uci_to_move(board, "e2e4")
        self.assertEqual(move_to_uci(m), "e2e4")


if __name__ == "__main__":
    unittest.main()
