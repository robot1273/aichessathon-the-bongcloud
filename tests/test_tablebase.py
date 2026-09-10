import unittest
from unittest.mock import patch

import chess

from src.board import Board, move_is_capture, move_to_uci
from src.tablebase import tb_probe_root


class _FakeTablebase:
    def __init__(
        self,
        root_wdl: int,
        child_wdl: dict[str, int],
        child_dtz: dict[str, int],
        default_child_wdl: int = 2,
    ) -> None:
        self.root_wdl = root_wdl
        self.child_wdl = child_wdl
        self.child_dtz = child_dtz
        self.default_child_wdl = default_child_wdl

    def get_wdl(self, board: chess.Board) -> int:
        if not board.move_stack:
            return self.root_wdl
        return self.child_wdl.get(board.peek().uci(), self.default_child_wdl)

    def get_dtz(self, board: chess.Board) -> int | None:
        return self.child_dtz.get(board.peek().uci())


class TablebaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board.from_fen("7k/8/8/8/8/8/6Q1/6K1 w - - 0 1")
        self.moves = self.board.generate_moves()

    def test_draw_plan_rejects_losing_search_moves(self) -> None:
        drawing = self.moves[0]
        uci = move_to_uci(drawing)
        fake = _FakeTablebase(0, {uci: 0}, {uci: 0})

        with patch("src.tablebase._get_tablebase", return_value=fake):
            plan = tb_probe_root(self.board, self.moves)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.wdl, 0)
        self.assertIn(drawing, plan.search_moves)
        losing = next(
            move
            for move in self.moves[1:]
            if move not in plan.search_moves
        )
        self.assertNotIn(losing, plan.search_moves)

    def test_win_plan_limits_search_to_shortest_safe_dtz(self) -> None:
        short, long = self.moves[:2]
        short_uci = move_to_uci(short)
        long_uci = move_to_uci(long)
        fake = _FakeTablebase(
            2,
            {short_uci: -2, long_uci: -2},
            {short_uci: -2, long_uci: -8},
            default_child_wdl=0,
        )

        with patch("src.tablebase._get_tablebase", return_value=fake):
            plan = tb_probe_root(self.board, self.moves)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.search_moves, frozenset({short}))
        self.assertEqual(plan.fallback_move, short)

    def test_win_plan_rejects_third_repetition(self) -> None:
        repeating, safe = self.moves[:2]
        repeating_uci = move_to_uci(repeating)
        safe_uci = move_to_uci(safe)
        repeated = self.board.copy()
        repeated.make_move(repeating)
        fake = _FakeTablebase(
            2,
            {repeating_uci: -2, safe_uci: -2},
            {repeating_uci: -1, safe_uci: -5},
            default_child_wdl=0,
        )

        with patch("src.tablebase._get_tablebase", return_value=fake):
            plan = tb_probe_root(self.board, self.moves, [repeated.hash, repeated.hash])

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertNotIn(repeating, plan.search_moves)
        self.assertIn(safe, plan.search_moves)
        self.assertEqual(plan.fallback_move, safe)

    def test_win_plan_rejects_move_allowing_reply_repetition(self) -> None:
        repeating, safe = self.moves[:2]
        repeating_uci = move_to_uci(repeating)
        safe_uci = move_to_uci(safe)
        repeated = self.board.copy()
        repeated.make_move(repeating)
        reply = repeated.generate_moves()[0]
        repeated.make_move(reply)
        fake = _FakeTablebase(
            2,
            {repeating_uci: -2, safe_uci: -2},
            {repeating_uci: -1, safe_uci: -5},
            default_child_wdl=0,
        )

        with patch("src.tablebase._get_tablebase", return_value=fake):
            plan = tb_probe_root(self.board, self.moves, [repeated.hash, repeated.hash])

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertNotIn(repeating, plan.search_moves)
        self.assertIn(safe, plan.search_moves)

    def test_win_plan_rejects_claimable_repetition_after_reply(self) -> None:
        repeating, safe = self.moves[:2]
        repeating_uci = move_to_uci(repeating)
        safe_uci = move_to_uci(safe)
        repeated = self.board.copy()
        repeated.make_move(repeating)
        repeated.make_move(repeated.generate_moves()[0])
        repeated.make_move(repeated.generate_moves()[0])
        fake = _FakeTablebase(
            2,
            {repeating_uci: -2, safe_uci: -2},
            {repeating_uci: -1, safe_uci: -5},
            default_child_wdl=0,
        )

        with patch("src.tablebase._get_tablebase", return_value=fake):
            plan = tb_probe_root(self.board, self.moves, [repeated.hash, repeated.hash])

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertNotIn(repeating, plan.search_moves)
        self.assertIn(safe, plan.search_moves)

    def test_five_piece_capture_into_tablebase_win(self) -> None:
        board = Board.from_fen("7k/8/8/8/8/8/P5Qr/6K1 w - - 0 1")
        moves = board.generate_moves()
        capture = next(move for move in moves if move_is_capture(move))
        uci = move_to_uci(capture)
        fake = _FakeTablebase(0, {uci: -2}, {uci: -1}, default_child_wdl=0)

        with patch("src.tablebase._get_tablebase", return_value=fake):
            plan = tb_probe_root(board, moves)

        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.wdl, 2)
        self.assertEqual(plan.search_moves, frozenset({capture}))
        self.assertEqual(plan.fallback_move, capture)


if __name__ == "__main__":
    unittest.main()
