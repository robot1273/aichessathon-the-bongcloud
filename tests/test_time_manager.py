import unittest

from src.board import Board
from src.time_manager import TimeManager


class TimeManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board.from_fen()

    def test_initial_score_does_not_extend_budget(self) -> None:
        manager = TimeManager(increment_s=0.5)
        manager.start(120_000, self.board)
        soft_limit = manager.soft_limit
        hard_limit = manager.hard_limit

        manager.extend_if_unstable(None, 800)

        self.assertEqual(manager.soft_limit, soft_limit)
        self.assertEqual(manager.hard_limit, hard_limit)

    def test_instability_extension_does_not_compound(self) -> None:
        manager = TimeManager(increment_s=0.5)
        manager.start(120_000, self.board)
        base_soft_limit = manager.soft_limit
        base_hard_limit = manager.hard_limit

        manager.extend_if_unstable(0, 400)
        extended_soft_limit = manager.soft_limit
        extended_hard_limit = manager.hard_limit
        manager.extend_if_unstable(400, 0)

        self.assertAlmostEqual(extended_soft_limit, base_soft_limit * 1.6)
        self.assertAlmostEqual(extended_hard_limit, base_hard_limit * 1.6)
        self.assertEqual(manager.soft_limit, extended_soft_limit)
        self.assertEqual(manager.hard_limit, extended_hard_limit)

    def test_stable_win_shortens_only_soft_budget(self) -> None:
        manager = TimeManager(increment_s=0.5)
        manager.start(120_000, self.board)
        soft_limit = manager.soft_limit
        hard_limit = manager.hard_limit

        manager.shorten_for_stable_win()

        self.assertAlmostEqual(manager.soft_limit, soft_limit * 0.80)
        self.assertEqual(manager.hard_limit, hard_limit)

    def test_opening_positions_receive_extra_budget(self) -> None:
        first_move = Board.from_fen()
        later_move = Board.from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 20")
        first_manager = TimeManager(increment_s=0.5)
        later_manager = TimeManager(increment_s=0.5)
        first_manager.start(120_000, first_move)
        later_manager.start(120_000, later_move)

        self.assertAlmostEqual(later_manager.soft_limit, first_manager.soft_limit * 1.20)
        self.assertAlmostEqual(later_manager.hard_limit, first_manager.hard_limit * 1.20)


if __name__ == "__main__":
    unittest.main()
