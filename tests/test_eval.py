import unittest

import chess

from src.scripts.eval import AdaptiveEvaluator, EvalStats, GameResult


class EvalStatsTests(unittest.TestCase):
    @staticmethod
    def _result(score: float, skill: int) -> GameResult:
        return GameResult(
            game_id=1,
            bot_color=chess.WHITE,
            result="1-0",
            bot_score=score,
            termination_reason="checkmate",
            stockfish_skill=skill,
            bot_time_left_ms=0.0,
            stockfish_time_left_ms=0.0,
            total_moves=1,
        )

    def test_rating_uses_each_game_opponent_skill(self) -> None:
        stats = EvalStats()
        stats.add_result(self._result(1.0, 4))
        stats.add_result(self._result(0.0, 6))

        elo, margin = stats.calculate_elo()

        self.assertAlmostEqual(elo, 1320.0 + 5 * 93.5, places=6)
        self.assertGreater(margin, 0.0)

    def test_equal_score_against_one_skill_matches_its_anchor(self) -> None:
        stats = EvalStats()
        stats.add_result(self._result(1.0, 5))
        stats.add_result(self._result(0.0, 5))

        elo, _ = stats.calculate_elo()

        self.assertAlmostEqual(elo, 1320.0 + 5 * 93.5, places=6)

    def test_parallel_evaluation_requires_fixed_skill(self) -> None:
        evaluator = AdaptiveEvaluator()

        with self.assertRaisesRegex(ValueError, "fixed Stockfish skill"):
            evaluator.run_benchmark(workers=2)


if __name__ == "__main__":
    unittest.main()
