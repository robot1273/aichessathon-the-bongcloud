import argparse
import math
import time
from dataclasses import dataclass, field

import chess
import chess.engine

from src.board import Board, move_to_uci
from src.search import Bot


@dataclass
class GameResult:
    game_id: int
    bot_color: chess.Color
    result: str  # "1-0", "0-1", "1/2-1/2", "void"
    bot_score: float  # 1.0, 0.5, 0.0
    termination_reason: str  # "mate", "flag", "illegal", etc.
    stockfish_skill: int
    bot_time_left_ms: float
    stockfish_time_left_ms: float
    total_moves: int
    time_violations: int = 0
    bot_max_move_time_ms: float = 0.0


@dataclass
class EvalStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_games: int = 0
    total_time_violations: int = 0
    flag_failures: int = 0
    results: list[GameResult] = field(default_factory=list)

    def add_result(self, res: GameResult) -> None:
        self.results.append(res)
        self.total_games += 1
        self.total_time_violations += res.time_violations

        if "flag" in res.termination_reason.lower() and res.bot_score == 0.0:
            self.flag_failures += 1

        if res.bot_score == 1.0:
            self.wins += 1
        elif res.bot_score == 0.5:
            self.draws += 1
        else:
            self.losses += 1

    @property
    def score_percentage(self) -> float:
        if self.total_games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.total_games

    def calculate_elo(self) -> tuple[float, float]:
        """Fit a rating against the skill level actually used in every game."""
        if self.total_games == 0:
            return 0.0, 0.0

        opponent_elos = [1320.0 + res.stockfish_skill * 93.5 for res in self.results]
        target_score = max(0.001, min(0.999, self.score_percentage))

        def expected_score(rating: float) -> float:
            total = 0.0
            for opponent_elo in opponent_elos:
                total += 1.0 / (1.0 + 10.0 ** ((opponent_elo - rating) / 400.0))
            return total / self.total_games

        lower, upper = -1000.0, 5000.0
        for _ in range(60):
            midpoint = (lower + upper) / 2.0
            if expected_score(midpoint) < target_score:
                lower = midpoint
            else:
                upper = midpoint

        bot_elo = (lower + upper) / 2.0
        information = sum(
            (math.log(10.0) / 400.0) ** 2
            * (1.0 / (1.0 + 10.0 ** ((opponent_elo - bot_elo) / 400.0)))
            * (1.0 - 1.0 / (1.0 + 10.0 ** ((opponent_elo - bot_elo) / 400.0)))
            for opponent_elo in opponent_elos
        )
        error_margin = 1.96 / math.sqrt(information) if information else 0.0

        return bot_elo, error_margin


class AdaptiveEvaluator:
    def __init__(
        self,
        stockfish_path: str = "stockfish",
        initial_skill: int = 0,
        base_time_ms: int = 2000,
        inc_ms: int = 50,
        tolerance_buffer_ms: int = 150,  # Max allowed engine latency over clock limit
    ) -> None:
        self.stockfish_path = stockfish_path
        self.current_skill = max(0, min(20, initial_skill))
        self.base_time_ms = base_time_ms
        self.inc_ms = inc_ms
        self.tolerance_buffer_ms = tolerance_buffer_ms

    def run_benchmark(self, total_games: int = 12, adapt_skill: bool = True) -> EvalStats:
        stats = EvalStats()

        print(f"Starting Engine Match ({total_games} Games)")
        print(
            f"Time Control: {self.base_time_ms}ms + {self.inc_ms}ms inc "
            f"| Adaptive Skill: {adapt_skill}"
        )
        print("=" * 65)

        for game_idx in range(1, total_games + 1):
            # Alternate colors every game
            bot_color = chess.WHITE if game_idx % 2 != 0 else chess.BLACK
            color_str = "White" if bot_color == chess.WHITE else "Black"

            print(
                f"Game {game_idx:02d}/{total_games:02d} | "
                f"Bot ({color_str}) vs Stockfish (Skill {self.current_skill})",
                end=" | ",
            )

            res = self._play_game(game_idx, bot_color)
            stats.add_result(res)

            # Adapt Stockfish Skill level dynamically using a win/loss step scheme
            if adapt_skill:
                if res.bot_score == 1.0 and self.current_skill < 20:
                    self.current_skill += 1
                elif res.bot_score == 0.0 and self.current_skill > 0:
                    self.current_skill -= 1

            bot_result = (
                "win" if res.bot_score == 1.0 else "draw" if res.bot_score == 0.5 else "loss"
            )
            print(
                f"Bot: {bot_result} | Board: {res.result} ({res.termination_reason}) | "
                f"Max Move Time: {res.bot_max_move_time_ms:.0f}ms | "
                f"Violations: {res.time_violations}"
            )

        self._print_summary(stats)
        return stats

    def _play_game(self, game_id: int, bot_color: chess.Color) -> GameResult:
        board = chess.Board()
        bot = Bot(increment_s=self.inc_ms / 1000)

        bot_clock = float(self.base_time_ms)
        sf_clock = float(self.base_time_ms)

        time_violations = 0
        max_move_time = 0.0
        total_moves = 0

        # Launch Stockfish engine
        try:
            engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            engine.configure({"Skill Level": self.current_skill})
        except Exception as e:
            print(f"\n[ERROR] Stockfish launch failed: {e}")
            return GameResult(
                game_id=game_id,
                bot_color=bot_color,
                result="void",
                bot_score=0.5,
                termination_reason="Stockfish launch error",
                stockfish_skill=self.current_skill,
                bot_time_left_ms=bot_clock,
                stockfish_time_left_ms=sf_clock,
                total_moves=0,
            )

        while not board.is_game_over(claim_draw=True):
            total_moves += 1
            mover = board.turn

            if mover == bot_color:
                # --- BOT'S TURN ---
                if bot_clock <= 0:
                    engine.quit()
                    return GameResult(
                        game_id=game_id,
                        bot_color=bot_color,
                        result="0-1" if bot_color == chess.WHITE else "1-0",
                        bot_score=0.0,
                        termination_reason="flag",  # Engine flagged
                        stockfish_skill=self.current_skill,
                        bot_time_left_ms=0.0,
                        stockfish_time_left_ms=sf_clock,
                        total_moves=total_moves,
                        time_violations=time_violations + 1,
                        bot_max_move_time_ms=max_move_time,
                    )

                start_t = time.perf_counter()
                try:
                    # Pass custom board to bot
                    custom_board = Board.from_fen(board.fen())
                    move_int = bot.get_best_move(custom_board, time_left_ms=int(bot_clock))
                    move = chess.Move.from_uci(move_to_uci(move_int))
                except Exception as e:
                    engine.quit()
                    return GameResult(
                        game_id=game_id,
                        bot_color=bot_color,
                        result="0-1" if bot_color == chess.WHITE else "1-0",
                        bot_score=0.0,
                        termination_reason=f"crash ({type(e).__name__})",
                        stockfish_skill=self.current_skill,
                        bot_time_left_ms=bot_clock,
                        stockfish_time_left_ms=sf_clock,
                        total_moves=total_moves,
                        bot_max_move_time_ms=max_move_time,
                    )

                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                max_move_time = max(max_move_time, elapsed_ms)

                # STRESS-TEST TIMING CHECK:
                if elapsed_ms > (bot_clock + self.inc_ms + self.tolerance_buffer_ms):
                    time_violations += 1

                bot_clock -= elapsed_ms

                if bot_clock < 0:
                    engine.quit()
                    return GameResult(
                        game_id=game_id,
                        bot_color=bot_color,
                        result="0-1" if bot_color == chess.WHITE else "1-0",
                        bot_score=0.0,
                        termination_reason="flag",
                        stockfish_skill=self.current_skill,
                        bot_time_left_ms=0.0,
                        stockfish_time_left_ms=sf_clock,
                        total_moves=total_moves,
                        time_violations=time_violations,
                        bot_max_move_time_ms=max_move_time,
                    )

                bot_clock += self.inc_ms

                if move not in board.legal_moves:
                    engine.quit()
                    return GameResult(
                        game_id=game_id,
                        bot_color=bot_color,
                        result="0-1" if bot_color == chess.WHITE else "1-0",
                        bot_score=0.0,
                        termination_reason="illegal",
                        stockfish_skill=self.current_skill,
                        bot_time_left_ms=bot_clock,
                        stockfish_time_left_ms=sf_clock,
                        total_moves=total_moves,
                        bot_max_move_time_ms=max_move_time,
                    )

                board.push(move)

            else:
                # --- STOCKFISH'S TURN ---
                start_t = time.perf_counter()

                limit = chess.engine.Limit(
                    white_clock=sf_clock / 1000.0
                    if board.turn == chess.WHITE
                    else bot_clock / 1000.0,
                    black_clock=bot_clock / 1000.0
                    if board.turn == chess.WHITE
                    else sf_clock / 1000.0,
                    white_inc=self.inc_ms / 1000.0,
                    black_inc=self.inc_ms / 1000.0,
                )

                sf_res = engine.play(board, limit)
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                sf_clock = max(0.0, sf_clock - elapsed_ms + self.inc_ms)

                if sf_res.move is None:
                    engine.quit()
                    return GameResult(
                        game_id=game_id,
                        bot_color=bot_color,
                        result="1-0" if bot_color == chess.WHITE else "0-1",
                        bot_score=1.0,
                        termination_reason="stockfish_forfeit",
                        stockfish_skill=self.current_skill,
                        bot_time_left_ms=bot_clock,
                        stockfish_time_left_ms=sf_clock,
                        total_moves=total_moves,
                        time_violations=time_violations,
                        bot_max_move_time_ms=max_move_time,
                    )

                board.push(sf_res.move)

        engine.quit()

        # Determine outcome
        outcome = board.outcome(claim_draw=True)
        assert outcome is not None

        if outcome.winner is None:
            res_str = "1/2-1/2"
            score = 0.5
        elif outcome.winner == bot_color:
            res_str = "1-0" if bot_color == chess.WHITE else "0-1"
            score = 1.0
        else:
            res_str = "0-1" if bot_color == chess.WHITE else "1-0"
            score = 0.0

        return GameResult(
            game_id=game_id,
            bot_color=bot_color,
            result=res_str,
            bot_score=score,
            termination_reason=outcome.termination.name.lower(),
            stockfish_skill=self.current_skill,
            bot_time_left_ms=bot_clock,
            stockfish_time_left_ms=sf_clock,
            total_moves=total_moves,
            time_violations=time_violations,
            bot_max_move_time_ms=max_move_time,
        )

    def _print_summary(self, stats: EvalStats) -> None:
        print("\n" + "=" * 65)
        print("MATCH EVALUATION SUMMARY")
        print("=" * 65)
        print(f"Total Games Played:     {stats.total_games}")
        print(f"Record (W-L-D):         {stats.wins} - {stats.losses} - {stats.draws}")
        print(f"Score:                  {stats.score_percentage * 100:.1f}%")
        print(f"Total Time Violations:  {stats.total_time_violations}")
        print(f"Flag Rate Failures:     {stats.flag_failures}")

        elo, margin = stats.calculate_elo()
        skills = [res.stockfish_skill for res in stats.results]
        print(
            f"Estimated Bot Rating:   {elo:.0f} ± {margin:.0f} Elo "
            f"(vs SF Skills {min(skills)}-{max(skills)})"
        )
        print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate engine against Stockfish using time controls."
    )
    parser.add_argument("--stockfish", default="stockfish", help="Path to Stockfish binary")
    parser.add_argument("--games", type=int, default=12, help="Number of games to play")
    parser.add_argument("--skill", type=int, default=3, help="Initial Stockfish skill level (0-20)")
    parser.add_argument("--base-time", type=int, default=9000, help="Base time per game (ms)")
    parser.add_argument("--inc", type=int, default=50, help="Time increment per move (ms)")
    parser.add_argument(
        "--static-skill", action="store_true", help="Disable adaptive skill adjustments"
    )

    args = parser.parse_args()

    evaluator = AdaptiveEvaluator(
        stockfish_path=args.stockfish,
        initial_skill=args.skill,
        base_time_ms=args.base_time,
        inc_ms=args.inc,
    )

    evaluator.run_benchmark(total_games=args.games, adapt_skill=not args.static_skill)


if __name__ == "__main__":
    main()
