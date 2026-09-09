import argparse
import math
import time
from dataclasses import dataclass, field

import chess
import chess.engine

# Import your bot architecture
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

    def calculate_elo(self, sf_skill: int) -> tuple[float, float]:
        """Calculates estimated absolute Elo based on Stockfish Skill level anchor."""
        if self.total_games == 0:
            return 0.0, 0.0

        p = max(0.001, min(0.999, self.score_percentage))

        # Approximate Stockfish skill level to base Elo (Skill 0 ~= 1320, Skill 20 ~= 3190)
        sf_estimated_elo = 1320 + (sf_skill * 93.5)

        # Relative Elo formula (Logistic distribution)
        elo_diff = -400.0 * math.log10(1.0 / p - 1.0)
        bot_elo = sf_estimated_elo + elo_diff

        # Standard Error (95% confidence bounds)
        stdev = math.sqrt(
            (self.wins * (1 - p) ** 2 + self.draws * (0.5 - p) ** 2 + self.losses * (0 - p) ** 2)
            / self.total_games
        )
        error_margin = (1.96 * stdev / math.sqrt(self.total_games)) * (400.0 / (math.log(10) * p * (1 - p)))

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
        print(f"Time Control: {self.base_time_ms}ms + {self.inc_ms}ms inc | Adaptive Skill: {adapt_skill}")
        print("=" * 65)

        for game_idx in range(1, total_games + 1):
            # Alternate colors every game
            bot_color = chess.WHITE if game_idx % 2 != 0 else chess.BLACK
            color_str = "White" if bot_color == chess.WHITE else "Black"

            print(f"Game {game_idx:02d}/{total_games:02d} | Bot ({color_str}) vs Stockfish (Skill {self.current_skill})", end=" | ")

            res = self._play_game(game_idx, bot_color)
            stats.add_result(res)

            # Adapt Stockfish Skill level dynamically using a win/loss step scheme
            if adapt_skill:
                if res.bot_score == 1.0 and self.current_skill < 20:
                    self.current_skill += 1
                elif res.bot_score == 0.0 and self.current_skill > 0:
                    self.current_skill -= 1

            print(
                f"Result: {res.result} ({res.termination_reason}) | "
                f"Max Move Time: {res.bot_max_move_time_ms:.0f}ms | Violations: {res.time_violations}"
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
                    # Pass remaining clock to bot
                    move = bot.get_best_move(board, time_left_ms=int(bot_clock))
                except Exception as e:
                    engine.quit()
                    return GameResult(
                        game_id=game_id,
                        bot_color=bot_color,
                        result="0-1" if bot_color == chess.WHITE else "1-0",
                        bot_score=0.0,
                        termination_reason=f"crash ({type(e).__name__})",  #[cite: 1]
                        stockfish_skill=self.current_skill,
                        bot_time_left_ms=bot_clock,
                        stockfish_time_left_ms=sf_clock,
                        total_moves=total_moves,
                        bot_max_move_time_ms=max_move_time,
                    )

                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                max_move_time = max(max_move_time, elapsed_ms)

                # STRESS-TEST TIMING CHECK:
                # Check if single move execution exceeded available time + allowed tolerance
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
                        termination_reason="flag",  #[cite: 1]
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
                        termination_reason="illegal",  #[cite: 1]
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
                    white_clock=sf_clock / 1000.0 if board.turn == chess.WHITE else bot_clock / 1000.0,
                    black_clock=bot_clock / 1000.0 if board.turn == chess.WHITE else sf_clock / 1000.0,
                    white_inc=self.inc_ms / 1000.0,
                    black_inc=self.inc_ms / 1000.0,
                )

                sf_res = engine.play(board, limit)
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                sf_clock = max(0.0, sf_clock - elapsed_ms + self.inc_ms)

                if sf_res.move is None:
                    break
                board.push(sf_res.move)

        engine.quit()

        # Parse match conclusion
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            score = 0.5
            result_str = "1/2-1/2"
            reason = outcome.termination.name.lower() if outcome else "draw"  #[cite: 1]
        elif outcome.winner == bot_color:
            score = 1.0
            result_str = "1-0" if bot_color == chess.WHITE else "0-1"
            reason = outcome.termination.name.lower()  #[cite: 1]
        else:
            score = 0.0
            result_str = "0-1" if bot_color == chess.WHITE else "1-0"
            reason = outcome.termination.name.lower()  #[cite: 1]

        return GameResult(
            game_id=game_id,
            bot_color=bot_color,
            result=result_str,
            bot_score=score,
            termination_reason=reason,
            stockfish_skill=self.current_skill,
            bot_time_left_ms=bot_clock,
            stockfish_time_left_ms=sf_clock,
            total_moves=total_moves,
            time_violations=time_violations,
            bot_max_move_time_ms=max_move_time,
        )

    def _print_summary(self, stats: EvalStats) -> None:
        estimated_elo, err = stats.calculate_elo(self.current_skill)
        print("\n================ BENCHMARK SUMMARY ================")
        print(f"Total Match Games:    {stats.total_games}")
        print(f"Record:               {stats.wins}W / {stats.draws}D / {stats.losses}L ({stats.score_percentage * 100:.1f}%)")
        print(f"Ending SF Skill:      {self.current_skill}")
        print(f"Estimated Bot Elo:    ~{estimated_elo:.0f} +/- {err:.0f}")
        print(f"Flag Losses:          {stats.flag_failures}")
        print(f"Timing Violations:    {stats.total_time_violations}")
        print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Stockfish Evaluator with Time Stress Testing.")
    parser.add_argument("--stockfish-path", type=str, default="stockfish", help="Path to Stockfish binary.")
    parser.add_argument("--games", type=int, default=25, help="Number of games to run.")
    parser.add_argument("--skill", type=int, default=3, help="Starting Stockfish Skill Level (0-20).")

    # Stress-test presets
    parser.add_argument("--stress_test", choices=["bullet", "sudden_death", "std"], default="std", help="Time stress test mode")

    args = parser.parse_args()

    # Configure time controls for stress tests
    if args.stress_test == "bullet":
        base_ms, inc_ms = 1000, 50
    elif args.stress_test == "sudden_death":
        base_ms, inc_ms = 2000, 0
    elif args.stress_test == "std":
        base_ms, inc_ms = 120_000, 500 # 2m 0.5s increment

    evaluator = AdaptiveEvaluator(
        stockfish_path=args.stockfish_path,
        initial_skill=args.skill,
        base_time_ms=base_ms,
        inc_ms=inc_ms,
    )
    evaluator.run_benchmark(total_games=args.games, adapt_skill=True)
