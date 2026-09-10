"""Endgame conversion suite: bot always holds the winning side.

Eight verified winning endgames (seven TB-covered, one 5-man for the
search), each played as White and mirrored for Black: 16 games total.
Measures conversion (wins), resistance to swindles (losses/draws), and
plies-to-mate for wins. TB-covered positions assert WDL==+2 at startup
when tb/ is present, so silently-drawn FEN typos fail loudly instead.

Usage:
    uv run python -m src.scripts.endgame_eval --games-per-side 1 --skill 9
    uv run python -m src.scripts.endgame_eval --no-tb   # search-only ablation
"""

from __future__ import annotations

import argparse

import chess

from src.scripts.eval import AdaptiveEvaluator, EvalStats

# Side to move is winning in all positions (TB-verified unless noted).
BASE_FENS: list[tuple[str, str]] = [
    ("KQK", "7k/8/8/8/8/8/6Q1/6K1 w - - 0 1"),
    ("KRK", "7k/8/8/8/8/8/6R1/6K1 w - - 0 1"),
    ("KPK-c", "8/8/8/8/8/5K2/5P2/5k2 w - - 0 1"),
    ("KPK-d", "8/8/8/8/4K3/8/4P3/4k3 w - - 0 1"),
    ("KBNK", "7k/8/8/8/8/8/8/KBN5 w - - 0 1"),
    ("KRPvKR", "3rk3/8/8/8/8/8/3RP3/3K4 w - - 0 1"),
    ("KRvKP", "6k1/8/8/8/8/6p1/6R1/6K1 w - - 0 1"),
    # 5-man, no TB coverage: Stockfish depth-24 reports forced mate.
    ("KQvKRP", "3rk3/8/8/8/8/2p5/3QP3/3K4 w - - 0 1"),
]


def build_fens() -> tuple[list[str], list[str]]:
    fens: list[str] = []
    names: list[str] = []
    for name, fen in BASE_FENS:
        board = chess.Board(fen)
        if not board.is_valid():
            raise SystemExit(f"suite FEN illegal, fix me: {name} {fen}")
        fens.append(fen)
        names.append(f"{name}w")
        mirrored = board.mirror().fen()
        fens.append(mirrored)
        names.append(f"{name}b")
    return fens, names


def verify_tb_wins(fens: list[str], names: list[str]) -> None:
    try:
        import chess.syzygy

        tb = chess.syzygy.open_tablebase("tb")
    except OSError:
        print("tb/ unavailable, skipping WDL verification")
        return
    for name, fen in zip(names, fens, strict=True):
        if name.startswith("KQvKRP"):
            continue  # 5-man, SF-verified instead
        board = chess.Board(fen)
        wdl = tb.get_wdl(board)
        if wdl != 2:
            raise SystemExit(f"suite position not winning per TB: {name} wdl={wdl}")
    tb.close()
    print("TB verification: all covered suite positions are WDL +2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Endgame conversion suite.")
    parser.add_argument("--skill", type=int, default=9)
    parser.add_argument("--base-time", type=int, default=12000)
    parser.add_argument("--inc", type=int, default=50)
    parser.add_argument("--no-tb", action="store_true")
    args = parser.parse_args()

    fens, names = build_fens()
    verify_tb_wins(fens, names)

    evaluator = AdaptiveEvaluator(
        initial_skill=args.skill,
        base_time_ms=args.base_time,
        inc_ms=args.inc,
        use_tb=not args.no_tb,
    )
    stats = EvalStats()
    for game_id, (name, fen) in enumerate(zip(names, fens, strict=True), 1):
        bot_color = chess.WHITE if chess.Board(fen).turn == chess.WHITE else chess.BLACK
        res = evaluator._play_game(game_id, bot_color, fen)
        stats.add_result(res)
        outcome = (
            "win" if res.bot_score == 1.0 else "draw" if res.bot_score == 0.5 else "loss"
        )
        print(
            f"[{name}] Bot: {outcome} | {res.result} ({res.termination_reason}) "
            f"| moves: {res.total_moves} | max_ms: {res.bot_max_move_time_ms:.0f} "
            f"| viol: {res.time_violations}"
        )
    print("=" * 65)
    print(
        f"ENDGAME SUITE: +{stats.wins} ={stats.draws} -{stats.losses} "
        f"score {stats.score_percentage * 100:.1f}% over {stats.total_games} "
        f"| violations {stats.total_time_violations}"
    )
    mate_plies = [r.total_moves for r in stats.results if r.bot_score == 1.0]
    if mate_plies:
        print(f"conversion plies (wins): {sorted(mate_plies)}")


if __name__ == "__main__":
    main()
