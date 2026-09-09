import chess

from src.search import Bot

bot = Bot()


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    return bot.get_best_move(board, time_left_ms).uci()
