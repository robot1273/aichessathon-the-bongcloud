from src.board import Board, move_to_uci
from src.search import Bot

bot = Bot()


def get_move(fen: str, time_left_ms: int) -> str:
    board = Board.from_fen(fen)
    best_move = bot.get_best_move(board, time_left_ms)
    return move_to_uci(best_move)
