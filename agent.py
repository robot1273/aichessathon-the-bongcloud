"""The submission entrypoint. The platform imports this file and calls get_move."""

import chess
from src.search import Bot

# Import time runs at the game start. 90s to import packages, build tables etc.

bot = Bot()

def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in, and your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion

    The process stays alive, but suspended between your moves, so state you keep on a module or in a
    closure survives to the next call. It does not survive to the next game

    print() is safe. Your stdout is redirected away from the protocol stream and kept in a
    log only your team can read, after validation and after every rated game.
    """
    board = chess.Board(fen)
    return bot.get_best_move(board, 4).uci()
