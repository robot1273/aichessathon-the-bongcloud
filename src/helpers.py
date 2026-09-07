import chess
import numpy as np

from .types import BoardArray, Bitboard

def encode_board(board: chess.Board) -> tuple[BoardArray, Bitboard]:
    """Given a board, returns a set of all pieces in the board, and a bitboard mask of the current player's pieces"""
    pieces = np.zeros(64, dtype=np.int32)
    player_mask = np.zeros(64, dtype=np.bool_)

    for square, piece in board.piece_map().items():
        pieces[square] = piece.piece_type
        player_mask[square] = piece.color == board.turn

    return pieces, player_mask
