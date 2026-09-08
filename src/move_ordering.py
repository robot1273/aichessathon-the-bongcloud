from typing import Final

import chess

PIECE_VALUES: Final[tuple[int, ...]] = (0, 100, 320, 330, 500, 900, 0)


def order_moves(board: chess.Board) -> list[chess.Move]:
    """MVV-LVA move ordering"""
    moves = list(board.legal_moves)
    if len(moves) <= 1:
        return moves

    def score_move(move: chess.Move) -> int:
        score = 0

        if board.is_capture(move):
            attacker = (
                board.piece_type_at(move.from_square) or 1
            )  # or 1 fallback in case type is none, god forbid
            if board.is_en_passant(move):
                victim = chess.PAWN
            else:
                victim = board.piece_type_at(move.to_square) or 1

            score = 10000 + (PIECE_VALUES[victim] * 10) - PIECE_VALUES[attacker]
            # we scale victim higher: eg. pawn is 100, queen is 900, score for PxQ = 7900, QxP = 100
            # the 10,000 added is to ensure all capture moves are scored above non-capture moves
        elif move.promotion:
            score = 9000 if move.promotion == chess.QUEEN else 2000

        return score

    moves.sort(key=score_move, reverse=True)
    return moves
