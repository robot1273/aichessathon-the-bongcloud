import chess

from .constants import PIECE_VALUES

def order_moves(board: chess.Board) -> list[chess.Move]:
    """MVV-LVA move ordering"""
    moves = list(board.legal_moves)
    if len(moves) <= 1:
        return moves

    def score_move(move: chess.Move) -> int:
        score = 0

        if board.is_capture(move):
            attacker = board.piece_type_at(move.from_square) or 1 #or 1 fallback in case type is none, god forbid
            if board.is_en_passant(move):
                victim = chess.PAWN
            else:
                victim = board.piece_type_at(move.to_square) or 1

            score = 10000 + (PIECE_VALUES[victim] * 10) - PIECE_VALUES[attacker]
            # we scale victim higher: e.g. pawn is 100, queen is 900, score for PxQ = 7900, QxP = 100
            # the 10,000 added is to ensure all capture moves are scored above non-capture moves

        elif move.promotion: # promotion score bonus, and we REALLY like queen promo!
            score = 9000 if move.promotion == chess.QUEEN else 2000

        return score

    moves.sort(key=score_move, reverse=True)
    return moves
