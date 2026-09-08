from typing import Final

import chess

PIECE_VALUES: Final[tuple[int, ...]] = (0, 100, 320, 330, 500, 900, 0)
PROMOTION_SCORE: Final[int] = 90000
CAPTURE_SCORE_BASE: Final[int] = 70000
CASTLE_SCORE: Final[int] = 60000


def score_move(board: chess.Board, move: chess.Move) -> int:
    """Score a move for move ordering (promotions > captures with MVV-LVA > castling > quiet)."""
    promo = move.promotion
    if promo:
        return PROMOTION_SCORE + (9000 if promo == chess.QUEEN else 2000)

    to_sq = move.to_square
    if to_sq == board.ep_square and (board.pawns & (1 << move.from_square)):
        return CAPTURE_SCORE_BASE + (PIECE_VALUES[chess.PAWN] * 10) - PIECE_VALUES[chess.PAWN]

    victim = board.piece_type_at(to_sq)
    if victim:
        attacker = board.piece_type_at(move.from_square) or 1
        return CAPTURE_SCORE_BASE + (PIECE_VALUES[victim] * 10) - PIECE_VALUES[attacker]

    if board.is_castling(move):
        return CASTLE_SCORE

    return 0


def generate_quiescence_moves(board: chess.Board) -> list[chess.Move]:
    """Generate candidate tactical moves for quiescence search: captures and queen promotions."""
    moves = list(board.generate_legal_captures())
    turn = board.turn
    promo_rank = chess.BB_RANK_7 if turn == chess.WHITE else chess.BB_RANK_2
    if board.pawns & board.occupied_co[turn] & promo_rank:
        target_rank = chess.BB_RANK_8 if turn == chess.WHITE else chess.BB_RANK_1
        for move in board.generate_legal_moves(from_mask=promo_rank, to_mask=target_rank):
            if move.promotion == chess.QUEEN and not board.is_capture(move):
                moves.append(move)

    return moves


def order_moves(
    board: chess.Board,
    moves: list[chess.Move] | None = None,
) -> list[chess.Move]:
    """Order moves by priority (promotions, captures with MVV-LVA, castling, quiet)."""
    if moves is None:
        moves = list(board.legal_moves)
    if len(moves) <= 1:
        return moves

    moves.sort(key=lambda m: score_move(board, m), reverse=True)
    return moves
