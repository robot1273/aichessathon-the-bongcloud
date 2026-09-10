from __future__ import annotations

from typing import TYPE_CHECKING, Final

from src.constants import (
    EN_PASSANT,
    KNIGHT_PROMO,
    NO_MOVE,
    PAWN,
    QUEEN,
)

if TYPE_CHECKING:
    from src.board import Board

# Piece values indexed by piece constants: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
PIECE_VALUES: Final[tuple[int, ...]] = (100, 320, 330, 500, 900, 0)
MVV_LVA: Final[tuple[tuple[int, ...], ...]] = tuple(
    tuple(PIECE_VALUES[v] * 10 - PIECE_VALUES[a] for a in range(6)) for v in range(6)
)


def pick_fallback_move(
    board: Board,
    moves: list[int],
    tt_move: int = NO_MOVE,
) -> int:
    """Choose a safe move before the timed search has completed an iteration."""
    if tt_move in moves:
        return tt_move

    piece_at_sq = board.piece_at_sq
    mvv_lva = MVV_LVA

    def score(move: int) -> int:
        flags = (move >> 12) & 0xF
        if flags >= KNIGHT_PROMO:
            promo_piece = (flags & 0x3) + 1
            return 90_000 + (9_000 if promo_piece == QUEEN else 2_000)

        from_square = move & 0x3F
        to_square = (move >> 6) & 0x3F

        if flags & 0x4:
            victim = PAWN if flags == EN_PASSANT else piece_at_sq[to_square]
            attacker = piece_at_sq[from_square]
            return 70_000 + mvv_lva[victim][attacker]
        return 0

    return max(moves, key=score)
