from __future__ import annotations

from typing import TYPE_CHECKING, Final

from src.board import (
    EN_PASSANT,
    KNIGHT_PROMO,
    PAWN,
    QUEEN,
    move_from,
    move_is_capture,
    move_promo_piece,
    move_to,
)

if TYPE_CHECKING:
    from src.board import Board

# Piece values indexed by piece constants: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
PIECE_VALUES: Final[tuple[int, ...]] = (100, 320, 330, 500, 900, 0)
TT_MOVE_SCORE: Final[int] = 100_000
PROMOTION_SCORE: Final[int] = 90_000
CAPTURE_SCORE_BASE: Final[int] = 70_000
KILLER_SCORE_1: Final[int] = 50_000
KILLER_SCORE_2: Final[int] = 49_000
MAX_PLY: Final[int] = 128

MVV_LVA: Final[tuple[tuple[int, ...], ...]] = tuple(
    tuple(PIECE_VALUES[v] * 10 - PIECE_VALUES[a] for a in range(6)) for v in range(6)
)


class KillerTable:
    __slots__ = ("_table",)

    def __init__(self) -> None:
        self._table: list[list[int]] = [[0, 0] for _ in range(MAX_PLY)]

    def store(self, ply: int, move: int) -> None:
        if ply >= MAX_PLY or move == 0:
            return
        slot = self._table[ply]
        if move == slot[0]:
            return
        slot[1] = slot[0]
        slot[0] = move

    def get(self, ply: int) -> tuple[int, int]:
        if ply >= MAX_PLY:
            return 0, 0
        slot = self._table[ply]
        return slot[0], slot[1]

    def clear(self) -> None:
        for slot in self._table:
            slot[0] = 0
            slot[1] = 0


class HistoryTable:
    __slots__ = ("_table",)

    def __init__(self) -> None:
        self._table: list[list[list[int]]] = [[[0] * 64 for _ in range(64)] for _ in range(2)]

    def update(self, color: bool, move: int, depth: int) -> None:
        c = 0 if color else 1
        from_sq = move_from(move)
        to_sq = move_to(move)
        bonus = depth * depth
        val = self._table[c][from_sq][to_sq]
        self._table[c][from_sq][to_sq] = min(val + bonus, 16384)

    def penalise(self, color: bool, move: int, depth: int) -> None:
        c = 0 if color else 1
        from_sq = move_from(move)
        to_sq = move_to(move)
        penalty = depth * depth
        val = self._table[c][from_sq][to_sq]
        self._table[c][from_sq][to_sq] = max(val - penalty, -16384)

    def get(self, color: bool, move: int) -> int:
        c = 0 if color else 1
        return self._table[c][move_from(move)][move_to(move)]


def order_moves(
    board: Board,
    moves: list[int] | None = None,
    tt_move: int = 0,
    ply: int = 0,
    killers: KillerTable | None = None,
    history: HistoryTable | None = None,
) -> list[int]:
    if moves is None:
        moves = board.generate_moves()
    if len(moves) <= 1:
        return moves

    k1, k2 = killers.get(ply) if killers is not None else (0, 0)
    history_scores = history._table[0 if board.turn else 1] if history is not None else None
    piece_at_sq = board.piece_at_sq
    mvv_lva = MVV_LVA

    def score(move: int) -> int:
        if tt_move != 0 and move == tt_move:
            return TT_MOVE_SCORE
        flags = (move >> 12) & 0xF
        if flags >= KNIGHT_PROMO:
            return PROMOTION_SCORE + (9000 if move_promo_piece(move) == QUEEN else 2000)

        from_square = move & 0x3F
        to_square = (move >> 6) & 0x3F

        if move_is_capture(move):
            victim = PAWN if flags == EN_PASSANT else piece_at_sq[to_square]
            attacker = piece_at_sq[from_square]
            return CAPTURE_SCORE_BASE + mvv_lva[victim][attacker]

        if k1 != 0 and move == k1:
            return KILLER_SCORE_1
        if k2 != 0 and move == k2:
            return KILLER_SCORE_2

        return history_scores[from_square][to_square] if history_scores is not None else 0

    moves.sort(key=score, reverse=True)
    return moves
