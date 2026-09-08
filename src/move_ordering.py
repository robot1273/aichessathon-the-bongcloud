from typing import Final

import chess

PIECE_VALUES: Final[tuple[int, ...]] = (0, 100, 320, 330, 500, 900, 0)
TT_MOVE_SCORE: Final[int] = 100_000
PROMOTION_SCORE: Final[int] = 90_000
CAPTURE_SCORE_BASE: Final[int] = 70_000
CASTLE_SCORE: Final[int] = 60_000
KILLER_SCORE_1: Final[int] = 50_000
KILLER_SCORE_2: Final[int] = 49_000
MAX_PLY: Final[int] = 128

MVV_LVA: Final[tuple[tuple[int, ...], ...]] = tuple(
    tuple(PIECE_VALUES[v] * 10 - PIECE_VALUES[a] for a in range(7))
    for v in range(7)
)


class KillerTable:
    __slots__ = ("_table",)

    def __init__(self) -> None:
        self._table: list[list[chess.Move | None]] = [
            [None, None] for _ in range(MAX_PLY)
        ]

    def store(self, ply: int, move: chess.Move) -> None:
        if ply >= MAX_PLY:
            return
        slot = self._table[ply]
        if move == slot[0]:
            return
        slot[1] = slot[0]
        slot[0] = move

    def get(self, ply: int) -> tuple[chess.Move | None, chess.Move | None]:
        if ply >= MAX_PLY:
            return None, None
        slot = self._table[ply]
        return slot[0], slot[1]

    def clear(self) -> None:
        for slot in self._table:
            slot[0] = None
            slot[1] = None


class HistoryTable:
    __slots__ = ("_table",)

    def __init__(self) -> None:
        self._table: list[list[list[int]]] = [
            [[0] * 64 for _ in range(64)] for _ in range(2)
        ]

    def update(self, color: bool, move: chess.Move, depth: int) -> None:
        c = 0 if color else 1
        bonus = depth * depth
        val = self._table[c][move.from_square][move.to_square]
        self._table[c][move.from_square][move.to_square] = min(val + bonus, 16384)

    def penalise(self, color: bool, move: chess.Move, depth: int) -> None:
        c = 0 if color else 1
        penalty = depth * depth
        val = self._table[c][move.from_square][move.to_square]
        self._table[c][move.from_square][move.to_square] = max(val - penalty, -16384)

    def get(self, color: bool, move: chess.Move) -> int:
        c = 0 if color else 1
        return self._table[c][move.from_square][move.to_square]


def score_move(
    board: chess.Board,
    move: chess.Move,
    tt_move: chess.Move | None = None,
    ply: int = 0,
    killers: KillerTable | None = None,
    history: HistoryTable | None = None,
) -> int:
    if move == tt_move:
        return TT_MOVE_SCORE

    promo = move.promotion
    if promo:
        return PROMOTION_SCORE + (9000 if promo == chess.QUEEN else 2000)

    to_sq = move.to_square
    from_sq = move.from_square

    if to_sq == board.ep_square and (board.pawns & (1 << from_sq)):
        return CAPTURE_SCORE_BASE + MVV_LVA[chess.PAWN][chess.PAWN]
    victim = board.piece_type_at(to_sq) or 0
    if victim:
        attacker = board.piece_type_at(from_sq) or 1
        return CAPTURE_SCORE_BASE + MVV_LVA[victim][attacker]

    if board.is_castling(move):
        return CASTLE_SCORE

    if killers is not None:
        k1, k2 = killers.get(ply)
        if move == k1:
            return KILLER_SCORE_1
        if move == k2:
            return KILLER_SCORE_2

    if history is not None:
        return history.get(board.turn, move)

    return 0


def generate_quiescence_moves(board: chess.Board) -> list[chess.Move]:
    """Generate candidate tactical moves for quiescence search: captures and queen promotions."""
    moves = list(board.generate_legal_captures())
    turn = board.turn
    promo_rank = chess.BB_RANK_7 if turn == chess.WHITE else chess.BB_RANK_2
    pawn_mask = board.pawns & board.occupied_co[turn] & promo_rank
    if pawn_mask:
        target_rank = chess.BB_RANK_8 if turn == chess.WHITE else chess.BB_RANK_1
        for move in board.generate_legal_moves(from_mask=pawn_mask, to_mask=target_rank):
            if move.promotion == chess.QUEEN and not board.is_capture(move):
                moves.append(move)

    return moves


def order_moves(
    board: chess.Board,
    moves: list[chess.Move] | None = None,
    tt_move: chess.Move | None = None,
    ply: int = 0,
    killers: KillerTable | None = None,
    history: HistoryTable | None = None,
) -> list[chess.Move]:
    if moves is None:
        moves = list(board.legal_moves)
    if len(moves) <= 1:
        return moves

    moves.sort(
        key=lambda m: score_move(board, m, tt_move, ply, killers, history),
        reverse=True,
    )
    return moves
