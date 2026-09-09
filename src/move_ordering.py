from typing import Final

import chess

PIECE_VALUES: Final[tuple[int, ...]] = (0, 100, 320, 330, 500, 900, 0)
TT_MOVE_SCORE: Final[int] = 100_000
PROMOTION_SCORE: Final[int] = 90_000
CAPTURE_SCORE_BASE: Final[int] = 70_000
KILLER_SCORE_1: Final[int] = 50_000
KILLER_SCORE_2: Final[int] = 49_000
MAX_PLY: Final[int] = 128

MVV_LVA: Final[tuple[tuple[int, ...], ...]] = tuple(
    tuple(PIECE_VALUES[v] * 10 - PIECE_VALUES[a] for a in range(7)) for v in range(7)
)


class KillerTable:
    __slots__ = ("_table",)

    def __init__(self) -> None:
        self._table: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY)]

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
        self._table: list[list[list[int]]] = [[[0] * 64 for _ in range(64)] for _ in range(2)]

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

    k1, k2 = killers.get(ply) if killers is not None else (None, None)
    turn = board.turn
    ep_square = board.ep_square
    occupied = board.occupied
    pawns = board.pawns
    squares = chess.BB_SQUARES
    piece_type_at = board.piece_type_at
    mvv_lva = MVV_LVA
    history_scores = history._table[0 if turn else 1] if history is not None else None

    def score(move: chess.Move) -> int:
        if tt_move is not None and move == tt_move:
            return TT_MOVE_SCORE
        if move.promotion:
            return PROMOTION_SCORE + (9000 if move.promotion == chess.QUEEN else 2000)

        from_square = move.from_square
        to_square = move.to_square
        if occupied & squares[to_square] or (
            to_square == ep_square and pawns & squares[from_square]
        ):
            victim = chess.PAWN if to_square == ep_square else piece_type_at(to_square)
            attacker = piece_type_at(from_square)
            return CAPTURE_SCORE_BASE + mvv_lva[victim or chess.PAWN][attacker or chess.PAWN]
        if k1 is not None and move == k1:
            return KILLER_SCORE_1
        if k2 is not None and move == k2:
            return KILLER_SCORE_2
        return history_scores[from_square][to_square] if history_scores is not None else 0

    moves.sort(key=score, reverse=True)
    return moves
