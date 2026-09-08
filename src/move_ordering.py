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
    scored: list[tuple[int, chess.Move]] = []
    for move in moves:
        if move == tt_move:
            score = TT_MOVE_SCORE
        elif move.promotion:
            score = PROMOTION_SCORE + (9000 if move.promotion == chess.QUEEN else 2000)
        elif board.is_capture(move):
            victim = (
                chess.PAWN
                if move.to_square == ep_square
                else board.piece_type_at(move.to_square)
            )
            attacker = board.piece_type_at(move.from_square)
            score = CAPTURE_SCORE_BASE + MVV_LVA[victim or chess.PAWN][attacker or chess.PAWN]
        elif move == k1:
            score = KILLER_SCORE_1
        elif move == k2:
            score = KILLER_SCORE_2
        elif history is not None:
            score = history.get(turn, move)
        else:
            score = 0
        scored.append((score, move))

    scored.sort(key=lambda item: item[0], reverse=True)
    moves[:] = [move for _, move in scored]
    return moves
