from typing import Any, Final

import chess
import numpy as np
from numba import njit

MG_PIECE_VALUES: Final[tuple[int, ...]] = (82, 337, 365, 477, 1025, 0)
EG_PIECE_VALUES: Final[tuple[int, ...]] = (94, 281, 297, 512, 936, 0)
GAMEPHASE_INC: Final[tuple[int, ...]] = (0, 155, 305, 405, 1050, 0)
_GAMEPHASE_INC_NP: Final[np.ndarray] = np.array(GAMEPHASE_INC, dtype=np.int16)

# fmt: off
MG_PAWN_TABLE: Final[tuple[int, ...]] = (
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
)

EG_PAWN_TABLE: Final[tuple[int, ...]] = (
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
)

MG_KNIGHT_TABLE: Final[tuple[int, ...]] = (
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
)

EG_KNIGHT_TABLE: Final[tuple[int, ...]] = (
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
)

MG_BISHOP_TABLE: Final[tuple[int, ...]] = (
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
)

EG_BISHOP_TABLE: Final[tuple[int, ...]] = (
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
)

MG_ROOK_TABLE: Final[tuple[int, ...]] = (
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
)

EG_ROOK_TABLE: Final[tuple[int, ...]] = (
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
)

MG_QUEEN_TABLE: Final[tuple[int, ...]] = (
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
)

EG_QUEEN_TABLE: Final[tuple[int, ...]] = (
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
)

MG_KING_TABLE: Final[tuple[int, ...]] = (
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
)

EG_KING_TABLE: Final[tuple[int, ...]] = (
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
)
# fmt: on

MG_PESTO_TABLES: Final[tuple[tuple[int, ...], ...]] = (
    MG_PAWN_TABLE,
    MG_KNIGHT_TABLE,
    MG_BISHOP_TABLE,
    MG_ROOK_TABLE,
    MG_QUEEN_TABLE,
    MG_KING_TABLE,
)

EG_PESTO_TABLES: Final[tuple[tuple[int, ...], ...]] = (
    EG_PAWN_TABLE,
    EG_KNIGHT_TABLE,
    EG_BISHOP_TABLE,
    EG_ROOK_TABLE,
    EG_QUEEN_TABLE,
    EG_KING_TABLE,
)

# ---------------------------------------------------------------------------
# Precomputed tables: table[color][piece_index][sq]
# colour: 0 for WHITE, 1 for BLACK
# piece: 0..5 (Pawn, Knight, Bishop, Rook, Queen, King)
# ---------------------------------------------------------------------------

MG_TABLE: Final[tuple[tuple[tuple[int, ...], ...], ...]] = (
    tuple(
        tuple(MG_PIECE_VALUES[p] + MG_PESTO_TABLES[p][sq ^ 56] for sq in range(64))
        for p in range(6)
    ),
    tuple(tuple(MG_PIECE_VALUES[p] + MG_PESTO_TABLES[p][sq] for sq in range(64)) for p in range(6)),
)

EG_TABLE: Final[tuple[tuple[tuple[int, ...], ...], ...]] = (
    tuple(
        tuple(EG_PIECE_VALUES[p] + EG_PESTO_TABLES[p][sq ^ 56] for sq in range(64))
        for p in range(6)
    ),
    tuple(tuple(EG_PIECE_VALUES[p] + EG_PESTO_TABLES[p][sq] for sq in range(64)) for p in range(6)),
)
_MG_TABLE_NP: Final[np.ndarray] = np.array(MG_TABLE, dtype=np.int16)
_EG_TABLE_NP: Final[np.ndarray] = np.array(EG_TABLE, dtype=np.int16)

GAMEPHASE_SUM: Final[int] = int(
    sum(inc * count * 2 for inc, count in zip(GAMEPHASE_INC, (8, 2, 2, 2, 1, 1), strict=True))
)  # 5560

BISHOP_PAIR_MG: Final[int] = 20
BISHOP_PAIR_EG: Final[int] = 30
ISOLATED_PAWN_MG: Final[int] = 7
ISOLATED_PAWN_EG: Final[int] = 10
DOUBLED_PAWN_MG: Final[int] = 8
DOUBLED_PAWN_EG: Final[int] = 12
ROOK_SEMI_OPEN_MG: Final[int] = 5
ROOK_SEMI_OPEN_EG: Final[int] = 3
ROOK_OPEN_MG: Final[int] = 10
ROOK_OPEN_EG: Final[int] = 6
PASSED_PAWN_MG: Final[tuple[int, ...]] = (0, 0, 3, 7, 12, 20, 32, 0)
PASSED_PAWN_EG: Final[tuple[int, ...]] = (0, 0, 6, 12, 22, 38, 65, 0)
_PASSED_PAWN_MG_NP: Final[np.ndarray] = np.array(PASSED_PAWN_MG, dtype=np.int16)
_PASSED_PAWN_EG_NP: Final[np.ndarray] = np.array(PASSED_PAWN_EG, dtype=np.int16)

FILE_MASKS: Final[np.ndarray] = np.asarray(chess.BB_FILES, dtype=np.uint64)
ADJACENT_FILE_MASKS: Final[np.ndarray] = np.asarray(
    tuple(
        (chess.BB_FILES[file_index - 1] if file_index > 0 else 0)
        | (chess.BB_FILES[file_index + 1] if file_index < 7 else 0)
        for file_index in range(8)
    ),
    dtype=np.uint64,
)
FORWARD_FILE_MASKS: Final[np.ndarray] = np.asarray(
    tuple(
        tuple(
            sum(
                1 << target
                for target in range(64)
                if target & 7 == square & 7
                and (
                    (color_index == 0 and target >> 3 > square >> 3)
                    or (color_index == 1 and target >> 3 < square >> 3)
                )
            )
            for square in range(64)
        )
        for color_index in range(2)
    ),
    dtype=np.uint64,
)
PASSED_PAWN_MASKS: Final[np.ndarray] = np.asarray(
    tuple(
        tuple(
            sum(
                1 << target
                for target in range(64)
                if abs((target & 7) - (square & 7)) <= 1
                and (
                    (color_index == 0 and target >> 3 > square >> 3)
                    or (color_index == 1 and target >> 3 < square >> 3)
                )
            )
            for square in range(64)
        )
        for color_index in range(2)
    ),
    dtype=np.uint64,
)


# fmt: off
_DEBRUIJN64: Final = np.uint64(0x03F79D71B4CB0A89)
_LSB_INDEX: Final = np.asarray(
    (
        0,  1, 48,  2, 57, 49, 28,  3,
        61, 58, 50, 42, 38, 29, 17,  4,
        62, 55, 59, 36, 53, 51, 43, 22,
        45, 39, 33, 30, 24, 18, 12,  5,
        63, 47, 56, 27, 60, 41, 37, 16,
        54, 35, 52, 21, 44, 32, 23, 11,
        46, 26, 40, 15, 34, 20, 31, 10,
        25, 14, 19,  9, 13,  8,  7,  6
    ),
    dtype=np.uint8,
)
# fmt: on


@njit(cache=False)
def _evaluate_kernel(
    pawns: np.uint64,
    knights: np.uint64,
    bishops: np.uint64,
    rooks: np.uint64,
    queens: np.uint64,
    kings: np.uint64,
    occupied_white: np.uint64,
    occupied_black: np.uint64,
    white_to_move: bool,
) -> tuple[int, int]:
    """Return (score, raw_game_phase) — score from the side-to-move's perspective."""
    piece_masks = (pawns, knights, bishops, rooks, queens, kings)
    occupancies = (occupied_white, occupied_black)

    mg_white = 0
    mg_black = 0
    eg_white = 0
    eg_black = 0
    game_phase = 0

    for color_index in range(2):
        occupancy = occupancies[color_index]

        for piece_index in range(6):
            bb = piece_masks[piece_index] & occupancy

            while bb != 0:
                lsb = bb & (np.uint64(0) - bb)
                square = _LSB_INDEX[(lsb * _DEBRUIJN64) >> np.uint64(58)]

                if color_index == 0:
                    mg_white += _MG_TABLE_NP[color_index, piece_index, square]
                    eg_white += _EG_TABLE_NP[color_index, piece_index, square]
                else:
                    mg_black += _MG_TABLE_NP[color_index, piece_index, square]
                    eg_black += _EG_TABLE_NP[color_index, piece_index, square]
                game_phase += _GAMEPHASE_INC_NP[piece_index]

                bb ^= lsb

        own_pawns = pawns & occupancy
        enemy_pawns = pawns & occupancies[1 - color_index]
        mg_bonus = 0
        eg_bonus = 0

        own_bishops = bishops & occupancy
        if own_bishops != 0 and own_bishops & (own_bishops - np.uint64(1)) != 0:
            mg_bonus += BISHOP_PAIR_MG
            eg_bonus += BISHOP_PAIR_EG

        bb = own_pawns
        while bb != 0:
            lsb = bb & (np.uint64(0) - bb)
            square = _LSB_INDEX[(lsb * _DEBRUIJN64) >> np.uint64(58)]
            file_index = square & 7
            if own_pawns & ADJACENT_FILE_MASKS[file_index] == 0:
                mg_bonus -= ISOLATED_PAWN_MG
                eg_bonus -= ISOLATED_PAWN_EG

            doubled = own_pawns & FORWARD_FILE_MASKS[color_index, square] != 0
            if doubled:
                mg_bonus -= DOUBLED_PAWN_MG
                eg_bonus -= DOUBLED_PAWN_EG
            elif enemy_pawns & PASSED_PAWN_MASKS[color_index, square] == 0:
                relative_rank = np.int64(square) >> np.int64(3)
                if color_index != 0:
                    relative_rank = np.int64(7) - relative_rank
                mg_bonus += _PASSED_PAWN_MG_NP[relative_rank]
                eg_bonus += _PASSED_PAWN_EG_NP[relative_rank]
            bb ^= lsb

        bb = rooks & occupancy
        while bb != 0:
            lsb = bb & (np.uint64(0) - bb)
            square = _LSB_INDEX[(lsb * _DEBRUIJN64) >> np.uint64(58)]
            file_mask = FILE_MASKS[square & 7]
            if own_pawns & file_mask == 0:
                if enemy_pawns & file_mask:
                    mg_bonus += ROOK_SEMI_OPEN_MG
                    eg_bonus += ROOK_SEMI_OPEN_EG
                else:
                    mg_bonus += ROOK_OPEN_MG
                    eg_bonus += ROOK_OPEN_EG
            bb ^= lsb

        if color_index == 0:
            mg_white += mg_bonus
            eg_white += eg_bonus
        else:
            mg_black += mg_bonus
            eg_black += eg_bonus

    side_index = 0 if white_to_move else 1

    if side_index == 0:
        mg_diff = mg_white - mg_black
        eg_diff = eg_white - eg_black
    else:
        mg_diff = mg_black - mg_white
        eg_diff = eg_black - eg_white

    mg_phase = min(game_phase, GAMEPHASE_SUM)
    eg_phase = GAMEPHASE_SUM - mg_phase

    score = mg_diff * mg_phase + eg_diff * eg_phase
    final_score = int(score // GAMEPHASE_SUM if score >= 0 else -((-score) // GAMEPHASE_SUM))

    return final_score, int(mg_phase)


def evaluate(board: Any) -> int:
    if hasattr(board, "evaluate"):
        return int(board.evaluate())
    score, _ = _evaluate_kernel(
        np.uint64(board.pawns),
        np.uint64(board.knights),
        np.uint64(board.bishops),
        np.uint64(board.rooks),
        np.uint64(board.queens),
        np.uint64(board.kings),
        np.uint64(board.occupied_co[chess.WHITE]),
        np.uint64(board.occupied_co[chess.BLACK]),
        board.turn == chess.WHITE,
    )
    return score


def evaluate_with_phase(board: Any) -> tuple[int, float]:
    if hasattr(board, "evaluate"):
        score = board.evaluate()
        phase = min(board.game_phase, GAMEPHASE_SUM) / GAMEPHASE_SUM
        return score, phase
    score, raw_phase = _evaluate_kernel(
        np.uint64(board.pawns),
        np.uint64(board.knights),
        np.uint64(board.bishops),
        np.uint64(board.rooks),
        np.uint64(board.queens),
        np.uint64(board.kings),
        np.uint64(board.occupied_co[chess.WHITE]),
        np.uint64(board.occupied_co[chess.BLACK]),
        board.turn == chess.WHITE,
    )
    return score, raw_phase / GAMEPHASE_SUM


evaluate(chess.Board())
