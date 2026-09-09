"""Constants for Numba-JIT bitboard engine."""

from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Piece Types
# ---------------------------------------------------------------------------
PAWN: Final[int] = 0
KNIGHT: Final[int] = 1
BISHOP: Final[int] = 2
ROOK: Final[int] = 3
QUEEN: Final[int] = 4
KING: Final[int] = 5

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
WHITE: Final[int] = 0
BLACK: Final[int] = 1

# ---------------------------------------------------------------------------
# Move Flags (16-bit encoding: from(6) | to(6) | flags(4))
# ---------------------------------------------------------------------------
QUIET: Final[int] = 0b0000
DOUBLE_PAWN_PUSH: Final[int] = 0b0001
KING_CASTLE: Final[int] = 0b0010
QUEEN_CASTLE: Final[int] = 0b0011
CAPTURE: Final[int] = 0b0100
EN_PASSANT: Final[int] = 0b0101
KNIGHT_PROMO: Final[int] = 0b1000
BISHOP_PROMO: Final[int] = 0b1001
ROOK_PROMO: Final[int] = 0b1010
QUEEN_PROMO: Final[int] = 0b1011
KNIGHT_PROMO_CAPTURE: Final[int] = 0b1100
BISHOP_PROMO_CAPTURE: Final[int] = 0b1101
ROOK_PROMO_CAPTURE: Final[int] = 0b1110
QUEEN_PROMO_CAPTURE: Final[int] = 0b1111

MAX_PLY: Final[int] = 128
INF: Final[int] = 2_000_000
MATE_SCORE: Final[int] = 1_000_000
MATE_THRESHOLD: Final[int] = 900_000
NO_MOVE: Final[int] = 0
PROMOTION_PIECES = (QUEEN_PROMO, KNIGHT_PROMO, ROOK_PROMO, BISHOP_PROMO)
PROMOTION_CAPTURE_PIECES = (
    QUEEN_PROMO_CAPTURE,
    KNIGHT_PROMO_CAPTURE,
    ROOK_PROMO_CAPTURE,
    BISHOP_PROMO_CAPTURE,
)

# ---------------------------------------------------------------------------
# State Array Indices (Size: 25 uint64)
# ---------------------------------------------------------------------------
# The board state is represented as a single numpy array of dtype=uint64.
# This avoids Python object overhead and enables Numba JIT compilation.

P_PAWN: Final[int] = 0
P_KNIGHT: Final[int] = 1
P_BISHOP: Final[int] = 2
P_ROOK: Final[int] = 3
P_QUEEN: Final[int] = 4
P_KING: Final[int] = 5

C_WHITE: Final[int] = 6
C_BLACK: Final[int] = 7

TURN: Final[int] = 8  # 0 for WHITE, 1 for BLACK
CASTLING: Final[int] = 9  # 4-bit mask (WK=1, WQ=2, BK=4, BQ=8)
EP_SQUARE: Final[int] = 10  # 0-63, or 64 if none
HALFMOVE: Final[int] = 11

HASH: Final[int] = 12  # Zobrist hash

MG_SCORE_W: Final[int] = 13
MG_SCORE_B: Final[int] = 14
EG_SCORE_W: Final[int] = 15
EG_SCORE_B: Final[int] = 16

GAME_PHASE: Final[int] = 17
NULL_SEARCH: Final[int] = 18

STATE_SIZE: Final[int] = 19

# ---------------------------------------------------------------------------
# Zobrist, Castling, and Move Helpers
# ---------------------------------------------------------------------------
CASTLE_WK: Final[int] = 1
CASTLE_WQ: Final[int] = 2
CASTLE_BK: Final[int] = 4
CASTLE_BQ: Final[int] = 8

_CASTLING_RIGHTS_MASK: list[int] = [15] * 64
_CASTLING_RIGHTS_MASK[0] = 13
_CASTLING_RIGHTS_MASK[7] = 14
_CASTLING_RIGHTS_MASK[4] = 12
_CASTLING_RIGHTS_MASK[56] = 7
_CASTLING_RIGHTS_MASK[63] = 11
_CASTLING_RIGHTS_MASK[60] = 3

NB_CASTLING_RIGHTS_MASK = np.array(_CASTLING_RIGHTS_MASK, dtype=np.uint8)
