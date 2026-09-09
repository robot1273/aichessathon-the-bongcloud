"""Polyglot Zobrist keys and repetition-safe hashing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import chess.polyglot
import numpy as np

from src.attacks import bishop_attacks, rook_attacks

if TYPE_CHECKING:
    from src.board import Board

_ARR: Final[list[int]] = chess.polyglot.POLYGLOT_RANDOM_ARRAY

# PIECE_KEYS[color][piece_type][square]
# color: WHITE=0, BLACK=1
# piece: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
PIECE_KEYS: Final[tuple[tuple[tuple[int, ...], ...], ...]] = tuple(
    tuple(tuple(_ARR[64 * (p * 2 + (1 - c)) + sq] for sq in range(64)) for p in range(6))
    for c in range(2)
)

CASTLING_KEYS: Final[tuple[int, ...]] = tuple(_ARR[768 + i] for i in range(4))

CASTLING_TABLE: Final[tuple[int, ...]] = tuple(
    (CASTLING_KEYS[0] if mask & 1 else 0)
    ^ (CASTLING_KEYS[1] if mask & 2 else 0)
    ^ (CASTLING_KEYS[2] if mask & 4 else 0)
    ^ (CASTLING_KEYS[3] if mask & 8 else 0)
    for mask in range(16)
)

EP_KEYS: Final[tuple[int, ...]] = tuple(_ARR[772 + f] for f in range(8))
TURN_KEY: Final[int] = _ARR[780]


def _gen_ep_candidate_masks() -> tuple[tuple[int, ...], tuple[int, ...]]:
    white_masks = [0] * 64
    black_masks = [0] * 64
    for sq in range(64):
        f = sq & 7
        r = sq >> 3
        # For White to capture en passant: ep square must be on rank 5 (r == 5)
        # White pawns on rank 4 (r == 4) on adjacent files can capture
        if r == 5:
            mask = 0
            if f > 0:
                mask |= 1 << (4 * 8 + (f - 1))
            if f < 7:
                mask |= 1 << (4 * 8 + (f + 1))
            white_masks[sq] = mask

        # For Black to capture en passant: ep square must be on rank 2 (r == 2)
        # Black pawns on rank 3 (r == 3) on adjacent files can capture
        if r == 2:
            mask = 0
            if f > 0:
                mask |= 1 << (3 * 8 + (f - 1))
            if f < 7:
                mask |= 1 << (3 * 8 + (f + 1))
            black_masks[sq] = mask
    return tuple(white_masks), tuple(black_masks)


EP_CANDIDATE_MASKS: Final[tuple[tuple[int, ...], ...]] = _gen_ep_candidate_masks()

NB_PIECE_KEYS = np.array(PIECE_KEYS, dtype=np.uint64)
NB_CASTLING_TABLE = np.array(CASTLING_TABLE, dtype=np.uint64)
NB_EP_KEYS = np.array(EP_KEYS, dtype=np.uint64)
NB_TURN_KEY = np.uint64(TURN_KEY)
NB_EP_CANDIDATE_MASKS = np.array(EP_CANDIDATE_MASKS, dtype=np.uint64)


def has_legal_en_passant(board: Board) -> bool:
    """Return whether the side to move has a legal en-passant capture."""
    if board.ep_square == -1:
        return False

    us = 0 if board.turn else 1
    them = 1 - us
    candidates = EP_CANDIDATE_MASKS[us][board.ep_square] & board.pieces[0] & board.colours[us]
    if not candidates:
        return False

    king_sq = board.king_sq[us]
    captured_sq = board.ep_square - 8 if us == 0 else board.ep_square + 8
    occupied = board.colours[0] | board.colours[1]
    enemy_diag = (board.pieces[2] | board.pieces[4]) & board.colours[them]
    enemy_orth = (board.pieces[3] | board.pieces[4]) & board.colours[them]

    while candidates:
        from_sq = (candidates & -candidates).bit_length() - 1
        candidates &= candidates - 1
        occupied_after = (occupied ^ (1 << from_sq) ^ (1 << captured_sq)) | (1 << board.ep_square)
        if not (bishop_attacks(king_sq, np.uint64(occupied_after)) & enemy_diag) and not (
            rook_attacks(king_sq, np.uint64(occupied_after)) & enemy_orth
        ):
            return True

    return False


def calculate_hash(board: Board) -> int:
    """Calculate the repetition-safe Zobrist hash for a Board instance."""
    h = 0
    # Pieces
    for sq in range(64):
        pt = board.piece_at_sq[sq]
        if pt >= 0:
            c = board.colour_at_sq[sq]
            h ^= PIECE_KEYS[c][pt][sq]

    # Castling
    h ^= CASTLING_TABLE[board.castling]

    # Turn (Polyglot convention: XOR turn key when White to move)
    if board.turn:
        h ^= TURN_KEY

    # Repetition identity includes en passant only when it is legal.
    if has_legal_en_passant(board):
        h ^= EP_KEYS[board.ep_square & 7]

    return h
