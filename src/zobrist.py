from typing import Final

import chess
import chess.polyglot

# Extract canonical 64-bit Zobrist keys from python-chess's precompiled Polyglot array
_ARR: Final[list[int]] = chess.polyglot.POLYGLOT_RANDOM_ARRAY

# Pre-indexed for O(1) piece key lookup: [color (0..1)][piece_type - 1 (0..5)][square (0..63)]
PIECE_KEYS: Final[tuple[tuple[tuple[int, ...], ...], ...]] = tuple(
    tuple(tuple(_ARR[64 * ((p - 1) * 2 + c) + sq] for sq in range(64)) for p in range(1, 7))
    for c in range(2)
)
CASTLING_KEYS: Final[tuple[int, ...]] = tuple(_ARR[768 + i] for i in range(4))
EP_KEYS: Final[tuple[int, ...]] = tuple(_ARR[772 + f] for f in range(8))
TURN_KEY: Final[int] = _ARR[780]

BB_H1: Final[int] = chess.BB_H1
BB_A1: Final[int] = chess.BB_A1
BB_H8: Final[int] = chess.BB_H8
BB_A8: Final[int] = chess.BB_A8


def calculate_hash(board: chess.Board) -> int:
    return chess.polyglot.zobrist_hash(board)


def push_hash(board: chess.Board, move: chess.Move, h: int) -> int:
    """Incrementally update the Zobrist hash across a move and push to board."""
    turn = board.turn
    c = 0 if turn == chess.WHITE else 1
    opp_c = 1 - c
    from_sq = move.from_square
    to_sq = move.to_square

    piece_type = board.piece_type_at(from_sq)
    assert piece_type is not None
    p = piece_type - 1

    h ^= TURN_KEY

    old_ep = board.ep_square
    if old_ep is not None:
        h ^= EP_KEYS[old_ep & 7]

    old_castling = board.castling_rights

    cap_type = board.piece_type_at(to_sq)
    if cap_type is not None:
        h ^= PIECE_KEYS[opp_c][cap_type - 1][to_sq]
    elif old_ep == to_sq and p == 0:
        ep_pawn_sq = to_sq - 8 if turn == chess.WHITE else to_sq + 8
        h ^= PIECE_KEYS[opp_c][0][ep_pawn_sq]

    h ^= PIECE_KEYS[c][p][from_sq]
    promo = move.promotion
    if promo:
        h ^= PIECE_KEYS[c][promo - 1][to_sq]
    else:
        h ^= PIECE_KEYS[c][p][to_sq]

    # Castling rook move
    if p == 5 and abs(to_sq - from_sq) == 2:
        if to_sq == 6:  # G1
            h ^= PIECE_KEYS[0][3][7] ^ PIECE_KEYS[0][3][5]
        elif to_sq == 2:  # C1
            h ^= PIECE_KEYS[0][3][0] ^ PIECE_KEYS[0][3][3]
        elif to_sq == 62:  # G8
            h ^= PIECE_KEYS[1][3][63] ^ PIECE_KEYS[1][3][61]
        elif to_sq == 58:  # C8
            h ^= PIECE_KEYS[1][3][56] ^ PIECE_KEYS[1][3][59]

    board.push(move)

    new_ep = board.ep_square
    if new_ep is not None:
        h ^= EP_KEYS[new_ep & 7]

    castling_diff = old_castling ^ board.castling_rights
    if castling_diff:
        if castling_diff & BB_H1:
            h ^= CASTLING_KEYS[0]
        if castling_diff & BB_A1:
            h ^= CASTLING_KEYS[1]
        if castling_diff & BB_H8:
            h ^= CASTLING_KEYS[2]
        if castling_diff & BB_A8:
            h ^= CASTLING_KEYS[3]

    return h
