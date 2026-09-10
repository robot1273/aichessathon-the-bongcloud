"""Core Numba-jitted primitives for board representation, move generation, and evaluation."""

# ruff: noqa
import numpy as np
from numba import njit

from src.attacks import (
    NB_KING_ATTACKS,
    NB_KNIGHT_ATTACKS,
    NB_PAWN_ATTACKS,
    NB_RAY_BETWEEN,
    bishop_attacks,
    queen_attacks,
    rook_attacks,
)
from src.constants import *

# Pull in the eval tables for incremental PeSTO evaluation.
from src.evaluation import (
    ADJACENT_FILE_MASKS,
    BISHOP_PAIR_EG,
    BISHOP_PAIR_MG,
    DOUBLED_PAWN_EG,
    DOUBLED_PAWN_MG,
    FILE_MASKS,
    FORWARD_FILE_MASKS,
    GAMEPHASE_SUM,
    ISOLATED_PAWN_EG,
    ISOLATED_PAWN_MG,
    PASSED_PAWN_MASKS,
    ROOK_OPEN_EG,
    ROOK_OPEN_MG,
    ROOK_SEMI_OPEN_EG,
    ROOK_SEMI_OPEN_MG,
    _EG_TABLE_NP,
    _GAMEPHASE_INC_NP,
    _MG_TABLE_NP,
    _PASSED_PAWN_EG_NP,
    _PASSED_PAWN_MG_NP,
)
from src.zobrist import (
    NB_CASTLING_TABLE,
    NB_EP_CANDIDATE_MASKS,
    NB_EP_KEYS,
    NB_PIECE_KEYS,
    NB_TURN_KEY,
)

_DEBRUIJN64 = np.uint64(0x03F79D71B4CB0A89)
_LSB_INDEX = np.array(
    (
        0,  1, 48,  2, 57, 49, 28,  3,
        61, 58, 50, 42, 38, 29, 17,  4,
        62, 55, 59, 36, 53, 51, 43, 22,
        45, 39, 33, 30, 24, 18, 12,  5,
        63, 47, 56, 27, 60, 41, 37, 16,
        54, 35, 52, 21, 44, 32, 23, 11,
        46, 26, 40, 15, 34, 20, 31, 10,
        25, 14, 19,  9, 13,  8,  7,  6,
    ),
    dtype=np.uint8,
)


@njit(cache=False)
def lsb_sq(bb: int | np.uint64) -> int:
    """Least significant bit index using 64-bit De Bruijn multiplication."""
    u = np.uint64(bb)
    lsb = u & (np.uint64(0) - u)
    return int(_LSB_INDEX[(lsb * _DEBRUIJN64) >> np.uint64(58)])


@njit(cache=False)
def _bit_length(n: int) -> int:
    """Fast bit length for uint64."""
    if n == 0:
        return 0
    res = 0
    while n > 0:
        res += 1
        n >>= 1
    return res


@njit(cache=False)
def _add_piece(state: np.ndarray, sq: int, piece: int, color: int) -> None:
    sq_bb = np.uint64(1) << np.uint64(sq)
    state[piece] |= sq_bb
    state[C_WHITE + color] |= sq_bb

    state[MG_SCORE_W + color] += _MG_TABLE_NP[color, piece, sq]
    state[EG_SCORE_W + color] += _EG_TABLE_NP[color, piece, sq]
    state[GAME_PHASE] += _GAMEPHASE_INC_NP[piece]
    state[HASH] ^= NB_PIECE_KEYS[color, piece, sq]


@njit(cache=False)
def _clear_piece(state: np.ndarray, sq: int, piece: int, color: int) -> None:
    sq_bb = ~(np.uint64(1) << np.uint64(sq))
    state[piece] &= sq_bb
    state[C_WHITE + color] &= sq_bb

    state[MG_SCORE_W + color] -= _MG_TABLE_NP[color, piece, sq]
    state[EG_SCORE_W + color] -= _EG_TABLE_NP[color, piece, sq]
    state[GAME_PHASE] -= _GAMEPHASE_INC_NP[piece]
    state[HASH] ^= NB_PIECE_KEYS[color, piece, sq]


@njit(cache=False)
def piece_type_at(state: np.ndarray, sq: int) -> int:
    sq_bb = np.uint64(1) << np.uint64(sq)
    for p in range(6):
        if state[p] & sq_bb:
            return p
    return -1


@njit(cache=False)
def color_at(state: np.ndarray, sq: int) -> int:
    sq_bb = np.uint64(1) << np.uint64(sq)
    if state[C_WHITE] & sq_bb:
        return WHITE
    if state[C_BLACK] & sq_bb:
        return BLACK
    return -1


@njit(cache=False)
def _has_legal_en_passant(state: np.ndarray, color: int, ep_sq: int) -> bool:
    candidates = state[P_PAWN] & state[C_WHITE + color] & NB_EP_CANDIDATE_MASKS[color, ep_sq]
    if not candidates:
        return False

    them = 1 - color
    king_sq = lsb_sq(state[P_KING] & state[C_WHITE + color])
    captured_sq = ep_sq - 8 if color == WHITE else ep_sq + 8
    occupied = state[C_WHITE] | state[C_BLACK]
    enemy_diag = (state[P_BISHOP] | state[P_QUEEN]) & state[C_WHITE + them]
    enemy_orth = (state[P_ROOK] | state[P_QUEEN]) & state[C_WHITE + them]

    while candidates:
        from_bit = candidates & (~candidates + np.uint64(1))
        from_sq = lsb_sq(from_bit)
        candidates &= candidates - np.uint64(1)
        occupied_after = (
            occupied
            ^ (np.uint64(1) << np.uint64(from_sq))
            ^ (np.uint64(1) << np.uint64(captured_sq))
        ) | (np.uint64(1) << np.uint64(ep_sq))
        if not (bishop_attacks(king_sq, occupied_after) & enemy_diag) and not (
            rook_attacks(king_sq, occupied_after) & enemy_orth
        ):
            return True

    return False


@njit(cache=False)
def make_move(state: np.ndarray, undo_stack: np.ndarray, ply: int, move: int) -> None:
    # Save current state to undo_stack
    for i in range(STATE_SIZE):
        undo_stack[ply, i] = state[i]

    us = int(state[TURN])
    them = 1 - us

    from_sq = move & 0x3F
    to_sq = (move >> 6) & 0x3F
    flags = (move >> 12) & 0xF

    moving_piece = piece_type_at(state, from_sq)
    captured_piece = -1
    if flags & CAPTURE and flags != EN_PASSANT:
        captured_piece = piece_type_at(state, to_sq)

    # Remove old EP hash if it was active
    ep_sq = state[EP_SQUARE]
    if ep_sq != 64:
        if _has_legal_en_passant(state, us, ep_sq):
            state[HASH] ^= NB_EP_KEYS[ep_sq & 7]
        state[EP_SQUARE] = 64

    # Halfmove clock
    if moving_piece == PAWN or captured_piece != -1:
        state[HALFMOVE] = 0
    else:
        state[HALFMOVE] += 1

    # Handle captures
    if captured_piece != -1:
        _clear_piece(state, to_sq, captured_piece, them)
    elif flags == EN_PASSANT:
        ep_cap_sq = to_sq - 8 if us == WHITE else to_sq + 8
        _clear_piece(state, ep_cap_sq, PAWN, them)

    # Move or promote piece
    _clear_piece(state, from_sq, moving_piece, us)
    if flags >= KNIGHT_PROMO:
        promo_piece = flags - 7 if flags < 12 else flags - 11  # mapping
        _add_piece(state, to_sq, promo_piece, us)
    else:
        _add_piece(state, to_sq, moving_piece, us)

    # Handle special moves
    if flags == DOUBLE_PAWN_PUSH:
        state[EP_SQUARE] = from_sq + 8 if us == WHITE else from_sq - 8
        if _has_legal_en_passant(state, them, state[EP_SQUARE]):
            state[HASH] ^= NB_EP_KEYS[state[EP_SQUARE] & 7]
    elif flags == KING_CASTLE:
        if us == WHITE:
            _clear_piece(state, 7, ROOK, WHITE)
            _add_piece(state, 5, ROOK, WHITE)
        else:
            _clear_piece(state, 63, ROOK, BLACK)
            _add_piece(state, 61, ROOK, BLACK)
    elif flags == QUEEN_CASTLE:
        if us == WHITE:
            _clear_piece(state, 0, ROOK, WHITE)
            _add_piece(state, 3, ROOK, WHITE)
        else:
            _clear_piece(state, 56, ROOK, BLACK)
            _add_piece(state, 59, ROOK, BLACK)

    # Castling rights update
    old_castling = state[CASTLING]
    new_castling = old_castling & NB_CASTLING_RIGHTS_MASK[from_sq] & NB_CASTLING_RIGHTS_MASK[to_sq]
    if new_castling != old_castling:
        state[CASTLING] = new_castling
        state[HASH] ^= NB_CASTLING_TABLE[old_castling] ^ NB_CASTLING_TABLE[new_castling]

    # S3: refresh incremental structural bonuses only when pawn structure,
    # bishop pair, or rook files could have changed (mirrors Board logic).
    # Unmake restores via the undo-stack copy, so no work there.
    if (
        moving_piece == PAWN
        or moving_piece == ROOK
        or moving_piece == BISHOP
        or captured_piece == PAWN
        or captured_piece == ROOK
        or captured_piece == BISHOP
        or flags == KING_CASTLE
        or flags == QUEEN_CASTLE
        or flags == EN_PASSANT
    ):
        _refresh_bonus(state, us)
        if (
            moving_piece == PAWN
            or captured_piece == PAWN
            or captured_piece == ROOK
            or captured_piece == BISHOP
            or flags == EN_PASSANT
        ):
            _refresh_bonus(state, them)

    state[TURN] = them
    state[HASH] ^= NB_TURN_KEY


@njit(cache=False)
def unmake_move(state: np.ndarray, undo_stack: np.ndarray, ply: int) -> None:
    for i in range(STATE_SIZE):
        state[i] = undo_stack[ply, i]


@njit(cache=False)
def make_null_move(state: np.ndarray, undo_stack: np.ndarray, ply: int) -> None:
    for i in range(STATE_SIZE):
        undo_stack[ply, i] = state[i]

    us = int(state[TURN])
    them = 1 - us

    ep_sq = state[EP_SQUARE]
    if ep_sq != 64:
        if _has_legal_en_passant(state, us, ep_sq):
            state[HASH] ^= NB_EP_KEYS[ep_sq & 7]
        state[EP_SQUARE] = 64

    state[TURN] = them
    state[HASH] ^= NB_TURN_KEY
    state[NULL_SEARCH] = 1


@njit(cache=False)
def unmake_null_move(state: np.ndarray, undo_stack: np.ndarray, ply: int) -> None:
    for i in range(STATE_SIZE):
        state[i] = undo_stack[ply, i]


@njit(cache=False)
def is_sq_attacked(state: np.ndarray, sq: int, by_color: int) -> bool:
    opp_pawns = state[P_PAWN] & state[C_WHITE + by_color]
    us = 1 - by_color
    if NB_PAWN_ATTACKS[us, sq] & opp_pawns:
        return True

    opp_knights = state[P_KNIGHT] & state[C_WHITE + by_color]
    if NB_KNIGHT_ATTACKS[sq] & opp_knights:
        return True

    opp_king_sq = lsb_sq(state[P_KING] & state[C_WHITE + by_color])
    if NB_KING_ATTACKS[opp_king_sq] & (np.uint64(1) << np.uint64(sq)):
        return True

    occupied = state[C_WHITE] | state[C_BLACK]
    opp_diag = (state[P_BISHOP] | state[P_QUEEN]) & state[C_WHITE + by_color]
    if opp_diag and (bishop_attacks(sq, occupied) & opp_diag):
        return True

    opp_orth = (state[P_ROOK] | state[P_QUEEN]) & state[C_WHITE + by_color]
    if opp_orth and (rook_attacks(sq, occupied) & opp_orth):
        return True

    return False


@njit(cache=False)
def is_in_check(state: np.ndarray) -> bool:
    us = int(state[TURN])
    king_bb = state[P_KING] & state[C_WHITE + us]
    if not king_bb:
        return False
    king_sq = lsb_sq(king_bb)
    return is_sq_attacked(state, king_sq, 1 - us)


@njit(cache=False)
def _structural_bonus(state: np.ndarray, color: int) -> tuple[int, int]:
    own_pieces = state[C_WHITE + color]
    own_pawns = state[P_PAWN] & own_pieces
    enemy_pawns = state[P_PAWN] & state[C_WHITE + (1 - color)]
    bishops = state[P_BISHOP] & own_pieces

    mg_bonus = 0
    eg_bonus = 0
    if bishops and bishops & (bishops - np.uint64(1)):
        mg_bonus += BISHOP_PAIR_MG
        eg_bonus += BISHOP_PAIR_EG

    pawns = own_pawns
    while pawns:
        pawn = pawns & (~pawns + np.uint64(1))
        square = lsb_sq(pawn)
        pawns &= pawns - np.uint64(1)
        file_index = square & 7

        if not own_pawns & ADJACENT_FILE_MASKS[file_index]:
            mg_bonus -= ISOLATED_PAWN_MG
            eg_bonus -= ISOLATED_PAWN_EG

        if own_pawns & FORWARD_FILE_MASKS[color, square]:
            mg_bonus -= DOUBLED_PAWN_MG
            eg_bonus -= DOUBLED_PAWN_EG
        elif not enemy_pawns & PASSED_PAWN_MASKS[color, square]:
            relative_rank = np.int64(square) >> np.int64(3)
            if color != WHITE:
                relative_rank = np.int64(7) - relative_rank
            mg_bonus += _PASSED_PAWN_MG_NP[relative_rank]
            eg_bonus += _PASSED_PAWN_EG_NP[relative_rank]

    rooks = state[P_ROOK] & own_pieces
    while rooks:
        rook = rooks & (~rooks + np.uint64(1))
        square = lsb_sq(rook)
        rooks &= rooks - np.uint64(1)
        file_mask = FILE_MASKS[square & 7]
        if not own_pawns & file_mask:
            if enemy_pawns & file_mask:
                mg_bonus += ROOK_SEMI_OPEN_MG
                eg_bonus += ROOK_SEMI_OPEN_EG
            else:
                mg_bonus += ROOK_OPEN_MG
                eg_bonus += ROOK_OPEN_EG

    return mg_bonus, eg_bonus


@njit(cache=False)
def evaluate(state: np.ndarray) -> int:
    us = int(state[TURN])
    them = 1 - us
    # S3: structural bonuses maintained incrementally in make_move; pure
    # arithmetic here (no pawn/rook scans). Slots wrap-uint64 like PST scores.
    mg = (
        np.int64(state[MG_SCORE_W + us])
        + np.int64(state[MG_BONUS_W + us])
        - np.int64(state[MG_SCORE_W + them])
        - np.int64(state[MG_BONUS_W + them])
    )
    eg = (
        np.int64(state[EG_SCORE_W + us])
        + np.int64(state[EG_BONUS_W + us])
        - np.int64(state[EG_SCORE_W + them])
        - np.int64(state[EG_BONUS_W + them])
    )
    phase = min(int(state[GAME_PHASE]), GAMEPHASE_SUM)
    eg_phase = GAMEPHASE_SUM - phase
    score = mg * phase + eg * eg_phase
    return int(score // GAMEPHASE_SUM if score >= 0 else -((-score) // GAMEPHASE_SUM))


@njit(cache=False)
def _refresh_bonus(state: np.ndarray, color: int) -> None:
    """Recompute and store one side's structural bonus (S3)."""
    mg, eg = _structural_bonus(state, color)
    state[MG_BONUS_W + color] = np.uint64(np.int64(mg))
    state[EG_BONUS_W + color] = np.uint64(np.int64(eg))


@njit(cache=False)
def gives_check(state: np.ndarray, move: int) -> bool:
    from_sq = move & 0x3F
    to_sq = (move >> 6) & 0x3F
    flags = (move >> 12) & 0xF

    us = int(state[TURN])
    them = 1 - us
    opp_king_bb = state[P_KING] & state[C_WHITE + them]
    if not opp_king_bb:
        return False
    opp_king_sq = lsb_sq(opp_king_bb)

    piece = piece_type_at(state, from_sq)
    moved_piece = piece
    opp_king_mask = np.uint64(1) << np.uint64(opp_king_sq)
    occupied = state[C_WHITE] | state[C_BLACK]
    occ = (occupied ^ (np.uint64(1) << np.uint64(from_sq))) | (np.uint64(1) << np.uint64(to_sq))

    if flags == EN_PASSANT:
        ep_cap_sq = to_sq - 8 if us == WHITE else to_sq + 8
        occ ^= np.uint64(1) << np.uint64(ep_cap_sq)
    elif flags >= KNIGHT_PROMO:
        moved_piece = flags - 7 if flags < KNIGHT_PROMO_CAPTURE else flags - 11
    elif flags == KING_CASTLE:
        rook_from = 7 if us == WHITE else 63
        rook_to = 5 if us == WHITE else 61
        occ ^= (np.uint64(1) << np.uint64(rook_from)) | (np.uint64(1) << np.uint64(rook_to))
    elif flags == QUEEN_CASTLE:
        rook_from = 0 if us == WHITE else 56
        rook_to = 3 if us == WHITE else 59
        occ ^= (np.uint64(1) << np.uint64(rook_from)) | (np.uint64(1) << np.uint64(rook_to))

    if moved_piece == KNIGHT:
        if NB_KNIGHT_ATTACKS[to_sq] & opp_king_mask:
            return True
    elif moved_piece == PAWN:
        if NB_PAWN_ATTACKS[us, to_sq] & opp_king_mask:
            return True
    elif moved_piece == BISHOP or moved_piece == QUEEN:
        if bishop_attacks(to_sq, occ) & opp_king_mask:
            return True
    if moved_piece == ROOK or moved_piece == QUEEN:
        if rook_attacks(to_sq, occ) & opp_king_mask:
            return True

    our_bishops_queens = (state[P_BISHOP] | state[P_QUEEN]) & state[C_WHITE + us]
    our_bishops_queens &= ~(np.uint64(1) << np.uint64(from_sq))
    if moved_piece == BISHOP or moved_piece == QUEEN:
        our_bishops_queens |= np.uint64(1) << np.uint64(to_sq)

    if bishop_attacks(opp_king_sq, occ) & our_bishops_queens:
        return True

    our_rooks_queens = (state[P_ROOK] | state[P_QUEEN]) & state[C_WHITE + us]
    our_rooks_queens &= ~(np.uint64(1) << np.uint64(from_sq))
    if moved_piece == ROOK or moved_piece == QUEEN:
        our_rooks_queens |= np.uint64(1) << np.uint64(to_sq)
    if flags == KING_CASTLE:
        our_rooks_queens ^= (np.uint64(1) << np.uint64(7 if us == WHITE else 63)) | (
            np.uint64(1) << np.uint64(5 if us == WHITE else 61)
        )
    elif flags == QUEEN_CASTLE:
        our_rooks_queens ^= (np.uint64(1) << np.uint64(0 if us == WHITE else 56)) | (
            np.uint64(1) << np.uint64(3 if us == WHITE else 59)
        )

    if rook_attacks(opp_king_sq, occ) & our_rooks_queens:
        return True

    return False


@njit(cache=False)
def _aligned(a: int, b: int) -> bool:
    """Same rank, file, or diagonal (necessary condition for slider contact)."""
    dr = (a >> 3) - (b >> 3)
    df = (a & 7) - (b & 7)
    return dr == 0 or df == 0 or dr == df or dr == -df


@njit(cache=False)
def may_give_check(
    state: np.ndarray,
    from_sq: int,
    to_sq: int,
    flags: int,
    opp_king_sq: int,
    us: int,
) -> bool:
    """Cheap necessary condition for check (S2 gate before raycasts).

    Returns False only when no direct or discovered check is geometrically
    possible, letting futility/LMR skip the up-to-4-raycast gives_check().
    Conservative: castling and promotions always return True.
    """
    if flags == KING_CASTLE or flags == QUEEN_CASTLE:
        return True
    if flags >= KNIGHT_PROMO:
        return True

    king_mask = np.uint64(1) << np.uint64(opp_king_sq)
    pt = piece_type_at(state, from_sq)
    if pt == PAWN:
        if NB_PAWN_ATTACKS[us, to_sq] & king_mask:
            return True
    elif pt == KNIGHT:
        if NB_KNIGHT_ATTACKS[to_sq] & king_mask:
            return True
    elif pt == KING:
        pass
    elif _aligned(to_sq, opp_king_sq):
        return True

    if _aligned(from_sq, opp_king_sq):
        return True
    if flags == EN_PASSANT:
        ep_cap_sq = to_sq - 8 if us == WHITE else to_sq + 8
        if _aligned(ep_cap_sq, opp_king_sq):
            return True
    return False


@njit(cache=False)
def _push_move(moves: np.ndarray, count: int, m: int) -> int:
    moves[count] = m
    return count + 1


@njit(cache=False)
def _lookup_pin_mask(
    from_sq: int,
    s0: int,
    m0: np.uint64,
    s1: int,
    m1: np.uint64,
    s2: int,
    m2: np.uint64,
    s3: int,
    m3: np.uint64,
    s4: int,
    m4: np.uint64,
    s5: int,
    m5: np.uint64,
    s6: int,
    m6: np.uint64,
    s7: int,
    m7: np.uint64,
) -> np.uint64:
    """Pin-ray mask for a pinned square (S1: stack scalars, no heap alloc).

    At most 8 pins can exist (one per king ray), so 8 slots are exhaustive.
    Callers only invoke this when the pinned bit is set.
    """
    if s0 == from_sq:
        return m0
    if s1 == from_sq:
        return m1
    if s2 == from_sq:
        return m2
    if s3 == from_sq:
        return m3
    if s4 == from_sq:
        return m4
    if s5 == from_sq:
        return m5
    if s6 == from_sq:
        return m6
    if s7 == from_sq:
        return m7
    return np.uint64(0xFFFFFFFFFFFFFFFF)


@njit(cache=False)
def generate_moves(state: np.ndarray, moves: np.ndarray, captures_only: bool = False) -> int:
    count = 0
    us = int(state[TURN])
    them = 1 - us
    own_pieces = state[C_WHITE + us]
    other_pieces = state[C_WHITE + them]
    occupied = own_pieces | other_pieces

    king_bb = state[P_KING] & own_pieces
    if not king_bb:
        return 0
    king_sq = lsb_sq(king_bb)

    occupied_no_king = occupied ^ king_bb
    opp_diag = (state[P_BISHOP] | state[P_QUEEN]) & other_pieces
    opp_orth = (state[P_ROOK] | state[P_QUEEN]) & other_pieces
    opp_pawns = state[P_PAWN] & other_pieces
    opp_knights = state[P_KNIGHT] & other_pieces
    opp_king_sq = lsb_sq(state[P_KING] & other_pieces)

    king_targets = NB_KING_ATTACKS[king_sq] & (other_pieces if captures_only else ~own_pieces)
    while king_targets:
        lsb = king_targets & (~king_targets + np.uint64(1))
        to_sq = lsb_sq(lsb)
        king_targets &= king_targets - np.uint64(1)

        # Check if to_sq is attacked
        attacked = False
        if (
            NB_PAWN_ATTACKS[us, to_sq] & opp_pawns
            or NB_KNIGHT_ATTACKS[to_sq] & opp_knights
            or NB_KING_ATTACKS[opp_king_sq] & (np.uint64(1) << np.uint64(to_sq))
            or (opp_diag and (bishop_attacks(to_sq, occupied_no_king) & opp_diag))
            or (opp_orth and (rook_attacks(to_sq, occupied_no_king) & opp_orth))
        ):
            attacked = True

        if not attacked:
            flag = CAPTURE if (other_pieces & (np.uint64(1) << np.uint64(to_sq))) else QUIET
            count = _push_move(moves, count, king_sq | (to_sq << 6) | (flag << 12))

    checkers = np.uint64(0)
    checkers |= NB_PAWN_ATTACKS[us, king_sq] & opp_pawns
    checkers |= NB_KNIGHT_ATTACKS[king_sq] & opp_knights
    checkers |= bishop_attacks(king_sq, occupied) & opp_diag
    checkers |= rook_attacks(king_sq, occupied) & opp_orth

    num_checkers = 0
    temp = checkers
    while temp:
        num_checkers += 1
        temp &= temp - np.uint64(1)

    if num_checkers >= 2:
        return count

    if num_checkers == 1:
        checker_sq = lsb_sq(checkers & (~checkers + np.uint64(1)))
        checker_type = piece_type_at(state, checker_sq)
        if checker_type == BISHOP or checker_type == ROOK or checker_type == QUEEN:
            check_mask = NB_RAY_BETWEEN[king_sq, checker_sq] | (
                np.uint64(1) << np.uint64(checker_sq)
            )
        else:
            check_mask = np.uint64(1) << np.uint64(checker_sq)
    else:
        check_mask = np.uint64(0xFFFFFFFFFFFFFFFF)

    # S1: pin storage as stack scalars (8 king rays = provable max pins).
    # Replaces pin_mask = np.empty(64) heap alloc per node (NRT pressure at
    # ~2M nodes/s). Lookup via _lookup_pin_mask only on pinned pieces (rare).
    pin_s0 = -1
    pin_s1 = -1
    pin_s2 = -1
    pin_s3 = -1
    pin_s4 = -1
    pin_s5 = -1
    pin_s6 = -1
    pin_s7 = -1
    pin_m0 = np.uint64(0)
    pin_m1 = np.uint64(0)
    pin_m2 = np.uint64(0)
    pin_m3 = np.uint64(0)
    pin_m4 = np.uint64(0)
    pin_m5 = np.uint64(0)
    pin_m6 = np.uint64(0)
    pin_m7 = np.uint64(0)
    num_pins = 0
    pinned_pieces = np.uint64(0)

    pinners = (bishop_attacks(king_sq, other_pieces) & opp_diag) | (
        rook_attacks(king_sq, other_pieces) & opp_orth
    )
    while pinners:
        lsb = pinners & (~pinners + np.uint64(1))
        pinner_sq = lsb_sq(lsb)
        pinners &= pinners - np.uint64(1)
        between = NB_RAY_BETWEEN[king_sq, pinner_sq]
        pieces_between = between & occupied
        if pieces_between != 0 and (pieces_between & (pieces_between - 1)) == 0:
            own_pinned = pieces_between & own_pieces
            if own_pinned:
                pinned_sq = lsb_sq(own_pinned)
                pin_ray = between | (np.uint64(1) << np.uint64(pinner_sq))
                if num_pins == 0:
                    pin_s0 = pinned_sq
                    pin_m0 = pin_ray
                elif num_pins == 1:
                    pin_s1 = pinned_sq
                    pin_m1 = pin_ray
                elif num_pins == 2:
                    pin_s2 = pinned_sq
                    pin_m2 = pin_ray
                elif num_pins == 3:
                    pin_s3 = pinned_sq
                    pin_m3 = pin_ray
                elif num_pins == 4:
                    pin_s4 = pinned_sq
                    pin_m4 = pin_ray
                elif num_pins == 5:
                    pin_s5 = pinned_sq
                    pin_m5 = pin_ray
                elif num_pins == 6:
                    pin_s6 = pinned_sq
                    pin_m6 = pin_ray
                elif num_pins == 7:
                    pin_s7 = pinned_sq
                    pin_m7 = pin_ray
                num_pins += 1
                pinned_pieces |= own_pinned

    active_own = own_pieces
    if num_checkers > 0:
        active_own &= ~pinned_pieces

    movable_mask = check_mask

    # Pawns
    pawns = state[P_PAWN] & active_own
    while pawns:
        lsb = pawns & (~pawns + np.uint64(1))
        from_sq = lsb_sq(lsb)
        pawns &= pawns - np.uint64(1)

        target_mask = movable_mask
        if pinned_pieces & lsb:
            target_mask &= _lookup_pin_mask(from_sq, pin_s0, pin_m0, pin_s1, pin_m1, pin_s2, pin_m2, pin_s3, pin_m3, pin_s4, pin_m4, pin_s5, pin_m5, pin_s6, pin_m6, pin_s7, pin_m7)
        file = from_sq & 7
        rank = from_sq >> 3

        if us == WHITE:
            push1_to = from_sq + 8
            if not (occupied & (np.uint64(1) << np.uint64(push1_to))) and (
                target_mask & (np.uint64(1) << np.uint64(push1_to))
            ):
                if rank == 6:
                    for promo in PROMOTION_PIECES:
                        count = _push_move(moves, count, from_sq | (push1_to << 6) | (promo << 12))
                elif not captures_only:
                    count = _push_move(moves, count, from_sq | (push1_to << 6) | (QUIET << 12))
            if (
                rank == 1
                and not captures_only
                and not (occupied & (np.uint64(1) << np.uint64(push1_to)))
            ):
                push2_to = from_sq + 16
                if not (occupied & (np.uint64(1) << np.uint64(push2_to))) and (
                    target_mask & (np.uint64(1) << np.uint64(push2_to))
                ):
                    count = _push_move(
                        moves, count, from_sq | (push2_to << 6) | (DOUBLE_PAWN_PUSH << 12)
                    )

            if file > 0:
                cap_to = from_sq + 7
                if (other_pieces & (np.uint64(1) << np.uint64(cap_to))) and (
                    target_mask & (np.uint64(1) << np.uint64(cap_to))
                ):
                    if rank == 6:
                        for promo in PROMOTION_CAPTURE_PIECES:
                            count = _push_move(
                                moves, count, from_sq | (cap_to << 6) | (promo << 12)
                            )
                    else:
                        count = _push_move(moves, count, from_sq | (cap_to << 6) | (CAPTURE << 12))
            if file < 7:
                cap_to = from_sq + 9
                if (other_pieces & (np.uint64(1) << np.uint64(cap_to))) and (
                    target_mask & (np.uint64(1) << np.uint64(cap_to))
                ):
                    if rank == 6:
                        for promo in PROMOTION_CAPTURE_PIECES:
                            count = _push_move(
                                moves, count, from_sq | (cap_to << 6) | (promo << 12)
                            )
                    else:
                        count = _push_move(moves, count, from_sq | (cap_to << 6) | (CAPTURE << 12))
        else:  # BLACK
            push1_to = from_sq - 8
            if not (occupied & (np.uint64(1) << np.uint64(push1_to))) and (
                target_mask & (np.uint64(1) << np.uint64(push1_to))
            ):
                if rank == 1:
                    for promo in PROMOTION_PIECES:
                        count = _push_move(moves, count, from_sq | (push1_to << 6) | (promo << 12))
                elif not captures_only:
                    count = _push_move(moves, count, from_sq | (push1_to << 6) | (QUIET << 12))
            if (
                rank == 6
                and not captures_only
                and not (occupied & (np.uint64(1) << np.uint64(push1_to)))
            ):
                push2_to = from_sq - 16
                if not (occupied & (np.uint64(1) << np.uint64(push2_to))) and (
                    target_mask & (np.uint64(1) << np.uint64(push2_to))
                ):
                    count = _push_move(
                        moves, count, from_sq | (push2_to << 6) | (DOUBLE_PAWN_PUSH << 12)
                    )
            if file > 0:
                cap_to = from_sq - 9
                if (other_pieces & (np.uint64(1) << np.uint64(cap_to))) and (
                    target_mask & (np.uint64(1) << np.uint64(cap_to))
                ):
                    if rank == 1:
                        for promo in PROMOTION_CAPTURE_PIECES:
                            count = _push_move(
                                moves, count, from_sq | (cap_to << 6) | (promo << 12)
                            )
                    else:
                        count = _push_move(moves, count, from_sq | (cap_to << 6) | (CAPTURE << 12))
            if file < 7:
                cap_to = from_sq - 7
                if (other_pieces & (np.uint64(1) << np.uint64(cap_to))) and (
                    target_mask & (np.uint64(1) << np.uint64(cap_to))
                ):
                    if rank == 1:
                        for promo in PROMOTION_CAPTURE_PIECES:
                            count = _push_move(
                                moves, count, from_sq | (cap_to << 6) | (promo << 12)
                            )
                    else:
                        count = _push_move(moves, count, from_sq | (cap_to << 6) | (CAPTURE << 12))

        # En Passant
        ep_square = state[EP_SQUARE]
        if ep_square != 64:
            if us == WHITE and rank == 4 and abs(file - (ep_square & 7)) == 1:
                ep_cap_sq = ep_square - 8
                occ_ep = (
                    occupied
                    ^ (np.uint64(1) << np.uint64(from_sq))
                    ^ (np.uint64(1) << np.uint64(ep_cap_sq))
                ) | (np.uint64(1) << np.uint64(ep_square))
                if not (rook_attacks(king_sq, occ_ep) & opp_orth) and not (
                    bishop_attacks(king_sq, occ_ep) & opp_diag
                ):
                    if (target_mask & (np.uint64(1) << np.uint64(ep_cap_sq))) or (
                        target_mask & (np.uint64(1) << np.uint64(ep_square))
                    ):
                        count = _push_move(
                            moves, count, from_sq | (ep_square << 6) | (EN_PASSANT << 12)
                        )
            elif us == BLACK and rank == 3 and abs(file - (ep_square & 7)) == 1:
                ep_cap_sq = ep_square + 8
                occ_ep = (
                    occupied
                    ^ (np.uint64(1) << np.uint64(from_sq))
                    ^ (np.uint64(1) << np.uint64(ep_cap_sq))
                ) | (np.uint64(1) << np.uint64(ep_square))
                if not (rook_attacks(king_sq, occ_ep) & opp_orth) and not (
                    bishop_attacks(king_sq, occ_ep) & opp_diag
                ):
                    if (target_mask & (np.uint64(1) << np.uint64(ep_cap_sq))) or (
                        target_mask & (np.uint64(1) << np.uint64(ep_square))
                    ):
                        count = _push_move(
                            moves, count, from_sq | (ep_square << 6) | (EN_PASSANT << 12)
                        )

    # Knights
    knights = state[P_KNIGHT] & active_own
    while knights:
        lsb = knights & (~knights + np.uint64(1))
        from_sq = lsb_sq(lsb)
        knights &= knights - np.uint64(1)
        target_mask = movable_mask
        if pinned_pieces & lsb:
            target_mask &= _lookup_pin_mask(from_sq, pin_s0, pin_m0, pin_s1, pin_m1, pin_s2, pin_m2, pin_s3, pin_m3, pin_s4, pin_m4, pin_s5, pin_m5, pin_s6, pin_m6, pin_s7, pin_m7)
        targets = NB_KNIGHT_ATTACKS[from_sq] & target_mask & (
            other_pieces if captures_only else ~own_pieces
        )
        while targets:
            tlsb = targets & (~targets + np.uint64(1))
            to_sq = lsb_sq(tlsb)
            targets &= targets - np.uint64(1)
            flag = CAPTURE if (other_pieces & (np.uint64(1) << np.uint64(to_sq))) else QUIET
            count = _push_move(moves, count, from_sq | (to_sq << 6) | (flag << 12))

    # Bishops
    bishops = state[P_BISHOP] & active_own
    while bishops:
        lsb = bishops & (~bishops + np.uint64(1))
        from_sq = lsb_sq(lsb)
        bishops &= bishops - np.uint64(1)
        target_mask = movable_mask
        if pinned_pieces & lsb:
            target_mask &= _lookup_pin_mask(from_sq, pin_s0, pin_m0, pin_s1, pin_m1, pin_s2, pin_m2, pin_s3, pin_m3, pin_s4, pin_m4, pin_s5, pin_m5, pin_s6, pin_m6, pin_s7, pin_m7)
        targets = bishop_attacks(from_sq, occupied) & target_mask & (
            other_pieces if captures_only else ~own_pieces
        )
        while targets:
            tlsb = targets & (~targets + np.uint64(1))
            to_sq = lsb_sq(tlsb)
            targets &= targets - np.uint64(1)
            flag = CAPTURE if (other_pieces & (np.uint64(1) << np.uint64(to_sq))) else QUIET
            count = _push_move(moves, count, from_sq | (to_sq << 6) | (flag << 12))

    # Rooks
    rooks = state[P_ROOK] & active_own
    while rooks:
        lsb = rooks & (~rooks + np.uint64(1))
        from_sq = lsb_sq(lsb)
        rooks &= rooks - np.uint64(1)
        target_mask = movable_mask
        if pinned_pieces & lsb:
            target_mask &= _lookup_pin_mask(from_sq, pin_s0, pin_m0, pin_s1, pin_m1, pin_s2, pin_m2, pin_s3, pin_m3, pin_s4, pin_m4, pin_s5, pin_m5, pin_s6, pin_m6, pin_s7, pin_m7)
        targets = rook_attacks(from_sq, occupied) & target_mask & (
            other_pieces if captures_only else ~own_pieces
        )
        while targets:
            tlsb = targets & (~targets + np.uint64(1))
            to_sq = lsb_sq(tlsb)
            targets &= targets - np.uint64(1)
            flag = CAPTURE if (other_pieces & (np.uint64(1) << np.uint64(to_sq))) else QUIET
            count = _push_move(moves, count, from_sq | (to_sq << 6) | (flag << 12))

    # Queens
    queens = state[P_QUEEN] & active_own
    while queens:
        lsb = queens & (~queens + np.uint64(1))
        from_sq = lsb_sq(lsb)
        queens &= queens - np.uint64(1)
        target_mask = movable_mask
        if pinned_pieces & lsb:
            target_mask &= _lookup_pin_mask(from_sq, pin_s0, pin_m0, pin_s1, pin_m1, pin_s2, pin_m2, pin_s3, pin_m3, pin_s4, pin_m4, pin_s5, pin_m5, pin_s6, pin_m6, pin_s7, pin_m7)
        targets = queen_attacks(from_sq, occupied) & target_mask & (
            other_pieces if captures_only else ~own_pieces
        )
        while targets:
            tlsb = targets & (~targets + np.uint64(1))
            to_sq = lsb_sq(tlsb)
            targets &= targets - np.uint64(1)
            flag = CAPTURE if (other_pieces & (np.uint64(1) << np.uint64(to_sq))) else QUIET
            count = _push_move(moves, count, from_sq | (to_sq << 6) | (flag << 12))

    # Castling
    if not captures_only and num_checkers == 0:
        c = state[CASTLING]
        if us == WHITE:
            if c & CASTLE_WK:
                if (
                    king_sq == 4
                    and (state[P_ROOK] & own_pieces & (np.uint64(1) << np.uint64(7)))
                    and not (occupied & np.uint64(0x60))\
                    and not is_sq_attacked(state, 5, BLACK)
                    and not is_sq_attacked(state, 6, BLACK)
                ):
                    count = _push_move(moves, count, 4 | (6 << 6) | (KING_CASTLE << 12))
            if c & CASTLE_WQ:
                if (
                    king_sq == 4
                    and (state[P_ROOK] & own_pieces & (np.uint64(1) << np.uint64(0)))
                    and not (occupied & np.uint64(0xE))
                    and not is_sq_attacked(state, 3, BLACK)
                    and not is_sq_attacked(state, 2, BLACK)
                ):
                    count = _push_move(moves, count, 4 | (2 << 6) | (QUEEN_CASTLE << 12))
        else:
            if c & CASTLE_BK:
                if (
                    king_sq == 60
                    and (state[P_ROOK] & own_pieces & (np.uint64(1) << np.uint64(63)))
                    and not (occupied & np.uint64(0x6000000000000000))
                    and not is_sq_attacked(state, 61, WHITE)
                    and not is_sq_attacked(state, 62, WHITE)
                ):
                    count = _push_move(moves, count, 60 | (62 << 6) | (KING_CASTLE << 12))
            if c & CASTLE_BQ:
                if (
                    king_sq == 60
                    and (state[P_ROOK] & own_pieces & (np.uint64(1) << np.uint64(56)))
                    and not (occupied & np.uint64(0x0E00000000000000))
                    and not is_sq_attacked(state, 59, WHITE)
                    and not is_sq_attacked(state, 58, WHITE)
                ):
                    count = _push_move(moves, count, 60 | (58 << 6) | (QUEEN_CASTLE << 12))

    return count
