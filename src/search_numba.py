# ruff: noqa
"""Numba-jitted search tree."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.board import Board

import numpy as np
from numba import njit

from src.board_primitives import (
    _bit_length,
    evaluate,
    generate_moves,
    gives_check,
    is_in_check,
    is_sq_attacked,
    make_move,
    make_null_move,
    piece_type_at,
    unmake_move,
    unmake_null_move,
)
from src.constants import (
    BISHOP,
    BISHOP_COUNT_B,
    BISHOP_COUNT_W,
    BLACK,
    C_BLACK,
    C_WHITE,
    CASTLING,
    EG_BONUS_B,
    EG_BONUS_W,
    EG_SCORE_B,
    EG_SCORE_W,
    EN_PASSANT,
    EP_SQUARE,
    FULLMOVE,
    GAME_PHASE,
    HALFMOVE,
    HASH,
    INF,
    KING,
    KNIGHT,
    KNIGHT_PROMO,
    MATE_SCORE,
    MATE_THRESHOLD,
    MAX_PLY,
    MG_BONUS_B,
    MG_BONUS_W,
    MG_SCORE_B,
    MG_SCORE_W,
    NO_MOVE,
    P_BISHOP,
    P_KING,
    P_KNIGHT,
    P_PAWN,
    P_QUEEN,
    P_ROOK,
    PAWN,
    QUEEN,
    ROOK,
    STATE_SIZE,
    TURN,
    WHITE,
)
from src.tt import (
    BOUND_EXACT,
    BOUND_LOWER,
    BOUND_NONE,
    BOUND_UPPER,
    probe_tt,
    score_from_tt,
    score_to_tt,
    store_tt,
)

# libc clock for microsecond timing inside Numba without Python GIL
libc = ctypes.CDLL(None)
clock = libc.clock
clock.restype = ctypes.c_long
clock.argtypes = []

# Precomputed Piece values and MVV-LVA scoring matrix
_PIECE_VALUES = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int32)
_MVV_LVA = np.zeros((6, 6), dtype=np.int32)
for _v in range(6):
    for _a in range(6):
        _MVV_LVA[_v, _a] = _PIECE_VALUES[_v] * 10 - _PIECE_VALUES[_a]


@njit(cache=False)
def is_draw(
    state: np.ndarray,
    undo_stack: np.ndarray,
    ply: int,
    hash_history: np.ndarray,
    hist_len: int,
) -> bool:
    if state[HALFMOVE] >= 100:
        return True

    curr_hash = state[HASH]
    halfmoves = int(state[HALFMOVE])

    matches = 1

    # Check prior positions in the search path.
    lookback = min(halfmoves, ply)
    for i in range(2, lookback + 1, 2):
        if undo_stack[ply - i, HASH] == curr_hash:
            matches += 1
            if matches == 3:
                return True

    # hash_history includes the root at its end. Continue the same-ply scan
    # through positions before the root once the search path is exhausted.
    for distance in range(ply + 2 if ply % 2 == 0 else ply + 1, halfmoves + 1, 2):
        history_index = hist_len - 1 - (distance - ply)
        if history_index < 0:
            break
        if hash_history[history_index] == curr_hash:
            matches += 1
            if matches == 3:
                return True

    return False


@njit(cache=False)
def score_move(
    state: np.ndarray,
    move: int,
    tt_move: int,
    ply: int,
    killers: np.ndarray,
    history: np.ndarray,
) -> int:
    if move == tt_move:
        return 2000000000

    from_sq = move & 0x3F
    to_sq = (move >> 6) & 0x3F
    flags = (move >> 12) & 0xF

    moving_piece = piece_type_at(state, from_sq)
    captured_piece = piece_type_at(state, to_sq)

    if captured_piece != -1 or flags == EN_PASSANT:
        if flags == EN_PASSANT:
            captured_piece = PAWN
        return 1000000000 + int(_MVV_LVA[captured_piece, moving_piece])

    if flags >= KNIGHT_PROMO:
        return 900000000 + flags

    if move == killers[ply, 0]:
        return 800000000
    if move == killers[ply, 1]:
        return 700000000

    us = int(state[TURN])
    return int(history[us, from_sq, to_sq])


@njit(cache=False)
def sort_moves(
    state: np.ndarray,
    moves: np.ndarray,
    num_moves: int,
    tt_move: int,
    ply: int,
    killers: np.ndarray,
    history: np.ndarray,
    scores: np.ndarray,
) -> None:
    for i in range(num_moves):
        scores[i] = score_move(state, moves[i], tt_move, ply, killers, history)

    for i in range(num_moves - 1):
        max_idx = i
        max_score = scores[i]
        for j in range(i + 1, num_moves):
            if scores[j] > max_score:
                max_score = scores[j]
                max_idx = j
        if max_idx != i:
            moves[i], moves[max_idx] = moves[max_idx], moves[i]
            scores[i], scores[max_idx] = scores[max_idx], scores[i]


@njit(cache=False)
def sort_captures(
    state: np.ndarray,
    moves: np.ndarray,
    num_moves: int,
    scores: np.ndarray,
) -> None:
    for i in range(num_moves):
        move = moves[i]
        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        flags = (move >> 12) & 0xF

        moving_piece = piece_type_at(state, from_sq)
        captured_piece = piece_type_at(state, to_sq)

        if captured_piece != -1 or flags == EN_PASSANT:
            if flags == EN_PASSANT:
                captured_piece = PAWN
            scores[i] = 1000000000 + _MVV_LVA[captured_piece, moving_piece]
        elif flags >= KNIGHT_PROMO:
            scores[i] = 900000000 + flags
        else:
            scores[i] = 0

    for i in range(num_moves - 1):
        max_idx = i
        max_score = scores[i]
        for j in range(i + 1, num_moves):
            if scores[j] > max_score:
                max_score = scores[j]
                max_idx = j
        if max_idx != i:
            moves[i], moves[max_idx] = moves[max_idx], moves[i]
            scores[i], scores[max_idx] = scores[max_idx], scores[i]


@njit(cache=False)
def quiescence(
    state: np.ndarray,
    undo_stack: np.ndarray,
    moves_stack: np.ndarray,
    scores_stack: np.ndarray,
    alpha: int,
    beta: int,
    ply: int,
    stats: np.ndarray,
) -> int:
    stats[0] += 1
    if (stats[0] & 1023) == 0:
        if stats[3] > 0 and clock() >= stats[3]:
            stats[1] = 1
            return 0

    if stats[1] == 1:
        return 0

    if ply >= MAX_PLY - 1:
        return evaluate(state)

    in_check = is_in_check(state)
    stand_pat = -INF

    if not in_check:
        stand_pat = evaluate(state)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

    moves = moves_stack[ply]
    num_moves = generate_moves(state, moves, captures_only=not in_check)
    sort_captures(state, moves, num_moves, scores_stack[ply])

    best_score = -INF if in_check else stand_pat
    legal_moves = 0

    for i in range(num_moves):
        move = moves[i]
        flags = (move >> 12) & 0xF
        to_sq = (move >> 6) & 0x3F

        # Delta pruning
        if not in_check and flags < KNIGHT_PROMO:
            cap_pt = piece_type_at(state, to_sq) if flags != EN_PASSANT else PAWN
            cap_val = _PIECE_VALUES[cap_pt] if cap_pt != -1 else 0
            if stand_pat + cap_val + 200 < alpha:
                continue

        make_move(state, undo_stack, ply, move)
        us = 1 - int(state[TURN])
        king_bb = state[P_KING] & state[C_WHITE + us]
        king_sq = _bit_length(king_bb) - 1
        if is_sq_attacked(state, king_sq, 1 - us):
            unmake_move(state, undo_stack, ply)
            continue

        legal_moves += 1
        score = -quiescence(
            state, undo_stack, moves_stack, scores_stack, -beta, -alpha, ply + 1, stats
        )
        unmake_move(state, undo_stack, ply)

        if stats[1] == 1:
            return 0

        if score > best_score:
            best_score = score
            if score > alpha:
                alpha = score
                if score >= beta:
                    return beta

    if in_check and legal_moves == 0:
        return -MATE_SCORE + ply

    return best_score


@njit(cache=False)
def alpha_beta(
    state: np.ndarray,
    undo_stack: np.ndarray,
    moves_stack: np.ndarray,
    scores_stack: np.ndarray,
    alpha: int,
    beta: int,
    depth: int,
    ply: int,
    null_move_allowed: bool,
    killers: np.ndarray,
    history: np.ndarray,
    stats: np.ndarray,
    hash_history: np.ndarray,
    hist_len: int,
    age: int,
    tt_hash: np.ndarray,
    tt_score: np.ndarray,
    tt_move: np.ndarray,
    tt_depth: np.ndarray,
    tt_bound: np.ndarray,
    tt_age: np.ndarray,
    tt_static_eval: np.ndarray,
) -> int:
    stats[0] += 1
    if (stats[0] & 1023) == 0:
        if stats[3] > 0 and clock() >= stats[3]:
            stats[1] = 1
            return 0

    if stats[1] == 1:
        return 0

    if ply > 0 and is_draw(state, undo_stack, ply, hash_history, hist_len):
        return 0

    if ply >= MAX_PLY - 1:
        return evaluate(state)

    # Mate distance pruning
    alpha = max(alpha, -MATE_SCORE + ply)
    beta = min(beta, MATE_SCORE - ply - 1)
    if alpha >= beta:
        return alpha

    is_pv = beta - alpha > 1
    hash_val = state[HASH]
    found, tt_move_val, tt_score_val, tt_depth_val, tt_bound_val, tt_static_val = probe_tt(
        hash_val, tt_hash, tt_score, tt_move, tt_depth, tt_bound, tt_age, tt_static_eval
    )

    if found and tt_depth_val >= depth and ply > 0 and not is_pv:
        adjusted_score = score_from_tt(tt_score_val, ply)
        if tt_bound_val == BOUND_EXACT:
            return adjusted_score
        if tt_bound_val == BOUND_LOWER and adjusted_score >= beta:
            return beta
        if tt_bound_val == BOUND_UPPER and adjusted_score <= alpha:
            return alpha

    in_check = is_in_check(state)
    if in_check and ply < MAX_PLY - 2:
        depth += 1

    if depth <= 0:
        return int(
            quiescence(state, undo_stack, moves_stack, scores_stack, alpha, beta, ply, stats)
        )

    # Static eval
    if in_check:
        static_eval = -INF
    elif found:
        static_eval = tt_static_val
    else:
        static_eval = evaluate(state)

    # Reverse Futility Pruning (RFP)
    if not is_pv and not in_check and depth <= 6 and abs(beta) < MATE_THRESHOLD:
        rfp_margin = 80 * depth
        if static_eval - rfp_margin >= beta:
            return static_eval

    # Null Move Pruning (NMP)
    us = int(state[TURN])
    has_pieces = bool(
        (state[P_KNIGHT] | state[P_BISHOP] | state[P_ROOK] | state[P_QUEEN]) & state[C_WHITE + us]
    )
    if (
        null_move_allowed
        and not is_pv
        and not in_check
        and depth >= 3
        and static_eval >= beta
        and has_pieces
    ):
        r = 3 + depth // 6
        make_null_move(state, undo_stack, ply)
        null_score = -alpha_beta(
            state,
            undo_stack,
            moves_stack,
            scores_stack,
            -beta,
            -beta + 1,
            depth - 1 - r,
            ply + 1,
            False,
            killers,
            history,
            stats,
            hash_history,
            hist_len,
            age,
            tt_hash,
            tt_score,
            tt_move,
            tt_depth,
            tt_bound,
            tt_age,
            tt_static_eval,
        )
        unmake_null_move(state, undo_stack, ply)

        if stats[1] == 1:
            return 0

        if null_score >= beta:
            return beta if null_score < MATE_THRESHOLD else null_score

    moves = moves_stack[ply]
    num_moves = generate_moves(state, moves, captures_only=False)
    sort_moves(state, moves, num_moves, tt_move_val, ply, killers, history, scores_stack[ply])

    best_move = NO_MOVE
    best_score = -INF
    alpha_orig = alpha
    legal_moves = 0

    for i in range(num_moves):
        move = moves[i]
        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        flags = (move >> 12) & 0xF

        is_tactical = (
            piece_type_at(state, to_sq) != -1 or flags == EN_PASSANT or flags >= KNIGHT_PROMO
        )

        # Futility Pruning
        if (
            not is_pv
            and not in_check
            and depth <= 3
            and legal_moves > 0
            and not is_tactical
            and not gives_check(state, move)
        ):
            if static_eval + 120 * depth <= alpha:
                continue

        make_move(state, undo_stack, ply, move)
        moved_side = 1 - int(state[TURN])
        king_bb = state[P_KING] & state[C_WHITE + moved_side]
        king_sq = _bit_length(king_bb) - 1

        if is_sq_attacked(state, king_sq, 1 - moved_side):
            unmake_move(state, undo_stack, ply)
            continue

        legal_moves += 1

        if legal_moves == 1:
            score = -alpha_beta(
                state,
                undo_stack,
                moves_stack,
                scores_stack,
                -beta,
                -alpha,
                depth - 1,
                ply + 1,
                True,
                killers,
                history,
                stats,
                hash_history,
                hist_len,
                age,
                tt_hash,
                tt_score,
                tt_move,
                tt_depth,
                tt_bound,
                tt_age,
                tt_static_eval,
            )
        else:
            # Late Move Reductions (LMR)
            reduction = 0
            if depth >= 3 and legal_moves >= 4 and not is_tactical and not in_check:
                reduction = 1
                if not is_pv and legal_moves >= 6:
                    reduction += 1
                reduction = min(reduction, depth - 2)

            score = -alpha_beta(
                state,
                undo_stack,
                moves_stack,
                scores_stack,
                -(alpha + 1),
                -alpha,
                depth - 1 - reduction,
                ply + 1,
                True,
                killers,
                history,
                stats,
                hash_history,
                hist_len,
                age,
                tt_hash,
                tt_score,
                tt_move,
                tt_depth,
                tt_bound,
                tt_age,
                tt_static_eval,
            )

            if score > alpha and reduction > 0:
                score = -alpha_beta(
                    state,
                    undo_stack,
                    moves_stack,
                    scores_stack,
                    -(alpha + 1),
                    -alpha,
                    depth - 1,
                    ply + 1,
                    True,
                    killers,
                    history,
                    stats,
                    hash_history,
                    hist_len,
                    age,
                    tt_hash,
                    tt_score,
                    tt_move,
                    tt_depth,
                    tt_bound,
                    tt_age,
                    tt_static_eval,
                )

            if is_pv and alpha < score < beta:
                score = -alpha_beta(
                    state,
                    undo_stack,
                    moves_stack,
                    scores_stack,
                    -beta,
                    -alpha,
                    depth - 1,
                    ply + 1,
                    True,
                    killers,
                    history,
                    stats,
                    hash_history,
                    hist_len,
                    age,
                    tt_hash,
                    tt_score,
                    tt_move,
                    tt_depth,
                    tt_bound,
                    tt_age,
                    tt_static_eval,
                )

        unmake_move(state, undo_stack, ply)

        if stats[1] == 1:
            return 0

        if score > best_score:
            best_score = score
            best_move = move

            if score > alpha:
                alpha = score

                if score >= beta:
                    if not is_tactical:
                        if killers[ply, 0] != move:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = move
                        history[us, from_sq, to_sq] += depth * depth

                    store_tt(
                        hash_val,
                        best_move,
                        score_to_tt(score, ply),
                        depth,
                        BOUND_LOWER,
                        static_eval,
                        age,
                        tt_hash,
                        tt_score,
                        tt_move,
                        tt_depth,
                        tt_bound,
                        tt_age,
                        tt_static_eval,
                    )
                    return beta

    if legal_moves == 0:
        if in_check:
            return -MATE_SCORE + ply
        return 0

    bound = BOUND_EXACT if best_score > alpha_orig else BOUND_UPPER
    store_tt(
        hash_val,
        best_move,
        score_to_tt(best_score, ply),
        depth,
        bound,
        static_eval,
        age,
        tt_hash,
        tt_score,
        tt_move,
        tt_depth,
        tt_bound,
        tt_age,
        tt_static_eval,
    )

    return best_score


@njit(cache=False)
def search_root(
    state: np.ndarray,
    undo_stack: np.ndarray,
    moves_stack: np.ndarray,
    scores_stack: np.ndarray,
    alpha: int,
    beta: int,
    depth: int,
    killers: np.ndarray,
    history: np.ndarray,
    stats: np.ndarray,
    hash_history: np.ndarray,
    hist_len: int,
    age: int,
    tt_hash: np.ndarray,
    tt_score: np.ndarray,
    tt_move: np.ndarray,
    tt_depth: np.ndarray,
    tt_bound: np.ndarray,
    tt_age: np.ndarray,
    tt_static_eval: np.ndarray,
) -> tuple[int, int, int, bool]:
    stats[0] += 1
    if (stats[0] & 1023) == 0:
        if stats[3] > 0 and clock() >= stats[3]:
            stats[1] = 1
            return 0, NO_MOVE, -INF, True

    if stats[1] == 1:
        return 0, NO_MOVE, -INF, True

    hash_val = state[HASH]
    found, tt_move_val, _, _, _, tt_static_val = probe_tt(
        hash_val, tt_hash, tt_score, tt_move, tt_depth, tt_bound, tt_age, tt_static_eval
    )

    in_check = is_in_check(state)
    static_eval = evaluate(state) if not in_check else -INF

    moves = moves_stack[0]
    num_moves = generate_moves(state, moves, captures_only=False)
    sort_moves(state, moves, num_moves, tt_move_val, 0, killers, history, scores_stack[0])

    best_move = NO_MOVE
    best_score = -INF
    runner_up_score = -INF
    alpha_orig = alpha
    legal_moves = 0
    us = int(state[TURN])

    for i in range(num_moves):
        move = moves[i]
        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        flags = (move >> 12) & 0xF

        is_tactical = (
            piece_type_at(state, to_sq) != -1 or flags == EN_PASSANT or flags >= KNIGHT_PROMO
        )

        make_move(state, undo_stack, 0, move)
        moved_side = 1 - int(state[TURN])
        king_bb = state[P_KING] & state[C_WHITE + moved_side]
        king_sq = _bit_length(king_bb) - 1

        if is_sq_attacked(state, king_sq, 1 - moved_side):
            unmake_move(state, undo_stack, 0)
            continue

        legal_moves += 1

        if legal_moves == 1:
            score = -alpha_beta(
                state,
                undo_stack,
                moves_stack,
                scores_stack,
                -beta,
                -alpha,
                depth - 1,
                1,
                True,
                killers,
                history,
                stats,
                hash_history,
                hist_len,
                age,
                tt_hash,
                tt_score,
                tt_move,
                tt_depth,
                tt_bound,
                tt_age,
                tt_static_eval,
            )
        else:
            reduction = 0
            if depth >= 3 and legal_moves >= 4 and not is_tactical and not in_check:
                reduction = 1
                reduction = min(reduction, depth - 2)

            score = -alpha_beta(
                state,
                undo_stack,
                moves_stack,
                scores_stack,
                -(alpha + 1),
                -alpha,
                depth - 1 - reduction,
                1,
                True,
                killers,
                history,
                stats,
                hash_history,
                hist_len,
                age,
                tt_hash,
                tt_score,
                tt_move,
                tt_depth,
                tt_bound,
                tt_age,
                tt_static_eval,
            )

            if score > alpha and reduction > 0:
                score = -alpha_beta(
                    state,
                    undo_stack,
                    moves_stack,
                    scores_stack,
                    -(alpha + 1),
                    -alpha,
                    depth - 1,
                    1,
                    True,
                    killers,
                    history,
                    stats,
                    hash_history,
                    hist_len,
                    age,
                    tt_hash,
                    tt_score,
                    tt_move,
                    tt_depth,
                    tt_bound,
                    tt_age,
                    tt_static_eval,
                )

            if alpha < score < beta:
                score = -alpha_beta(
                    state,
                    undo_stack,
                    moves_stack,
                    scores_stack,
                    -beta,
                    -alpha,
                    depth - 1,
                    1,
                    True,
                    killers,
                    history,
                    stats,
                    hash_history,
                    hist_len,
                    age,
                    tt_hash,
                    tt_score,
                    tt_move,
                    tt_depth,
                    tt_bound,
                    tt_age,
                    tt_static_eval,
                )

        unmake_move(state, undo_stack, 0)

        if stats[1] == 1:
            return best_score if best_score != -INF else 0, best_move, runner_up_score, True

        if score > best_score:
            runner_up_score = best_score
            best_score = score
            best_move = move
        elif score > runner_up_score:
            runner_up_score = score

        if score > alpha:
            alpha = score

            if score >= beta:
                if not is_tactical:
                    if killers[0, 0] != move:
                        killers[0, 1] = killers[0, 0]
                        killers[0, 0] = move
                    history[us, from_sq, to_sq] += depth * depth

                store_tt(
                    hash_val,
                    best_move,
                    score_to_tt(score, 0),
                    depth,
                    BOUND_LOWER,
                    static_eval,
                    age,
                    tt_hash,
                    tt_score,
                    tt_move,
                    tt_depth,
                    tt_bound,
                    tt_age,
                    tt_static_eval,
                )
                return beta, best_move, runner_up_score, False

    if legal_moves == 0:
        mate_val = -MATE_SCORE if in_check else 0
        return mate_val, NO_MOVE, -INF, False

    bound = BOUND_EXACT if best_score > alpha_orig else BOUND_UPPER
    store_tt(
        hash_val,
        best_move,
        score_to_tt(best_score, 0),
        depth,
        bound,
        static_eval,
        age,
        tt_hash,
        tt_score,
        tt_move,
        tt_depth,
        tt_bound,
        tt_age,
        tt_static_eval,
    )

    return best_score, best_move, runner_up_score, False


@njit(cache=False)
def iterative_deepening(
    state: np.ndarray,
    max_depth: int,
    max_time_ms: int,
    age: int,
    hash_history: np.ndarray,
    hist_len: int,
    tt_hash: np.ndarray,
    tt_score: np.ndarray,
    tt_move: np.ndarray,
    tt_depth: np.ndarray,
    tt_bound: np.ndarray,
    tt_age: np.ndarray,
    tt_static_eval: np.ndarray,
) -> tuple[int, int, int]:
    best_move_overall = NO_MOVE
    best_score_overall = 0

    undo_stack = np.zeros((MAX_PLY, STATE_SIZE), dtype=np.uint64)
    moves_stack = np.zeros((MAX_PLY, 256), dtype=np.int32)
    scores_stack = np.zeros((MAX_PLY, 256), dtype=np.int32)
    killers = np.zeros((MAX_PLY, 2), dtype=np.int32)
    history = np.zeros((2, 64, 64), dtype=np.int32)

    stats = np.zeros(4, dtype=np.int64)
    start_ticks = clock()
    stats[0] = 0
    stats[1] = 0
    stats[2] = start_ticks
    if max_time_ms > 0:
        stats[3] = start_ticks + int(max_time_ms * 1000)
    else:
        stats[3] = 0

    for depth in range(1, max_depth + 1):
        score, move, _, aborted = search_root(
            state,
            undo_stack,
            moves_stack,
            scores_stack,
            -INF,
            INF,
            depth,
            killers,
            history,
            stats,
            hash_history,
            hist_len,
            age,
            tt_hash,
            tt_score,
            tt_move,
            tt_depth,
            tt_bound,
            tt_age,
            tt_static_eval,
        )

        if aborted:
            break

        if move != NO_MOVE:
            best_move_overall = move
            best_score_overall = score

        if abs(score) > MATE_THRESHOLD:
            break

    return best_move_overall, best_score_overall, int(stats[0])


def board_to_state(board: "Board") -> np.ndarray:
    state = np.zeros(STATE_SIZE, dtype=np.uint64)
    state[P_PAWN] = np.uint64(board.pieces[PAWN])
    state[P_KNIGHT] = np.uint64(board.pieces[KNIGHT])
    state[P_BISHOP] = np.uint64(board.pieces[BISHOP])
    state[P_ROOK] = np.uint64(board.pieces[ROOK])
    state[P_QUEEN] = np.uint64(board.pieces[QUEEN])
    state[P_KING] = np.uint64(board.pieces[KING])

    state[C_WHITE] = np.uint64(board.colours[WHITE])
    state[C_BLACK] = np.uint64(board.colours[BLACK])

    state[TURN] = np.uint64(0 if board.turn else 1)
    state[CASTLING] = np.uint64(board.castling)
    state[EP_SQUARE] = np.uint64(board.ep_square if board.ep_square != -1 else 64)
    state[HALFMOVE] = np.uint64(board.halfmove)
    state[FULLMOVE] = np.uint64(board.fullmove)

    state[HASH] = np.uint64(board.hash)

    state[MG_SCORE_W] = np.uint64(board.mg_score[WHITE] & 0xFFFFFFFFFFFFFFFF)
    state[EG_SCORE_W] = np.uint64(board.eg_score[WHITE] & 0xFFFFFFFFFFFFFFFF)
    state[MG_SCORE_B] = np.uint64(board.mg_score[BLACK] & 0xFFFFFFFFFFFFFFFF)
    state[EG_SCORE_B] = np.uint64(board.eg_score[BLACK] & 0xFFFFFFFFFFFFFFFF)

    state[GAME_PHASE] = np.uint64(board.game_phase)

    state[BISHOP_COUNT_W] = np.uint64(board.bishop_count[WHITE])
    state[BISHOP_COUNT_B] = np.uint64(board.bishop_count[BLACK])

    state[MG_BONUS_W] = np.uint64(board.mg_bonus[WHITE] & 0xFFFFFFFFFFFFFFFF)
    state[EG_BONUS_W] = np.uint64(board.eg_bonus[WHITE] & 0xFFFFFFFFFFFFFFFF)
    state[MG_BONUS_B] = np.uint64(board.mg_bonus[BLACK] & 0xFFFFFFFFFFFFFFFF)
    state[EG_BONUS_B] = np.uint64(board.eg_bonus[BLACK] & 0xFFFFFFFFFFFFFFFF)

    return state
