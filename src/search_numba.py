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
    evaluate,
    generate_moves,
    gives_check,
    is_in_check,
    is_sq_attacked,
    lsb_sq,
    make_move,
    make_null_move,
    may_give_check,
    piece_type_at,
    unmake_move,
    unmake_null_move,
)
from src.constants import (
    BISHOP,
    BLACK,
    CAPTURE,
    C_BLACK,
    C_WHITE,
    CASTLING,
    EG_BONUS_B,
    EG_BONUS_W,
    EG_SCORE_B,
    EG_SCORE_W,
    EN_PASSANT,
    EP_SQUARE,
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
    NULL_SEARCH,
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
from src.move_ordering import MVV_LVA, PIECE_VALUES
from src.tt import (
    BOUND_EXACT,
    BOUND_LOWER,
    BOUND_UPPER,
    probe_tt,
    score_from_tt,
    score_to_tt,
    store_tt,
)

libc = ctypes.CDLL(None)
clock = libc.clock
clock.restype = ctypes.c_int64
clock.argtypes = []

_PIECE_VALUES = np.array(PIECE_VALUES, dtype=np.int32)
_MVV_LVA = np.array(MVV_LVA, dtype=np.int32)


@njit(cache=False)
def is_insufficient_material(state: np.ndarray) -> bool:
    if state[P_PAWN] or state[P_ROOK] or state[P_QUEEN]:
        return False

    knights = state[P_KNIGHT]
    bishops = state[P_BISHOP]
    if knights:
        return not bishops and not (knights & (knights - np.uint64(1)))
    if not bishops:
        return True

    light_squares = np.uint64(0x55AA55AA55AA55AA)
    dark_squares = np.uint64(0xAA55AA55AA55AA55)
    return not (bishops & light_squares) or not (bishops & dark_squares)


@njit(cache=False)
def is_draw(
    state: np.ndarray,
    undo_stack: np.ndarray,
    ply: int,
    hash_history: np.ndarray,
    hist_len: int,
) -> bool:
    if is_insufficient_material(state):
        return True
    if state[NULL_SEARCH]:
        return False
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
    if tt_move != NO_MOVE and move == tt_move:
        return 2000000000

    from_sq = move & 0x3F
    to_sq = (move >> 6) & 0x3F
    flags = (move >> 12) & 0xF

    if flags & CAPTURE:
        moving_piece = piece_type_at(state, from_sq)
        captured_piece = PAWN if flags == EN_PASSANT else piece_type_at(state, to_sq)
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
def pick_move(moves: np.ndarray, scores: np.ndarray, num_moves: int, current_index: int) -> None:
    """Lazy Selection / Pick-Next-Best: select highest scoring move and swap into current_index."""
    max_idx = current_index
    max_score = scores[current_index]
    for j in range(current_index + 1, num_moves):
        if scores[j] > max_score:
            max_score = scores[j]
            max_idx = j
    if max_idx != current_index:
        moves[current_index], moves[max_idx] = moves[max_idx], moves[current_index]
        scores[current_index], scores[max_idx] = scores[max_idx], scores[current_index]


@njit(cache=False)
def score_captures(
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

        if flags & CAPTURE:
            moving_piece = piece_type_at(state, from_sq)
            captured_piece = PAWN if flags == EN_PASSANT else piece_type_at(state, to_sq)
            scores[i] = 1000000000 + _MVV_LVA[captured_piece, moving_piece]
        elif flags >= KNIGHT_PROMO:
            scores[i] = 900000000 + flags
        else:
            scores[i] = 0


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
    stats[4] += 1  # qnodes (S7)
    if (stats[0] & 1023) == 0:
        if stats[3] > 0 and clock() >= stats[3]:
            stats[1] = 1
            return 0

    if stats[1] == 1:
        return 0

    if ply >= MAX_PLY - 1:
        return evaluate(state)

    in_check = is_in_check(state)
    if (state[HALFMOVE] >= 100 and not state[NULL_SEARCH]) or is_insufficient_material(state):
        if not in_check:
            return 0
        moves = moves_stack[ply]
        if generate_moves(state, moves, captures_only=False) == 0:
            return -MATE_SCORE + ply
        return 0

    stand_pat = -INF
    if not in_check:
        stand_pat = evaluate(state)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

    moves = moves_stack[ply]
    num_moves = generate_moves(state, moves, captures_only=not in_check)
    if not in_check and num_moves == 0:
        non_kings = (state[C_WHITE] | state[C_BLACK]) & ~state[P_KING]
        piece_count = 0
        while non_kings and piece_count <= 3:
            non_kings &= non_kings - np.uint64(1)
            piece_count += 1
        if (piece_count <= 3 or abs(stand_pat) >= 800) and generate_moves(
            state, moves, captures_only=False
        ) == 0:
            return 0
    scores = scores_stack[ply]
    score_captures(state, moves, num_moves, scores)

    best_score = -INF if in_check else stand_pat
    legal_moves = 0

    for i in range(num_moves):
        pick_move(moves, scores, num_moves, i)
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
    extensions: int = 0,
) -> int:
    stats[0] += 1
    if (stats[0] & 1023) == 0:
        if stats[3] > 0 and clock() >= stats[3]:
            stats[1] = 1
            return 0

    if stats[1] == 1:
        return 0

    in_check = is_in_check(state)
    # S6: bound check extensions. Skip when near fifty-move draw (line draws
    # anyway) and cap horizon extensions (depth<=0 already goes to quiescence
    # evasion search; extending there converts cheap qnodes into full-width
    # nodes and risks perpetual-check explosion).
    if in_check and ply < MAX_PLY - 2 and extensions < 4 and state[HALFMOVE] < 90:
        if depth > 0 or extensions < 2:
            depth += 1
            extensions += 1

    # Checkmate takes precedence over claimed draws.
    if ply > 0 and is_draw(state, undo_stack, ply, hash_history, hist_len):
        if in_check:
            moves = moves_stack[ply]
            if generate_moves(state, moves, captures_only=False) == 0:
                return -MATE_SCORE + ply
        return 0

    if ply >= MAX_PLY - 1:
        return evaluate(state)

    # Mate-distance pruning narrows forced-mate subtrees without changing scores.
    alpha = max(alpha, -MATE_SCORE + ply)
    beta = min(beta, MATE_SCORE - ply - 1)
    if alpha >= beta:
        return alpha

    is_pv = beta - alpha > 1
    hash_val = state[HASH]

    # Transposition Table probe
    found, tt_move_val, tt_score_val, tt_depth_val, tt_bound_val, tt_static_val = probe_tt(
        hash_val, tt_hash, tt_score, tt_move, tt_depth, tt_bound, tt_age, tt_static_eval
    )
    stats[5] += 1  # tt_probes (S7)
    if found:
        stats[6] += 1  # tt_hits

    if found and not is_pv and tt_depth_val >= depth:
        adjusted_score = score_from_tt(tt_score_val, ply)
        if tt_bound_val == BOUND_EXACT:
            stats[7] += 1
            return adjusted_score
        if tt_bound_val == BOUND_LOWER and adjusted_score >= beta:
            stats[7] += 1
            return beta
        if tt_bound_val == BOUND_UPPER and adjusted_score <= alpha:
            stats[7] += 1
            return alpha

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
            stats[9] += 1
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
            extensions,
        )
        unmake_null_move(state, undo_stack, ply)

        if stats[1] == 1:
            return 0

        if null_score >= beta:
            stats[10] += 1
            return beta

    moves = moves_stack[ply]
    num_moves = generate_moves(state, moves, captures_only=False)
    scores = scores_stack[ply]
    for i in range(num_moves):
        scores[i] = score_move(state, moves[i], tt_move_val, ply, killers, history)

    best_move = NO_MOVE
    best_score = -INF
    alpha_orig = alpha
    legal_moves = 0
    # S2: enemy king square for the may_give_check gate (1 lsb per node).
    _opp_king_bb = state[P_KING] & state[C_WHITE + (1 - us)]
    opp_king_sq = lsb_sq(_opp_king_bb) if _opp_king_bb != np.uint64(0) else -1

    for i in range(num_moves):
        pick_move(moves, scores, num_moves, i)
        move = moves[i]
        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        flags = (move >> 12) & 0xF

        is_tactical = bool(flags & CAPTURE or flags >= KNIGHT_PROMO)

        # Futility Pruning: verify arithmetic margin before expensive raycast
        # S2: may_give_check gate (coordinate/table pre-filter) avoids the
        # up-to-4-raycast gives_check() on geometrically impossible checks.
        if (
            not is_pv
            and not in_check
            and depth <= 3
            and legal_moves > 0
            and not is_tactical
            and (static_eval + 120 * depth <= alpha)
            and opp_king_sq >= 0
            and not (
                may_give_check(state, from_sq, to_sq, flags, opp_king_sq, us)
                and gives_check(state, move)
            )
        ):
            stats[11] += 1
            continue

        make_move(state, undo_stack, ply, move)
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
                extensions,
            )
        else:
            # Modernized Late Move Reductions (LMR)
            reduction = 0
            if depth >= 3 and legal_moves >= 4 and not is_tactical and not in_check:
                reduction = 1
                if legal_moves >= 6:
                    reduction += 1
                if depth >= 6 and legal_moves >= 12:
                    reduction += 1
                if is_pv:
                    reduction -= 1
                h = history[us, from_sq, to_sq]
                if h > 4000:
                    reduction -= 1
                elif h < -4000:
                    reduction += 1
                # S2 gate: skip raycasts unless geometrically possible.
                if (
                    reduction > 0
                    and opp_king_sq >= 0
                    and may_give_check(state, from_sq, to_sq, flags, opp_king_sq, us)
                    and gives_check(state, move)
                ):
                    reduction -= 1
                reduction = max(0, min(reduction, depth - 2))
                if reduction > 0:
                    stats[12] += 1

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
                extensions,
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
                    extensions,
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
                    extensions,
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
                        # History gravity on beta-cutoff
                        bonus = min(depth * depth, 400)
                        cur = history[us, from_sq, to_sq]
                        history[us, from_sq, to_sq] = cur + bonus - (cur * abs(bonus)) // 16384
                        # Apply malus to non-cutoff quiet moves previously tried at this node
                        for j in range(i):
                            prev_m = moves[j]
                            prev_to = (prev_m >> 6) & 0x3F
                            prev_flags = (prev_m >> 12) & 0xF
                            if not prev_flags & CAPTURE and prev_flags < KNIGHT_PROMO:
                                prev_from = prev_m & 0x3F
                                prev_cur = history[us, prev_from, prev_to]
                                history[us, prev_from, prev_to] = (
                                    prev_cur - bonus - (prev_cur * abs(bonus)) // 16384
                                )

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
                    stats[8] += 1
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
    found, tt_move_val, _, _, _, _ = probe_tt(
        hash_val, tt_hash, tt_score, tt_move, tt_depth, tt_bound, tt_age, tt_static_eval
    )
    stats[5] += 1
    if found:
        stats[6] += 1

    in_check = is_in_check(state)
    static_eval = evaluate(state) if not in_check else -INF

    moves = moves_stack[0]
    num_moves = generate_moves(state, moves, captures_only=False)
    scores = scores_stack[0]
    for i in range(num_moves):
        scores[i] = score_move(state, moves[i], tt_move_val, 0, killers, history)

    best_move = NO_MOVE
    best_score = -INF
    runner_up_score = -INF
    alpha_orig = alpha
    legal_moves = 0
    us = int(state[TURN])
    _root_opp_bb = state[P_KING] & state[C_WHITE + (1 - us)]
    root_opp_king_sq = lsb_sq(_root_opp_bb) if _root_opp_bb != np.uint64(0) else -1

    for i in range(num_moves):
        pick_move(moves, scores, num_moves, i)
        move = moves[i]
        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        flags = (move >> 12) & 0xF

        is_tactical = bool(flags & CAPTURE or flags >= KNIGHT_PROMO)

        may_check = (
            depth >= 3
            and legal_moves >= 3
            and not is_tactical
            and not in_check
            and root_opp_king_sq >= 0
            and may_give_check(state, from_sq, to_sq, flags, root_opp_king_sq, us)
            and gives_check(state, move)
        )

        make_move(state, undo_stack, 0, move)
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
                0,
            )
        else:
            reduction = 0
            if depth >= 3 and legal_moves >= 4 and not is_tactical and not in_check:
                reduction = 1
                if legal_moves >= 6:
                    reduction += 1
                if depth >= 6 and legal_moves >= 12:
                    reduction += 1
                h = history[us, from_sq, to_sq]
                if h > 4000:
                    reduction -= 1
                elif h < -4000:
                    reduction += 1
                # S2 gate (root): skip raycasts unless geometrically possible.
                if reduction > 0 and may_check:
                    reduction -= 1
                reduction = max(0, min(reduction, depth - 2))
                if reduction > 0:
                    stats[12] += 1

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
                0,
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
                    0,
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
                    0,
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
                    bonus = min(depth * depth, 400)
                    cur = history[us, from_sq, to_sq]
                    history[us, from_sq, to_sq] = cur + bonus - (cur * abs(bonus)) // 16384
                    for j in range(i):
                        prev_m = moves[j]
                        prev_to = (prev_m >> 6) & 0x3F
                        prev_flags = (prev_m >> 12) & 0xF
                        if not prev_flags & CAPTURE and prev_flags < KNIGHT_PROMO:
                            prev_from = prev_m & 0x3F
                            prev_cur = history[us, prev_from, prev_to]
                            history[us, prev_from, prev_to] = (
                                prev_cur - bonus - (prev_cur * abs(bonus)) // 16384
                            )

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

    state[HASH] = np.uint64(board.hash)

    state[MG_SCORE_W] = np.uint64(board.mg_score[WHITE] & 0xFFFFFFFFFFFFFFFF)
    state[EG_SCORE_W] = np.uint64(board.eg_score[WHITE] & 0xFFFFFFFFFFFFFFFF)
    state[MG_SCORE_B] = np.uint64(board.mg_score[BLACK] & 0xFFFFFFFFFFFFFFFF)
    state[EG_SCORE_B] = np.uint64(board.eg_score[BLACK] & 0xFFFFFFFFFFFFFFFF)

    state[GAME_PHASE] = np.uint64(board.game_phase)

    # S3: seed incremental structural bonuses from the Python board (which
    # maintains them via _update_bonus). Same wrap-uint64 encoding as scores.
    state[MG_BONUS_W] = np.uint64(board.mg_bonus[WHITE] & 0xFFFFFFFFFFFFFFFF)
    state[MG_BONUS_B] = np.uint64(board.mg_bonus[BLACK] & 0xFFFFFFFFFFFFFFFF)
    state[EG_BONUS_W] = np.uint64(board.eg_bonus[WHITE] & 0xFFFFFFFFFFFFFFFF)
    state[EG_BONUS_B] = np.uint64(board.eg_bonus[BLACK] & 0xFFFFFFFFFFFFFFFF)

    return state
