from __future__ import annotations

# ruff: noqa
import numpy as np
from numba import njit

BOUND_NONE = 0
BOUND_EXACT = 1
BOUND_LOWER = 2
BOUND_UPPER = 3

MATE_SCORE = 1_000_000
MATE_THRESHOLD = 900_000
NO_MOVE = 0


@njit(cache=False)
def score_to_tt(score: int, ply: int) -> int:
    if score > MATE_THRESHOLD:
        return score + ply
    if score < -MATE_THRESHOLD:
        return score - ply
    return score


@njit(cache=False)
def score_from_tt(score: int, ply: int) -> int:
    if score > MATE_THRESHOLD:
        return score - ply
    if score < -MATE_THRESHOLD:
        return score + ply
    return score


def create_tt_arrays(
    exp_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if exp_size < 1:
        raise ValueError("Transposition table exponent must be positive")

    table_size = 1 << exp_size
    return (
        np.zeros(table_size, dtype=np.uint64),
        np.zeros(table_size, dtype=np.int32),
        np.zeros(table_size, dtype=np.uint16),
        np.zeros(table_size, dtype=np.int16),
        np.zeros(table_size, dtype=np.uint8),
        np.zeros(table_size, dtype=np.uint8),
        np.zeros(table_size, dtype=np.int32),
    )


@njit(cache=False)
def probe_tt(
    hash_val: int | np.uint64,
    tt_hash: np.ndarray,
    tt_score: np.ndarray,
    tt_move: np.ndarray,
    tt_depth: np.ndarray,
    tt_bound: np.ndarray,
    tt_age: np.ndarray,
    tt_static_eval: np.ndarray,
) -> tuple[bool, int, int, int, int, int]:
    """Returns (found, move, score, depth, bound, static_eval)"""
    idx = hash_val & (tt_hash.size - 1)
    if tt_hash[idx] == hash_val and tt_bound[idx] != BOUND_NONE:
        return (
            True,
            int(tt_move[idx]),
            int(tt_score[idx]),
            int(tt_depth[idx]),
            int(tt_bound[idx]),
            int(tt_static_eval[idx]),
        )
    return False, NO_MOVE, 0, 0, BOUND_NONE, 0


@njit(cache=False)
def store_tt(
    hash_val: int | np.uint64,
    best_move: int,
    score: int,
    depth: int,
    bound: int,
    static_eval: int,
    current_age: int,
    tt_hash: np.ndarray,
    tt_score: np.ndarray,
    tt_move: np.ndarray,
    tt_depth: np.ndarray,
    tt_bound: np.ndarray,
    tt_age: np.ndarray,
    tt_static_eval: np.ndarray,
) -> None:
    idx = hash_val & (tt_hash.size - 1)
    old_hash = tt_hash[idx]
    current_age &= 0xFF

    replace = False
    if tt_bound[idx] == BOUND_NONE:
        replace = True
    elif old_hash != hash_val:
        old_depth = tt_depth[idx]
        age_diff = (current_age - tt_age[idx]) & 0xFF
        # Keep recent deep work, but let moderately close or stale entries be replaced.
        replace = age_diff > 2 or depth >= old_depth - 2
    else:
        old_depth = tt_depth[idx]
        age_diff = (current_age - tt_age[idx]) & 0xFF
        if depth > old_depth or age_diff > 2:
            replace = True
        elif depth == old_depth and bound == BOUND_EXACT and tt_bound[idx] != BOUND_EXACT:
            replace = True

    if replace:
        tt_hash[idx] = hash_val
        tt_score[idx] = score
        tt_move[idx] = best_move
        tt_depth[idx] = depth
        tt_bound[idx] = bound
        tt_age[idx] = current_age
        tt_static_eval[idx] = static_eval
