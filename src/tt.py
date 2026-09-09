from enum import IntEnum

import chess


class Bound(IntEnum):
    NONE = 0  # entry is empty/invalid
    EXACT = 1  # PV-node (exact score)
    LOWER = 2  # all-node (failed high, beta cutoff)
    UPPER = 3  # cut-node (failed low, no move improved alpha)


MATE_SCORE: int = 1_000_000
MATE_THRESHOLD: int = 900_000


def score_to_tt(score: int, ply: int) -> int:
    """Adjust mate score for transposition table storage (relative to position, not root)."""
    if abs(score) > MATE_THRESHOLD:
        return score + ply if score > 0 else score - ply
    return score


def score_from_tt(score: int, ply: int) -> int:
    """Adjust mate score from transposition table retrieval (relative to root)."""
    if abs(score) > MATE_THRESHOLD:
        return score - ply if score > 0 else score + ply
    return score


class TTEntry:
    __slots__ = ("age", "best_move", "bound", "depth", "hash", "score", "static_eval")

    def __init__(
        self,
        hash: int = 0,
        score: int = 0,
        best_move: chess.Move | None = None,
        depth: int = 0,
        bound: Bound = Bound.NONE,
        age: int = 0,
        static_eval: int = 0,
    ) -> None:
        self.hash = hash
        self.score = score
        self.best_move = best_move
        self.depth = depth
        self.bound = bound
        self.age = age
        self.static_eval = static_eval


class TT:
    """Transposition table with depth-preferred replacement and age-based eviction."""

    def __init__(self, exp_size: int = 20) -> None:
        self.exp_size = exp_size
        self.table_size = 1 << exp_size
        self.mask = self.table_size - 1
        self.table: list[TTEntry | None] = [None] * self.table_size
        self.current_age: int = 0

    def clear(self) -> None:
        """Reset transposition table entries and search age."""
        self.table = [None] * self.table_size
        self.current_age = 0

    def new_search(self) -> None:
        """Advance age counter at the start of a new root search."""
        self.current_age += 1

    def probe(self, hash_val: int) -> TTEntry | None:
        """Look up an entry by 64-bit Zobrist hash."""
        entry = self.table[hash_val & self.mask]
        if entry is not None and entry.bound != Bound.NONE and entry.hash == hash_val:
            return entry
        return None

    def store(
        self,
        hash_val: int,
        best_move: chess.Move | None,
        score: int,
        depth: int,
        bound: Bound,
        static_eval: int = 0,
    ) -> None:
        """Store an entry into the transposition table with replacement logic.

        Replacement policy:
        - Always replace if the slot is empty or has a different hash.
        - For same-hash entries: replace if new depth >= old depth, or
          the old entry is from a stale search (age difference > 2),
          or the new entry is EXACT bound.
        """
        index = hash_val & self.mask
        entry = self.table[index]

        if entry is None:
            self.table[index] = TTEntry(
                hash=hash_val,
                score=score,
                best_move=best_move,
                depth=depth,
                bound=bound,
                age=self.current_age,
                static_eval=static_eval,
            )
            return

        # Different hash — replace if new entry is deeper or old entry is stale.
        if entry.hash != hash_val:
            if depth >= entry.depth or (self.current_age - entry.age) > 2:
                entry.hash = hash_val
                entry.best_move = best_move
                entry.score = score
                entry.depth = depth
                entry.bound = bound
                entry.age = self.current_age
                entry.static_eval = static_eval
            return

        # Same hash — always update if new depth >= old depth, stale, or EXACT.
        if depth >= entry.depth or (self.current_age - entry.age) > 2 or bound == Bound.EXACT:
            # Preserve the best move if the new search didn't find one.
            if best_move is None:
                best_move = entry.best_move
            entry.hash = hash_val
            entry.best_move = best_move
            entry.score = score
            entry.depth = depth
            entry.bound = bound
            entry.age = self.current_age
            entry.static_eval = static_eval
