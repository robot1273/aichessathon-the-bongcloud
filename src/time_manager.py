from __future__ import annotations

import time
from typing import Any


class TimeManager:
    """Decide how long to think on a single move."""

    SAFETY_S: float = 1.0

    def __init__(self, increment_s: float) -> None:
        self._start: float = 0.0
        self._soft_limit: float = 0.0
        self._hard_limit: float = 0.0
        self._base_soft_limit: float = 0.0
        self._base_hard_limit: float = 0.0
        self._is_fixed_time: bool = False
        self._usable: float = 0.0
        self._increment_s: float = increment_s

    def start(
        self,
        time_left_ms: int,
        board: Any,
        movetime_ms: int | None = None,
        started_at: float | None = None,
    ) -> None:
        """Begin the clock for one move."""
        self._start = time.monotonic() if started_at is None else started_at

        if movetime_ms is not None:
            self._is_fixed_time = True
            usable = max(movetime_ms / 1000.0 - 0.002, 0.001)
            self._usable = usable
            self._soft_limit = usable * 0.80
            self._hard_limit = usable
            self._base_soft_limit = self._soft_limit
            self._base_hard_limit = self._hard_limit
            return

        self._is_fixed_time = False
        usable = max(time_left_ms / 1000.0 - self.SAFETY_S, 0.01)
        self._usable = usable

        if hasattr(board, "game_phase"):
            phase = min(board.game_phase, 5560) / 5560.0
        else:
            from .evaluation import evaluate_with_phase

            _, phase = evaluate_with_phase(board)

        move_number = getattr(board, "fullmove", getattr(board, "fullmove_number", 1))
        base = self._base_time(usable, move_number, phase)

        self._soft_limit = min(base, usable * 0.20)
        self._hard_limit = min(base * 3.0, usable * 0.40)
        self._base_soft_limit = self._soft_limit
        self._base_hard_limit = self._hard_limit

    def extend_if_unstable(self, prev_score: int | None, curr_score: int | None) -> None:
        """Widen limits when the score swings between iterations."""
        if self._is_fixed_time:
            return
        if prev_score is None or curr_score is None:
            return
        swing = abs(curr_score - prev_score)
        if swing >= 50:
            # Always scale from the original allocation so successive swings do
            # not compound into a clock-consuming search.
            factor = min(1.0 + swing / 300.0, 1.6)
            self._soft_limit = min(self._base_soft_limit * factor, self._usable * 0.45)
            self._hard_limit = min(self._base_hard_limit * factor, self._usable * 0.65)

    def shorten_for_stable_win(self) -> None:
        """Stop earlier when several completed iterations confirm a clear win."""
        if not self._is_fixed_time:
            self._soft_limit = min(self._soft_limit, self._base_soft_limit * 0.80)

    @property
    def soft_limit(self) -> float:
        return self._soft_limit

    @property
    def hard_limit(self) -> float:
        return self._hard_limit

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def is_time_up(self) -> bool:
        return self.elapsed() >= self._hard_limit

    def should_stop_iterating(self) -> bool:
        return self.elapsed() >= self._soft_limit

    def _base_time(self, usable: float, move_number: int, phase: float) -> float:
        # Material alone is a poor proxy for remaining moves: many won rook and
        # pawn endings still need dozens of accurate conversion moves.
        expected_remaining = 28.0 + 17.0 * phase
        base = (usable / expected_remaining) + (self._increment_s * 0.8)
        if 5 <= move_number <= 25:
            base *= 1.20
        return base
