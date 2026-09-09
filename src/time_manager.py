from __future__ import annotations

import time

import chess

from .evaluation import evaluate_with_phase


class TimeManager:
    """Decide how long to think on a single move."""
    SAFETY_S: float = 1.0

    def __init__(self, increment_s: float) -> None:
        self._start: float = 0.0
        self._soft_limit: float = 0.0
        self._hard_limit: float = 0.0
        self._is_fixed_time: bool = False
        self._usable: float = 0.0
        self._increment_s: float = increment_s

    def start(
        self,
        time_left_ms: int,
        board: chess.Board,
        movetime_ms: int | None = None,
    ) -> None:
        """Begin the clock for one move."""
        self._start = time.monotonic()

        if movetime_ms is not None:
            self._is_fixed_time = True
            usable = max(movetime_ms / 1000.0 - 0.002, 0.001)
            self._usable = usable
            self._soft_limit = usable * 0.80
            self._hard_limit = usable
            return

        self._is_fixed_time = False
        usable = max(time_left_ms / 1000.0 - self.SAFETY_S, 0.01)
        self._usable = usable
        _, phase = evaluate_with_phase(board)
        base = self._base_time(usable, board.fullmove_number, phase)

        self._soft_limit = min(base, usable * 0.20)
        self._hard_limit = min(base * 3.0, usable * 0.40)

    def extend_if_unstable(self, prev_score: int | None, curr_score: int | None) -> None:
        """Widen limits when the score swings between iterations."""
        if self._is_fixed_time:
            return
        if prev_score is None or curr_score is None:
            return
        swing = abs(curr_score - prev_score)
        if swing >= 50:
            factor = min(1.0 + swing / 200.0, 1.8)
            self._soft_limit = min(self._soft_limit * factor, self._usable * 0.50)
            self._hard_limit = min(self._hard_limit * factor, self._usable * 0.70)

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
        expected_remaining = 20.0 + 25.0 * phase
        base = (usable / expected_remaining) + (self._increment_s * 0.8)
        if 5 <= move_number <= 25:
            base *= 1.25
        return base
