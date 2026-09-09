from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TimeConfig:
    """Explicit tuning surface for move allocation and search timing."""

    safety_s: float = 1.0
    fixed_overhead_s: float = 0.002
    fixed_soft_ratio: float = 0.80
    low_clock_increments: float = 20.0
    panic_clock_increments: float = 6.0
    low_clock_floor_s: float = 1.0
    panic_clock_floor_s: float = 0.25

    expected_moves_base: float = 28.0
    expected_moves_phase: float = 17.0
    pressure_moves: float = 12.0
    increment_share: float = 0.80
    opening_first_move: int = 5
    opening_last_move: int = 25
    opening_boost: float = 1.20

    soft_usable_cap: float = 0.20
    hard_base_ratio: float = 3.0
    hard_usable_cap: float = 0.40
    panic_soft_increment: float = 0.65
    panic_hard_increment: float = 0.95
    panic_soft_floor_s: float = 0.010
    panic_hard_floor_s: float = 0.020

    instability_threshold: int = 50
    instability_scale: float = 300.0
    instability_max_factor: float = 1.60
    instability_soft_cap: float = 0.45
    instability_hard_cap: float = 0.65
    uncertainty_soft_factor: float = 1.20
    uncertainty_hard_factor: float = 1.25
    uncertainty_soft_cap: float = 0.45
    uncertainty_hard_cap: float = 0.65
    stable_win_soft_factor: float = 0.80

    stable_win_score: int = 600
    stable_score_delta: int = 35
    stable_win_iterations: int = 3
    iteration_safety_factor: float = 1.10
    normal_min_growth: float = 1.60
    busy_min_growth: float = 1.90
    uncertain_min_growth: float = 2.10
    max_growth: float = 3.0
    busy_prediction_safety: float = 1.20
    uncertain_prediction_safety: float = 1.25
    busy_root_moves: int = 30
    root_uncertainty_gap: int = 50
    require_positive_root_gap: bool = False
    prediction_min_depth: int = 5
    panic_max_depth: int = 2
    aspiration_min_depth: int = 5
    aspiration_delta: int = 30
    aspiration_retry_reserve: float = 2.0


DEFAULT_TIME_CONFIG = TimeConfig()


class TimeManager:
    """Decide how long to think on a single move."""

    def __init__(self, increment_s: float, config: TimeConfig = DEFAULT_TIME_CONFIG) -> None:
        self.config = config
        self._start: float = 0.0
        self._soft_limit: float = 0.0
        self._hard_limit: float = 0.0
        self._base_soft_limit: float = 0.0
        self._base_hard_limit: float = 0.0
        self._is_fixed_time: bool = False
        self._usable: float = 0.0
        self._clock_left_s: float = 0.0
        self._increment_s: float = increment_s
        self._policy = "unstarted"

    def start(
        self,
        time_left_ms: int,
        board: Any,
        movetime_ms: int | None = None,
        started_at: float | None = None,
    ) -> None:
        """Begin the clock for one move."""
        self._start = time.monotonic() if started_at is None else started_at
        self._clock_left_s = max(time_left_ms / 1000.0, 0.0)

        if movetime_ms is not None:
            self._is_fixed_time = True
            usable = max(movetime_ms / 1000.0 - self.config.fixed_overhead_s, 0.001)
            self._usable = usable
            self._soft_limit = usable * self.config.fixed_soft_ratio
            self._hard_limit = usable
            self._base_soft_limit = self._soft_limit
            self._base_hard_limit = self._hard_limit
            self._policy = "fixed"
            return

        self._is_fixed_time = False
        usable = max(time_left_ms / 1000.0 - self.config.safety_s, 0.01)
        self._usable = usable

        if hasattr(board, "game_phase"):
            phase = min(board.game_phase, 5560) / 5560.0
        else:
            from .evaluation import evaluate_with_phase

            _, phase = evaluate_with_phase(board)

        move_number = getattr(board, "fullmove", getattr(board, "fullmove_number", 1))
        base = self._base_time(usable, move_number, phase)

        self._soft_limit = min(base, usable * self.config.soft_usable_cap)
        self._hard_limit = min(
            base * self.config.hard_base_ratio,
            usable * self.config.hard_usable_cap,
        )
        self._policy = "normal"
        if self.is_panic_clock():
            self._soft_limit = min(
                self._soft_limit,
                max(
                    self._increment_s * self.config.panic_soft_increment,
                    self.config.panic_soft_floor_s,
                ),
            )
            self._hard_limit = min(
                self._hard_limit,
                max(
                    self._increment_s * self.config.panic_hard_increment,
                    self.config.panic_hard_floor_s,
                ),
            )
            self._policy = "panic"
        elif self.is_low_clock():
            self._policy = "low"
        self._base_soft_limit = self._soft_limit
        self._base_hard_limit = self._hard_limit

    def extend_if_unstable(self, prev_score: int | None, curr_score: int | None) -> None:
        """Widen limits when the score swings between iterations."""
        if self._is_fixed_time or prev_score is None or curr_score is None:
            return
        swing = abs(curr_score - prev_score)
        if swing >= self.config.instability_threshold:
            factor = min(
                1.0 + swing / self.config.instability_scale,
                self.config.instability_max_factor,
            )
            self._soft_limit = min(
                self._base_soft_limit * factor,
                self._usable * self.config.instability_soft_cap,
            )
            self._hard_limit = min(
                self._base_hard_limit * factor,
                self._usable * self.config.instability_hard_cap,
            )
            self._policy = f"{self.clock_band}+unstable"

    def extend_for_root_uncertainty(self) -> None:
        """Reserve more time when multiple root moves remain competitive."""
        if self._is_fixed_time or self.is_low_clock():
            return
        self._soft_limit = max(
            self._soft_limit,
            min(
                self._base_soft_limit * self.config.uncertainty_soft_factor,
                self._usable * self.config.uncertainty_soft_cap,
            ),
        )
        self._hard_limit = max(
            self._hard_limit,
            min(
                self._base_hard_limit * self.config.uncertainty_hard_factor,
                self._usable * self.config.uncertainty_hard_cap,
            ),
        )
        self._policy = f"{self.clock_band}+uncertain"

    def shorten_for_stable_win(self) -> None:
        """Stop earlier when several completed iterations confirm a clear win."""
        if not self._is_fixed_time:
            self._soft_limit = min(
                self._soft_limit,
                self._base_soft_limit * self.config.stable_win_soft_factor,
            )
            self._policy = f"{self.clock_band}+stable-win"

    @property
    def soft_limit(self) -> float:
        return self._soft_limit

    @property
    def hard_limit(self) -> float:
        return self._hard_limit

    @property
    def clock_left_s(self) -> float:
        return self._clock_left_s

    @property
    def clock_band(self) -> str:
        if self._is_fixed_time:
            return "fixed"
        if self.is_panic_clock():
            return "panic"
        if self.is_low_clock():
            return "low"
        return "normal"

    @property
    def policy(self) -> str:
        return self._policy

    def elapsed(self) -> float:
        return time.monotonic() - self._start

    def is_time_up(self) -> bool:
        return self.elapsed() >= self._hard_limit

    def should_stop_iterating(self) -> bool:
        return self.elapsed() >= self._soft_limit

    def is_low_clock(self) -> bool:
        return not self._is_fixed_time and self._clock_left_s <= self._low_clock_threshold()

    def is_panic_clock(self) -> bool:
        return not self._is_fixed_time and self._clock_left_s <= self._panic_clock_threshold()

    def _low_clock_threshold(self) -> float:
        return max(
            self._increment_s * self.config.low_clock_increments,
            self.config.low_clock_floor_s,
        )

    def _panic_clock_threshold(self) -> float:
        return max(
            self._increment_s * self.config.panic_clock_increments,
            self.config.panic_clock_floor_s,
        )

    def _base_time(self, usable: float, move_number: int, phase: float) -> float:
        expected_remaining = (
            self.config.expected_moves_base + self.config.expected_moves_phase * phase
        )
        pressure_threshold = self._low_clock_threshold()
        if self._clock_left_s < pressure_threshold:
            expected_remaining += (
                self.config.pressure_moves
                * (pressure_threshold - self._clock_left_s)
                / pressure_threshold
            )
        base = (usable / expected_remaining) + (
            self._increment_s * self.config.increment_share
        )
        if self.config.opening_first_move <= move_number <= self.config.opening_last_move:
            base *= self.config.opening_boost
        return base
