"""Benchmark positions and helpers for profiling scripts."""

from __future__ import annotations

BENCHMARK_POSITIONS: list[tuple[str, str]] = [
    ("Start Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    (
        "Kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    ),
    ("Endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
]


def get_positions(
    position_arg: str | None = None,
    fen_arg: str | None = None,
) -> list[tuple[str, str]]:
    """Resolve benchmark positions based on user filters."""
    if fen_arg:
        return [("Custom Position", fen_arg.strip())]

    if not position_arg or position_arg.lower() in ("all", "*"):
        return BENCHMARK_POSITIONS

    # Check numeric 1-based index
    if position_arg.isdigit():
        idx = int(position_arg) - 1
        if 0 <= idx < len(BENCHMARK_POSITIONS):
            return [BENCHMARK_POSITIONS[idx]]
        raise ValueError(
            f"Position index {position_arg} out of range (1..{len(BENCHMARK_POSITIONS)})"
        )

    # Substring search by name
    target = position_arg.lower()
    matches = [pos for pos in BENCHMARK_POSITIONS if target in pos[0].lower()]
    if matches:
        return matches

    available = ", ".join(f"'{p[0]}'" for p in BENCHMARK_POSITIONS)
    raise ValueError(
        f"Unknown position '{position_arg}'. Available positions: {available}"
    )
