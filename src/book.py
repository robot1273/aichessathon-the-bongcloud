"""Polyglot opening-book probe (vendored gm2001.bin).

Master-game statistics (GM games 2001-2013, >=2530 Elo), not an engine book.
Root-only: returns a legal move or None. Never raises on clock: a missing or
unreadable book file simply disables the book.

Variety comes from weight-proportional choice with our own RNG. The harness
sets HARNESS_SEED for baselines; the platform sets nothing, and our agent
must not read it, so we seed from pid + monotonic clock (distinct per game).
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final

import chess
import chess.polyglot

from src.board import move_to_uci

if TYPE_CHECKING:
    from src.board import Board

BOOK_FILE: Final[str] = "book.bin"
BOOK_MIN_WEIGHT: Final[int] = 1


def _book_path() -> Path:
    # src/book.py -> <root>/book/book.bin, both in repo and in the
    # extracted submission zip (agent.py + book/ ship side by side).
    return Path(__file__).resolve().parent.parent / "book" / BOOK_FILE


_reader: chess.polyglot.MemoryMappedReader | None = None
_reader_failed: bool = False


def _get_reader() -> chess.polyglot.MemoryMappedReader | None:
    global _reader, _reader_failed
    if _reader is not None:
        return _reader
    if _reader_failed:
        return None
    try:
        _reader = chess.polyglot.MemoryMappedReader(str(_book_path()))
    except OSError:
        _reader_failed = True
        return None
    return _reader


def make_rng() -> random.Random:
    return random.Random((os.getpid() << 32) ^ time.monotonic_ns())


def book_move(board: Board, legal_moves: list[int], rng: random.Random) -> int | None:
    """Return a legal book move, or None when out of book / no book file."""
    reader = _get_reader()
    if reader is None or not legal_moves:
        return None
    try:
        chess_board = chess.Board(board.fen())
    except ValueError:
        return None
    try:
        entries = list(reader.find_all(chess_board, minimum_weight=BOOK_MIN_WEIGHT))
    except (ValueError, OSError):
        return None
    if not entries:
        return None
    by_uci = {move_to_uci(m): m for m in legal_moves}
    weighted: list[tuple[int, int]] = []
    for entry in entries:
        try:
            uci = entry.move.uci()
        except ValueError:
            continue
        if uci in by_uci:
            weighted.append((entry.weight, by_uci[uci]))
    if not weighted:
        return None
    total = sum(w for w, _ in weighted)
    if total <= 0:
        return weighted[0][1]
    pick = rng.randrange(total)
    for weight, move in weighted:
        pick -= weight
        if pick < 0:
            return move
    return weighted[-1][1]
