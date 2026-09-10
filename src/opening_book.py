"""Root Polyglot opening-book probing for the custom board."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import chess.polyglot

from src.board import move_is_promo, move_promo_piece

if TYPE_CHECKING:
    from src.board import Board

BOOK_DIR: Final[str] = "book"
BOOK_FILE: Final[str] = "book.bin"
OPENING_MAX_FULLMOVE: Final[int] = 20
MIN_WEIGHT_FRACTION: Final[int] = 8

_CASTLING_TARGETS: Final[dict[tuple[int, int], int]] = {
    (4, 7): 6,
    (4, 0): 2,
    (60, 63): 62,
    (60, 56): 58,
}


def _book_path() -> Path:
    return Path(__file__).resolve().parent.parent / BOOK_DIR / BOOK_FILE


_reader: chess.polyglot.MemoryMappedReader | None = None
_reader_failed = False


def _get_reader() -> chess.polyglot.MemoryMappedReader | None:
    global _reader, _reader_failed
    if _reader is not None:
        return _reader
    if _reader_failed:
        return None
    try:
        _reader = chess.polyglot.open_reader(_book_path())
    except (OSError, ValueError):
        _reader_failed = True
        return None
    return _reader


def initialize_opening_book() -> bool:
    """Memory-map the shipped book during the platform's init budget."""
    return _get_reader() is not None


def _encoded_move_key(move: int) -> tuple[int, int, int | None]:
    promotion = move_promo_piece(move) if move_is_promo(move) else None
    return move & 0x3F, (move >> 6) & 0x3F, promotion


def _entry_move_key(entry: chess.polyglot.Entry) -> tuple[int, int, int | None]:
    move = entry.move
    to_square = _CASTLING_TARGETS.get(
        (move.from_square, move.to_square), move.to_square
    )
    promotion = move.promotion - 1 if move.promotion is not None else None
    return move.from_square, to_square, promotion


def book_move(board: Board, legal_moves: list[int]) -> int | None:
    """Return the engine-preferred move among credible opening entries."""
    if board.fullmove > OPENING_MAX_FULLMOVE or not legal_moves:
        return None
    reader = _get_reader()
    if reader is None:
        return None

    legal_by_key = {_encoded_move_key(move): move for move in legal_moves}
    try:
        candidates: list[tuple[int, int]] = []
        for entry in reader.find_all(board.hash):
            move = legal_by_key.get(_entry_move_key(entry))
            if move is not None:
                candidates.append((entry.weight, move))
        if not candidates:
            return None

        max_weight = max(weight for weight, _ in candidates)
        credible = (
            (weight, move)
            for weight, move in candidates
            if weight * MIN_WEIGHT_FRACTION >= max_weight
        )

        def score(candidate: tuple[int, int]) -> tuple[int, int]:
            weight, move = candidate
            child = board.copy()
            child.make_move(move)
            return -child.evaluate(), weight

        return max(credible, key=score)[1]
    except (IndexError, OSError, ValueError):
        return None
