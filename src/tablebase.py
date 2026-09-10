"""Root Syzygy probing through python-chess.

The search board stays custom and Numba-friendly. At roots with at most four
pieces, or five-piece roots that can capture into the tables, FEN provides a
small and reliable boundary to python-chess. The resulting plan lets search
choose the most natural move without allowing it to discard an exact result.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import chess
import chess.polyglot
import chess.syzygy

from src.board import move_to_uci

if TYPE_CHECKING:
    from src.board import Board

TB_DIR: Final[str] = "tb"
TB_MAX_PIECES: Final[int] = 4
TB_TRANSITION_PIECES: Final[int] = TB_MAX_PIECES + 1
DTZ_FIFTY_MARGIN: Final[int] = 2


@dataclass(frozen=True)
class RootTablebase:
    """Exact root policy prepared before the timed search."""

    wdl: int
    search_moves: frozenset[int]
    fallback_move: int


@dataclass(frozen=True)
class _Candidate:
    move: int
    wdl: int
    dtz: int | None
    safe_win: bool
    claims_draw: bool


def _tb_path() -> Path:
    return Path(__file__).resolve().parent.parent / TB_DIR


_tablebase: chess.syzygy.Tablebase | None = None
_tablebase_failed = False


def _get_tablebase() -> chess.syzygy.Tablebase | None:
    global _tablebase, _tablebase_failed
    if _tablebase is not None:
        return _tablebase
    if _tablebase_failed:
        return None
    try:
        _tablebase = chess.syzygy.open_tablebase(str(_tb_path()))
    except OSError:
        _tablebase_failed = True
        return None
    return _tablebase


def initialize_tablebase() -> bool:
    """Open and index shipped tables during the platform's init budget."""
    return _get_tablebase() is not None


def _result_context(board: Board, move: int, game_hashes: list[int]) -> tuple[int, bool]:
    result = board.copy()
    result.make_move(move)
    claims_draw = result.halfmove >= 100 or game_hashes.count(result.hash) >= 2
    return result.halfmove, claims_draw


def _allows_repetition_draw(cb: chess.Board, game_hashes: list[int]) -> bool:
    if not game_hashes:
        return False
    counts = Counter(game_hashes)
    candidate_hash = chess.polyglot.zobrist_hash(cb)
    counts[candidate_hash] += 1
    if counts[candidate_hash] >= 3:
        return True
    for reply in cb.legal_moves:
        cb.push(reply)
        try:
            reply_hash = chess.polyglot.zobrist_hash(cb)
            counts[reply_hash] += 1
            if counts[reply_hash] >= 3:
                return True
            # The referee claims automatically before our next turn, even if
            # we would decline a claim and choose a different winning move.
            for next_move in cb.legal_moves:
                cb.push(next_move)
                try:
                    if counts[chess.polyglot.zobrist_hash(cb)] >= 2:
                        return True
                finally:
                    cb.pop()
            counts[reply_hash] -= 1
        finally:
            cb.pop()
    return False


def _candidate(
    tb: chess.syzygy.Tablebase,
    cb: chess.Board,
    board: Board,
    move: int,
    chess_move: chess.Move,
    game_hashes: list[int],
) -> _Candidate | None:
    zeroing = cb.is_zeroing(chess_move)
    cb.push(chess_move)
    try:
        halfmove, claims_draw = _result_context(board, move, game_hashes)
        if cb.is_checkmate():
            return _Candidate(move, 2, 0, True, False)
        if cb.is_stalemate() or cb.is_insufficient_material():
            return _Candidate(move, 0, 0, False, True)

        child_wdl = tb.get_wdl(cb)
        if child_wdl is None:
            return None
        move_wdl = -child_wdl
        try:
            child_dtz = tb.get_dtz(cb)
        except (OSError, ValueError):
            child_dtz = None
        dtz = abs(child_dtz) if child_dtz is not None else None
        permits_repetition = move_wdl == 2 and _allows_repetition_draw(cb, game_hashes)
        safe_win = move_wdl == 2 and not claims_draw and not permits_repetition and (
            zeroing
            or (dtz is not None and halfmove + dtz + DTZ_FIFTY_MARGIN <= 100)
        )
        return _Candidate(move, move_wdl, dtz, safe_win, claims_draw)
    except (KeyError, OSError, ValueError, IndexError):
        return None
    finally:
        cb.pop()


def _fallback(candidates: list[_Candidate], root_wdl: int) -> _Candidate:
    preserving = [candidate for candidate in candidates if candidate.wdl == root_wdl]
    pool = preserving or candidates

    if root_wdl <= 0:
        drawing = [candidate for candidate in pool if candidate.claims_draw]
        if drawing:
            return drawing[0]
    if root_wdl == 2:
        safe = [candidate for candidate in pool if candidate.safe_win]
        if safe:
            pool = safe

    with_dtz = [candidate for candidate in pool if candidate.dtz is not None]
    if with_dtz:
        if root_wdl > 0:
            return min(with_dtz, key=lambda candidate: candidate.dtz or 0)
        if root_wdl < 0:
            return max(with_dtz, key=lambda candidate: candidate.dtz or 0)
    return pool[0]


def tb_probe_root(
    board: Board,
    legal_moves: list[int],
    game_hashes: list[int] | None = None,
) -> RootTablebase | None:
    """Build an exact move policy, returning ``None`` when tables do not apply."""
    if board.castling != 0 or not legal_moves:
        return None
    piece_count = (board.colours[0] | board.colours[1]).bit_count()
    if piece_count > TB_TRANSITION_PIECES:
        return None

    tb = _get_tablebase()
    if tb is None:
        return None
    try:
        cb = chess.Board(board.fen())
    except ValueError:
        return None

    root_wdl: int | None = None
    if piece_count <= TB_MAX_PIECES:
        try:
            root_wdl = tb.get_wdl(cb)
        except (OSError, ValueError):
            return None
        if root_wdl is None:
            return None

    encoded = {move_to_uci(move): move for move in legal_moves}
    hashes = game_hashes if game_hashes is not None else []
    candidates: list[_Candidate] = []
    for chess_move in cb.legal_moves:
        move = encoded.get(chess_move.uci())
        if move is None:
            continue
        if piece_count == TB_TRANSITION_PIECES and not cb.is_capture(chess_move):
            continue
        candidate = _candidate(tb, cb, board, move, chess_move, hashes)
        if candidate is not None:
            candidates.append(candidate)

    if piece_count == TB_TRANSITION_PIECES:
        candidates = [candidate for candidate in candidates if candidate.safe_win]
        if not candidates:
            return None
        root_wdl = 2
    if root_wdl is None or not candidates:
        return None

    fallback = _fallback(candidates, root_wdl)
    if root_wdl == 2:
        winning = [
            candidate
            for candidate in candidates
            if candidate.wdl == 2 and candidate.safe_win
        ]
        known_dtz = [candidate.dtz for candidate in winning if candidate.dtz is not None]
        best_dtz = min(known_dtz) if known_dtz else None
        search_moves = frozenset(
            candidate.move
            for candidate in winning
            if best_dtz is None or candidate.dtz == best_dtz
        )
    else:
        search_moves = frozenset(
            candidate.move
            for candidate in candidates
            if candidate.wdl == root_wdl or (root_wdl < 0 and candidate.claims_draw)
        )
    return RootTablebase(root_wdl, search_moves, fallback.move)
