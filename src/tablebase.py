"""Root-only Syzygy probing (vendored 3+4-man WDL+DTZ in tb/).

In-search probing is impossible here (Numba tree cannot call Python per
node, and per-node Python probes would destroy the ~2M NPS), so the design
is root-only, which is also the highest-leverage point: exact conversion of
won endings and exact defence of lost ones.

Policy (all game-legal positions only; never raises on clock):
- bare position (<=4 pieces, no castling rights) with WDL +-2  -> DTZ move,
  with fifty-move and repetition guards, else fall back to search.
- WDL 0 / cursed (+1) / blessed (-1) -> search (swindles + progress).
- Missing tb/ dir or missing sub-table -> search.

DTZ50 note: python-chess WDL assumes a reset counter, so wins additionally
require halfmove + |dtz| + margin <= 100, else they may bust the fifty-move
rule and we search instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

import chess
import chess.syzygy

from src.board import PAWN, move_is_capture, move_to_uci

if TYPE_CHECKING:
    from src.board import Board

TB_DIR: Final[str] = "tb"
TB_MAX_PIECES: Final[int] = 4
DTZ_FIFTY_MARGIN: Final[int] = 2


def _tb_path() -> Path:
    # src/tablebase.py -> <root>/tb, both in repo and extracted zip.
    return Path(__file__).resolve().parent.parent / TB_DIR


_tablebase: chess.syzygy.Tablebase | None = None
_tablebase_failed: bool = False


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


def _result_hash(board: Board, move: int) -> int | None:
    """Hash after `move`, without disturbing `board` (for repetition guard)."""
    tmp = board.copy()
    try:
        tmp.make_move(move)
    except (ValueError, IndexError):
        return None
    return tmp.hash


def _repeats(hash_value: int, game_hashes: object, times: int) -> bool:
    if not isinstance(game_hashes, list):
        return False
    seen = 0
    for hashed in game_hashes:
        if hashed == hash_value:
            seen += 1
            if seen >= times:
                return True
    return False


def _pick_win(
    tb: chess.syzygy.Tablebase,
    cb: chess.Board,
    board: Board,
    legal_moves: list[int],
    game_hashes: list[int],
) -> int | None:
    tier_win: list[tuple[int, int, bool]] = []  # (dtz, move, zeroing_ours)
    tier_cursed: list[tuple[int, int]] = []
    halfmove = board.halfmove
    for move in legal_moves:
        uci = move_to_uci(move)
        try:
            chess_move = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if chess_move not in cb.legal_moves:
            continue
        cb.push(chess_move)
        popped = False
        try:
            if cb.is_checkmate():
                return move
            if cb.is_stalemate() or cb.is_insufficient_material():
                continue  # winning line never settles for less (search backup)
            wdl = tb.get_wdl(cb)
            if wdl == -2:
                dtz = tb.get_dtz(cb)
                if dtz is not None:
                    from_sq = move & 0x3F
                    zeroing = move_is_capture(move) or board.piece_at_sq[from_sq] == PAWN
                    tier_win.append((abs(dtz), move, zeroing))
            elif wdl == -1:
                dtz = tb.get_dtz(cb)
                if dtz is not None:
                    tier_cursed.append((abs(dtz), move))
        finally:
            if not popped:
                cb.pop()
                popped = True
    for dtz, move, zeroing in sorted(tier_win):
        if not zeroing and halfmove + dtz + DTZ_FIFTY_MARGIN > 100:
            continue  # would risk busting the fifty-move rule
        hashed = _result_hash(board, move)
        if hashed is not None and _repeats(hashed, game_hashes, 2):
            continue  # would claim a draw by repetition while winning
        return move
    for _, move in sorted(tier_cursed):
        hashed = _result_hash(board, move)
        if hashed is not None and _repeats(hashed, game_hashes, 2):
            continue
        return move
    return None


def _pick_loss(
    tb: chess.syzygy.Tablebase,
    cb: chess.Board,
    legal_moves: list[int],
) -> int | None:
    best_move: int | None = None
    best_dtz = -10**9
    for move in legal_moves:
        uci = move_to_uci(move)
        try:
            chess_move = chess.Move.from_uci(uci)
        except ValueError:
            continue
        if chess_move not in cb.legal_moves:
            continue
        cb.push(chess_move)
        popped = False
        try:
            if cb.is_checkmate():
                return move  # mate for us even in a "lost" TB score
            if cb.is_stalemate() or cb.is_insufficient_material():
                return move  # hold the draw
            wdl = tb.get_wdl(cb)
            if wdl == 0 or wdl == -1:
                # Opponent draws or is blessed-losing: take it immediately.
                return move
            if wdl == 2:
                dtz = tb.get_dtz(cb)
                if dtz is not None and dtz > best_dtz:
                    best_dtz = dtz
                    best_move = move
        finally:
            if not popped:
                cb.pop()
                popped = True
    return best_move


def tb_root_move(
    board: Board,
    legal_moves: list[int],
    game_hashes: list[int] | None = None,
) -> int | None:
    """Return an exact tablebase move, or None to fall through to search."""
    if board.castling != 0 or not legal_moves:
        return None
    occupied = board.colours[0] | board.colours[1]
    if occupied.bit_count() > TB_MAX_PIECES:
        return None
    tb = _get_tablebase()
    if tb is None:
        return None
    try:
        cb = chess.Board(board.fen())
    except ValueError:
        return None
    try:
        wdl = tb.get_wdl(cb)
    except (KeyError, OSError, ValueError):
        return None
    if wdl is None or wdl in (0, 1, -1):
        return None
    hashes: list[int] = game_hashes if game_hashes is not None else []
    try:
        if wdl == 2:
            return _pick_win(tb, cb, board, legal_moves, hashes)
        return _pick_loss(tb, cb, legal_moves)
    except (KeyError, OSError, ValueError, IndexError):
        return None
