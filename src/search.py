from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import chess

from .evaluation import evaluate
from .move_ordering import (
    MVV_LVA,
    PIECE_VALUES,
    HistoryTable,
    KillerTable,
    order_moves,
)
from .time_manager import TimeManager
from .tt import TT, Bound, score_from_tt, score_to_tt
from .zobrist import calculate_hash, push_hash

MATE_SCORE: Final[int] = 1_000_000
MATE_THRESHOLD: Final[int] = 900_000
MAX_DEPTH: Final[int] = 100
MAX_QUIESCENCE_DEPTH: Final[int] = 16
DELTA_PRUNING: Final[int] = 200
INF: Final[int] = MATE_SCORE + 1

ASP_INITIAL_DELTA: Final[int] = 25
ASP_MAX_DELTA: Final[int] = 500
ASP_MIN_DEPTH: Final[int] = 5

NMP_MIN_DEPTH: Final[int] = 3
LMR_MIN_DEPTH: Final[int] = 3
LMR_MIN_MOVE_INDEX: Final[int] = 3
FUTILITY_MAX_DEPTH: Final[int] = 3
FUTILITY_MARGIN_PER_DEPTH: Final[int] = 120
RFP_MAX_DEPTH: Final[int] = 6
RFP_MARGIN_PER_DEPTH: Final[int] = 80
NODE_CHECK_INTERVAL: Final[int] = 512

_LMR_TABLE: Final[list[list[int]]] = [
    [
        max(0, int(1.0 + math.log(d) * math.log(m) / 2.0)) if d > 0 and m > 0 else 0
        for m in range(64)
    ]
    for d in range(64)
]


class SearchAborted(Exception):
    pass


@dataclass(frozen=True, slots=True)
class SearchStats:
    nodes: int
    qnodes: int
    move_generations: int
    tt_probes: int
    tt_hits: int
    tt_cutoffs: int
    beta_cutoffs: int
    rfp_prunes: int
    futility_prunes: int
    null_prunes: int
    lmr_reductions: int


class Bot:
    def __init__(self, tt_exp_size: int = 22, collect_stats: bool = False) -> None:
        self.tt = TT(exp_size=tt_exp_size)
        self.time_mgr = TimeManager()
        self.killers = KillerTable()
        self.history = HistoryTable()
        self.game_positions: dict[int, int] = {}
        self.collect_stats = collect_stats
        self.nodes: int = 0
        self.sel_depth: int = 0
        self.completed_depth: int = 0
        self.best_move: chess.Move | None = None
        self.best_score: int = 0
        self.qnodes: int = 0
        self.move_generations: int = 0
        self.tt_probes: int = 0
        self.tt_hits: int = 0
        self.tt_cutoffs: int = 0
        self.beta_cutoffs: int = 0
        self.rfp_prunes: int = 0
        self.futility_prunes: int = 0
        self.null_prunes: int = 0
        self.lmr_reductions: int = 0

    @property
    def search_stats(self) -> SearchStats | None:
        if not self.collect_stats:
            return None
        return SearchStats(
            nodes=self.nodes,
            qnodes=self.qnodes,
            move_generations=self.move_generations,
            tt_probes=self.tt_probes,
            tt_hits=self.tt_hits,
            tt_cutoffs=self.tt_cutoffs,
            beta_cutoffs=self.beta_cutoffs,
            rfp_prunes=self.rfp_prunes,
            futility_prunes=self.futility_prunes,
            null_prunes=self.null_prunes,
            lmr_reductions=self.lmr_reductions,
        )

    def get_best_move(
        self,
        board: chess.Board,
        time_left_ms: int = 100_000,
        depth: int | None = None,
        movetime_ms: int | None = None,
        verbose: bool = False,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> chess.Move:
        self.time_mgr.start(time_left_ms, board, movetime_ms=movetime_ms)
        self.tt.new_search()
        self.killers.clear()

        self.nodes = 0
        self.sel_depth = 0
        self.completed_depth = 0
        self.best_move = None
        self.best_score = 0
        self.qnodes = 0
        self.move_generations = 0
        self.tt_probes = 0
        self.tt_hits = 0
        self.tt_cutoffs = 0
        self.beta_cutoffs = 0
        self.rfp_prunes = 0
        self.futility_prunes = 0
        self.null_prunes = 0
        self.lmr_reductions = 0

        root_hash = calculate_hash(board)
        self.game_positions[root_hash] = self.game_positions.get(root_hash, 0) + 1
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        if len(legal_moves) == 1:
            self.best_move = legal_moves[0]
            return legal_moves[0]

        best_move = legal_moves[0]
        best_score = -INF
        prev_score = 0
        prev_best: int | None = None
        target_depth = depth if depth is not None else MAX_DEPTH
        root_ply = len(board.move_stack)

        for current_depth in range(1, target_depth + 1):
            self.sel_depth = 0

            try:
                score, move = self._search_root(
                    board, legal_moves, current_depth, prev_score, root_hash
                )
            except SearchAborted:
                while len(board.move_stack) > root_ply:
                    board.pop()
                break

            if move is not None:
                best_move = move
                best_score = score
                self.completed_depth = current_depth

            self.time_mgr.extend_if_unstable(prev_best, best_score)
            prev_best = best_score
            prev_score = best_score

            elapsed = max(self.time_mgr.elapsed(), 0.001)
            nps = int(self.nodes / elapsed)
            elapsed_ms = int(elapsed * 1000)
            score_str = self._format_score(best_score)

            iter_info: dict[str, Any] = {
                "depth": current_depth,
                "sel_depth": self.sel_depth,
                "score": best_score,
                "score_str": score_str,
                "nodes": self.nodes,
                "nps": nps,
                "time": elapsed,
                "time_ms": elapsed_ms,
                "pv": best_move,
            }
            if callback is not None:
                callback(iter_info)
            elif verbose:
                print(
                    f"depth {current_depth:>2}/{self.sel_depth:<3} "
                    f"score {score_str:>8} "
                    f"nodes {self.nodes:>9,} "
                    f"nps {nps:>9,} "
                    f"time {elapsed_ms:>6}ms "
                    f"pv {best_move.uci()}"
                )

            if abs(best_score) > MATE_THRESHOLD:
                break
            if (depth is None or movetime_ms is not None) and self.time_mgr.should_stop_iterating():
                break

        self.best_move = best_move
        self.best_score = best_score
        return best_move

    def _search_root(
        self,
        board: chess.Board,
        root_moves: list[chess.Move],
        depth: int,
        prev_score: int,
        root_hash: int,
    ) -> tuple[int, chess.Move | None]:
        if depth < ASP_MIN_DEPTH:
            return self._pvs_root(board, root_moves, depth, -INF, INF, root_hash)

        delta = ASP_INITIAL_DELTA
        alpha = prev_score - delta
        beta = prev_score + delta

        while True:
            score, move = self._pvs_root(board, root_moves, depth, alpha, beta, root_hash)
            if score <= alpha:
                delta *= 2
                alpha = -INF if delta >= ASP_MAX_DELTA else prev_score - delta
                beta = (alpha + beta) // 2 + delta
            elif score >= beta:
                delta *= 2
                beta = INF if delta >= ASP_MAX_DELTA else prev_score + delta
                alpha = (alpha + beta) // 2 - delta
            else:
                return score, move

    def _pvs_root(
        self,
        board: chess.Board,
        root_moves: list[chess.Move],
        depth: int,
        alpha: int,
        beta: int,
        root_hash: int,
    ) -> tuple[int, chess.Move | None]:
        tt_entry = self.tt.probe(root_hash)
        tt_move = tt_entry.best_move if tt_entry is not None else None

        moves = order_moves(
            board,
            root_moves.copy(),
            tt_move=tt_move,
            ply=0,
            killers=self.killers,
            history=self.history,
        )
        if not moves:
            return (-MATE_SCORE if board.is_check() else 0), None

        best_move = moves[0]
        best_score = -INF
        alpha_orig = alpha
        searched_quiets: list[chess.Move] = []

        for i, move in enumerate(moves):
            is_capture = bool(
                board.occupied & chess.BB_SQUARES[move.to_square]
                or (
                    move.to_square == board.ep_square
                    and board.pawns & chess.BB_SQUARES[move.from_square]
                )
            )
            new_hash = push_hash(board, move, root_hash)
            child_in_check = bool(board.checkers_mask())

            if i == 0:
                score = -self._pvs(
                    board, depth - 1, -beta, -alpha, 1, new_hash, False, child_in_check
                )
            else:
                score = -self._pvs(
                    board, depth - 1, -alpha - 1, -alpha, 1, new_hash, False, child_in_check
                )
                if alpha < score < beta:
                    score = -self._pvs(
                        board, depth - 1, -beta, -alpha, 1, new_hash, False, child_in_check
                    )

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if not is_capture and not move.promotion:
                    self.killers.store(0, move)
                    self.history.update(board.turn, move, depth)
                    for prev in searched_quiets:
                        self.history.penalise(board.turn, prev, depth)
                break
            if not is_capture and not move.promotion:
                searched_quiets.append(move)

        bound = (
            Bound.UPPER
            if best_score <= alpha_orig
            else (Bound.LOWER if best_score >= beta else Bound.EXACT)
        )
        self.tt.store(
            root_hash,
            best_move,
            score_to_tt(best_score, 0),
            depth,
            bound,
            evaluate(board),
        )
        return best_score, best_move

    def _pvs(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        current_hash: int,
        null_move_made: bool,
        in_check: bool | None = None,
    ) -> int:
        if self.game_positions.get(current_hash, 0) >= 2:
            return 0
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply, in_check, 0)

        self.nodes += 1
        if self.nodes & (NODE_CHECK_INTERVAL - 1) == 0 and self.time_mgr.is_time_up():
            raise SearchAborted

        if ply > self.sel_depth:
            self.sel_depth = ply

        if self._is_draw(board):
            return 0

        alpha = max(alpha, -(MATE_SCORE - ply))
        beta = min(beta, MATE_SCORE - ply - 1)
        if alpha >= beta:
            return alpha

        is_pv = beta - alpha > 1
        tt_move: chess.Move | None = None
        tt_static_eval: int | None = None

        if self.collect_stats:
            self.tt_probes += 1
        tt_entry = self.tt.probe(current_hash)
        if tt_entry is not None:
            if self.collect_stats:
                self.tt_hits += 1
            tt_move = tt_entry.best_move
            tt_static_eval = tt_entry.static_eval
            if tt_entry.depth >= depth and not is_pv:
                tt_score = score_from_tt(tt_entry.score, ply)
                if (
                    tt_entry.bound == Bound.EXACT
                    or (tt_entry.bound == Bound.LOWER and tt_score >= beta)
                    or (tt_entry.bound == Bound.UPPER and tt_score <= alpha)
                ):
                    if self.collect_stats:
                        self.tt_cutoffs += 1
                    return tt_score

        if in_check is None:
            in_check = bool(board.checkers_mask())
        if in_check:
            static_eval = -INF
        elif tt_static_eval is not None:
            static_eval = tt_static_eval
        else:
            static_eval = evaluate(board)

        if (
            not is_pv
            and not in_check
            and depth <= RFP_MAX_DEPTH
            and static_eval - RFP_MARGIN_PER_DEPTH * depth >= beta
            and abs(beta) < MATE_THRESHOLD
            and next(board.generate_legal_moves(), None) is not None
        ):
            if self.collect_stats:
                self.rfp_prunes += 1
            return static_eval

        if (
            not is_pv
            and not in_check
            and not null_move_made
            and depth >= NMP_MIN_DEPTH
            and static_eval >= beta
            and bool(
                board.occupied_co[board.turn]
                & (board.knights | board.bishops | board.rooks | board.queens)
            )
            and next(board.generate_legal_moves(), None) is not None
        ):
            r = min(3 + depth // 6, depth - 1)
            null_hash = push_hash(board, chess.Move.null(), current_hash)
            null_score = -self._pvs(
                board, depth - 1 - r, -beta, -beta + 1, ply + 1, null_hash, True, False
            )
            board.pop()
            if null_score >= beta:
                if self.collect_stats:
                    self.null_prunes += 1
                return beta if null_score > MATE_THRESHOLD else null_score

        can_futility = (
            not is_pv
            and not in_check
            and depth <= FUTILITY_MAX_DEPTH
            and static_eval + FUTILITY_MARGIN_PER_DEPTH * depth <= alpha
            and abs(alpha) < MATE_THRESHOLD
        )

        alpha_orig = alpha
        if self.collect_stats:
            self.move_generations += 1
        moves = order_moves(
            board,
            list(board.legal_moves),
            tt_move=tt_move,
            ply=ply,
            killers=self.killers,
            history=self.history,
        )
        if not moves:
            return -(MATE_SCORE - ply) if in_check else 0

        best_score = -INF
        best_move: chess.Move | None = None
        searched_quiets: list[chess.Move] = []
        moves_tried = 0

        for move in moves:
            is_capture = bool(
                board.occupied & chess.BB_SQUARES[move.to_square]
                or (
                    move.to_square == board.ep_square
                    and board.pawns & chess.BB_SQUARES[move.from_square]
                )
            )
            is_promotion = move.promotion is not None
            is_tactical = is_capture or is_promotion

            if can_futility and moves_tried > 0 and not is_tactical and not board.gives_check(move):
                if self.collect_stats:
                    self.futility_prunes += 1
                continue

            new_hash = push_hash(board, move, current_hash)
            child_in_check = bool(board.checkers_mask())
            new_depth = depth - 1

            reduction = 0
            if (
                moves_tried >= LMR_MIN_MOVE_INDEX
                and depth >= LMR_MIN_DEPTH
                and not is_tactical
                and not in_check
                and not child_in_check
            ):
                reduction = _LMR_TABLE[min(depth, 63)][min(moves_tried, 63)]
                if is_pv:
                    reduction = max(0, reduction - 1)
                if self.history.get(not board.turn, move) > 0:
                    reduction = max(0, reduction - 1)
                reduction = min(reduction, new_depth - 1)
                if reduction and self.collect_stats:
                    self.lmr_reductions += 1

            if moves_tried == 0:
                score = -self._pvs(
                    board, new_depth, -beta, -alpha, ply + 1, new_hash, False, child_in_check
                )
            else:
                score = -self._pvs(
                    board,
                    new_depth - reduction,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    new_hash,
                    False,
                    child_in_check,
                )
                if reduction > 0 and score > alpha:
                    score = -self._pvs(
                        board,
                        new_depth,
                        -alpha - 1,
                        -alpha,
                        ply + 1,
                        new_hash,
                        False,
                        child_in_check,
                    )
                if is_pv and alpha < score < beta:
                    score = -self._pvs(
                        board,
                        new_depth,
                        -beta,
                        -alpha,
                        ply + 1,
                        new_hash,
                        False,
                        child_in_check,
                    )

            board.pop()
            moves_tried += 1

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if self.collect_stats:
                    self.beta_cutoffs += 1
                if not is_tactical:
                    self.killers.store(ply, move)
                    self.history.update(board.turn, move, depth)
                    for prev in searched_quiets:
                        self.history.penalise(board.turn, prev, depth)
                break
            if not is_tactical:
                searched_quiets.append(move)

        bound = (
            Bound.UPPER
            if best_score <= alpha_orig
            else (Bound.LOWER if best_score >= beta else Bound.EXACT)
        )
        self.tt.store(
            current_hash,
            best_move,
            score_to_tt(best_score, ply),
            depth,
            bound,
            static_eval if static_eval != -INF else 0,
        )
        return best_score

    def _quiescence(
        self,
        board: chess.Board,
        alpha: int,
        beta: int,
        ply: int,
        in_check: bool | None = None,
        qply: int = 0,
    ) -> int:
        self.nodes += 1
        if self.collect_stats:
            self.qnodes += 1
        if self.nodes & (NODE_CHECK_INTERVAL - 1) == 0 and self.time_mgr.is_time_up():
            raise SearchAborted

        if ply > self.sel_depth:
            self.sel_depth = ply
        if in_check is None:
            in_check = bool(board.checkers_mask())
        if self._is_draw(board):
            return 0

        alpha = max(alpha, -(MATE_SCORE - ply))
        beta = min(beta, MATE_SCORE - ply - 1)
        if alpha >= beta:
            return alpha

        if qply >= MAX_QUIESCENCE_DEPTH and not in_check:
            return evaluate(board)

        if in_check:
            if self.collect_stats:
                self.move_generations += 1
            moves = list(board.legal_moves)
            if not moves:
                return -(MATE_SCORE - ply)
            occupied = board.occupied
            ep_square = board.ep_square
            pawns = board.pawns
            moves.sort(
                key=lambda move: bool(
                    occupied & chess.BB_SQUARES[move.to_square]
                    or (move.to_square == ep_square and pawns & chess.BB_SQUARES[move.from_square])
                ),
                reverse=True,
            )
            best_score = -INF
            for move in moves:
                board.push(move)
                score = -self._quiescence(board, -beta, -alpha, ply + 1, None, qply + 1)
                board.pop()
                if score > best_score:
                    best_score = score
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
            return best_score

        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        best_score = stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        candidates: list[tuple[int, chess.Move]] = []
        if self.collect_stats:
            self.move_generations += 1
        for move in board.generate_legal_captures():
            to_sq = move.to_square
            victim = (
                chess.PAWN
                if to_sq == board.ep_square
                else (board.piece_type_at(to_sq) or chess.PAWN)
            )
            if not move.promotion and (stand_pat + PIECE_VALUES[victim] + DELTA_PRUNING < alpha):
                continue
            attacker = board.piece_type_at(move.from_square) or chess.PAWN
            score = MVV_LVA[victim][attacker]
            if move.promotion == chess.QUEEN:
                score += 10_000
            candidates.append((score, move))

        turn = board.turn
        promo_rank = chess.BB_RANK_7 if turn == chess.WHITE else chess.BB_RANK_2
        pawn_promos = board.pawns & board.occupied_co[turn] & promo_rank
        if pawn_promos:
            step = 8 if turn == chess.WHITE else -8
            while pawn_promos:
                from_sq = (pawn_promos & -pawn_promos).bit_length() - 1
                to_sq = from_sq + step
                if board.piece_at(to_sq) is None:
                    m = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                    if board.is_legal(m):
                        candidates.append((10_000, m))
                pawn_promos &= pawn_promos - 1

        if not candidates:
            return 0 if next(board.generate_legal_moves(), None) is None else best_score

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, move in candidates:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1, None, qply + 1)
            board.pop()

            if score > best_score:
                best_score = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        return best_score

    @staticmethod
    def _is_draw(board: chess.Board) -> bool:
        if board.halfmove_clock >= 100 and board.is_fifty_moves():
            return True
        if board.halfmove_clock >= 4 and len(board.move_stack) >= 4 and board.is_repetition(2):
            return True
        return not (board.pawns | board.rooks | board.queens) and board.is_insufficient_material()

    @staticmethod
    def _format_score(score: int) -> str:
        if abs(score) > MATE_THRESHOLD:
            plies = MATE_SCORE - abs(score)
            mate_in = (plies + 1) // 2
            return f"mate {mate_in}" if score > 0 else f"mate -{mate_in}"
        return f"cp {score}"
