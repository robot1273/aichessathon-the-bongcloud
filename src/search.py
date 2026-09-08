from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import Any, Final

import chess

from .evaluation import Evaluator
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
NODE_CHECK_INTERVAL: Final[int] = 2048

_LMR_TABLE: Final[list[list[int]]] = [
    [
        max(0, int(1.0 + math.log(d) * math.log(m) / 2.0)) if d > 0 and m > 0 else 0
        for m in range(64)
    ]
    for d in range(64)
]


class SearchAborted(Exception):
    pass


class Bot:
    def __init__(self, tt_exp_size: int = 22) -> None:
        self.tt = TT(exp_size=tt_exp_size)
        self.time_mgr = TimeManager()
        self.killers = KillerTable()
        self.history = HistoryTable()
        self.nodes: int = 0
        self.iter_nodes: int = 0
        self.sel_depth: int = 0
        self.completed_depth: int = 0
        self.best_move: chess.Move | None = None
        self.best_score: int = 0
        self.iterations: list[dict[str, Any]] = []

    @property
    def nodes_visited(self) -> int:
        return self.nodes

    @nodes_visited.setter
    def nodes_visited(self, value: int) -> None:
        self.nodes = value

    def get_best_move(
        self,
        board: chess.Board,
        time_left_ms: int = 100_000,
        depth: int | None = None,
        movetime_ms: int | None = None,
        verbose: bool = True,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> chess.Move:
        self.time_mgr.start(time_left_ms, board, movetime_ms=movetime_ms)
        self.tt.new_search()
        self.killers.clear()

        root_hash = calculate_hash(board)
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return chess.Move.null()
        if len(legal_moves) == 1:
            return legal_moves[0]

        best_move = legal_moves[0]
        best_score = -INF
        prev_score = 0
        prev_best: int | None = None
        target_depth = depth if depth is not None else MAX_DEPTH

        self.nodes = 0
        self.completed_depth = 0
        self.iterations = []

        for current_depth in range(1, target_depth + 1):
            self.iter_nodes = 0
            self.sel_depth = 0

            try:
                score, move = self._search_root(
                    board, current_depth, prev_score, root_hash
                )
            except SearchAborted:
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
                "iter_nodes": self.iter_nodes,
                "nps": nps,
                "time": elapsed,
                "time_ms": elapsed_ms,
                "pv": best_move,
            }
            self.iterations.append(iter_info)

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
        depth: int,
        prev_score: int,
        root_hash: int,
    ) -> tuple[int, chess.Move | None]:
        if depth < ASP_MIN_DEPTH:
            return self._pvs_root(board, depth, -INF, INF, root_hash)

        delta = ASP_INITIAL_DELTA
        alpha = prev_score - delta
        beta = prev_score + delta

        while True:
            score, move = self._pvs_root(board, depth, alpha, beta, root_hash)
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
        depth: int,
        alpha: int,
        beta: int,
        root_hash: int,
    ) -> tuple[int, chess.Move | None]:
        tt_entry = self.tt.probe(root_hash)
        tt_move = tt_entry.best_move if tt_entry is not None else None

        moves = order_moves(
            board,
            tt_move=tt_move,
            ply=0,
            killers=self.killers,
            history=self.history,
        )
        if not moves:
            return (-MATE_SCORE if board.is_check() else 0), None

        best_move = moves[0]
        best_score = -INF
        searched_quiets: list[chess.Move] = []

        for i, move in enumerate(moves):
            is_capture = board.is_capture(move)
            new_hash = push_hash(board, move, root_hash)

            if i == 0:
                score = -self._pvs(board, depth - 1, -beta, -alpha, 1, new_hash, False)
            else:
                score = -self._pvs(
                    board, depth - 1, -alpha - 1, -alpha, 1, new_hash, False
                )
                if alpha < score < beta:
                    score = -self._pvs(
                        board, depth - 1, -beta, -alpha, 1, new_hash, False
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
            if best_score <= alpha
            else (Bound.LOWER if best_score >= beta else Bound.EXACT)
        )
        self.tt.store(
            root_hash,
            best_move,
            score_to_tt(best_score, 0),
            depth,
            bound,
            Evaluator.evaluate(board),
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
    ) -> int:
        self.nodes += 1
        self.iter_nodes += 1
        if self.nodes & (NODE_CHECK_INTERVAL - 1) == 0 and self.time_mgr.is_time_up():
            raise SearchAborted

        if ply > self.sel_depth:
            self.sel_depth = ply

        if ply > 0:
            if board.is_fifty_moves():
                return 0
            if board.is_repetition(2):
                return 0
            if board.is_insufficient_material():
                return 0

        alpha = max(alpha, -(MATE_SCORE - ply))
        beta = min(beta, MATE_SCORE - ply - 1)
        if alpha >= beta:
            return alpha

        is_pv = beta - alpha > 1
        tt_move: chess.Move | None = None
        tt_static_eval: int | None = None

        tt_entry = self.tt.probe(current_hash)
        if tt_entry is not None:
            tt_move = tt_entry.best_move
            tt_static_eval = tt_entry.static_eval
            if tt_entry.depth >= depth and not is_pv:
                tt_score = score_from_tt(tt_entry.score, ply)
                if (
                    tt_entry.bound == Bound.EXACT
                    or (tt_entry.bound == Bound.LOWER and tt_score >= beta)
                    or (tt_entry.bound == Bound.UPPER and tt_score <= alpha)
                ):
                    return tt_score

        in_check = board.is_check()
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply)

        if in_check:
            static_eval = -INF
        elif tt_static_eval is not None:
            static_eval = tt_static_eval
        else:
            static_eval = Evaluator.evaluate(board)

        if (
            not is_pv
            and not in_check
            and depth <= RFP_MAX_DEPTH
            and static_eval - RFP_MARGIN_PER_DEPTH * depth >= beta
            and abs(beta) < MATE_THRESHOLD
        ):
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
        ):
            r = min(3 + depth // 6, depth - 1)
            board.push(chess.Move.null())
            null_hash = calculate_hash(board)
            null_score = -self._pvs(
                board, depth - 1 - r, -beta, -beta + 1, ply + 1, null_hash, True
            )
            board.pop()
            if null_score >= beta:
                return beta if null_score > MATE_THRESHOLD else null_score

        can_futility = (
            not is_pv
            and not in_check
            and depth <= FUTILITY_MAX_DEPTH
            and static_eval + FUTILITY_MARGIN_PER_DEPTH * depth <= alpha
            and abs(alpha) < MATE_THRESHOLD
        )

        alpha_orig = alpha
        best_score = -INF
        best_move: chess.Move | None = None
        searched_quiets: list[chess.Move] = []
        moves_tried = 0

        opp_king = board.king(not board.turn)
        opp_king_bb = chess.BB_SQUARES[opp_king] if opp_king is not None else 0

        k1, k2 = self.killers.get(ply)
        seen_moves: set[chess.Move] = set()

        if tt_move is not None and board.is_legal(tt_move):
            seen_moves.add(tt_move)
            is_capture = board.is_capture(tt_move)
            is_tactical = is_capture or (tt_move.promotion is not None)

            new_hash = push_hash(board, tt_move, current_hash)
            child_in_check = board.is_check()
            ext = 1 if child_in_check else 0
            new_depth = depth - 1 + ext

            score = -self._pvs(
                board, new_depth, -beta, -alpha, ply + 1, new_hash, False
            )
            board.pop()

            moves_tried = 1
            best_move = tt_move
            best_score = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if not is_tactical:
                    self.killers.store(ply, tt_move)
                    self.history.update(board.turn, tt_move, depth)
                self.tt.store(
                    current_hash,
                    best_move,
                    score_to_tt(best_score, ply),
                    depth,
                    Bound.LOWER,
                    static_eval if static_eval != -INF else 0,
                )
                return best_score
            if not is_tactical:
                searched_quiets.append(tt_move)

        def move_generator() -> Iterator[chess.Move]:
            captures: list[tuple[int, chess.Move]] = []
            for m in board.generate_legal_captures():
                if m in seen_moves:
                    continue
                seen_moves.add(m)
                victim = (
                    chess.PAWN
                    if m.to_square == board.ep_square
                    else (board.piece_type_at(m.to_square) or chess.PAWN)
                )
                attacker = board.piece_type_at(m.from_square) or 1
                captures.append((MVV_LVA[victim][attacker], m))

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
                        if m not in seen_moves and board.is_legal(m):
                            seen_moves.add(m)
                            captures.append((20000, m))
                    pawn_promos &= pawn_promos - 1

            captures.sort(key=lambda x: x[0], reverse=True)
            for _, m in captures:
                yield m

            if k1 is not None and k1 not in seen_moves and board.is_legal(k1):
                seen_moves.add(k1)
                yield k1
            if k2 is not None and k2 not in seen_moves and board.is_legal(k2):
                seen_moves.add(k2)
                yield k2

            quiets: list[tuple[int, chess.Move]] = []
            for m in board.generate_legal_moves():
                if m in seen_moves:
                    continue
                seen_moves.add(m)
                quiets.append((self.history.get(board.turn, m), m))
            quiets.sort(key=lambda x: x[0], reverse=True)
            for _, m in quiets:
                yield m

        has_any_move = moves_tried > 0

        for move in move_generator():
            has_any_move = True
            is_capture = board.is_capture(move)
            is_promotion = move.promotion is not None
            is_tactical = is_capture or is_promotion

            if (
                can_futility
                and moves_tried > 0
                and not is_tactical
                and not (board.attacks_mask(move.to_square) & opp_king_bb)
            ):
                continue

            new_hash = push_hash(board, move, current_hash)
            child_in_check = board.is_check()
            ext = 1 if child_in_check else 0
            new_depth = depth - 1 + ext

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
                if self.history.get(board.turn, move) > 0:
                    reduction = max(0, reduction - 1)
                reduction = min(reduction, new_depth - 1)

            if moves_tried == 0:
                score = -self._pvs(
                    board, new_depth, -beta, -alpha, ply + 1, new_hash, False
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
                    )
                if is_pv and alpha < score < beta:
                    score = -self._pvs(
                        board, new_depth, -beta, -alpha, ply + 1, new_hash, False
                    )

            board.pop()
            moves_tried += 1

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if not is_tactical:
                    self.killers.store(ply, move)
                    self.history.update(board.turn, move, depth)
                    for prev in searched_quiets:
                        self.history.penalise(board.turn, prev, depth)
                break
            if not is_tactical:
                searched_quiets.append(move)

        if not has_any_move:
            return -(MATE_SCORE - ply) if in_check else 0

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
        self, board: chess.Board, alpha: int, beta: int, ply: int
    ) -> int:
        self.nodes += 1
        self.iter_nodes += 1
        if self.nodes & (NODE_CHECK_INTERVAL - 1) == 0 and self.time_mgr.is_time_up():
            raise SearchAborted

        if ply > self.sel_depth:
            self.sel_depth = ply
        if ply >= MAX_DEPTH:
            return Evaluator.evaluate(board)

        if board.is_check():
            moves = list(board.legal_moves)
            if not moves:
                return -(MATE_SCORE - ply)
            moves.sort(key=lambda m: 1 if board.is_capture(m) else 0, reverse=True)
            best_score = -INF
            for move in moves:
                board.push(move)
                score = -self._quiescence(board, -beta, -alpha, ply + 1)
                board.pop()
                if score > best_score:
                    best_score = score
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
            return best_score

        stand_pat = Evaluator.evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        best_score = stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        candidates: list[tuple[int, chess.Move]] = []
        for move in board.generate_legal_captures():
            to_sq = move.to_square
            victim = (
                chess.PAWN
                if to_sq == board.ep_square
                else (board.piece_type_at(to_sq) or chess.PAWN)
            )
            if not move.promotion and (
                stand_pat + PIECE_VALUES[victim] + DELTA_PRUNING < alpha
            ):
                continue
            attacker = board.piece_type_at(move.from_square) or 1
            score = MVV_LVA[victim][attacker]
            if move.promotion == chess.QUEEN:
                score += 10000
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
                        candidates.append((10000, m))
                pawn_promos &= pawn_promos - 1

        if not candidates:
            return best_score

        candidates.sort(key=lambda x: x[0], reverse=True)

        for _, move in candidates:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, ply + 1)
            board.pop()

            if score > best_score:
                best_score = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        return best_score

    @staticmethod
    def _format_score(score: int) -> str:
        if abs(score) > MATE_THRESHOLD:
            plies = MATE_SCORE - abs(score)
            mate_in = (plies + 1) // 2
            return f"mate {mate_in}" if score > 0 else f"mate -{mate_in}"
        return f"cp {score}"
