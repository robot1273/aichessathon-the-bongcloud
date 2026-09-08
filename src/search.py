from __future__ import annotations

from typing import Final

import chess

from .evaluation import Evaluator
from .move_ordering import (
    PIECE_VALUES,
    HistoryTable,
    KillerTable,
    generate_quiescence_moves,
    order_moves,
)
from .time_manager import TimeManager
from .tt import TT, Bound, score_from_tt, score_to_tt
from .zobrist import calculate_hash, push_hash

MATE_SCORE: Final[int] = 1_000_000
MATE_THRESHOLD: Final[int] = 900_000
INF: Final[int] = MATE_SCORE + 1
DELTA_PRUNING: Final[int] = 2 * PIECE_VALUES[chess.PAWN]
MAX_QUIECENCE_DEPTH: Final[int] = 16
MAX_DEPTH: Final[int] = 100
NODE_CHECK_INTERVAL: Final[int] = 2048

ASP_INITIAL_DELTA: Final[int] = 25
ASP_MAX_DELTA: Final[int] = 500
ASP_MIN_DEPTH: Final[int] = 5


class SearchAborted(Exception):
    pass


class Bot:
    def __init__(self, tt_exp_size: int = 22) -> None:
        self.tt = TT(exp_size=tt_exp_size)
        self.time_mgr = TimeManager()
        self.killers = KillerTable()
        self.history = HistoryTable()
        self.nodes: int = 0
        self.sel_depth: int = 0

    @property
    def nodes_visited(self) -> int:
        return self.nodes

    @nodes_visited.setter
    def nodes_visited(self, value: int) -> None:
        self.nodes = value

    def clear_tt(self) -> None:
        self.tt.clear()

    def quiescence(
        self,
        board: chess.Board,
        alpha: int,
        beta: int,
        ply: int,
        quicience_ply: int,
    ) -> int:
        self.nodes += 1
        if ply > self.sel_depth:
            self.sel_depth = ply

        if quicience_ply >= MAX_QUIECENCE_DEPTH:
            return Evaluator.evaluate(board)

        if board.is_check():
            moves = order_moves(
                board,
                ply=ply,
                killers=self.killers,
                history=self.history,
            )
            if not moves:
                return -(MATE_SCORE - ply)

            best_score = -INF
            for move in moves:
                board.push(move)
                score = -self.quiescence(board, -beta, -alpha, ply + 1, quicience_ply + 1)
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

        candidates = []
        for move in generate_quiescence_moves(board):
            if not move.promotion:
                to_sq = move.to_square
                victim = (
                    chess.PAWN if to_sq == board.ep_square else (board.piece_type_at(to_sq) or 1)
                )
                gain = PIECE_VALUES[victim]
                # skip move if we can't improve the score enough to beat alpha
                if stand_pat + gain + DELTA_PRUNING < alpha:
                    continue
            candidates.append(move)

        for move in order_moves(
            board,
            candidates,
            ply=ply,
            killers=self.killers,
            history=self.history,
        ):
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1, quicience_ply + 1)
            board.pop()

            if score > best_score:
                best_score = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        return best_score

    def _pvs(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        current_hash: int,
        null_move_made: bool = False,
    ) -> int:
        self.nodes += 1
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

        tt_entry = self.tt.probe(current_hash)
        if tt_entry is not None:
            tt_move = tt_entry.best_move

            if tt_entry.depth >= depth and not is_pv:
                tt_score = score_from_tt(tt_entry.score, ply)

                if (
                    (tt_entry.bound == Bound.EXACT)
                    or (tt_entry.bound == Bound.LOWER and tt_score >= beta)
                    or (tt_entry.bound == Bound.UPPER and tt_score <= alpha)
                ):
                    return tt_score

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply, quicience_ply=0)

        alpha_orig = alpha
        best_score = -INF
        best_move = tt_move
        moves_tried = 0
        searched_quiets: list[chess.Move] = []

        # Early TT move test
        if tt_move is not None and board.is_legal(tt_move):
            is_capture = board.is_capture(tt_move)
            is_tactical = is_capture or (tt_move.promotion is not None)

            new_hash = push_hash(board, tt_move, current_hash)
            score = -self._pvs(board, depth - 1, -beta, -alpha, ply + 1, new_hash, False)
            board.pop()

            moves_tried = 1
            best_score = score
            best_move = tt_move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if not is_tactical:
                    self.killers.store(ply, tt_move)
                    self.history.update(board.turn, tt_move, depth)
                tt_score = score_to_tt(best_score, ply)
                self.tt.store(
                    current_hash,
                    best_move,
                    tt_score,
                    depth,
                    Bound.LOWER,
                    Evaluator.evaluate(board),
                )
                return best_score
            if not is_tactical:
                searched_quiets.append(tt_move)

        moves = order_moves(
            board,
            tt_move=tt_move,
            ply=ply,
            killers=self.killers,
            history=self.history,
        )
        if not moves and moves_tried == 0:
            return -(MATE_SCORE - ply) if board.is_check() else 0

        for move in moves:
            if move == tt_move:
                continue

            is_capture = board.is_capture(move)
            is_tactical = is_capture or (move.promotion is not None)

            new_hash = push_hash(board, move, current_hash)
            if moves_tried == 0:
                score = -self._pvs(board, depth - 1, -beta, -alpha, ply + 1, new_hash, False)
            else:
                score = -self._pvs(board, depth - 1, -alpha - 1, -alpha, ply + 1, new_hash, False)
                if is_pv and alpha < score < beta:
                    score = -self._pvs(board, depth - 1, -beta, -alpha, ply + 1, new_hash, False)
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

        bound = (
            Bound.UPPER
            if best_score <= alpha_orig
            else (Bound.LOWER if best_score >= beta else Bound.EXACT)
        )

        tt_score = score_to_tt(best_score, ply)
        self.tt.store(
            current_hash,
            best_move,
            tt_score,
            depth,
            bound,
            Evaluator.evaluate(board),
        )

        return best_score

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

    def get_best_move(
        self,
        board: chess.Board,
        time_left_ms: int = 100_000,
        depth: int | None = None,
    ) -> chess.Move:
        self.time_mgr.start(time_left_ms, board)
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

        for current_depth in range(1, target_depth + 1):
            self.nodes = 0
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
            self.time_mgr.extend_if_unstable(prev_best, best_score)
            prev_best = best_score
            prev_score = best_score

            elapsed = max(self.time_mgr.elapsed(), 0.001)
            nps = int(self.nodes / elapsed)
            elapsed_ms = int(elapsed * 1000)
            score_str = self._format_score(best_score)
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
            if depth is None and self.time_mgr.should_stop_iterating():
                break

        return best_move

    @staticmethod
    def _format_score(score: int) -> str:
        if abs(score) > MATE_THRESHOLD:
            plies = MATE_SCORE - abs(score)
            mate_in = (plies + 1) // 2
            return f"mate {mate_in}" if score > 0 else f"mate -{mate_in}"
        return f"cp {score}"
