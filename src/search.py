from __future__ import annotations

import os
import random
from typing import Final

import chess

from .evaluation import Evaluator
from .move_ordering import (
    PIECE_VALUES,
    generate_quiescence_moves,
    order_moves,
)
from .tt import TT, Bound, score_from_tt, score_to_tt
from .zobrist import calculate_hash, push_hash

RNG = random.Random(os.environ.get("HARNESS_SEED", "69"))  # Not-so constant RNG constnat
MATE_SCORE: Final[int] = 1_000_000
INF: Final[int] = MATE_SCORE + 1
DELTA_PRUNING: Final[int] = 2 * PIECE_VALUES[chess.PAWN]
MAX_QUIECENCE_DEPTH: Final[int] = 16


class Bot:
    def __init__(self, tt_exp_size: int = 20) -> None:
        self.nodes_visited = 0
        self.tt = TT(exp_size=tt_exp_size)

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
        self.nodes_visited += 1

        if quicience_ply >= MAX_QUIECENCE_DEPTH:
            return Evaluator.evaluate(board)

        if board.is_check():
            moves = order_moves(board)
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

        for move in order_moves(board, candidates):
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

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        ply: int,
        current_hash: int | None = None,
    ) -> int:
        self.nodes_visited += 1

        if current_hash is None:
            current_hash = calculate_hash(board)

        if board.halfmove_clock >= 100:
            return 0  # 50 move rule
        if chess.popcount(board.occupied) <= 4 and board.is_insufficient_material():
            return 0  # insufficient material

        alpha_orig = alpha
        is_pv_node = (beta - alpha) > 1
        tt_move: chess.Move | None = None

        tt_entry = self.tt.probe(current_hash)
        if tt_entry is not None:
            tt_move = tt_entry.best_move

            if tt_entry.depth >= depth and not is_pv_node:
                tt_score = score_from_tt(tt_entry.score, ply)

                if (
                    (tt_entry.bound == Bound.EXACT)
                    or (tt_entry.bound == Bound.LOWER and tt_score >= beta)
                    or (tt_entry.bound == Bound.UPPER and tt_score <= alpha)
                ):
                    return tt_score

        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply, quicience_ply=0)

        best_score = -INF
        best_move = tt_move

        # early TT move test: try tt_move before generating and ordering all other moves!
        # in cut-nodes, tt_move causes an immediate beta cutoff, avoiding move generation/sorting.
        if tt_move is not None and board.is_legal(tt_move):
            new_hash = push_hash(board, tt_move, current_hash)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, current_hash=new_hash)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = tt_move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                tt_score = score_to_tt(best_score, ply)
                self.tt.store(current_hash, best_move, tt_score, depth, Bound.LOWER)
                return best_score

        moves = order_moves(board)
        if not moves:  # we are being checkmated! (or drawing)
            return -(MATE_SCORE - ply) if board.is_check() else 0

        if best_move is None:
            best_move = moves[0]

        for move in moves:
            if move == tt_move:
                continue

            new_hash = push_hash(board, move, current_hash)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1, current_hash=new_hash)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        if best_score <= alpha_orig:
            bound = Bound.UPPER
        elif best_score >= beta:
            bound = Bound.LOWER
        else:
            bound = Bound.EXACT

        tt_score = score_to_tt(best_score, ply)
        self.tt.store(current_hash, best_move, tt_score, depth, bound)

        return best_score

    def get_best_move(self, board: chess.Board, depth: int) -> chess.Move:
        self.tt.new_search()
        root_hash = calculate_hash(board)

        alpha, beta = -INF, INF
        best_moves: list[chess.Move] = []
        best_score = -INF

        tt_entry = self.tt.probe(root_hash)
        tt_move = tt_entry.best_move if tt_entry is not None else None

        moves = order_moves(board, tt_move=tt_move)
        if not moves:
            return chess.Move.null()

        for move in moves:
            new_hash = push_hash(board, move, root_hash)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply=1, current_hash=new_hash)
            board.pop()

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

            alpha = max(alpha, score)

        chosen_move = RNG.choice(best_moves)
        self.tt.store(
            root_hash,
            chosen_move,
            score_to_tt(best_score, ply=0),
            depth,
            Bound.EXACT,
        )
        return chosen_move
