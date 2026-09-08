import math
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

RNG = random.Random(os.environ.get("HARNESS_SEED", "69"))  # Not-so constant RNG constnat
MATE_SCORE = int(1e6)
DELTA_PRUNING: Final[int] = 2 * PIECE_VALUES[chess.PAWN]
MAX_QUIECENCE_DEPTH: Final[int] = 16

class Bot:
    def __init__(self) -> None:
        self.nodes_visited = 0

    def quiescence(
        self,
        board: chess.Board,
        alpha: float,
        beta: float,
        ply: int,
        quicience_ply: int
    ) -> float:
        self.nodes_visited += 1

        if quicience_ply >= MAX_QUIECENCE_DEPTH:
            return Evaluator.evaluate(board)

        if board.is_check():
            moves = order_moves(board)
            if not moves:
                return -(MATE_SCORE - ply)

            best_score = -math.inf
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
        alpha: float,
        beta: float,
        ply: int,
    ) -> float:
        self.nodes_visited += 1

        if board.halfmove_clock >= 100:
            return 0  # 50 move rule
        if chess.popcount(board.occupied) <= 4: # only bother checking insufficient if low material count
            if board.is_insufficient_material():
                return 0  # insufficient material
        if depth <= 0:
            return self.quiescence(board, alpha, beta, ply, quicience_ply = 0)

        moves = order_moves(board)

        if not moves:  # we are being checkmated! (or drawing)
            return -(MATE_SCORE - ply) if board.is_check() else 0

        best_score = -math.inf
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > best_score:
                best_score = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        return best_score

    def get_best_move(self, board: chess.Board, depth: int) -> chess.Move:
        alpha, beta = -math.inf, math.inf
        best_moves: list[chess.Move] = []
        best_score = -math.inf

        moves = order_moves(board)

        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply=1)
            board.pop()

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

            alpha = max(alpha, score)

        return RNG.choice(best_moves)
