import math

import chess

from .constants import MATE_SCORE, RNG
from .evaluation import Evaluator
from .move_ordering import order_moves

class Bot:
    def __init__(self) -> None:
        self.nodes_visited = 0

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
    ) -> float:
        self.nodes_visited += 1

        if board.halfmove_clock >= 100:      return 0  # 50 move rule
        if board.is_insufficient_material(): return 0  # insufficient material
        if depth <= 0:                       return Evaluator.evaluate(board) # root evaluation call

        moves = order_moves(board)

        if not moves: # we are being checkmated! (or drawing)
            return -(MATE_SCORE - ply) if board.is_check() else 0

        best_score = -math.inf
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > best_score: best_score = score
            if score > alpha: alpha = score
            if alpha >= beta: break

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
