import chess

from numba import njit

from .types import BoardArray, Bitboard
from .constants import PIECE_VALUES, MOBILITY_WEIGHT
from .helpers import encode_board

# We move this aspect of the evaluation to a seperate kernel function to call njit on it
# Moving forward, _evaluate_kernel will be a numba compiled function and needs warming up
@njit(cache=False)
def _evaluate_kernel(pieces: BoardArray, player_mask: Bitboard, legal_move_count: int) -> int:
    """Naiive evluation of pieces and piece mobility for the player masked by the player_mask
    Returns positive for the current player (+ does not mean white or black specifically, depends on turn)"""
    material = 0
    for square in range(64):
        piece_type = pieces[square]
        if piece_type == 0:
            continue
        val = PIECE_VALUES[piece_type]
        material += val if player_mask[square] else -val
    return int(material + MOBILITY_WEIGHT * legal_move_count)

class Evaluator:
    """Main evaluation class"""
    _kernel = staticmethod(_evaluate_kernel)

    @classmethod
    def evaluate(cls, board: chess.Board, legal_moves: list[chess.Move]) -> float:
        """Naiive evaluation"""
        pieces, mine = encode_board(board)
        return float(cls._kernel(pieces, mine, len(legal_moves)))

    @classmethod
    def warmup_evaluator(cls) -> None:
        """Warms up the evaluation by evaluating the start chess position"""
        pieces, mine = encode_board(chess.Board())
        cls._kernel(pieces, mine, 20)
