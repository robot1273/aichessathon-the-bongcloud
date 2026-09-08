import chess

from numba import njit

from .types import BoardArray, Bitboard
from .constants import PIECE_VALUES, MOBILITY_WEIGHT
from .helpers import encode_board

# We move this aspect of the evaluation to a seperate kernel function to call njit on it
# Moving forward, _evaluate_kernel will be a numba compiled function and needs warming up
@njit(fastmath=True, cache=False)
def _evaluate_kernel(pieces: BoardArray, player_mask: Bitboard) -> int:
    """Naiive evluation of pieces and piece mobility for the player masked by the player_mask
    Returns positive for the current player (+ does not mean white or black specifically, depends on turn)"""
    material = 0
    active_squares = pieces.nonzero()[0] # iterate over all non-empty squares
    for square in active_squares:
        piece_type = pieces[square]
        val = PIECE_VALUES[piece_type]
        material += val if player_mask[square] else -val

    return int(material)

class Evaluator:
    """Main evaluation class"""
    _kernel = staticmethod(_evaluate_kernel)

    @classmethod
    def evaluate(cls, board: chess.Board) -> float:
        """Naiive evaluation, no mobility as it is incredibly slow"""
        pieces, mine = encode_board(board)
        return float(cls._kernel(pieces, mine))

    @classmethod
    def warmup_evaluator(cls) -> None:
        """Warms up the evaluation by evaluating the start chess position"""
        pieces, mine = encode_board(chess.Board())
        cls._kernel(pieces, mine)

Evaluator.warmup_evaluator()
