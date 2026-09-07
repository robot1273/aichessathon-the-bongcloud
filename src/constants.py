import numpy as np
import os
import random

RNG = random.Random(os.environ.get("HARNESS_SEED", "69")) # Not-so constant RNG constnat

PIECE_VALUES: np.ndarray = np.array([0, 100, 320, 330, 500, 900, 0], dtype=np.int32)
MOBILITY_WEIGHT: int = 4
MATE_SCORE: float = 1e6
