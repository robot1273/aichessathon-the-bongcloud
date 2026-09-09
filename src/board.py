"""Custom Bitboard Chess Board with legal-only movegen and incremental evaluation.

Ported from stargaze and bobot-chess C++ engines.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from src.attacks import (
    BISHOP_MASKS,
    KING_ATTACKS,
    KNIGHT_ATTACKS,
    PAWN_ATTACKS,
    RAY_BETWEEN,
    ROOK_MASKS,
    bishop_attacks,
    queen_attacks,
    rook_attacks,
)
from src.evaluation import (
    ADJACENT_FILE_MASKS,
    EG_TABLE,
    FILE_MASKS,
    FORWARD_FILE_MASKS,
    GAMEPHASE_INC,
    GAMEPHASE_SUM,
    MG_TABLE,
    PASSED_PAWN_EG,
    PASSED_PAWN_MASKS,
    PASSED_PAWN_MG,
)
from src.zobrist import (
    CASTLING_TABLE,
    EP_KEYS,
    PIECE_KEYS,
    TURN_KEY,
    calculate_hash,
    has_legal_en_passant,
)

# ---------------------------------------------------------------------------
# Constants: Pieces and Colours
# ---------------------------------------------------------------------------

PAWN: Final[int] = 0
KNIGHT: Final[int] = 1
BISHOP: Final[int] = 2
ROOK: Final[int] = 3
QUEEN: Final[int] = 4
KING: Final[int] = 5

WHITE: Final[int] = 0
BLACK: Final[int] = 1

_REL_RANK: Final[tuple[tuple[int, ...], ...]] = tuple(
    tuple(sq >> 3 if c == WHITE else 7 - (sq >> 3) for sq in range(64)) for c in range(2)
)
_PASSED_MG: Final[tuple[tuple[int, ...], ...]] = tuple(
    tuple(PASSED_PAWN_MG[_REL_RANK[c][sq]] for sq in range(64)) for c in range(2)
)
_PASSED_EG: Final[tuple[tuple[int, ...], ...]] = tuple(
    tuple(PASSED_PAWN_EG[_REL_RANK[c][sq]] for sq in range(64)) for c in range(2)
)
_SQ_FILE: Final[tuple[int, ...]] = tuple(sq & 7 for sq in range(64))
_SQ_FILE_MASK: Final[tuple[int, ...]] = tuple(int(FILE_MASKS[sq & 7]) for sq in range(64))

# ---------------------------------------------------------------------------
# Move Flags (16-bit encoding matching stargaze)
# bits 0-5: from_square
# bits 6-11: to_square
# bits 12-15: flags
# ---------------------------------------------------------------------------

QUIET: Final[int] = 0b0000
DOUBLE_PAWN_PUSH: Final[int] = 0b0001
KING_CASTLE: Final[int] = 0b0010
QUEEN_CASTLE: Final[int] = 0b0011
CAPTURE: Final[int] = 0b0100
EN_PASSANT: Final[int] = 0b0101
KNIGHT_PROMO: Final[int] = 0b1000
BISHOP_PROMO: Final[int] = 0b1001
ROOK_PROMO: Final[int] = 0b1010
QUEEN_PROMO: Final[int] = 0b1011
KNIGHT_PROMO_CAPTURE: Final[int] = 0b1100
BISHOP_PROMO_CAPTURE: Final[int] = 0b1101
ROOK_PROMO_CAPTURE: Final[int] = 0b1110
QUEEN_PROMO_CAPTURE: Final[int] = 0b1111

PROMOTION_PIECES: Final[tuple[int, ...]] = (
    KNIGHT_PROMO,
    BISHOP_PROMO,
    ROOK_PROMO,
    QUEEN_PROMO,
)
PROMOTION_CAPTURE_PIECES: Final[tuple[int, ...]] = (
    KNIGHT_PROMO_CAPTURE,
    BISHOP_PROMO_CAPTURE,
    ROOK_PROMO_CAPTURE,
    QUEEN_PROMO_CAPTURE,
)

SQUARE_NAMES: Final[tuple[str, ...]] = tuple(
    f"{chr(ord('a') + (sq & 7))}{1 + (sq >> 3)}" for sq in range(64)
)
SQUARE_NAME_TO_SQ: Final[dict[str, int]] = {name: idx for idx, name in enumerate(SQUARE_NAMES)}

PROMO_CHARS: Final[dict[int, str]] = {
    KNIGHT: "n",
    BISHOP: "b",
    ROOK: "r",
    QUEEN: "q",
}
CHAR_TO_PROMO_PIECE: Final[dict[str, int]] = {
    "n": KNIGHT,
    "b": BISHOP,
    "r": ROOK,
    "q": QUEEN,
}


def make_move(from_sq: int, to_sq: int, flags: int = QUIET) -> int:
    return from_sq | (to_sq << 6) | (flags << 12)


def move_from(m: int) -> int:
    return m & 0x3F


def move_to(m: int) -> int:
    return (m >> 6) & 0x3F


def move_flags(m: int) -> int:
    return (m >> 12) & 0xF


def move_is_capture(m: int) -> bool:
    return bool(m & 0x4000)


def move_is_promo(m: int) -> bool:
    return bool(m & 0x8000)


def move_is_quiet(m: int) -> bool:
    return (m >> 12) & 0xC == 0


def move_promo_piece(m: int) -> int:
    return ((m >> 12) & 0x3) + 1


def move_to_uci(m: int) -> str:
    from_sq = m & 0x3F
    to_sq = (m >> 6) & 0x3F
    s = SQUARE_NAMES[from_sq] + SQUARE_NAMES[to_sq]
    if move_is_promo(m):
        s += PROMO_CHARS[move_promo_piece(m)]
    return s


# Castling rights update masks indexed by square
_CASTLING_RIGHTS_MASK: list[int] = [15] * 64
_CASTLING_RIGHTS_MASK[0] = 13  # A1 rook moved / captured: clears WQ (bit 1)
_CASTLING_RIGHTS_MASK[7] = 14  # H1 rook moved / captured: clears WK (bit 0)
_CASTLING_RIGHTS_MASK[4] = 12  # E1 king moved: clears WK and WQ (bits 0, 1)
_CASTLING_RIGHTS_MASK[56] = 7  # A8 rook moved / captured: clears BQ (bit 3)
_CASTLING_RIGHTS_MASK[63] = 11  # H8 rook moved / captured: clears BK (bit 2)
_CASTLING_RIGHTS_MASK[60] = 3  # E8 king moved: clears BK and BQ (bits 2, 3)
CASTLING_RIGHTS_MASK: Final[tuple[int, ...]] = tuple(_CASTLING_RIGHTS_MASK)


class Board:
    """Full-featured Bitboard Board with incremental evaluation and Zobrist hashing."""

    __slots__ = (
        "_stack",
        "bishop_count",
        "castling",
        "colour_at_sq",
        "colours",
        "eg_bonus",
        "eg_score",
        "ep_square",
        "fullmove",
        "game_phase",
        "halfmove",
        "hash",
        "history",
        "king_sq",
        "mg_bonus",
        "mg_score",
        "piece_at_sq",
        "pieces",
        "turn",
    )

    def __init__(self) -> None:
        self.pieces: list[int] = [0] * 6
        self.colours: list[int] = [0] * 2
        self.piece_at_sq: list[int] = [-1] * 64
        self.colour_at_sq: list[int] = [-1] * 64
        self.king_sq: list[int] = [-1, -1]
        self.turn: bool = True  # True = WHITE
        self.castling: int = 15  # WK=1, WQ=2, BK=4, BQ=8
        self.ep_square: int = -1
        self.halfmove: int = 0
        self.fullmove: int = 1
        self.hash: int = 0
        self.mg_score: list[int] = [0, 0]
        self.eg_score: list[int] = [0, 0]
        self.game_phase: int = 0
        self.bishop_count: list[int] = [0, 0]
        self.mg_bonus: list[int] = [0, 0]
        self.eg_bonus: list[int] = [0, 0]
        self._stack: list[
            tuple[
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
                int,
            ]
        ] = []
        self.history: list[int] = []

    @classmethod
    def from_fen(
        cls, fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    ) -> Board:
        board = cls()
        tokens = fen.split()
        board_part = tokens[0]
        turn_part = tokens[1] if len(tokens) > 1 else "w"
        castling_part = tokens[2] if len(tokens) > 2 else "-"
        ep_part = tokens[3] if len(tokens) > 3 else "-"
        halfmove_part = tokens[4] if len(tokens) > 4 else "0"
        fullmove_part = tokens[5] if len(tokens) > 5 else "1"

        piece_map = {
            "p": (PAWN, BLACK),
            "n": (KNIGHT, BLACK),
            "b": (BISHOP, BLACK),
            "r": (ROOK, BLACK),
            "q": (QUEEN, BLACK),
            "k": (KING, BLACK),
            "P": (PAWN, WHITE),
            "N": (KNIGHT, WHITE),
            "B": (BISHOP, WHITE),
            "R": (ROOK, WHITE),
            "Q": (QUEEN, WHITE),
            "K": (KING, WHITE),
        }

        rank = 7
        file = 0
        for char in board_part:
            if char == "/":
                rank -= 1
                file = 0
            elif char.isdigit():
                file += int(char)
            else:
                piece, color = piece_map[char]
                sq = rank * 8 + file
                board._add_piece(sq, piece, color)
                file += 1

        board.turn = turn_part == "w"

        board.castling = 0
        if "K" in castling_part:
            board.castling |= 1
        if "Q" in castling_part:
            board.castling |= 2
        if "k" in castling_part:
            board.castling |= 4
        if "q" in castling_part:
            board.castling |= 8

        if ep_part != "-" and ep_part in SQUARE_NAME_TO_SQ:
            board.ep_square = SQUARE_NAME_TO_SQ[ep_part]
        else:
            board.ep_square = -1

        board.halfmove = int(halfmove_part)
        board.fullmove = int(fullmove_part)

        # Initialise bonus scores
        board._update_bonus(WHITE)
        board._update_bonus(BLACK)

        board.hash = calculate_hash(board)
        board.history.append(board.hash)
        return board

    def copy(self) -> Board:
        b = Board()
        b.pieces = list(self.pieces)
        b.colours = list(self.colours)
        b.piece_at_sq = list(self.piece_at_sq)
        b.colour_at_sq = list(self.colour_at_sq)
        b.king_sq = list(self.king_sq)
        b.turn = self.turn
        b.castling = self.castling
        b.ep_square = self.ep_square
        b.halfmove = self.halfmove
        b.fullmove = self.fullmove
        b.hash = self.hash
        b.mg_score = list(self.mg_score)
        b.eg_score = list(self.eg_score)
        b.game_phase = self.game_phase
        b.bishop_count = list(self.bishop_count)
        b.mg_bonus = list(self.mg_bonus)
        b.eg_bonus = list(self.eg_bonus)
        b._stack = list(self._stack)
        b.history = list(self.history)
        return b

    # -----------------------------------------------------------------------
    # Piece Placement & Mutation
    # -----------------------------------------------------------------------

    def piece_type_at(self, sq: int) -> int | None:
        pt = self.piece_at_sq[sq]
        return pt if pt >= 0 else None

    def color_at(self, sq: int) -> int | None:
        c = self.colour_at_sq[sq]
        return c if c >= 0 else None

    def _add_piece(self, sq: int, piece: int, color: int) -> None:
        sq = int(sq)
        piece = int(piece)
        color = int(color)
        sq_bb = 1 << sq
        self.pieces[piece] |= sq_bb
        self.colours[color] |= sq_bb
        self.piece_at_sq[sq] = piece
        self.colour_at_sq[sq] = color

        if piece == KING:
            self.king_sq[color] = sq
        elif piece == BISHOP:
            self.bishop_count[color] += 1

        self.mg_score[color] += MG_TABLE[color][piece][sq]
        self.eg_score[color] += EG_TABLE[color][piece][sq]
        self.game_phase += GAMEPHASE_INC[piece]
        self.hash ^= PIECE_KEYS[color][piece][sq]

    def _clear_piece(self, sq: int) -> int:
        sq = int(sq)
        piece = self.piece_at_sq[sq]
        color = self.colour_at_sq[sq]
        sq_bb = ~(1 << sq)
        self.pieces[piece] &= sq_bb
        self.colours[color] &= sq_bb
        self.piece_at_sq[sq] = -1
        self.colour_at_sq[sq] = -1

        if piece == BISHOP:
            self.bishop_count[color] -= 1

        self.mg_score[color] -= MG_TABLE[color][piece][sq]
        self.eg_score[color] -= EG_TABLE[color][piece][sq]
        self.game_phase -= GAMEPHASE_INC[piece]
        self.hash ^= PIECE_KEYS[color][piece][sq]
        return piece

    # -----------------------------------------------------------------------
    # Evaluation Bonus Calculation
    # -----------------------------------------------------------------------

    def _update_bonus(self, color: int) -> None:
        """Compute pawn structure, bishop pair, and rook file bonuses for `color`."""
        opp = 1 - color
        own_pawns = self.pieces[PAWN] & self.colours[color]
        opp_pawns = self.pieces[PAWN] & self.colours[opp]
        own_rooks = self.pieces[ROOK] & self.colours[color]

        mg = 20 if self.bishop_count[color] >= 2 else 0
        eg = 30 if self.bishop_count[color] >= 2 else 0

        adj_masks = ADJACENT_FILE_MASKS
        fwd_masks = FORWARD_FILE_MASKS[color]
        passed_masks = PASSED_PAWN_MASKS[color]
        passed_mg = _PASSED_MG[color]
        passed_eg = _PASSED_EG[color]
        sq_file = _SQ_FILE
        sq_file_mask = _SQ_FILE_MASK

        # Pawns
        bb = own_pawns
        while bb:
            lsb = bb & -bb
            sq = lsb.bit_length() - 1
            bb &= bb - 1

            if not (own_pawns & adj_masks[sq_file[sq]]):
                mg -= 7
                eg -= 10

            if own_pawns & fwd_masks[sq]:
                mg -= 8
                eg -= 12
            elif not (opp_pawns & passed_masks[sq]):
                mg += passed_mg[sq]
                eg += passed_eg[sq]

        # Rooks on semi-open or open files
        bb = own_rooks
        while bb:
            lsb = bb & -bb
            sq = lsb.bit_length() - 1
            bb &= bb - 1

            file_mask = sq_file_mask[sq]
            if not (own_pawns & file_mask):
                if opp_pawns & file_mask:
                    mg += 5
                    eg += 3
                else:
                    mg += 10
                    eg += 6

        self.mg_bonus[color] = mg
        self.eg_bonus[color] = eg

    def evaluate(self) -> int:
        """Return static evaluation in centipawns from the perspective of active turn."""
        us = WHITE if self.turn else BLACK
        them = 1 - us
        mg = (self.mg_score[us] + self.mg_bonus[us]) - (self.mg_score[them] + self.mg_bonus[them])
        eg = (self.eg_score[us] + self.eg_bonus[us]) - (self.eg_score[them] + self.eg_bonus[them])
        phase = min(self.game_phase, GAMEPHASE_SUM)
        eg_phase = GAMEPHASE_SUM - phase
        score = mg * phase + eg * eg_phase
        return score // GAMEPHASE_SUM if score >= 0 else -((-score) // GAMEPHASE_SUM)

    # -----------------------------------------------------------------------
    # Make / Unmake Move
    # -----------------------------------------------------------------------

    def make_move(self, move: int) -> None:
        move = int(move)
        us = WHITE if self.turn else BLACK
        them = 1 - us
        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        flags = (move >> 12) & 0xF

        moving_piece = self.piece_at_sq[from_sq]
        captured_piece = self.piece_at_sq[to_sq]

        # Save undo state
        self._stack.append(
            (
                move,
                moving_piece,
                captured_piece,
                self.castling,
                self.ep_square,
                self.halfmove,
                self.hash,
                self.mg_score[0],
                self.mg_score[1],
                self.eg_score[0],
                self.eg_score[1],
                self.game_phase,
                self.mg_bonus[0],
                self.mg_bonus[1],
                self.eg_bonus[0],
                self.eg_bonus[1],
            )
        )

        # Remove old EP hash if it was active
        if self.ep_square != -1:
            if has_legal_en_passant(self):
                self.hash ^= EP_KEYS[self.ep_square & 7]
            self.ep_square = -1

        # Halfmove clock
        if moving_piece == PAWN or captured_piece != -1:
            self.halfmove = 0
        else:
            self.halfmove += 1

        # Handle captures
        if captured_piece != -1:
            self._clear_piece(to_sq)
        elif flags == EN_PASSANT:
            ep_cap_sq = to_sq - 8 if us == WHITE else to_sq + 8
            self._clear_piece(ep_cap_sq)

        # Move or promote piece
        self._clear_piece(from_sq)
        if flags >= KNIGHT_PROMO:
            promo_piece = move_promo_piece(move)
            self._add_piece(to_sq, promo_piece, us)
        else:
            self._add_piece(to_sq, moving_piece, us)

        # Handle special moves
        if flags == DOUBLE_PAWN_PUSH:
            self.ep_square = from_sq + 8 if us == WHITE else from_sq - 8
            self.turn = not self.turn
            if has_legal_en_passant(self):
                self.hash ^= EP_KEYS[self.ep_square & 7]
            self.turn = not self.turn
        elif flags == KING_CASTLE:
            if us == WHITE:
                self._clear_piece(7)
                self._add_piece(5, ROOK, WHITE)
            else:
                self._clear_piece(63)
                self._add_piece(61, ROOK, BLACK)
        elif flags == QUEEN_CASTLE:
            if us == WHITE:
                self._clear_piece(0)
                self._add_piece(3, ROOK, WHITE)
            else:
                self._clear_piece(56)
                self._add_piece(59, ROOK, BLACK)

        # Update castling rights
        old_castling = self.castling
        new_castling = old_castling & CASTLING_RIGHTS_MASK[from_sq] & CASTLING_RIGHTS_MASK[to_sq]
        if new_castling != old_castling:
            self.castling = new_castling
            self.hash ^= CASTLING_TABLE[old_castling] ^ CASTLING_TABLE[new_castling]

        # Update bonus scores if pawns/rooks/bishops were involved
        if (
            moving_piece in (PAWN, ROOK, BISHOP)
            or captured_piece in (PAWN, ROOK, BISHOP)
            or flags in (KING_CASTLE, QUEEN_CASTLE, EN_PASSANT)
        ):
            self._update_bonus(us)
            if (
                moving_piece == PAWN
                or captured_piece in (PAWN, ROOK, BISHOP)
                or flags == EN_PASSANT
            ):
                self._update_bonus(them)

        # Flip turn
        self.turn = not self.turn
        self.hash ^= TURN_KEY
        if us == BLACK:
            self.fullmove += 1

        self.history.append(self.hash)

    def unmake_move(self) -> None:
        (
            move,
            moving_piece,
            captured_piece,
            self.castling,
            self.ep_square,
            self.halfmove,
            saved_hash,
            mg0,
            mg1,
            eg0,
            eg1,
            saved_phase,
            b_mg0,
            b_mg1,
            b_eg0,
            b_eg1,
        ) = self._stack.pop()

        self.history.pop()

        self.turn = not self.turn
        us = WHITE if self.turn else BLACK
        them = 1 - us
        if us == BLACK:
            self.fullmove -= 1

        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        flags = (move >> 12) & 0xF

        # Remove placed piece from to_sq
        self._clear_piece(to_sq)

        # Restore moving piece to from_sq
        if flags >= KNIGHT_PROMO:
            self._add_piece(from_sq, PAWN, us)
        else:
            self._add_piece(from_sq, moving_piece, us)

        # Handle castling rook reversal
        if flags == KING_CASTLE:
            if us == WHITE:
                self._clear_piece(5)
                self._add_piece(7, ROOK, WHITE)
            else:
                self._clear_piece(61)
                self._add_piece(63, ROOK, BLACK)
        elif flags == QUEEN_CASTLE:
            if us == WHITE:
                self._clear_piece(3)
                self._add_piece(0, ROOK, WHITE)
            else:
                self._clear_piece(59)
                self._add_piece(56, ROOK, BLACK)
        elif flags == EN_PASSANT:
            ep_cap_sq = to_sq - 8 if us == WHITE else to_sq + 8
            self._add_piece(ep_cap_sq, PAWN, them)

        # Restore captured piece if any
        if captured_piece != -1:
            self._add_piece(to_sq, captured_piece, them)

        # Restore evaluation and bonus scores, phase, and exact hash in O(1)
        self.mg_score[0] = mg0
        self.mg_score[1] = mg1
        self.eg_score[0] = eg0
        self.eg_score[1] = eg1
        self.mg_bonus[0] = b_mg0
        self.mg_bonus[1] = b_mg1
        self.eg_bonus[0] = b_eg0
        self.eg_bonus[1] = b_eg1
        self.game_phase = saved_phase
        self.hash = saved_hash

    def make_null_move(self) -> None:
        us = WHITE if self.turn else BLACK
        # Save undo state
        self._stack.append(
            (
                0,
                -1,
                -1,
                self.castling,
                self.ep_square,
                self.halfmove,
                self.hash,
                self.mg_score[0],
                self.mg_score[1],
                self.eg_score[0],
                self.eg_score[1],
                self.game_phase,
                self.mg_bonus[0],
                self.mg_bonus[1],
                self.eg_bonus[0],
                self.eg_bonus[1],
            )
        )

        if self.ep_square != -1:
            if has_legal_en_passant(self):
                self.hash ^= EP_KEYS[self.ep_square & 7]
            self.ep_square = -1

        self.halfmove += 1
        self.turn = not self.turn
        self.hash ^= TURN_KEY
        if us == BLACK:
            self.fullmove += 1
        self.history.append(self.hash)

    def unmake_null_move(self) -> None:
        (
            _,
            _,
            _,
            self.castling,
            self.ep_square,
            self.halfmove,
            self.hash,
            self.mg_score[0],
            self.mg_score[1],
            self.eg_score[0],
            self.eg_score[1],
            self.game_phase,
            self.mg_bonus[0],
            self.mg_bonus[1],
            self.eg_bonus[0],
            self.eg_bonus[1],
        ) = self._stack.pop()

        self.history.pop()
        self.turn = not self.turn
        if not self.turn:
            self.fullmove -= 1

    # -----------------------------------------------------------------------
    # Move Generation (Legal-only, Stargaze port)
    # -----------------------------------------------------------------------

    def generate_moves(self, captures_only: bool = False) -> list[int]:
        legal_moves: list[int] = []

        us = WHITE if self.turn else BLACK
        them = 1 - us
        own_pieces = self.colours[us]
        other_pieces = self.colours[them]
        occupied = own_pieces | other_pieces

        king_sq = self.king_sq[us]

        # 1. Candidate king moves and opponent attack checking
        occupied_no_king = occupied ^ (1 << king_sq)
        opp_pawns = self.pieces[PAWN] & other_pieces
        opp_knights = self.pieces[KNIGHT] & other_pieces
        opp_king_sq = self.king_sq[them]
        opp_diag = (self.pieces[BISHOP] | self.pieces[QUEEN]) & other_pieces
        opp_orth = (self.pieces[ROOK] | self.pieces[QUEEN]) & other_pieces

        def is_sq_attacked(sq: int) -> bool:
            if PAWN_ATTACKS[us][sq] & opp_pawns:
                return True
            if KNIGHT_ATTACKS[sq] & opp_knights:
                return True
            if KING_ATTACKS[opp_king_sq] & (1 << sq):
                return True
            if opp_diag and (bishop_attacks(sq, np.uint64(occupied_no_king)) & opp_diag):
                return True
            return bool(opp_orth and (rook_attacks(sq, np.uint64(occupied_no_king)) & opp_orth))

        # 2. King moves
        king_targets = KING_ATTACKS[king_sq] & (other_pieces if captures_only else ~own_pieces)
        while king_targets:
            lsb = king_targets & -king_targets
            to_sq = lsb.bit_length() - 1
            king_targets &= king_targets - 1
            if not is_sq_attacked(to_sq):
                flag = CAPTURE if (other_pieces & (1 << to_sq)) else QUIET
                legal_moves.append(make_move(king_sq, to_sq, flag))

        # 3. Find checkers of the king
        checkers = 0
        checkers |= PAWN_ATTACKS[us][king_sq] & (self.pieces[PAWN] & other_pieces)
        checkers |= KNIGHT_ATTACKS[king_sq] & (self.pieces[KNIGHT] & other_pieces)
        checkers |= bishop_attacks(king_sq, np.uint64(occupied)) & (
            (self.pieces[BISHOP] | self.pieces[QUEEN]) & other_pieces
        )
        checkers |= rook_attacks(king_sq, np.uint64(occupied)) & (
            (self.pieces[ROOK] | self.pieces[QUEEN]) & other_pieces
        )

        num_checkers = checkers.bit_count()

        # Double check: only king moves are legal
        if num_checkers >= 2:
            return legal_moves

        # 4. Check mask
        if num_checkers == 1:
            checker_sq = (checkers & -checkers).bit_length() - 1
            checker_type = self.piece_at_sq[checker_sq]
            if checker_type in (BISHOP, ROOK, QUEEN):
                check_mask = RAY_BETWEEN[king_sq][checker_sq] | (1 << checker_sq)
            else:
                check_mask = 1 << checker_sq
        else:
            check_mask = 0xFFFFFFFFFFFFFFFF

        # 5. Pinned pieces and pin masks
        pin_mask = [0xFFFFFFFFFFFFFFFF] * 64
        pinned_pieces = 0

        pinners = (
            BISHOP_MASKS[king_sq] & ((self.pieces[BISHOP] | self.pieces[QUEEN]) & other_pieces)
        ) | (ROOK_MASKS[king_sq] & ((self.pieces[ROOK] | self.pieces[QUEEN]) & other_pieces))

        while pinners:
            lsb = pinners & -pinners
            pinner_sq = lsb.bit_length() - 1
            pinners &= pinners - 1

            between = RAY_BETWEEN[king_sq][pinner_sq]
            pieces_between = between & occupied
            if pieces_between.bit_count() == 1:
                own_pinned = pieces_between & own_pieces
                if own_pinned:
                    pinned_sq = (own_pinned & -own_pinned).bit_length() - 1
                    pin_mask[pinned_sq] = between | (1 << pinner_sq)
                    pinned_pieces |= own_pinned

        # Pinned pieces cannot move when in check
        active_own = own_pieces
        if num_checkers > 0:
            active_own &= ~pinned_pieces

        movable_mask = check_mask

        # -------------------------------------------------------------------
        # Pawns
        # -------------------------------------------------------------------
        pawns = self.pieces[PAWN] & active_own
        while pawns:
            lsb = pawns & -pawns
            from_sq = lsb.bit_length() - 1
            pawns &= pawns - 1

            target_mask = movable_mask & pin_mask[from_sq]
            file = from_sq & 7
            rank = from_sq >> 3

            if us == WHITE:
                push1_to = from_sq + 8
                if not (occupied & (1 << push1_to)) and (target_mask & (1 << push1_to)):
                    if rank == 6:  # Promo rank
                        for promo in PROMOTION_PIECES:
                            legal_moves.append(make_move(from_sq, push1_to, promo))
                    elif not captures_only:
                        legal_moves.append(make_move(from_sq, push1_to, QUIET))

                # Push two
                if (
                    rank == 1
                    and not captures_only
                    and not (occupied & (1 << push1_to))
                    and not (occupied & (1 << (from_sq + 16)))
                    and (target_mask & (1 << (from_sq + 16)))
                ):
                    legal_moves.append(make_move(from_sq, from_sq + 16, DOUBLE_PAWN_PUSH))

                # Left capture (file - 1)
                if file > 0:
                    cap_to = from_sq + 7
                    if (other_pieces & (1 << cap_to)) and (target_mask & (1 << cap_to)):
                        if rank == 6:
                            for promo in PROMOTION_CAPTURE_PIECES:
                                legal_moves.append(make_move(from_sq, cap_to, promo))
                        else:
                            legal_moves.append(make_move(from_sq, cap_to, CAPTURE))

                # Right capture (file + 1)
                if file < 7:
                    cap_to = from_sq + 9
                    if (other_pieces & (1 << cap_to)) and (target_mask & (1 << cap_to)):
                        if rank == 6:
                            for promo in PROMOTION_CAPTURE_PIECES:
                                legal_moves.append(make_move(from_sq, cap_to, promo))
                        else:
                            legal_moves.append(make_move(from_sq, cap_to, CAPTURE))

                # En Passant
                if self.ep_square != -1:
                    ep = self.ep_square
                    cap_pawn_sq = ep - 8
                    can_ep = (file > 0 and ep == from_sq + 7) or (file < 7 and ep == from_sq + 9)
                    if (
                        can_ep
                        and (movable_mask & ((1 << cap_pawn_sq) | (1 << ep)))
                        and (pin_mask[from_sq] & (1 << ep))
                    ):
                        occ_after = (occupied ^ (1 << from_sq) ^ (1 << cap_pawn_sq)) | (1 << ep)
                        diag_sliders = (self.pieces[BISHOP] | self.pieces[QUEEN]) & other_pieces
                        orth_sliders = (self.pieces[ROOK] | self.pieces[QUEEN]) & other_pieces
                        if not (
                            rook_attacks(king_sq, np.uint64(occ_after)) & orth_sliders
                        ) and not (bishop_attacks(king_sq, np.uint64(occ_after)) & diag_sliders):
                            legal_moves.append(make_move(from_sq, ep, EN_PASSANT))
            else:
                push1_to = from_sq - 8
                if not (occupied & (1 << push1_to)) and (target_mask & (1 << push1_to)):
                    if rank == 1:  # Promo rank
                        for promo in PROMOTION_PIECES:
                            legal_moves.append(make_move(from_sq, push1_to, promo))
                    elif not captures_only:
                        legal_moves.append(make_move(from_sq, push1_to, QUIET))

                # Push two
                if (
                    rank == 6
                    and not captures_only
                    and not (occupied & (1 << push1_to))
                    and not (occupied & (1 << (from_sq - 16)))
                    and (target_mask & (1 << (from_sq - 16)))
                ):
                    legal_moves.append(make_move(from_sq, from_sq - 16, DOUBLE_PAWN_PUSH))

                # Left capture (file - 1)
                if file > 0:
                    cap_to = from_sq - 9
                    if (other_pieces & (1 << cap_to)) and (target_mask & (1 << cap_to)):
                        if rank == 1:
                            for promo in PROMOTION_CAPTURE_PIECES:
                                legal_moves.append(make_move(from_sq, cap_to, promo))
                        else:
                            legal_moves.append(make_move(from_sq, cap_to, CAPTURE))

                # Right capture (file + 1)
                if file < 7:
                    cap_to = from_sq - 7
                    if (other_pieces & (1 << cap_to)) and (target_mask & (1 << cap_to)):
                        if rank == 1:
                            for promo in PROMOTION_CAPTURE_PIECES:
                                legal_moves.append(make_move(from_sq, cap_to, promo))
                        else:
                            legal_moves.append(make_move(from_sq, cap_to, CAPTURE))

                # En Passant
                if self.ep_square != -1:
                    ep = self.ep_square
                    cap_pawn_sq = ep + 8
                    can_ep = (file > 0 and ep == from_sq - 9) or (file < 7 and ep == from_sq - 7)
                    if (
                        can_ep
                        and (movable_mask & ((1 << cap_pawn_sq) | (1 << ep)))
                        and (pin_mask[from_sq] & (1 << ep))
                    ):
                        occ_after = (occupied ^ (1 << from_sq) ^ (1 << cap_pawn_sq)) | (1 << ep)
                        diag_sliders = (self.pieces[BISHOP] | self.pieces[QUEEN]) & other_pieces
                        orth_sliders = (self.pieces[ROOK] | self.pieces[QUEEN]) & other_pieces
                        if not (
                            rook_attacks(king_sq, np.uint64(occ_after)) & orth_sliders
                        ) and not (bishop_attacks(king_sq, np.uint64(occ_after)) & diag_sliders):
                            legal_moves.append(make_move(from_sq, ep, EN_PASSANT))

        # -------------------------------------------------------------------
        # Knights
        # -------------------------------------------------------------------
        knights = self.pieces[KNIGHT] & active_own
        while knights:
            lsb = knights & -knights
            from_sq = lsb.bit_length() - 1
            knights &= knights - 1

            targets = KNIGHT_ATTACKS[from_sq] & ~own_pieces & movable_mask & pin_mask[from_sq]
            if captures_only:
                targets &= other_pieces

            while targets:
                t_lsb = targets & -targets
                to_sq = t_lsb.bit_length() - 1
                targets &= targets - 1
                flag = CAPTURE if (other_pieces & (1 << to_sq)) else QUIET
                legal_moves.append(make_move(from_sq, to_sq, flag))

        # -------------------------------------------------------------------
        # Bishops
        # -------------------------------------------------------------------
        bishops = self.pieces[BISHOP] & active_own
        while bishops:
            lsb = bishops & -bishops
            from_sq = lsb.bit_length() - 1
            bishops &= bishops - 1

            targets = (
                bishop_attacks(from_sq, np.uint64(occupied))
                & ~own_pieces
                & movable_mask
                & pin_mask[from_sq]
            )
            if captures_only:
                targets &= other_pieces

            while targets:
                t_lsb = targets & -targets
                to_sq = t_lsb.bit_length() - 1
                targets &= targets - 1
                flag = CAPTURE if (other_pieces & (1 << to_sq)) else QUIET
                legal_moves.append(make_move(from_sq, to_sq, flag))

        # -------------------------------------------------------------------
        # Rooks
        # -------------------------------------------------------------------
        rooks = self.pieces[ROOK] & active_own
        while rooks:
            lsb = rooks & -rooks
            from_sq = lsb.bit_length() - 1
            rooks &= rooks - 1

            targets = (
                rook_attacks(from_sq, np.uint64(occupied))
                & ~own_pieces
                & movable_mask
                & pin_mask[from_sq]
            )
            if captures_only:
                targets &= other_pieces

            while targets:
                t_lsb = targets & -targets
                to_sq = t_lsb.bit_length() - 1
                targets &= targets - 1
                flag = CAPTURE if (other_pieces & (1 << to_sq)) else QUIET
                legal_moves.append(make_move(from_sq, to_sq, flag))

        # -------------------------------------------------------------------
        # Queens
        # -------------------------------------------------------------------
        queens = self.pieces[QUEEN] & active_own
        while queens:
            lsb = queens & -queens
            from_sq = lsb.bit_length() - 1
            queens &= queens - 1

            targets = (
                queen_attacks(from_sq, np.uint64(occupied))
                & ~own_pieces
                & movable_mask
                & pin_mask[from_sq]
            )
            if captures_only:
                targets &= other_pieces

            while targets:
                t_lsb = targets & -targets
                to_sq = t_lsb.bit_length() - 1
                targets &= targets - 1
                flag = CAPTURE if (other_pieces & (1 << to_sq)) else QUIET
                legal_moves.append(make_move(from_sq, to_sq, flag))

        # -------------------------------------------------------------------
        # Castling
        # -------------------------------------------------------------------
        if not captures_only and num_checkers == 0:
            if us == WHITE:
                # White Kingside: E1(4), F1(5), G1(6)
                if (
                    king_sq == 4
                    and (self.castling & 1)
                    and (self.pieces[ROOK] & own_pieces & (1 << 7))
                    and not (occupied & ((1 << 5) | (1 << 6)))
                    and not is_sq_attacked(4)
                    and not is_sq_attacked(5)
                    and not is_sq_attacked(6)
                ):
                    legal_moves.append(make_move(4, 6, KING_CASTLE))

                # White Queenside: E1(4), D1(3), C1(2), B1(1)
                if (
                    king_sq == 4
                    and (self.castling & 2)
                    and (self.pieces[ROOK] & own_pieces & (1 << 0))
                    and not (occupied & ((1 << 1) | (1 << 2) | (1 << 3)))
                    and not is_sq_attacked(4)
                    and not is_sq_attacked(3)
                    and not is_sq_attacked(2)
                ):
                    legal_moves.append(make_move(4, 2, QUEEN_CASTLE))
            else:
                # Black Kingside: E8(60), F8(61), G8(62)
                if (
                    king_sq == 60
                    and (self.castling & 4)
                    and (self.pieces[ROOK] & own_pieces & (1 << 63))
                    and not (occupied & ((1 << 61) | (1 << 62)))
                    and not is_sq_attacked(60)
                    and not is_sq_attacked(61)
                    and not is_sq_attacked(62)
                ):
                    legal_moves.append(make_move(60, 62, KING_CASTLE))

                # Black Queenside: E8(60), D8(59), C8(58), B8(57)
                if (
                    king_sq == 60
                    and (self.castling & 8)
                    and (self.pieces[ROOK] & own_pieces & (1 << 56))
                    and not (occupied & ((1 << 57) | (1 << 58) | (1 << 59)))
                    and not is_sq_attacked(60)
                    and not is_sq_attacked(59)
                    and not is_sq_attacked(58)
                ):
                    legal_moves.append(make_move(60, 58, QUEEN_CASTLE))

        return legal_moves

    # -----------------------------------------------------------------------
    # Check Detection & gives_check Pre-Move
    # -----------------------------------------------------------------------

    def is_in_check(self) -> bool:
        """Is the active side's king currently under attack?"""
        us = WHITE if self.turn else BLACK
        them = 1 - us
        king_sq = self.king_sq[us]
        other_pieces = self.colours[them]
        occupied = self.colours[0] | self.colours[1]

        return bool(
            (PAWN_ATTACKS[us][king_sq] & (self.pieces[PAWN] & other_pieces))
            or (KNIGHT_ATTACKS[king_sq] & (self.pieces[KNIGHT] & other_pieces))
            or (
                bishop_attacks(king_sq, np.uint64(occupied))
                & ((self.pieces[BISHOP] | self.pieces[QUEEN]) & other_pieces)
            )
            or (
                rook_attacks(king_sq, np.uint64(occupied))
                & ((self.pieces[ROOK] | self.pieces[QUEEN]) & other_pieces)
            )
        )

    def gives_check(self, move: int) -> bool:
        """Does `move` give check to the opponent king? Computed pre-move via attack tables."""
        from_sq = move & 0x3F
        to_sq = (move >> 6) & 0x3F
        flags = (move >> 12) & 0xF
        ep_cap_sq = to_sq - 8 if self.turn else to_sq + 8

        us = WHITE if self.turn else BLACK
        them = 1 - us
        opp_king_sq = self.king_sq[them]
        occupied = self.colours[0] | self.colours[1]

        # 1. Direct check
        promo = -1
        if flags >= KNIGHT_PROMO:
            promo = move_promo_piece(move)
            if promo == KNIGHT:
                if KNIGHT_ATTACKS[to_sq] & (1 << opp_king_sq):
                    return True
            elif promo == BISHOP:
                occ = (occupied ^ (1 << from_sq)) | (1 << to_sq)
                if bishop_attacks(to_sq, np.uint64(occ)) & (1 << opp_king_sq):
                    return True
            elif promo == ROOK:
                occ = (occupied ^ (1 << from_sq)) | (1 << to_sq)
                if rook_attacks(to_sq, np.uint64(occ)) & (1 << opp_king_sq):
                    return True
            elif promo == QUEEN:
                occ = (occupied ^ (1 << from_sq)) | (1 << to_sq)
                if queen_attacks(to_sq, np.uint64(occ)) & (1 << opp_king_sq):
                    return True
        elif flags == KING_CASTLE:
            rook_to = 5 if us == WHITE else 61
            occ = (
                (occupied ^ (1 << from_sq) ^ (1 << (7 if us == WHITE else 63)))
                | (1 << to_sq)
                | (1 << rook_to)
            )
            if rook_attacks(rook_to, np.uint64(occ)) & (1 << opp_king_sq):
                return True
        elif flags == QUEEN_CASTLE:
            rook_to = 3 if us == WHITE else 59
            occ = (
                (occupied ^ (1 << from_sq) ^ (1 << (0 if us == WHITE else 56)))
                | (1 << to_sq)
                | (1 << rook_to)
            )
            if rook_attacks(rook_to, np.uint64(occ)) & (1 << opp_king_sq):
                return True
        else:
            pt = self.piece_at_sq[from_sq]
            if pt == PAWN:
                if PAWN_ATTACKS[us][to_sq] & (1 << opp_king_sq):
                    return True
            elif pt == KNIGHT:
                if KNIGHT_ATTACKS[to_sq] & (1 << opp_king_sq):
                    return True
            elif pt == BISHOP:
                occ = (occupied ^ (1 << from_sq)) | (1 << to_sq)
                if bishop_attacks(to_sq, np.uint64(occ)) & (1 << opp_king_sq):
                    return True
            elif pt == ROOK:
                occ = (occupied ^ (1 << from_sq)) | (1 << to_sq)
                if rook_attacks(to_sq, np.uint64(occ)) & (1 << opp_king_sq):
                    return True
            elif pt == QUEEN:
                occ = (occupied ^ (1 << from_sq)) | (1 << to_sq)
                if queen_attacks(to_sq, np.uint64(occ)) & (1 << opp_king_sq):
                    return True

        # 2. Discovered check (sliders attacking opp_king through vacated square)
        if flags == EN_PASSANT:
            occ = (occupied ^ (1 << from_sq) ^ (1 << ep_cap_sq)) | (1 << to_sq)
        else:
            occ = (occupied ^ (1 << from_sq)) | (1 << to_sq)

        # Check if friendly slider has discovered attack on opp_king
        diag_sliders = (
            (self.pieces[BISHOP] | self.pieces[QUEEN]) & self.colours[us] & ~(1 << from_sq)
        )
        if (flags >= KNIGHT_PROMO and promo in (BISHOP, QUEEN)) or (
            flags < KNIGHT_PROMO and self.piece_at_sq[from_sq] in (BISHOP, QUEEN)
        ):
            diag_sliders |= 1 << to_sq

        if (
            (BISHOP_MASKS[opp_king_sq] & (1 << from_sq))
            or (flags == EN_PASSANT and (BISHOP_MASKS[opp_king_sq] & (1 << ep_cap_sq)))
        ) and (bishop_attacks(opp_king_sq, np.uint64(occ)) & diag_sliders):
            return True

        orth_sliders = (self.pieces[ROOK] | self.pieces[QUEEN]) & self.colours[us] & ~(1 << from_sq)
        if (flags >= KNIGHT_PROMO and promo in (ROOK, QUEEN)) or (
            flags < KNIGHT_PROMO and self.piece_at_sq[from_sq] in (ROOK, QUEEN)
        ):
            orth_sliders |= 1 << to_sq

        return bool(
            (
                (ROOK_MASKS[opp_king_sq] & (1 << from_sq))
                or (flags == EN_PASSANT and (ROOK_MASKS[opp_king_sq] & (1 << ep_cap_sq)))
            )
            and (rook_attacks(opp_king_sq, np.uint64(occ)) & orth_sliders)
        )

    # -----------------------------------------------------------------------
    # Draw and Game State
    # -----------------------------------------------------------------------

    def is_repetition(self, count: int = 3) -> bool:
        curr_hash = self.hash
        limit = max(0, len(self.history) - 1 - self.halfmove)
        if count == 2:
            return any(
                self.history[idx] == curr_hash
                for idx in range(len(self.history) - 2, limit - 1, -2)
            )
        seen = 1
        for idx in range(len(self.history) - 2, limit - 1, -2):
            if self.history[idx] == curr_hash:
                seen += 1
                if seen >= count:
                    return True
        return False

    def is_halfmove_draw(self) -> bool:
        return self.halfmove >= 100

    def is_insufficient_material(self) -> bool:
        if self.pieces[PAWN] or self.pieces[ROOK] or self.pieces[QUEEN]:
            return False
        # Only kings, knights, and bishops
        w_minors = (self.pieces[KNIGHT] | self.pieces[BISHOP]) & self.colours[WHITE]
        b_minors = (self.pieces[KNIGHT] | self.pieces[BISHOP]) & self.colours[BLACK]
        w_count = w_minors.bit_count()
        b_count = b_minors.bit_count()
        return (
            (w_count == 0 and b_count == 0)
            or (w_count == 1 and b_count == 0)
            or (w_count == 0 and b_count == 1)
        )

    def is_game_over(self) -> bool:
        if self.is_halfmove_draw() or self.is_insufficient_material() or self.is_repetition(3):
            return True
        return len(self.generate_moves()) == 0

    def fen(self) -> str:
        """Export current position as standard FEN string."""
        parts = []
        piece_chars = {
            (PAWN, WHITE): "P",
            (KNIGHT, WHITE): "N",
            (BISHOP, WHITE): "B",
            (ROOK, WHITE): "R",
            (QUEEN, WHITE): "Q",
            (KING, WHITE): "K",
            (PAWN, BLACK): "p",
            (KNIGHT, BLACK): "n",
            (BISHOP, BLACK): "b",
            (ROOK, BLACK): "r",
            (QUEEN, BLACK): "q",
            (KING, BLACK): "k",
        }
        for rank in range(7, -1, -1):
            empty = 0
            row = ""
            for file in range(8):
                sq = rank * 8 + file
                pt = self.piece_at_sq[sq]
                if pt == -1:
                    empty += 1
                else:
                    if empty > 0:
                        row += str(empty)
                        empty = 0
                    row += piece_chars[(pt, self.colour_at_sq[sq])]
            if empty > 0:
                row += str(empty)
            parts.append(row)
        board_str = "/".join(parts)

        turn_str = "w" if self.turn else "b"

        castling_str = ""
        if self.castling & 1:
            castling_str += "K"
        if self.castling & 2:
            castling_str += "Q"
        if self.castling & 4:
            castling_str += "k"
        if self.castling & 8:
            castling_str += "q"
        if not castling_str:
            castling_str = "-"

        ep_str = SQUARE_NAMES[self.ep_square] if self.ep_square != -1 else "-"

        return f"{board_str} {turn_str} {castling_str} {ep_str} {self.halfmove} {self.fullmove}"


def parse_uci_to_move(board: Board, uci_str: str) -> int:
    """Parse UCI move string against board's legal moves."""
    for m in board.generate_moves():
        if move_to_uci(m) == uci_str:
            return m
    raise ValueError(f"Illegal or unknown UCI move '{uci_str}' in position: {board.fen()}")
