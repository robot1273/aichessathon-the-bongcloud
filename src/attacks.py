"""Precomputed attack tables and magic bitboard lookups.

Ported from stargaze C++ engine (magic.hpp and mask.hpp).
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Magic Numbers (Stargaze)
# ---------------------------------------------------------------------------

BISHOP_MAGICS: Final[tuple[int, ...]] = (
    0xA0043002232020,
    0x8620A10A4A014000,
    0xC1000822F400080,
    0x80404008A900004,
    0x6021180000020,
    0x1042004400001,
    0x411010803408204,
    0x4000104E08201810,
    0x641080230443102,
    0x10002106A020148,
    0x5000844403821000,
    0x160051040088000A,
    0x50040420000004,
    0x400008821081440,
    0x21008948C104020,
    0xC0C044400880800,
    0x20401004018800,
    0x408001242483209,
    0x1510002A04009322,
    0xC10C210802002002,
    0x4104020280E00080,
    0x40002002100C2002,
    0x2010600C02080400,
    0x8000208042080400,
    0x1404220010200120,
    0x511900004105A00,
    0x10A80800040A2220,
    0x8080000820002,
    0x20840204802002,
    0x18020000490404,
    0x7230024240B02,
    0x2108508085040100,
    0x202082000042000,
    0x1084640440121001,
    0x1082010644140800,
    0x12008020420201,
    0x40010200010880,
    0x8000850100021000,
    0x4041220422008408,
    0x4E08211820004205,
    0x401303010840508,
    0xA282108000422,
    0x2409004822007000,
    0xAC0004200800802,
    0xC81340094000200,
    0x2040500050C02180,
    0x2408100900400218,
    0x2020405020420,
    0xA110622210400080,
    0x20904110100000,
    0x300011041103000,
    0x2252110210440010,
    0x8209640620820028,
    0xC0052004010210,
    0x8004080288120000,
    0x22928401060193,
    0x5892410040400,
    0x204422080250,
    0x104901404C1022,
    0x6010262001421604,
    0x104A01310820202,
    0x1000402044010204,
    0x40001020028C0440,
    0x20E0040508002980,
)

ROOK_MAGICS: Final[tuple[int, ...]] = (
    0x80008020400010,
    0x40200040001000,
    0x1280082001100080,
    0x60008200C704200,
    0x2500041048010002,
    0x500040008210012,
    0x100010045A40A00,
    0x1200020021104084,
    0x800080204000,
    0x210140002000D000,
    0x420801000802004,
    0x4C08800800100481,
    0x810808004008800,
    0x812001082000408,
    0x874000488010210,
    0x20028001C2800100,
    0x30410020800104,
    0x404000201003,
    0x2890002008002400,
    0x40C1010008100020,
    0x3000808004000802,
    0x202008080040002,
    0x8543010100040200,
    0x40004200210290CC,
    0x920800280204008,
    0x4000200080400080,
    0x490040020080020,
    0x400401200082201,
    0x8004040080800800,
    0x83020080040080,
    0x1000080400020110,
    0x141000100285082,
    0x420324001800080,
    0x2000200080804000,
    0x802008801000,
    0x4180805002802801,
    0x4A18010045001048,
    0x404020080800400,
    0x840100204000108,
    0x10088842000401,
    0x80800040028020,
    0x200500020004001,
    0x1400200702410010,
    0xD800100300090020,
    0x48008004008008,
    0x101100440080120,
    0x2080020001008080,
    0x1100005081020004,
    0x20800020400080,
    0x8000400020009080,
    0x2400102001044500,
    0x2000080180100180,
    0x4C004800816580,
    0x8102820080840080,
    0x1381000200241100,
    0x80004C08812200,
    0x2000800102C06315,
    0x400085110260C3,
    0x2021008214082,
    0x2042101100009,
    0x202013060082402,
    0x4002000810010402,
    0x2308500A8204,
    0x251000040802201,
)

# ---------------------------------------------------------------------------
# Simple non-sliding piece attacks and masks
# ---------------------------------------------------------------------------


def _gen_knight_attacks() -> tuple[int, ...]:
    attacks = []
    for sq in range(64):
        r, f = divmod(sq, 8)
        mask = 0
        for dr, df in (
            (-2, -1),
            (-2, 1),
            (-1, -2),
            (-1, 2),
            (1, -2),
            (1, 2),
            (2, -1),
            (2, 1),
        ):
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                mask |= 1 << (nr * 8 + nf)
        attacks.append(mask)
    return tuple(attacks)


def _gen_king_attacks() -> tuple[int, ...]:
    attacks = []
    for sq in range(64):
        mask = 0
        r, f = divmod(sq, 8)
        for dr in (-1, 0, 1):
            for df in (-1, 0, 1):
                if dr == 0 and df == 0:
                    continue
                nr, nf = r + dr, f + df
                if 0 <= nr < 8 and 0 <= nf < 8:
                    mask |= 1 << (nr * 8 + nf)
        attacks.append(mask)
    return tuple(attacks)


def _gen_pawn_attacks() -> tuple[tuple[int, ...], tuple[int, ...]]:
    white_attacks = []
    black_attacks = []
    for sq in range(64):
        r, f = divmod(sq, 8)
        # White pawns attack north-west (dr=1, df=-1) and north-east (dr=1, df=1)
        w_mask = 0
        if r < 7:
            if f > 0:
                w_mask |= 1 << ((r + 1) * 8 + (f - 1))
            if f < 7:
                w_mask |= 1 << ((r + 1) * 8 + (f + 1))
        white_attacks.append(w_mask)

        # Black pawns attack south-west (dr=-1, df=-1) and south-east (dr=-1, df=1)
        b_mask = 0
        if r > 0:
            if f > 0:
                b_mask |= 1 << ((r - 1) * 8 + (f - 1))
            if f < 7:
                b_mask |= 1 << ((r - 1) * 8 + (f + 1))
        black_attacks.append(b_mask)

    return tuple(white_attacks), tuple(black_attacks)


def _gen_ray_between() -> tuple[tuple[int, ...], ...]:
    table: list[list[int]] = [[0] * 64 for _ in range(64)]
    dirs = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for sq1 in range(64):
        r1, f1 = divmod(sq1, 8)
        for sq2 in range(64):
            if sq1 == sq2:
                continue
            r2, f2 = divmod(sq2, 8)
            dr = r2 - r1
            df = f2 - f1
            step_r = 0 if dr == 0 else (1 if dr > 0 else -1)
            step_f = 0 if df == 0 else (1 if df > 0 else -1)
            if (dr == 0 or df == 0 or abs(dr) == abs(df)) and (step_r, step_f) in dirs:
                path = 0
                cr, cf = r1 + step_r, f1 + step_f
                while (cr, cf) != (r2, f2) and 0 <= cr < 8 and 0 <= cf < 8:
                    path |= 1 << (cr * 8 + cf)
                    cr += step_r
                    cf += step_f
                if (cr, cf) == (r2, f2):
                    table[sq1][sq2] = path
    return tuple(tuple(row) for row in table)


def _gen_bishop_masks() -> tuple[int, ...]:
    """Full diagonal ray masks from square to all edges (for pinners/checkers)."""
    masks = []
    for sq in range(64):
        r, f = divmod(sq, 8)
        mask = 0
        for dr, df in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nf = r + dr, f + df
            while 0 <= nr < 8 and 0 <= nf < 8:
                mask |= 1 << (nr * 8 + nf)
                nr += dr
                nf += df
        masks.append(mask)
    return tuple(masks)


def _gen_rook_masks() -> tuple[int, ...]:
    """Full cardinal ray masks from square to all edges (for pinners/checkers)."""
    masks = []
    for sq in range(64):
        r, f = divmod(sq, 8)
        mask = 0
        for dr, df in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nf = r + dr, f + df
            while 0 <= nr < 8 and 0 <= nf < 8:
                mask |= 1 << (nr * 8 + nf)
                nr += dr
                nf += df
        masks.append(mask)
    return tuple(masks)


KNIGHT_ATTACKS: Final[tuple[int, ...]] = _gen_knight_attacks()
KING_ATTACKS: Final[tuple[int, ...]] = _gen_king_attacks()
PAWN_ATTACKS: Final[tuple[tuple[int, ...], tuple[int, ...]]] = _gen_pawn_attacks()
RAY_BETWEEN: Final[tuple[tuple[int, ...], ...]] = _gen_ray_between()
BISHOP_MASKS: Final[tuple[int, ...]] = _gen_bishop_masks()
ROOK_MASKS: Final[tuple[int, ...]] = _gen_rook_masks()


# ---------------------------------------------------------------------------
# Magic Bitboard Slider Masks & Slow Attacks
# ---------------------------------------------------------------------------


def _bishop_slider_mask(sq: int) -> int:
    r, f = divmod(sq, 8)
    mask = 0
    for dr, df in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        nr, nf = r + dr, f + df
        while 0 < nr < 7 and 0 < nf < 7:
            mask |= 1 << (nr * 8 + nf)
            nr += dr
            nf += df
    return mask


def _rook_slider_mask(sq: int) -> int:
    r, f = divmod(sq, 8)
    mask = 0
    for nr in range(r + 1, 7):
        mask |= 1 << (nr * 8 + f)
    for nr in range(r - 1, 0, -1):
        mask |= 1 << (nr * 8 + f)
    for nf in range(f + 1, 7):
        mask |= 1 << (r * 8 + nf)
    for nf in range(f - 1, 0, -1):
        mask |= 1 << (r * 8 + nf)
    return mask


def _bishop_attacks_slow(sq: int, blockers: int) -> int:
    r, f = divmod(sq, 8)
    attacks = 0
    for dr, df in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        nr, nf = r + dr, f + df
        while 0 <= nr < 8 and 0 <= nf < 8:
            target = nr * 8 + nf
            attacks |= 1 << target
            if (blockers >> target) & 1:
                break
            nr += dr
            nf += df
    return attacks


def _rook_attacks_slow(sq: int, blockers: int) -> int:
    r, f = divmod(sq, 8)
    attacks = 0
    for nr in range(r + 1, 8):
        target = nr * 8 + f
        attacks |= 1 << target
        if (blockers >> target) & 1:
            break
    for nr in range(r - 1, -1, -1):
        target = nr * 8 + f
        attacks |= 1 << target
        if (blockers >> target) & 1:
            break
    for nf in range(f + 1, 8):
        target = r * 8 + nf
        attacks |= 1 << target
        if (blockers >> target) & 1:
            break
    for nf in range(f - 1, -1, -1):
        target = r * 8 + nf
        attacks |= 1 << target
        if (blockers >> target) & 1:
            break
    return attacks


# ---------------------------------------------------------------------------
# Build Magic Tables at Module Import Time
# ---------------------------------------------------------------------------

_bishop_table: list[int] = [0] * 5248
_bishop_masks_magic: list[int] = [0] * 64
_bishop_shifts: list[int] = [0] * 64
_bishop_offsets: list[int] = [0] * 64

_offset = 0
for _sq in range(64):
    _m = _bishop_slider_mask(_sq)
    _bits = _m.bit_count()
    _n = 1 << _bits
    _shift = 64 - _bits
    _magic = BISHOP_MAGICS[_sq]
    _bishop_masks_magic[_sq] = _m
    _bishop_shifts[_sq] = _shift
    _bishop_offsets[_sq] = _offset

    _occ = 0
    for _ in range(_n):
        _idx = ((_occ * _magic) & 0xFFFFFFFFFFFFFFFF) >> _shift
        _bishop_table[_offset + _idx] = _bishop_attacks_slow(_sq, _occ)
        _occ = (_occ - _m) & _m
    _offset += _n

BISHOP_TABLE: Final[tuple[int, ...]] = tuple(_bishop_table)
BISHOP_MASKS_MAGIC: Final[tuple[int, ...]] = tuple(_bishop_masks_magic)
BISHOP_SHIFTS: Final[tuple[int, ...]] = tuple(_bishop_shifts)
BISHOP_OFFSETS: Final[tuple[int, ...]] = tuple(_bishop_offsets)


_rook_table: list[int] = [0] * 102400
_rook_masks_magic: list[int] = [0] * 64
_rook_shifts: list[int] = [0] * 64
_rook_offsets: list[int] = [0] * 64

_offset = 0
for _sq in range(64):
    _m = _rook_slider_mask(_sq)
    _bits = _m.bit_count()
    _n = 1 << _bits
    _shift = 64 - _bits
    _magic = ROOK_MAGICS[_sq]
    _rook_masks_magic[_sq] = _m
    _rook_shifts[_sq] = _shift
    _rook_offsets[_sq] = _offset

    _occ = 0
    for _ in range(_n):
        _idx = ((_occ * _magic) & 0xFFFFFFFFFFFFFFFF) >> _shift
        _rook_table[_offset + _idx] = _rook_attacks_slow(_sq, _occ)
        _occ = (_occ - _m) & _m
    _offset += _n

ROOK_TABLE: Final[tuple[int, ...]] = tuple(_rook_table)
ROOK_MASKS_MAGIC: Final[tuple[int, ...]] = tuple(_rook_masks_magic)
ROOK_SHIFTS: Final[tuple[int, ...]] = tuple(_rook_shifts)
ROOK_OFFSETS: Final[tuple[int, ...]] = tuple(_rook_offsets)


# ---------------------------------------------------------------------------
# Numba-compatible Arrays
# ---------------------------------------------------------------------------

NB_KNIGHT_ATTACKS = np.array(KNIGHT_ATTACKS, dtype=np.uint64)
NB_KING_ATTACKS = np.array(KING_ATTACKS, dtype=np.uint64)
NB_PAWN_ATTACKS = np.array(PAWN_ATTACKS, dtype=np.uint64)
NB_RAY_BETWEEN = np.array(RAY_BETWEEN, dtype=np.uint64)

NB_BISHOP_TABLE = np.array(BISHOP_TABLE, dtype=np.uint64)
NB_BISHOP_MASKS_MAGIC = np.array(BISHOP_MASKS_MAGIC, dtype=np.uint64)
NB_BISHOP_MAGICS = np.array(BISHOP_MAGICS, dtype=np.uint64)
NB_BISHOP_SHIFTS = np.array(BISHOP_SHIFTS, dtype=np.uint8)
NB_BISHOP_OFFSETS = np.array(BISHOP_OFFSETS, dtype=np.uint32)

NB_ROOK_TABLE = np.array(ROOK_TABLE, dtype=np.uint64)
NB_ROOK_MASKS_MAGIC = np.array(ROOK_MASKS_MAGIC, dtype=np.uint64)
NB_ROOK_MAGICS = np.array(ROOK_MAGICS, dtype=np.uint64)
NB_ROOK_SHIFTS = np.array(ROOK_SHIFTS, dtype=np.uint8)
NB_ROOK_OFFSETS = np.array(ROOK_OFFSETS, dtype=np.uint32)

# ---------------------------------------------------------------------------
# Lookup Functions
# ---------------------------------------------------------------------------


@njit(cache=False)
def bishop_attacks(sq: int, occupied: int | np.uint64) -> int:
    """Return bishop attacks bitboard for square `sq` given board occupancy."""
    mask = NB_BISHOP_MASKS_MAGIC[sq]
    magic = NB_BISHOP_MAGICS[sq]
    shift = NB_BISHOP_SHIFTS[sq]
    offset = NB_BISHOP_OFFSETS[sq]
    idx = ((occupied & mask) * magic) >> shift
    return int(NB_BISHOP_TABLE[offset + idx])


@njit(cache=False)
def rook_attacks(sq: int, occupied: int | np.uint64) -> int:
    """Return rook attacks bitboard for square `sq` given board occupancy."""
    mask = NB_ROOK_MASKS_MAGIC[sq]
    magic = NB_ROOK_MAGICS[sq]
    shift = NB_ROOK_SHIFTS[sq]
    offset = NB_ROOK_OFFSETS[sq]
    idx = ((occupied & mask) * magic) >> shift
    return int(NB_ROOK_TABLE[offset + idx])


@njit(cache=False)
def queen_attacks(sq: int, occupied: int | np.uint64) -> int:
    """Return queen attacks bitboard for square `sq` given board occupancy."""
    return int(bishop_attacks(sq, occupied) | rook_attacks(sq, occupied))
