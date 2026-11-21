"""Generator kandidat S-Box jalur Modifikasi AES pada medan hingga GF(2^8).

Menyediakan enumerasi seluruh 30 polinomial irreducible berderajat 8,
tabel invers multiplikatif GF(2^8), pembangkit matriks affine invertibel
GL(8, F_2), konstruksi baseline AES Standar & K44 (Alamsyah et al.),
serta pencari konstanta bebas fixed point.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

AES_IRREDUCIBLE_POLY = 0x11B
AES_AFFINE_CONSTANT = 0x63
_AES_ROW_INTS = (0xF1, 0xE3, 0xC7, 0x8F, 0x1F, 0x3E, 0x7C, 0xF8)
_K44_MSB_ROW_INTS = (0x57, 0xAB, 0xD5, 0xEA, 0x75, 0xBA, 0x5D, 0xAE)
_BIT_WEIGHTS = (1 << np.arange(8, dtype=np.uint16)).astype(np.uint8)


def _is_irreducible_deg8(poly: int) -> bool:
    if not (0x100 < poly <= 0x1FF) or (poly & 1) == 0:
        return False
    for divisor in range(2, 32):  # Semua polinomial berderajat 1..4
        rem = poly
        deg_div = divisor.bit_length() - 1
        while rem.bit_length() - 1 >= deg_div:
            rem ^= divisor << ((rem.bit_length() - 1) - deg_div)
        if rem == 0:
            return False
    return True


IRREDUCIBLE_POLYS_GF256: tuple[int, ...] = tuple(
    p for p in range(0x101, 0x200, 2) if _is_irreducible_deg8(p)
)
_IRREDUCIBLE_SET = frozenset(IRREDUCIBLE_POLYS_GF256)


def gf256_mul(a: int, b: int, poly: int = AES_IRREDUCIBLE_POLY) -> int:
    """Perkalian dua elemen GF(2^8) modulo polinomial irreducible `poly`."""
    mod = poly & 0xFF
    res = 0
    aa, bb = a & 0xFF, b & 0xFF
    for _ in range(8):
        if bb & 1:
            res ^= aa
        hi = aa & 0x80
        aa = (aa << 1) & 0xFF
        if hi:
            aa ^= mod
        bb >>= 1
    return res


@lru_cache(maxsize=32)
def gf256_inverse_table(poly: int = AES_IRREDUCIBLE_POLY) -> np.ndarray:
    """Mengembalikan tabel invers multiplikatif GF(2^8) dengan 0^-1 := 0."""
    if poly not in _IRREDUCIBLE_SET:
        raise ValueError(f"Polinomial 0x{poly:03X} bukan polinomial irreducible GF(2^8).")
    inv = np.zeros(256, dtype=np.uint8)
    for a in range(1, 256):
        if inv[a] != 0:
            continue
        for b in range(a, 256):
            if gf256_mul(a, b, poly) == 1:
                inv[a] = b
                inv[b] = a
                break
    inv.setflags(write=False)
    return inv


@lru_cache(maxsize=32)
def _gf256_inverse_bits(poly: int) -> np.ndarray:
    inv = gf256_inverse_table(poly)
    bits = ((inv[:, None] >> np.arange(8, dtype=np.uint8)[None, :]) & 1).astype(np.uint8)
    bits.setflags(write=False)
    return bits


def gf2_rank(matrix: np.ndarray) -> int:
    """Menghitung rank matriks biner 8x8 di atas GF(2)."""
    mat = (np.asarray(matrix, dtype=np.uint8) & 1).copy()
    if mat.shape != (8, 8):
        raise ValueError(f"Matriks affine harus berukuran (8, 8), diterima {mat.shape}.")
    rank = 0
    for col in range(8):
        pivots = np.where(mat[rank:, col] == 1)[0]
        if pivots.size == 0:
            continue
        piv_row = rank + int(pivots[0])
        if piv_row != rank:
            mat[[rank, piv_row]] = mat[[piv_row, rank]]
        for row in range(8):
            if row != rank and mat[row, col] == 1:
                mat[row] ^= mat[rank]
        rank += 1
        if rank == 8:
            break
    return rank


def rows_to_matrix(row_ints: tuple[int, ...] | list[int], msb_columns: bool = False) -> np.ndarray:
    """Mengonversi 8 bilangan bulat baris menjadi matriks biner (8, 8) LSB-first."""
    if len(row_ints) != 8:
        raise ValueError("Jumlah baris matriks harus tepat 8.")
    mat = np.zeros((8, 8), dtype=np.uint8)
    for i, val in enumerate(row_ints):
        for bit in range(8):
            src_bit = (7 - bit) if msb_columns else bit
            mat[i, bit] = (int(val) >> src_bit) & 1
    return mat


def random_invertible_matrix(rng: np.random.Generator) -> np.ndarray:
    """Membangkitkan matriks 8x8 invertibel (rank 8 di GF(2)) secara seragam."""
    while True:
        mat = rng.integers(0, 2, size=(8, 8), dtype=np.uint8)
        if gf2_rank(mat) == 8:
            return mat


def build_aes_affine_sbox(
    matrix: np.ndarray,
    constant: int = AES_AFFINE_CONSTANT,
    poly: int = AES_IRREDUCIBLE_POLY,
) -> np.ndarray:
    """Membentuk S-box S(x) = matrix * inv(x) XOR constant di GF(2^8) (LSB-first)."""
    if poly not in _IRREDUCIBLE_SET:
        raise ValueError(f"Polinomial 0x{poly:03X} bukan polinomial irreducible GF(2^8).")
    mat = np.asarray(matrix, dtype=np.uint8) & 1
    if gf2_rank(mat) != 8:
        raise ValueError("Matriks affine harus invertible (rank 8 di GF(2)).")
    inv_bits = _gf256_inverse_bits(poly)
    out_bits = (inv_bits @ mat.T) & 1
    sbox = (out_bits @ _BIT_WEIGHTS).astype(np.uint8) ^ np.uint8(constant & 0xFF)
    return sbox.astype(np.uint8)


def build_standard_aes_sbox() -> np.ndarray:
    """Membentuk S-Box AES standar sesuai FIPS-197."""
    mat = rows_to_matrix(_AES_ROW_INTS, msb_columns=False)
    return build_aes_affine_sbox(mat, constant=AES_AFFINE_CONSTANT, poly=AES_IRREDUCIBLE_POLY)


def build_k44_sbox() -> np.ndarray:
    """Membentuk S-Box44 (matriks sirkulan K44) sesuai Alamsyah et al. (2024/2025)."""
    mat = rows_to_matrix(_K44_MSB_ROW_INTS, msb_columns=True)
    return build_aes_affine_sbox(mat, constant=AES_AFFINE_CONSTANT, poly=AES_IRREDUCIBLE_POLY)


def find_zero_fixed_point_constant(
    base_sbox: np.ndarray,
    preferred: int = AES_AFFINE_CONSTANT,
) -> Optional[int]:
    """Mencari konstanta C in [0..255] agar S(x)=base(x)^C bebas fixed & opposite-fixed point."""
    base = np.asarray(base_sbox, dtype=np.uint8)
    x = np.arange(256, dtype=np.uint8)
    forbidden = set((base ^ x).tolist()) | set((base ^ (x ^ np.uint8(0xFF))).tolist())
    if preferred not in forbidden:
        return int(preferred)
    for c in range(256):
        if c not in forbidden:
            return c
    return None


def generate_aes_affine_candidates(
    n: int,
    seed: int = 42,
    polys: Optional[tuple[int, ...] | list[int]] = None,
    avoid_fixed_points: bool = True,
) -> list[dict]:
    """Membangkitkan `n` kandidat S-box unik jalur Modifikasi AES untuk dataset/seleksi ML."""
    if n <= 0:
        return []
    poly_pool = tuple(polys) if polys is not None else IRREDUCIBLE_POLYS_GF256
    for p in poly_pool:
        if p not in _IRREDUCIBLE_SET:
            raise ValueError(f"Polinomial 0x{p:03X} bukan polinomial irreducible GF(2^8).")

    rng = np.random.default_rng(seed)
    candidates: list[dict] = []
    seen: set[bytes] = set()
    max_attempts = max(n * 20, 200)

    attempts = 0
    while len(candidates) < n and attempts < max_attempts:
        attempts += 1
        poly = int(poly_pool[int(rng.integers(0, len(poly_pool)))])
        mat = random_invertible_matrix(rng)
        base = build_aes_affine_sbox(mat, constant=0, poly=poly)
        if avoid_fixed_points:
            const = find_zero_fixed_point_constant(base, preferred=int(rng.integers(0, 256)))
            if const is None:
                continue
        else:
            const = int(rng.integers(0, 256))
        sbox = (base ^ np.uint8(const)).astype(np.uint8)
        key = bytes(sbox.tolist())
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "sbox": sbox,
                "poly": poly,
                "matrix": mat,
                "constant": const,
                "source": "aes_affine",
            }
        )
    return candidates
