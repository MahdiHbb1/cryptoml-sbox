import numpy as np
import pytest

from src.aes_affine_generator import (
    IRREDUCIBLE_POLYS_GF256,
    build_aes_affine_sbox,
    build_k44_sbox,
    build_standard_aes_sbox,
    find_zero_fixed_point_constant,
    generate_aes_affine_candidates,
    gf2_rank,
    gf256_inverse_table,
    gf256_mul,
    random_invertible_matrix,
)


def _is_bijective_and_balanced(sbox: np.ndarray) -> bool:
    if sbox.dtype != np.uint8 or sbox.shape != (256,):
        return False
    if len(set(sbox.tolist())) != 256:
        return False
    for bit in range(8):
        if int(np.sum((sbox >> bit) & 1)) != 128:
            return False
    return True


def test_enumerates_all_30_irreducible_degree_8_polynomials():
    assert len(IRREDUCIBLE_POLYS_GF256) == 30
    assert 0x11B in IRREDUCIBLE_POLYS_GF256
    for poly in IRREDUCIBLE_POLYS_GF256:
        assert 0x100 < poly <= 0x1FF
        inv = gf256_inverse_table(poly)
        assert inv[0] == 0
        for a in range(1, 256):
            assert gf256_mul(a, int(inv[a]), poly) == 1


def test_standard_aes_sbox_matches_fips197_entries():
    sbox = build_standard_aes_sbox()
    assert _is_bijective_and_balanced(sbox)
    assert sbox[:5].tolist() == [0x63, 0x7C, 0x77, 0x7B, 0xF2]
    assert int(sbox[255]) == 0x16
    x = np.arange(256, dtype=np.uint8)
    assert int(np.sum(sbox == x)) == 0
    assert int(np.sum(sbox == (x ^ 0xFF))) == 0


def test_k44_sbox_matches_alamsyah_reference_table():
    sbox = build_k44_sbox()
    assert _is_bijective_and_balanced(sbox)
    assert sbox[:12].tolist() == [99, 205, 85, 71, 25, 127, 113, 219, 63, 244, 109, 159]


def test_random_invertible_matrix_is_rank_8_over_gf2():
    rng = np.random.default_rng(123)
    for _ in range(20):
        mat = random_invertible_matrix(rng)
        assert mat.shape == (8, 8)
        assert mat.dtype == np.uint8
        assert set(np.unique(mat).tolist()).issubset({0, 1})
        assert gf2_rank(mat) == 8


def test_find_zero_fixed_point_constant_eliminates_fixed_and_opposite_points():
    rng = np.random.default_rng(7)
    mat = random_invertible_matrix(rng)
    base = build_aes_affine_sbox(mat, constant=0, poly=0x11B)
    const = find_zero_fixed_point_constant(base)
    assert const is not None
    sbox = (base ^ np.uint8(const)).astype(np.uint8)
    x = np.arange(256, dtype=np.uint8)
    assert int(np.sum(sbox == x)) == 0
    assert int(np.sum(sbox == (x ^ 0xFF))) == 0


def test_generate_aes_affine_candidates_produces_unique_bijective_sboxes():
    candidates = generate_aes_affine_candidates(n=20, seed=42, avoid_fixed_points=True)
    assert len(candidates) == 20
    seen = set()
    x = np.arange(256, dtype=np.uint8)
    for item in candidates:
        sbox = item["sbox"]
        assert _is_bijective_and_balanced(sbox)
        assert item["poly"] in IRREDUCIBLE_POLYS_GF256
        assert int(np.sum(sbox == x)) == 0
        assert int(np.sum(sbox == (x ^ 0xFF))) == 0
        seen.add(bytes(sbox.tolist()))
    assert len(seen) == 20


def test_rejects_singular_matrix_and_reducible_polynomial():
    singular = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="invertible|rank"):
        build_aes_affine_sbox(singular, constant=0x63, poly=0x11B)
    eye = np.eye(8, dtype=np.uint8)
    with pytest.raises(ValueError, match="irreducible"):
        build_aes_affine_sbox(eye, constant=0x63, poly=0x101)
