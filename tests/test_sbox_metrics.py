import numpy as np
import pytest

from src.sbox_metrics import evaluate_sbox


def _gmul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        if a & 0x100:
            a ^= 0x11B
        b >>= 1
    return r


def aes_sbox():
    inv = [0] * 256
    for a in range(1, 256):
        for b in range(1, 256):
            if _gmul(a, b) == 1:
                inv[a] = b
                break
    rotl = lambda x, n: ((x << n) | (x >> (8 - n))) & 0xFF
    out = [inv[x] ^ rotl(inv[x], 1) ^ rotl(inv[x], 2) ^ rotl(inv[x], 3)
           ^ rotl(inv[x], 4) ^ 0x63 for x in range(256)]
    return np.array(out, dtype=np.uint8)


def test_aes_standard():
    s = aes_sbox()
    assert s[0] == 0x63 and s[1] == 0x7C
    r = evaluate_sbox(s)
    assert r["bijective"] and r["balanced"]
    assert r["nl"] == 112 and r["bic_nl"] == 112
    assert r["du"] == 4
    assert r["dap"] == pytest.approx(0.015625)
    assert r["lap"] == pytest.approx(0.0625)
    assert r["sac_mean"] == pytest.approx(0.5049, abs=1e-4)
    assert r["fixed_points"] == 0


def test_identity_is_bad():
    r = evaluate_sbox(np.arange(256, dtype=np.uint8))
    assert r["bijective"] and r["nl"] == 0 and r["du"] == 256


def test_matches_fallback_evaluator():
    """Hasil harus identik dengan evaluate_sbox_default agar dataset tidak bergeser."""
    from src.train_xgboost import evaluate_sbox_default

    rng = np.random.default_rng(0)
    boxes = [aes_sbox()] + [rng.permutation(256).astype(np.uint8) for _ in range(5)]
    for sb in boxes:
        a, b = evaluate_sbox(sb), evaluate_sbox_default(sb)
        assert a.keys() == b.keys()
        for k in a:
            assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-9), k


def test_pipeline_uses_this_module():
    """ImportError ditelan try/except, jadi pastikan resolver benar-benar memilih modul ini."""
    from src.dataset_builder import resolve_evaluator
    assert resolve_evaluator() is evaluate_sbox


def test_k44_alamsyah():
    """Validasi baseline S-Box44 (K44 Alamsyah et al.) sesuai ARCHITECTURE_CONTRACT.md."""
    from src.aes_affine_generator import build_k44_sbox

    k44 = build_k44_sbox()
    r = evaluate_sbox(k44)
    assert r["bijective"] and r["balanced"]
    assert r["nl"] == 112 and r["bic_nl"] == 112
    assert r["du"] == 4
    assert r["dap"] == pytest.approx(0.015625)
    assert r["lap"] == pytest.approx(0.0625)
    assert r["sac_mean"] == pytest.approx(0.50073, abs=1e-4)
    assert r["bic_sac"] == pytest.approx(0.50237, abs=1e-4)
    assert r["fixed_points"] == 1
