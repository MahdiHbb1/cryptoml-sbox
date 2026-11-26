"""Konsistensi evaluator, baseline AES/K44, dan Composite Score."""
from pathlib import Path

import numpy as np
import pytest

from src.composite_score import (
    COMPOSITE_SCORE_VERSION,
    compute_composite_score as cs_new,
)
from src.sbox_metrics import (
    EVALUATOR_VERSION,
    evaluate_sbox as eval_new,
)
from src.train_xgboost import (
    compute_composite_score as cs_old,
    evaluate_sbox_default as eval_old,
)

BASE_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "baselines"
)

def test_versions_exist():
    assert isinstance(EVALUATOR_VERSION, str)
    assert isinstance(COMPOSITE_SCORE_VERSION, str)

    assert len(EVALUATOR_VERSION) > 0
    assert len(COMPOSITE_SCORE_VERSION) > 0


def test_float_input_rejected():
    """Input bertipe float harus ditolak."""

    with pytest.raises(TypeError):
        eval_new(
            np.linspace(
                0,
                255,
                256,
                dtype=float,
            )
        )


def _baseline(name):
    path = BASE_DIR / f"{name}.npy"
    if not path.exists():
        pytest.skip(f"{path} belum ada; simpan dulu tabel baseline.")
    return np.load(path).astype(np.uint8)


def _random_perms(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.permutation(256).astype(np.uint8) for _ in range(n)]


def _diff(a, b):
    assert set(a) == set(b), f"Field beda: {set(a) ^ set(b)}"
    out = {}
    for k in a:
        if isinstance(a[k], bool) or isinstance(b[k], bool):
            out[k] = 0.0 if a[k] == b[k] else 1.0
        else:
            out[k] = abs(float(a[k]) - float(b[k]))
    return out


@pytest.mark.parametrize("sbox", _random_perms(), ids=[f"perm{i}" for i in range(20)])
def test_new_vs_fallback_random(sbox):
    assert max(_diff(eval_new(sbox), eval_old(sbox)).values()) < 1e-9


@pytest.mark.parametrize("name", ["aes_standard", "k44"])
def test_new_vs_fallback_baselines(name):
    s = _baseline(name)
    assert max(_diff(eval_new(s), eval_old(s)).values()) < 1e-9


def test_aes_fips197_points():
    s = _baseline("aes_standard")
    assert (int(s[0x00]), int(s[0x01]), int(s[0x53]), int(s[0xFF])) == (0x63, 0x7C, 0xED, 0x16)


def test_aes_reference_values():
    m = eval_new(_baseline("aes_standard"))
    assert (m["nl"], m["bic_nl"], m["du"], m["fixed_points"]) == (112, 112, 4, 0)
    assert m["lap"] == pytest.approx(0.0625)
    assert m["dap"] == pytest.approx(0.015625)
    assert m["sac_max_err"] == pytest.approx(0.0625)
    assert m["sac_mean"] == pytest.approx(0.50488, abs=1e-5)
    assert m["bic_sac"] == pytest.approx(0.50460, abs=1e-5)


def test_k44_reference_values():
    m = eval_new(_baseline("k44"))
    assert (m["nl"], m["bic_nl"], m["du"], m["fixed_points"]) == (112, 112, 4, 1)
    assert m["sac_max_err"] == pytest.approx(0.0625)
    assert m["sac_mean"] == pytest.approx(0.50073, abs=1e-5)
    assert m["bic_sac"] == pytest.approx(0.50237, abs=1e-5)


@pytest.mark.parametrize("fn", [eval_new, eval_old], ids=["sbox_metrics", "fallback"])
def test_invalid_inputs_rejected(fn):
    with pytest.raises(ValueError):
        fn(np.arange(255, dtype=np.uint8))  # panjang salah
    with pytest.raises(ValueError):
        fn(np.arange(256) + 1)  # ada nilai 256
    dup = np.arange(256, dtype=np.uint8)
    dup[1] = dup[0]
    with pytest.raises(ValueError):
        fn(dup)  # tidak bijektif


def test_composite_hand_computed():
    m = dict(bijective=True, nl=100, bic_nl=100, du=8, sac_max_err=0.1,
             sac_mean=0.5, bic_sac=0.5, fixed_points=0, whs_cost=20.0)
    assert cs_new(m) == pytest.approx(96.0)


@pytest.mark.parametrize("sbox", _random_perms(10, seed=1), ids=[f"perm{i}" for i in range(10)])
def test_composite_matches_train_xgboost(sbox):
    m = eval_new(sbox)
    assert cs_new(m) == pytest.approx(cs_old(m), abs=1e-12)


def test_composite_rejects_non_bijective():
    m = {**eval_new(_random_perms(1)[0]), "bijective": False}
    with pytest.raises(ValueError):
        cs_new(m)


def test_report_worst_case_diff(capsys):
    sboxes = _random_perms()
    for n in ("aes_standard", "k44"):
        if (BASE_DIR / f"{n}.npy").exists():
            sboxes.append(np.load(BASE_DIR / f"{n}.npy").astype(np.uint8))
    worst = {}
    for s in sboxes:
        for k, v in _diff(eval_new(s), eval_old(s)).items():
            worst[k] = max(worst.get(k, 0.0), v)
    with capsys.disabled():
        print("\nSelisih terbesar sbox_metrics vs fallback:")
        for k, v in worst.items():
            print(f"  {k:14s} {v:.3e}")