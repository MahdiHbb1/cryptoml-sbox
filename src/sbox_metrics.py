"""Evaluator Metrik 1 untuk S-box 8x8 (kontrak: evaluate_sbox).

Sengaja mandiri: tidak mengimpor train_xgboost / dataset_builder
agar tidak terjadi circular import (ImportError yang ditelan try/except).
"""
import numpy as np

_N = 256
_IDX = np.arange(_N)
_PAR = np.array([bin(i).count("1") & 1 for i in range(_N)], dtype=np.uint8)

# Hadamard H[a, x] = (-1)^{parity(a & x)}; float32 agar matmul memakai BLAS.
# PENTING: konversi ke float SEBELUM 1 - 2*p (uint8 akan wraparound).
_H = 1.0 - 2.0 * _PAR[_IDX[:, None] & _IDX[None, :]].astype(np.float32)

_PAIR_MASKS = [(1 << j) | (1 << k) for j in range(8) for k in range(j + 1, 8)]
_PAIR_ROWS = [m - 1 for m in _PAIR_MASKS]  # baris Walsh untuk mask m


def _walsh_all_masks(sbox: np.ndarray) -> np.ndarray:
    """W[b-1, a] untuk mask b = 1..255 dan a = 0..255 (int32)."""
    masks = np.arange(1, _N)
    f = _PAR[masks[:, None] & sbox[None, :]].astype(np.float32)   # (255, 256)
    F = 1.0 - 2.0 * f
    return np.rint(F @ _H).astype(np.int32)                        # H simetris


def evaluate_sbox(sbox: np.ndarray) -> dict:
    arr = np.asarray(sbox)
    if arr.shape != (_N,) or arr.min() < 0 or arr.max() > 255:
        raise ValueError("S-box harus berbentuk (256,) dengan nilai 0..255")
    sb = arr.astype(np.uint8)
    bijective = len(np.unique(sb)) == _N

    # Walsh: NL, balance, LAP, WHS
    W = _walsh_all_masks(sb)
    absW = np.abs(W)
    max_w = int(absW.max())
    nl = int(128 - max_w // 2)
    lap = max_w / 512.0
    balanced = bool(np.all(W[:, 0] == 0))
    whs_cost = float(np.mean((absW - 16.0) ** 4) ** 0.25)   # sama dengan fallback

    # BIC-NL: minimum NL atas 28 pasangan bit output
    bic_nl = int(128 - int(absW[_PAIR_ROWS].max()) // 2)

    # SAC & BIC-SAC
    d = np.stack([sb ^ sb[_IDX ^ (1 << i)] for i in range(8)])     # (8, 256)
    sac = ((d[:, :, None] >> np.arange(8)) & 1).mean(axis=1)       # (8, 8)
    sac_mean = float(sac.mean())
    sac_max_err = float(np.abs(sac - 0.5).max())
    bic_sac = float(np.mean([_PAR[d & m].mean(axis=1) for m in _PAIR_MASKS]))

    # DDT
    du = 0
    for dx in range(1, _N):
        du = max(du, int(np.bincount(sb ^ sb[_IDX ^ dx], minlength=_N).max()))

    return {
        "bijective": bool(bijective),
        "balanced": balanced,
        "nl": nl,
        "sac_mean": sac_mean,
        "sac_max_err": sac_max_err,
        "bic_nl": bic_nl,
        "bic_sac": bic_sac,
        "du": du,
        "dap": float(du / 256.0),
        "lap": float(lap),
        "fixed_points": int(np.sum(sb == _IDX)),
        "whs_cost": whs_cost,
    }