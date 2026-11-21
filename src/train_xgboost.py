"""Training, evaluasi, dan penyaringan kandidat S-Box menggunakan XGBoost.

Modul ini merupakan implementasi jobdesk Mahdi (Issue #6):
- Memuat dataset bersama ``data/sbox_dataset.csv`` (kontrak: ``f_*`` + ``composite_score``),
  atau otomatis membangkitkan dataset sintetis beragam jika file belum tersedia.
- Melatih & mengevaluasi ``XGBRegressor`` (opsional hyperparameter tuning) dengan
  split train/test ``random_state=42`` agar adil dibandingkan dengan Random Forest & MLP.
- Menggunakan model XGBoost terlatih untuk menyaring ribuan kandidat S-box secara cepat,
  mengevaluasi metrik penuh hanya pada ``top_k`` terbaik, lalu menyimpan S-box pemenang
  dalam format baku ``np.uint8`` berukuran ``(256,)`` ke ``results/sbox_xgb.npy``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

from src.aes_affine_generator import generate_aes_affine_candidates
from src.aes_cipher_runner import validate_sbox

_X_IDX = np.arange(256, dtype=np.int32)
_BITS_IDX = np.arange(8, dtype=np.int32)
_FAST_MASKS = np.array(
    [1, 2, 4, 8, 16, 32, 64, 128, 3, 5, 12, 17, 85, 170, 15, 240],
    dtype=np.int32,
)
_ALL_MASKS = np.arange(1, 256, dtype=np.int32)


def _fwht_batch(matrix_pm1: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard Transform tervektorisasi untuk batch (M, 256)."""
    a = matrix_pm1.astype(np.int32, copy=True)
    m = a.shape[0]
    h = 1
    while h < 256:
        a = a.reshape(m, -1, 2 * h)
        u = a[:, :, :h]
        v = a[:, :, h:]
        a = np.concatenate([u + v, u - v], axis=2)
        h *= 2
    return a.reshape(m, 256)


def _parity_pm1(sbox_int: np.ndarray, masks: np.ndarray) -> np.ndarray:
    dot = np.bitwise_and(masks[:, None], sbox_int[None, :])
    par = np.zeros_like(dot)
    for k in range(8):
        par ^= (dot >> k) & 1
    return 1 - 2 * par


def _sac_matrix(sbox_int: np.ndarray) -> np.ndarray:
    diffs = sbox_int[_X_IDX[:, None] ^ (1 << _BITS_IDX[None, :])] ^ sbox_int[:, None]
    return np.array(
        [np.mean((diffs[:, i][:, None] >> _BITS_IDX[None, :]) & 1, axis=0) for i in range(8)],
        dtype=np.float64,
    )


def extract_fast_features_default(sbox: np.ndarray) -> np.ndarray:
    """Ekstraksi fitur cepat (~0.7 ms/S-box) untuk prediksi kualitas kriptografi."""
    sb = validate_sbox(sbox).astype(np.int32)
    walsh_sub = np.abs(_fwht_batch(_parity_pm1(sb, _FAST_MASKS)))
    sub_nls = (128 - np.max(walsh_sub, axis=1) // 2).astype(np.float64)
    sub_whs = np.sqrt(np.mean((walsh_sub - 16.0) ** 2, axis=1))
    sac_mat = _sac_matrix(sb)
    fixed = float(np.sum(sb == _X_IDX))
    opp_fixed = float(np.sum(sb == (_X_IDX ^ 0xFF)))
    summary = np.array(
        [
            float(np.min(sub_nls)),
            float(np.mean(sub_nls)),
            float(np.std(sub_nls)),
            float(np.mean(sac_mat)),
            float(np.max(np.abs(sac_mat - 0.5))),
            float(np.std(sac_mat)),
            fixed,
            opp_fixed,
        ],
        dtype=np.float64,
    )
    return np.concatenate([sub_nls, sub_whs, summary])


def evaluate_sbox_default(sbox: np.ndarray) -> dict:
    """Fallback evaluator Metrik 1 sesuai kontrak docs/ARCHITECTURE_CONTRACT.md."""
    sb = validate_sbox(sbox).astype(np.int32)
    balanced = all(int(np.sum((sb >> k) & 1)) == 128 for k in range(8))

    walsh = _fwht_batch(_parity_pm1(sb, _ALL_MASKS))
    abs_w = np.abs(walsh)
    max_w = int(np.max(abs_w))
    nl = int(128 - max_w // 2)
    lap = float(max_w / 512.0)
    whs_cost = float(np.mean((abs_w - 16.0) ** 4) ** 0.25)

    ddt = np.zeros((256, 256), dtype=np.int32)
    for dx in range(1, 256):
        ddt[dx] = np.bincount(sb ^ sb[_X_IDX ^ dx], minlength=256)
    du = int(np.max(ddt[1:]))
    dap = float(du / 256.0)

    sac_mat = _sac_matrix(sb)
    sac_mean = float(np.mean(sac_mat))
    sac_max_err = float(np.max(np.abs(sac_mat - 0.5)))

    bic_nls: list[int] = []
    bic_sacs: list[float] = []
    for j in range(8):
        for k in range(j + 1, 8):
            idx = ((1 << j) | (1 << k)) - 1
            bic_nls.append(int(128 - int(np.max(abs_w[idx])) // 2))
            pair_y = ((sb >> j) & 1) ^ ((sb >> k) & 1)
            for i in range(8):
                bic_sacs.append(float(np.mean(pair_y ^ pair_y[_X_IDX ^ (1 << i)])))

    fixed_points = int(np.sum(sb == _X_IDX))
    return {
        "bijective": True,
        "balanced": balanced,
        "nl": nl,
        "sac_mean": sac_mean,
        "sac_max_err": sac_max_err,
        "bic_nl": int(min(bic_nls)),
        "bic_sac": float(np.mean(bic_sacs)),
        "du": du,
        "dap": dap,
        "lap": lap,
        "fixed_points": fixed_points,
        "whs_cost": whs_cost,
    }


def _resolve_team_functions() -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], dict]]:
    """Otomatis memakai modul Naila & Yafi bila sudah tersedia di repo, jika belum pakai fallback."""
    try:
        from src.dataset_builder import extract_fast_features as feat_fn  # type: ignore
    except ImportError:
        feat_fn = extract_fast_features_default
    try:
        from src.sbox_metrics import evaluate_sbox as eval_fn  # type: ignore
    except ImportError:
        eval_fn = evaluate_sbox_default
    return feat_fn, eval_fn


def compute_composite_score(metrics: dict) -> float:
    """Skor gabungan Metrik 1 (semakin tinggi semakin kuat)."""
    return float(
        metrics["nl"]
        + 0.25 * metrics["bic_nl"]
        - 2.0 * metrics["du"]
        - 120.0 * metrics["sac_max_err"]
        - 40.0 * abs(metrics["sac_mean"] - 0.5)
        - 30.0 * abs(metrics["bic_sac"] - 0.5)
        - 1.5 * metrics["fixed_points"]
        - 0.05 * metrics["whs_cost"]
    )


def _chaotic_logistic_sbox(r: float, x0: float, skip: int = 80) -> np.ndarray:
    val = x0
    for _ in range(skip):
        val = r * val * (1.0 - val)
    seq = np.empty(256, dtype=np.float64)
    for k in range(256):
        val = r * val * (1.0 - val)
        seq[k] = val
    return np.argsort(seq).astype(np.uint8)


def generate_candidate_pool(n: int, mode: str = "aes_affine", seed: int = 100) -> list[dict]:
    """Membangkitkan kandidat S-box bijektif untuk penyaringan oleh model XGBoost."""
    rng = np.random.default_rng(seed)
    if mode == "aes_affine":
        return generate_aes_affine_candidates(n=n, seed=seed, avoid_fixed_points=True)

    candidates: list[dict] = []
    if mode == "chaotic":
        for _ in range(n):
            r = float(rng.uniform(3.85, 3.9999))
            x0 = float(rng.uniform(0.05, 0.95))
            sbox = _chaotic_logistic_sbox(r, x0)
            candidates.append({"sbox": sbox, "source": "chaotic", "r": r, "x0": x0})
        return candidates

    if mode == "mixed":
        n_aes = n // 2
        n_chaotic = n - n_aes
        candidates.extend(generate_aes_affine_candidates(n=n_aes, seed=seed, avoid_fixed_points=True))
        candidates.extend(generate_candidate_pool(n=n_chaotic, mode="chaotic", seed=seed + 1))
        return candidates

    raise ValueError(f"pool_mode tidak dikenal: {mode!r}. Pilih 'aes_affine', 'chaotic', atau 'mixed'.")


def build_fallback_training_data(
    n_samples: int,
    feature_fn: Callable[[np.ndarray], np.ndarray],
    evaluate_fn: Callable[[np.ndarray], dict],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Membuat dataset latih beragam (chaotic, perturbed-AES, AES-affine) bila CSV bersama belum ada."""
    rng = np.random.default_rng(seed)
    n_aes = max(n_samples // 3, 10)
    n_perturbed = max(n_samples // 3, 10)
    n_chaotic = max(n_samples - n_aes - n_perturbed, 10)

    sboxes: list[np.ndarray] = []
    for item in generate_aes_affine_candidates(n=n_aes, seed=seed, avoid_fixed_points=False):
        sboxes.append(item["sbox"])

    base_aes_pool = generate_aes_affine_candidates(n=min(n_perturbed, 40), seed=seed + 10, avoid_fixed_points=False)
    for idx in range(n_perturbed):
        s = base_aes_pool[idx % len(base_aes_pool)]["sbox"].copy()
        n_swaps = int(rng.integers(1, 25))
        for _ in range(n_swaps):
            i, j = rng.choice(256, 2, replace=False)
            s[i], s[j] = s[j], s[i]
        sboxes.append(s)

    for _ in range(n_chaotic):
        sboxes.append(_chaotic_logistic_sbox(float(rng.uniform(3.85, 3.9999)), float(rng.uniform(0.05, 0.95))))

    X_rows: list[np.ndarray] = []
    y_vals: list[float] = []
    for sb in sboxes:
        X_rows.append(feature_fn(sb))
        y_vals.append(compute_composite_score(evaluate_fn(sb)))

    X = np.vstack(X_rows)
    y = np.array(y_vals, dtype=np.float64)
    feature_cols = [f"f_{i}" for i in range(X.shape[1])]
    return X, y, feature_cols


def load_dataset_csv(csv_path: str | Path, target_col: str = "composite_score") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Memuat dataset bersama sesuai kontrak docs/ARCHITECTURE_CONTRACT.md."""
    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col.startswith("f_")]
    if not feature_cols:
        raise ValueError("File dataset tidak memiliki kolom fitur berawalan 'f_'.")
    if target_col not in df.columns:
        raise ValueError(f"Kolom target {target_col!r} tidak ditemukan pada dataset.")
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df[target_col].to_numpy(dtype=np.float64)
    return X, y, feature_cols


def train_and_evaluate_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    test_size: float = 0.2,
    tune: bool = False,
) -> tuple[XGBRegressor, dict]:
    """Melatih dan mengevaluasi XGBRegressor dengan metrik R2, MAE, RMSE, dan waktu latih."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    t0 = time.perf_counter()
    if tune:
        base = XGBRegressor(objective="reg:squarederror", random_state=random_state, n_jobs=-1)
        param_dist = {
            "n_estimators": [150, 250, 400],
            "max_depth": [3, 4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=12,
            cv=3,
            scoring="r2",
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        model: XGBRegressor = search.best_estimator_
        best_params = search.best_params_
    else:
        best_params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.06,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        }
        model = XGBRegressor(
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            **best_params,
        )
        model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    preds = model.predict(X_test)
    r2 = float(r2_score(y_test, preds))
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    metrics = {
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "train_time_sec": float(train_time),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "best_params": best_params,
        "feature_importances": [float(v) for v in model.feature_importances_],
    }
    return model, metrics


def screen_and_select_best_sbox(
    model: XGBRegressor,
    candidates: list[dict],
    feature_fn: Callable[[np.ndarray], np.ndarray],
    evaluate_fn: Callable[[np.ndarray], dict],
    top_k: int = 20,
) -> dict:
    """Menyaring kandidat menggunakan prediksi XGBoost, lalu evaluasi penuh pada `top_k` teratas."""
    if not candidates:
        raise ValueError("Daftar kandidat S-box tidak boleh kosong.")
    k = max(1, min(int(top_k), len(candidates)))

    X_pool = np.vstack([feature_fn(item["sbox"]) for item in candidates])
    expected_n = getattr(model, "n_features_in_", None)
    if expected_n is not None and X_pool.shape[1] != int(expected_n):
        raise ValueError(
            f"Jumlah fitur ekstraktor ({X_pool.shape[1]}) tidak cocok dengan jumlah fitur model ({expected_n}). "
            "Pastikan extract_fast_features selaras dengan kolom 'f_*' pada sbox_dataset.csv."
        )
    preds = np.asarray(model.predict(X_pool), dtype=np.float64)
    top_indices = np.argsort(preds)[-k:][::-1]

    best_entry: Optional[dict] = None
    for idx in top_indices:
        cand = candidates[int(idx)]
        sb = validate_sbox(cand["sbox"])
        m = evaluate_fn(sb)
        score = compute_composite_score(m)
        if best_entry is None or score > best_entry["score"]:
            best_entry = {
                "sbox": sb,
                "metrics": m,
                "score": score,
                "predicted_score": float(preds[int(idx)]),
                "source": cand.get("source", "unknown"),
                "poly": cand.get("poly"),
                "constant": cand.get("constant"),
            }
    assert best_entry is not None
    return best_entry


def run_xgboost_pipeline(
    dataset_path: str | Path = Path("data/sbox_dataset.csv"),
    out_sbox_path: str | Path = Path("results/sbox_xgb.npy"),
    out_metrics_path: str | Path = Path("results/xgb_metrics.json"),
    n_train_samples: int = 300,
    n_screen_candidates: int = 1000,
    top_k: int = 20,
    tune: bool = False,
    seed: int = 42,
    pool_mode: str = "aes_affine",
) -> dict:
    """Menjalankan pipeline lengkap XGBoost dan menyimpan S-box pemenang + metrik JSON."""
    feature_fn, evaluate_fn = _resolve_team_functions()
    ds_path = Path(dataset_path)

    if ds_path.exists():
        X, y, feature_cols = load_dataset_csv(ds_path)
        dataset_origin = str(ds_path)
    else:
        X, y, feature_cols = build_fallback_training_data(
            n_samples=n_train_samples,
            feature_fn=feature_fn,
            evaluate_fn=evaluate_fn,
            seed=seed,
        )
        dataset_origin = f"fallback_synthetic(n={len(y)})"

    model, ml_metrics = train_and_evaluate_xgboost(X, y, random_state=seed, tune=tune)

    t_screen = time.perf_counter()
    candidates = generate_candidate_pool(n=n_screen_candidates, mode=pool_mode, seed=seed + 999)
    best = screen_and_select_best_sbox(
        model=model,
        candidates=candidates,
        feature_fn=feature_fn,
        evaluate_fn=evaluate_fn,
        top_k=top_k,
    )
    screen_time = time.perf_counter() - t_screen

    out_sbox = Path(out_sbox_path)
    out_sbox.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_sbox, best["sbox"])

    summary = {
        "dataset_origin": dataset_origin,
        "n_features": len(feature_cols),
        "pool_mode": pool_mode,
        "n_screen_candidates": len(candidates),
        "top_k_evaluated": min(top_k, len(candidates)),
        "screen_time_sec": float(screen_time),
        "ml_metrics": ml_metrics,
        "selected_sbox_source": best["source"],
        "selected_sbox_poly": f"0x{best['poly']:03X}" if best.get("poly") is not None else None,
        "selected_sbox_constant": f"0x{best['constant']:02X}" if best.get("constant") is not None else None,
        "selected_sbox_score": float(best["score"]),
        "selected_sbox_predicted_score": float(best["predicted_score"]),
        "selected_sbox_metrics": best["metrics"],
    }

    out_metrics = Path(out_metrics_path)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Training & seleksi S-Box berbasis XGBoost (Jobdesk Mahdi).")
    parser.add_argument("--dataset", type=str, default="data/sbox_dataset.csv")
    parser.add_argument("--out-sbox", type=str, default="results/sbox_xgb.npy")
    parser.add_argument("--out-metrics", type=str, default="results/xgb_metrics.json")
    parser.add_argument("--n-train", type=int, default=400, help="Jumlah sampel sintetis bila CSV belum ada.")
    parser.add_argument("--n-screen", type=int, default=2000, help="Jumlah kandidat baru untuk disaring XGBoost.")
    parser.add_argument("--top-k", type=int, default=25, help="Jumlah kandidat teratas yang diuji penuh.")
    parser.add_argument("--pool-mode", choices=["aes_affine", "chaotic", "mixed"], default="aes_affine")
    parser.add_argument("--tune", action="store_true", help="Aktifkan RandomizedSearchCV hyperparameter tuning.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_xgboost_pipeline(
        dataset_path=args.dataset,
        out_sbox_path=args.out_sbox,
        out_metrics_path=args.out_metrics,
        n_train_samples=args.n_train,
        n_screen_candidates=args.n_screen,
        top_k=args.top_k,
        tune=args.tune,
        seed=args.seed,
        pool_mode=args.pool_mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
