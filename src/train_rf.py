"""Training, evaluasi, dan seleksi S-Box dengan Random Forest."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.aes_affine_generator import generate_aes_affine_candidates
from src.dataset_builder import (
    extract_fast_features,
    generate_chaotic_swap_candidates,
    resolve_evaluator,
)
from src.train_xgboost import load_dataset_csv, screen_and_select_best_sbox


def train_and_evaluate_rf(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    test_size: float = 0.2,
    n_estimators: int = 300,
) -> tuple[RandomForestRegressor, dict]:
    """Melatih RF dengan split baku dan mengembalikan metrik test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - started
    predictions = model.predict(X_test)
    return model, {
        "r2": float(r2_score(y_test, predictions)),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "train_time_sec": float(train_time),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_estimators": n_estimators,
        "feature_importances": [float(value) for value in model.feature_importances_],
    }


def generate_candidate_pool(n: int, seed: int = 1041) -> list[dict]:
    """Membangkitkan pool seimbang dari jalur Non-AES dan AES-affine."""
    if n <= 0:
        raise ValueError("Jumlah kandidat penyaringan harus positif.")
    n_chaotic = n // 2
    return generate_chaotic_swap_candidates(n_chaotic, seed=seed) + generate_aes_affine_candidates(
        n=n - n_chaotic,
        seed=seed + 1,
        avoid_fixed_points=True,
    )


def run_rf_pipeline(
    dataset_path: str | Path = Path("data/sbox_dataset.csv"),
    out_sbox_path: str | Path = Path("results/sbox_rf.npy"),
    out_metrics_path: str | Path = Path("results/rf_metrics.json"),
    n_screen_candidates: int = 2000,
    top_k: int = 25,
    n_estimators: int = 300,
    seed: int = 42,
) -> dict:
    """Menjalankan pipeline RF lengkap dan menyimpan pemenang serta metrik."""
    dataset = Path(dataset_path)
    if not dataset.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {dataset}. Jalankan `python -m src.dataset_builder` lebih dulu."
        )

    X, y, feature_columns = load_dataset_csv(dataset)
    model, ml_metrics = train_and_evaluate_rf(
        X,
        y,
        random_state=seed,
        n_estimators=n_estimators,
    )
    started = time.perf_counter()
    candidates = generate_candidate_pool(n_screen_candidates, seed=seed + 999)
    best = screen_and_select_best_sbox(
        model=model,
        candidates=candidates,
        feature_fn=extract_fast_features,
        evaluate_fn=resolve_evaluator(),
        top_k=top_k,
    )
    screen_time = time.perf_counter() - started

    out_sbox = Path(out_sbox_path)
    out_sbox.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_sbox, best["sbox"])

    importance_order = np.argsort(model.feature_importances_)[::-1]
    summary = {
        "dataset_origin": str(dataset),
        "n_features": len(feature_columns),
        "n_screen_candidates": len(candidates),
        "top_k_evaluated": min(top_k, len(candidates)),
        "screen_time_sec": float(screen_time),
        "ml_metrics": ml_metrics,
        "feature_importance_ranked": [
            {"feature": feature_columns[int(index)], "importance": float(model.feature_importances_[index])}
            for index in importance_order
        ],
        "selected_sbox_source": best["source"],
        "selected_sbox_score": float(best["score"]),
        "selected_sbox_predicted_score": float(best["predicted_score"]),
        "selected_sbox_metrics": best["metrics"],
    }
    out_metrics = Path(out_metrics_path)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Training dan seleksi S-Box berbasis Random Forest.")
    parser.add_argument("--dataset", default="data/sbox_dataset.csv")
    parser.add_argument("--out-sbox", default="results/sbox_rf.npy")
    parser.add_argument("--out-metrics", default="results/rf_metrics.json")
    parser.add_argument("--n-screen", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = run_rf_pipeline(
        dataset_path=args.dataset,
        out_sbox_path=args.out_sbox,
        out_metrics_path=args.out_metrics,
        n_screen_candidates=args.n_screen,
        top_k=args.top_k,
        n_estimators=args.n_estimators,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
