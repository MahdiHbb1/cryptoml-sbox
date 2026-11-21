"""Training, evaluasi, dan seleksi S-Box dengan MLP (StandardScaler + MLPRegressor)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset_builder import extract_fast_features, resolve_evaluator
from src.train_rf import generate_candidate_pool  # pool kandidat sama dengan RF -> perbandingan adil
from src.train_xgboost import load_dataset_csv, screen_and_select_best_sbox


def build_mlp_model(
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    random_state: int = 42,
    max_iter: int = 500,
) -> TransformedTargetRegressor:
    """Fitur & target distandarisasi; prediksi tetap dalam skala asli composite_score."""
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        max_iter=max_iter,
        random_state=random_state,
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def train_and_evaluate_mlp(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    test_size: float = 0.2,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    max_iter: int = 500,
) -> tuple[TransformedTargetRegressor, dict]:
    """Melatih MLP dengan split baku (sama dengan XGBoost & RF) dan mengembalikan metrik test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = build_mlp_model(hidden_layer_sizes, random_state, max_iter)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - started

    pred = model.predict(X_test)
    mlp = model.regressor_.named_steps["mlp"]
    return model, {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "train_time_sec": float(train_time),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "hidden_layer_sizes": list(hidden_layer_sizes),
        "n_iter": int(mlp.n_iter_),
        "final_loss": float(mlp.loss_),
    }


def run_mlp_pipeline(
    dataset_path: str | Path = Path("data/sbox_dataset.csv"),
    out_sbox_path: str | Path = Path("results/sbox_mlp.npy"),
    out_metrics_path: str | Path = Path("results/mlp_metrics.json"),
    n_screen_candidates: int = 2000,
    top_k: int = 25,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
    max_iter: int = 500,
    seed: int = 42,
) -> dict:
    """Menjalankan pipeline MLP lengkap dan menyimpan S-box pemenang serta metrik."""
    dataset = Path(dataset_path)
    if not dataset.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {dataset}. Jalankan `python -m src.dataset_builder` lebih dulu."
        )

    X, y, feature_columns = load_dataset_csv(dataset)
    model, ml_metrics = train_and_evaluate_mlp(
        X, y, random_state=seed, hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter
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

    summary = {
        "dataset_origin": str(dataset),
        "n_features": len(feature_columns),
        "n_screen_candidates": len(candidates),
        "top_k_evaluated": min(top_k, len(candidates)),
        "screen_time_sec": float(screen_time),
        "ml_metrics": ml_metrics,
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
    parser = argparse.ArgumentParser(description="Training dan seleksi S-Box berbasis MLP.")
    parser.add_argument("--dataset", default="data/sbox_dataset.csv")
    parser.add_argument("--out-sbox", default="results/sbox_mlp.npy")
    parser.add_argument("--out-metrics", default="results/mlp_metrics.json")
    parser.add_argument("--n-screen", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--hidden", default="128,64", help="Ukuran hidden layer, dipisah koma.")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = run_mlp_pipeline(
        dataset_path=args.dataset,
        out_sbox_path=args.out_sbox,
        out_metrics_path=args.out_metrics,
        n_screen_candidates=args.n_screen,
        top_k=args.top_k,
        hidden_layer_sizes=tuple(int(v) for v in args.hidden.split(",")),
        max_iter=args.max_iter,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()