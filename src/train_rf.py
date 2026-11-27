"""Training, evaluasi, dan seleksi S-Box dengan Random Forest."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

def grouped_ranking_metrics(rows, predicted, k=3):
    """Measure global and per-family ranking quality without scoring test data twice."""
    predicted = np.asarray(predicted, dtype=float)
    if len(rows) != len(predicted) or not 1 <= k:
        raise ValueError('Expected aligned predictions and a positive Top-K')
    actual = np.asarray([row['composite_score'] for row in rows], dtype=float)
    global_rho = None
    if np.ptp(actual) > 0 and np.ptp(predicted) > 0:
        value = float(spearmanr(actual, predicted).statistic)
        global_rho = value if np.isfinite(value) else None
    pools = {}
    for row, y, prediction in zip(rows, actual, predicted):
        pools.setdefault(row['family_id'], [[], []])
        pools[row['family_id']][0].append(float(y))
        pools[row['family_id']][1].append(float(prediction))
    per_pool = []
    for family_id, (scores, estimates) in sorted(pools.items()):
        scores, estimates = np.asarray(scores), np.asarray(estimates)
        rho = None
        if np.ptp(scores) > 0 and np.ptp(estimates) > 0:
            value = float(spearmanr(scores, estimates).statistic)
            rho = value if np.isfinite(value) else None
        shortlist = np.argsort(-estimates, kind='stable')[:min(k, len(scores))]
        regret = float(scores.max() - scores[shortlist].max())
        per_pool.append({
            'family_id': family_id,
            'spearman': rho,
            'topk_regret': regret,
            'topk_hit': bool(scores[shortlist].max() == scores.max()),
        })
    valid_rhos = [pool['spearman'] for pool in per_pool if pool['spearman'] is not None]
    return {
        'spearman': global_rho,
        'spearman_per_pool_mean': float(np.mean(valid_rhos)) if valid_rhos else None,
        'topk_regret': float(np.mean([pool['topk_regret'] for pool in per_pool])),
        'topk_hit_rate': float(np.mean([pool['topk_hit'] for pool in per_pool])),
        'n_pools': len(per_pool),
        'pool_ranking': per_pool,
    }


def train_grouped_rf(rows, *, n_estimators=300, seed=42, estimator=None):
    """Fit on the fixed family split; report held-out metrics."""
    import pandas as pd
    from src.dataset_builder import FEATURE_NAMES

    groups = {name: [] for name in ('train', 'validation', 'test')}
    families, ids = {}, set()
    for row in rows:
        split = row['split']
        if split not in groups or families.setdefault(row['family_id'], split) != split:
            raise ValueError('Invalid split or family leakage')
        if row['sbox_id'] in ids or set(row['features']) != set(FEATURE_NAMES):
            raise ValueError('Duplicate S-Box or incompatible feature schema')
        ids.add(row['sbox_id'])
        groups[split].append(row)
    if not groups['train'] or len(groups['test']) < 2:
        raise ValueError('Need training rows and at least two test rows')

    def arrays(split):
        items = groups[split]
        return (pd.DataFrame([row['features'] for row in items], columns=FEATURE_NAMES),
                np.asarray([row['composite_score'] for row in items], dtype=float))

    x_train, y_train = arrays('train')
    x_validation, y_validation = arrays('validation')
    x_test, y_test = arrays('test')
    model = estimator if estimator is not None else RandomForestRegressor(
        n_estimators=n_estimators, random_state=seed, n_jobs=1)
    started = time.perf_counter()
    model.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    inference_started = time.perf_counter()
    predicted = model.predict(x_test)
    inference_time = time.perf_counter() - inference_started
    validation_prediction = model.predict(x_validation)
    mse = float(mean_squared_error(y_test, predicted))
    validation_mse = float(mean_squared_error(y_validation, validation_prediction))
    metrics = dict(n_train=len(y_train), n_test=len(y_test), mse=mse,
                       rmse=float(np.sqrt(mse)), mae=float(mean_absolute_error(y_test, predicted)),
                       r2=float(r2_score(y_test, predicted)), train_time_sec=elapsed,
                       inference_time_sec=inference_time, n_validation=len(y_validation),
                       validation_metrics={
                           'mae': float(mean_absolute_error(y_validation, validation_prediction)),
                           'mse': validation_mse,
                           'rmse': float(np.sqrt(validation_mse)),
                           'r2': float(r2_score(y_validation, validation_prediction)),
                       })
    metrics.update(grouped_ranking_metrics(groups['test'], predicted))
    metrics['validation_metrics'].update(
        grouped_ranking_metrics(groups['validation'], validation_prediction))
    if hasattr(model, "feature_importances_"):
        metrics["feature_importances"] = dict(zip(FEATURE_NAMES, model.feature_importances_.tolist()))
    return model, metrics


def save_grouped_rf(directory, model, metrics, rows, *, kind="rf"):
    """Export a fitted model and its split manifest to a local artifact directory."""
    from hashlib import sha256

    import joblib
    import sklearn
    import xgboost
    from src.dataset_builder import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

    if tuple(model.feature_names_in_) != FEATURE_NAMES:
        raise ValueError('Incompatible feature schema')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': '2.0.0',
        'model_kind': kind,
        'feature_schema_version': FEATURE_SCHEMA_VERSION,
        'feature_names': list(FEATURE_NAMES),
        'n_features': len(FEATURE_NAMES),
        'split_by_sbox': {row['sbox_id']: row['split'] for row in rows},
        'family_by_sbox': {row['sbox_id']: row['family_id'] for row in rows},
        'split_sha256': sha256(json.dumps(sorted(
            (row['sbox_id'], row['family_id'], row['split']) for row in rows
        )).encode()).hexdigest(),
        'dataset_sha256': sha256(json.dumps(sorted(rows, key=lambda row: row['sbox_id']),
                                             sort_keys=True, allow_nan=False).encode()).hexdigest(),
        'sklearn_version': sklearn.__version__,
        'xgboost_version': xgboost.__version__ if kind == 'xgboost' else None,
        'python_version': platform.python_version(),
        'model_parameters': (
            {f"mlp__{key}": value for key, value in model.named_steps["mlpregressor"].get_params().items()}
            if kind == "mlp" else model.get_params(deep=False)
        ),
        'metrics': metrics,
    }
    for key, value in manifest['model_parameters'].items():
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            manifest['model_parameters'][key] = None
    text = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)
    with (directory / f"{kind}.joblib").open('xb') as output:
        joblib.dump(model, output, compress=3)
    with (directory / 'manifest.json').open('x', encoding='utf-8') as output:
        output.write(text + '\n')


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
    from src.aes_affine_generator import generate_aes_affine_candidates
    from src.dataset_builder import generate_chaotic_swap_candidates

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
    from src.dataset_builder import (
        extract_fast_features,
        resolve_evaluator,
    )
    from src.train_xgboost import load_dataset_csv, screen_and_select_best_sbox

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
