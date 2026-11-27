"""Grouped, schema-versioned XGBoost training for permutation S-Boxes."""

from __future__ import annotations

import time
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from src.dataset_builder import FEATURE_NAMES, build_dataset
from src.train_rf import grouped_ranking_metrics, save_grouped_rf


def extract_crypto_cheap_features(sbox, *, include_sac_fp=True):
    """Extract polynomial-free permutation and cryptographic proxy features."""
    box = np.asarray(sbox)
    if (box.shape != (256,) or not np.issubdtype(box.dtype, np.integer)
            or np.any(box < 0) or np.any(box > 255) or np.unique(box).size != 256):
        raise ValueError('Expected a permutation of integers 0–255')
    box = box.astype(np.uint8, copy=False)
    values = np.arange(256, dtype=np.uint8)
    weights = np.array([value.bit_count() for value in range(256)])
    hamming = weights[box ^ values]
    displacement = np.abs(box.astype(np.int16) - values)
    features = {
        'hamming_mean': float(hamming.mean()),
        'hamming_std': float(hamming.std()),
        'displacement_mean': float(displacement.mean()),
        'displacement_std': float(displacement.std()),
    }

    differential_max = 0
    for bit in range(8):
        counts = np.bincount(box ^ box[values ^ (1 << bit)], minlength=256)
        differential_max = max(differential_max, int(counts.max()))
    features['differential_proxy'] = differential_max / 256.0

    if include_sac_fp:
        differences = np.stack([box ^ box[values ^ (1 << bit)] for bit in range(8)])
        sac = np.unpackbits(differences[:, :, None], axis=2, bitorder='little').mean(axis=1)
        features.update(
            sac_mean=float(sac.mean()),
            sac_max_err=float(np.max(np.abs(sac - 0.5))),
            fixed_points=float(np.count_nonzero(box == values)),
        )
    return features


def _regression_metrics(actual, predicted):
    mse = float(mean_squared_error(actual, predicted))
    rho = None
    if np.ptp(actual) > 0 and np.ptp(predicted) > 0:
        value = float(spearmanr(actual, predicted).statistic)
        rho = value if np.isfinite(value) else None
    return {
        'mae': float(mean_absolute_error(actual, predicted)),
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'r2': float(r2_score(actual, predicted)),
        'spearman': rho,
    }


def train_grouped_xgboost(rows, *, seed=42, tune=True, n_estimators=300):
    """Tune on validation only, then report once against the locked test split."""
    if not isinstance(n_estimators, int) or n_estimators < 1:
        raise ValueError('n_estimators must be a positive integer')
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
    if not groups['train'] or len(groups['validation']) < 2 or len(groups['test']) < 2:
        raise ValueError('Need training, validation, and test rows')

    def arrays(split):
        subset = groups[split]
        x = pd.DataFrame([row['features'] for row in subset], columns=FEATURE_NAMES)
        y = np.asarray([row['composite_score'] for row in subset], dtype=float)
        return x, y

    x_train, y_train = arrays('train')
    x_validation, y_validation = arrays('validation')
    x_test, y_test = arrays('test')
    candidates = [(n_estimators, 4, 0.05)]
    if tune:
        candidates = [(trees, depth, rate) for trees in (150, 300, n_estimators)
                      for depth in (3, 5) for rate in (0.03, 0.08)]

    started = time.perf_counter()
    best = None
    best_validation_rmse = float('inf')
    for trees, depth, rate in dict.fromkeys(candidates):
        candidate = XGBRegressor(
            n_estimators=trees, max_depth=depth, learning_rate=rate,
            objective='reg:squarederror', random_state=seed, n_jobs=1,
            verbosity=0,
        )
        candidate.fit(x_train, y_train)
        validation_prediction = candidate.predict(x_validation)
        validation_rmse = float(np.sqrt(mean_squared_error(y_validation, validation_prediction)))
        if validation_rmse < best_validation_rmse:
            best, best_validation_rmse = candidate, validation_rmse
    train_time = time.perf_counter() - started

    started = time.perf_counter()
    validation_prediction = best.predict(x_validation)
    validation_time = time.perf_counter() - started
    started = time.perf_counter()
    test_prediction = best.predict(x_test)
    inference_time = time.perf_counter() - started
    metrics = _regression_metrics(y_test, test_prediction)
    metrics.update(grouped_ranking_metrics(groups['test'], test_prediction))
    validation_metrics = _regression_metrics(y_validation, validation_prediction)
    validation_metrics.update(
        grouped_ranking_metrics(groups['validation'], validation_prediction))
    metrics.update(
        n_train=len(y_train),
        n_validation=len(y_validation),
        n_test=len(y_test),
        train_time_sec=train_time,
        validation_inference_time_sec=validation_time,
        inference_time_sec=inference_time,
        validation_metrics=validation_metrics,
        feature_importances=dict(zip(FEATURE_NAMES, best.feature_importances_.tolist())),
    )
    return best, metrics


def save_grouped_xgboost(directory, model, metrics, rows):
    """Persist model and the same canonical dataset/split fingerprints as RF."""
    save_grouped_rf(directory, model, metrics, rows, kind='xgboost')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', type=Path, default=Path('models/xgboost'))
    parser.add_argument('--families', type=int, default=150)
    parser.add_argument('--family-size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--no-tune', action='store_true')
    args = parser.parse_args()

    from src.train_rf_v2 import evaluate

    rows = build_dataset(args.families, args.family_size, args.seed, evaluate)
    model, metrics = train_grouped_xgboost(
        rows, seed=args.seed, tune=not args.no_tune, n_estimators=args.n_estimators)
    save_grouped_xgboost(args.output, model, metrics, rows)
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
