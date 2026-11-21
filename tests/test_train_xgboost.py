import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.aes_affine_generator import build_standard_aes_sbox, generate_aes_affine_candidates
from src.train_xgboost import (
    compute_composite_score,
    extract_fast_features_default,
    load_dataset_csv,
    run_xgboost_pipeline,
    screen_and_select_best_sbox,
    train_and_evaluate_xgboost,
)


def test_load_dataset_csv_reads_contract_schema_and_rejects_missing_columns(tmp_path: Path):
    csv_file = tmp_path / "sbox_dataset.csv"
    df = pd.DataFrame(
        {
            "sbox_id": ["s0", "s1", "s2", "s3"],
            "source": ["aes_affine", "chaotic_swap", "aes_affine", "chaotic_swap"],
            "f_0": [112.0, 94.0, 112.0, 90.0],
            "f_1": [0.05, 0.11, 0.04, 0.13],
            "composite_score": [100.0, 60.0, 101.0, 55.0],
        }
    )
    df.to_csv(csv_file, index=False)

    X, y, feature_cols = load_dataset_csv(csv_file, target_col="composite_score")
    assert feature_cols == ["f_0", "f_1"]
    assert X.shape == (4, 2)
    assert y.shape == (4,)

    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"sbox_id": ["s0"], "other": [1.0]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="fitur|kolom"):
        load_dataset_csv(bad_csv, target_col="composite_score")


def test_extract_fast_features_default_is_deterministic_and_finite():
    sbox = build_standard_aes_sbox()
    feats1 = extract_fast_features_default(sbox)
    feats2 = extract_fast_features_default(sbox)
    assert feats1.ndim == 1
    assert feats1.size >= 16
    assert np.all(np.isfinite(feats1))
    assert np.array_equal(feats1, feats2)


def test_train_and_evaluate_xgboost_learns_signal_from_sbox_features():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(250, 8))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(scale=0.1, size=250)

    model, metrics = train_and_evaluate_xgboost(X, y, random_state=42, tune=False)
    baseline_mae = float(np.mean(np.abs(y - np.mean(y))))
    assert metrics["r2"] > 0.85
    assert metrics["mae"] < 0.35 * baseline_mae
    assert metrics["rmse"] >= metrics["mae"]
    assert metrics["train_time_sec"] >= 0.0
    assert len(metrics["feature_importances"]) == 8
    assert model.predict(X[:3]).shape == (3,)


def test_screen_and_select_best_sbox_evaluates_only_top_k_candidates():
    candidates = generate_aes_affine_candidates(n=15, seed=7, avoid_fixed_points=True)
    eval_calls = []

    class DummyModel:
        def predict(self, X):
            return np.arange(len(X), dtype=float)

    def fake_feature(sbox):
        return np.array([float(sbox[0]), float(sbox[1])])

    def counting_eval(sbox):
        eval_calls.append(sbox)
        return {
            "bijective": True,
            "balanced": True,
            "nl": 112,
            "sac_mean": 0.5,
            "sac_max_err": 0.05,
            "bic_nl": 112,
            "bic_sac": 0.5,
            "du": 4,
            "dap": 0.015625,
            "lap": 0.0625,
            "fixed_points": 0,
            "whs_cost": 10.0,
        }

    best = screen_and_select_best_sbox(
        model=DummyModel(),
        candidates=candidates,
        feature_fn=fake_feature,
        evaluate_fn=counting_eval,
        top_k=4,
    )
    assert len(eval_calls) == 4
    assert best["sbox"].dtype == np.uint8
    assert best["sbox"].shape == (256,)
    assert len(set(best["sbox"].tolist())) == 256
    assert best["score"] == pytest.approx(compute_composite_score(best["metrics"]))


def test_run_xgboost_pipeline_end_to_end_saves_bijective_sbox_and_metrics(tmp_path: Path):
    out_sbox = tmp_path / "sbox_xgb.npy"
    out_metrics = tmp_path / "xgb_metrics.json"

    summary = run_xgboost_pipeline(
        dataset_path=tmp_path / "missing.csv",
        out_sbox_path=out_sbox,
        out_metrics_path=out_metrics,
        n_train_samples=60,
        n_screen_candidates=30,
        top_k=5,
        tune=False,
        seed=42,
        pool_mode="aes_affine",
    )
    assert out_sbox.exists()
    assert out_metrics.exists()

    loaded_sbox = np.load(out_sbox)
    assert loaded_sbox.dtype == np.uint8
    assert loaded_sbox.shape == (256,)
    assert len(set(loaded_sbox.tolist())) == 256

    saved_metrics = json.loads(out_metrics.read_text(encoding="utf-8"))
    assert "ml_metrics" in saved_metrics
    assert "selected_sbox_metrics" in saved_metrics
    assert saved_metrics["selected_sbox_metrics"]["nl"] == 112
    assert saved_metrics["selected_sbox_metrics"]["du"] == 4
    assert summary["ml_metrics"]["r2"] > 0.5
