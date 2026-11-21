import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_rf import run_rf_pipeline, train_and_evaluate_rf


def test_train_and_evaluate_rf_learns_signal_and_reports_importance():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(300, 8))
    y = 4.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(scale=0.1, size=300)

    model, metrics = train_and_evaluate_rf(X, y, n_estimators=80)

    assert metrics["r2"] > 0.85
    assert metrics["rmse"] >= metrics["mae"]
    assert len(metrics["feature_importances"]) == 8
    assert int(np.argmax(model.feature_importances_)) == 0


def test_run_rf_pipeline_saves_bijective_sbox_and_metrics(tmp_path: Path):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 40))
    dataset = pd.DataFrame(X, columns=[f"f_{index}" for index in range(40)])
    dataset.insert(0, "source", ["chaotic_swap", "aes_affine"] * 50)
    dataset.insert(0, "sbox_id", [f"sbox_{index:03d}" for index in range(100)])
    dataset["composite_score"] = 3.0 * X[:, 0] - X[:, 1]
    dataset_path = tmp_path / "sbox_dataset.csv"
    dataset.to_csv(dataset_path, index=False)
    out_sbox = tmp_path / "sbox_rf.npy"
    out_metrics = tmp_path / "rf_metrics.json"

    summary = run_rf_pipeline(
        dataset_path=dataset_path,
        out_sbox_path=out_sbox,
        out_metrics_path=out_metrics,
        n_screen_candidates=20,
        top_k=3,
        n_estimators=30,
    )

    sbox = np.load(out_sbox)
    saved = json.loads(out_metrics.read_text(encoding="utf-8"))
    assert sbox.dtype == np.uint8
    assert sbox.shape == (256,)
    assert np.unique(sbox).size == 256
    assert len(saved["feature_importance_ranked"]) == 40
    assert saved["selected_sbox_metrics"] == summary["selected_sbox_metrics"]
