"""Unit tests untuk pipeline XGBoost v2, loader model tepercaya, dan runner benchmark v2."""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.dataset_builder import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, build_dataset
from src.ml_reconstruction import load_trusted_model, reconstruct
from src.train_rf import save_grouped_rf, train_grouped_rf
from src.train_xgboost_v2 import (
    extract_crypto_cheap_features,
    save_grouped_xgboost,
    train_grouped_xgboost,
)


def _toy_evaluate(box: np.ndarray) -> dict:
    # Deterministic toy objective for fast unit testing
    score = float(box[0]) * 0.25 + float(box[1] ^ box[2]) * 0.1
    return {"composite_score": score}


def test_train_grouped_xgboost_respects_feature_order_and_rejects_leakage():
    rows = build_dataset(10, 4, 42, _toy_evaluate)
    model, metrics = train_grouped_xgboost(rows, seed=42, tune=False)

    assert tuple(model.feature_names_in_) == FEATURE_NAMES
    assert metrics["n_train"] == sum(r["split"] == "train" for r in rows)
    assert metrics["n_validation"] == sum(r["split"] == "validation" for r in rows)
    assert metrics["n_test"] == sum(r["split"] == "test" for r in rows)
    for key in ("mae", "mse", "rmse", "r2", "train_time_sec", "inference_time_sec"):
        assert np.isfinite(metrics[key])
    assert set(metrics["feature_importances"]) == set(FEATURE_NAMES)
    assert set(metrics["validation_metrics"]) >= {"mae", "mse", "rmse", "r2"}

    # Mutate one family row split -> must raise family leakage ValueError
    leaked = [dict(r) for r in rows]
    leaked[0]["split"] = "test" if leaked[0]["split"] != "test" else "train"
    with pytest.raises(ValueError, match="family"):
        train_grouped_xgboost(leaked, seed=42, tune=False)


def test_save_and_load_trusted_xgboost_roundtrip_and_split_match(tmp_path):
    rows = build_dataset(10, 4, 42, _toy_evaluate)
    rf_model, rf_metrics = train_grouped_rf(rows, n_estimators=5, seed=42)
    xgb_model, xgb_metrics = train_grouped_xgboost(rows, seed=42, tune=False)

    rf_dir = tmp_path / "rf"
    xgb_dir = tmp_path / "xgboost"
    save_grouped_rf(rf_dir, rf_model, rf_metrics, rows, kind="rf")
    save_grouped_xgboost(xgb_dir, xgb_model, xgb_metrics, rows)

    rf_manifest = json.loads((rf_dir / "manifest.json").read_text(encoding="utf-8"))
    loaded_model, xgb_manifest = load_trusted_model(xgb_dir)

    assert xgb_manifest["model_kind"] == "xgboost"
    assert xgb_manifest["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert xgb_manifest["split_sha256"] == rf_manifest["split_sha256"]
    assert xgb_manifest["dataset_sha256"] == rf_manifest["dataset_sha256"]

    frame = pd.DataFrame([r["features"] for r in rows[:5]], columns=FEATURE_NAMES)
    np.testing.assert_allclose(loaded_model.predict(frame), xgb_model.predict(frame), rtol=0, atol=1e-6)


def test_load_trusted_model_rejects_schema_mismatch_and_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_trusted_model(tmp_path / "does_not_exist")

    rows = build_dataset(8, 3, 42, _toy_evaluate)
    xgb_model, xgb_metrics = train_grouped_xgboost(rows, seed=42, tune=False)
    xgb_dir = tmp_path / "bad_schema"
    save_grouped_xgboost(xgb_dir, xgb_model, xgb_metrics, rows)

    manifest_path = xgb_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_schema_version"] = "incompatible-v99"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        load_trusted_model(xgb_dir)


def test_load_trusted_model_rejects_dataset_fingerprint_mismatch(tmp_path):
    rows = build_dataset(10, 4, 42, _toy_evaluate)
    model, metrics = train_grouped_xgboost(rows, seed=42, tune=False)
    root = tmp_path / 'repo'
    model_dir = root / 'models' / 'xgboost'
    save_grouped_xgboost(model_dir, model, metrics, rows)
    split_dir = root / 'datasets'
    split_dir.mkdir(parents=True)
    (split_dir / 'split_manifest.json').write_text(
        json.dumps({'dataset_sha256': 'wrong', 'split_sha256': 'wrong'}),
        encoding='utf-8',
    )
    with pytest.raises(ValueError, match='fingerprint'):
        load_trusted_model(model_dir)


def test_reconstruct_records_timing_and_cache_counters_and_cli_supports_xgboost(tmp_path):
    rows = build_dataset(8, 3, 42, _toy_evaluate)
    xgb_model, xgb_metrics = train_grouped_xgboost(rows, seed=42, tune=False)
    xgb_dir = tmp_path / "xgb_model"
    save_grouped_xgboost(xgb_dir, xgb_model, xgb_metrics, rows)

    res = reconstruct(
        lambda b: _toy_evaluate(b)["composite_score"],
        xgb_model.predict,
        seed=7,
        iterations=5,
        pool_size=6,
        top_k=2,
        max_evaluations=7,
    )
    assert res["full_evaluations"] == 7
    assert res["cache_misses"] == 7
    assert res["cache_hits"] >= 0
    for tkey in ("feature_time_sec", "inference_time_sec", "evaluation_time_sec", "total_time_sec"):
        assert res[tkey] >= 0.0

    # End-to-end CLI roundtrip with xgboost artifact
    out_json = tmp_path / "xgb_run.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.ml_reconstruction",
            str(out_json),
            "--model",
            str(xgb_dir),
            "--seed",
            "7",
            "--budget",
            "4",
        ],
        check=True,
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["full_evaluations"] == 4
    assert sorted(payload["sbox"]) == list(range(256))
    assert payload["model_kind"] == "xgboost"


def test_crypto_cheap_features_polynomial_free_and_finite():
    from src.sbox_metrics import evaluate_sbox

    rng = np.random.default_rng(99)
    sbox = rng.permutation(256).astype(np.uint8)
    feats_full = extract_crypto_cheap_features(sbox, include_sac_fp=True)
    feats_walsh = extract_crypto_cheap_features(sbox, include_sac_fp=False)
    ref = evaluate_sbox(sbox)

    assert len(feats_full) > len(feats_walsh) > 0
    assert all(np.isfinite(v) for v in feats_full.values())
    assert all(np.isfinite(v) for v in feats_walsh.values())
    assert "sac_mean" not in feats_walsh and "sac_max_err" not in feats_walsh and "fixed_points" not in feats_walsh
    assert feats_full["sac_mean"] == pytest.approx(ref["sac_mean"], abs=1e-9)
    assert feats_full["sac_max_err"] == pytest.approx(ref["sac_max_err"], abs=1e-9)
    assert feats_full["fixed_points"] == pytest.approx(float(ref["fixed_points"]), abs=1e-9)


def test_mandrill_v2_runner_lossless_and_pixel_correlation(tmp_path):
    from src.run_v2_benchmark import evaluate_mandrill_catalog

    rng = np.random.default_rng(12)
    img = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    img_path = tmp_path / "tiny_mandrill.png"
    Image.fromarray(img).save(img_path)

    sboxes = {
        "AES Standar (FIPS-197)": np.load("data/baselines/aes_standard.npy"),
        "S-Box XGBoost v2": rng.permutation(256).astype(np.uint8),
    }
    report = evaluate_mandrill_catalog(sboxes=sboxes, image_path=img_path, modes=("RGB",))
    assert len(report["runs"]) == 2
    assert all(r["lossless"] for r in report["runs"])
    assert all("pixel_correlation" in r and "cipher_variant" in r for r in report["runs"])
    assert report["runs"][1]["cipher_variant"]["is_standard_sbox"] is False


def test_v2_mandrill_benchmark_writes_selected_results_and_checks_inverse(tmp_path):
    from src.run_v2_benchmark import run_v2_mandrill_benchmark

    rng = np.random.default_rng(77)
    boxes = {
        method: rng.permutation(256).astype(np.uint8)
        for method in ('rf', 'xgboost', 'mlp')
    }
    initial = rng.permutation(256).astype(np.uint8)
    paired = []
    for method, box in boxes.items():
        paired.append({
            'method': method,
            'seed': 7,
            'initial_score': 1.0,
            'best_score': 2.0,
            'score_delta': 1.0,
            'initial_sbox': initial.tolist(),
            'final_sbox': box.tolist(),
            'inverse_sbox': np.argsort(box).astype(np.uint8).tolist(),
        })
    comparison = tmp_path / 'comparative.json'
    comparison.write_text(json.dumps({'paired': paired}), encoding='utf-8')
    image_path = tmp_path / 'tiny.png'
    Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(image_path)
    output_path = tmp_path / 'v2' / 'mandrill.json'

    report = run_v2_mandrill_benchmark(comparison, image_path, output_path, seed=7)
    assert len(report['runs']) == 10
    assert all(row['lossless'] for row in report['runs'])
    assert report['image_sha256']
    for method in boxes:
        result_dir = output_path.parent / method / 'seed_7'
        assert all((result_dir / name).is_file() for name in (
            'initial_sbox.json', 'final_sbox.json', 'inverse_sbox.json',
            'cryptographic_metrics.json'))
