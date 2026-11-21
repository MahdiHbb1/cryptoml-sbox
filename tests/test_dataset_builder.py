from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.aes_affine_generator import build_standard_aes_sbox
from src.dataset_builder import (
    build_sbox_dataset,
    extract_fast_features,
    generate_chaotic_swap_candidates,
)


def test_chaotic_swap_candidates_are_deterministic_unique_and_bijective():
    first = generate_chaotic_swap_candidates(12, seed=42)
    second = generate_chaotic_swap_candidates(12, seed=42)

    assert [item["sbox"].tobytes() for item in first] == [item["sbox"].tobytes() for item in second]
    assert len({item["sbox"].tobytes() for item in first}) == 12
    for item in first:
        assert item["source"] == "chaotic_swap"
        assert item["sbox"].dtype == np.uint8
        assert item["sbox"].shape == (256,)
        assert np.unique(item["sbox"]).size == 256


def test_extract_fast_features_returns_finite_40_values():
    features = extract_fast_features(build_standard_aes_sbox())
    assert features.shape == (40,)
    assert np.all(np.isfinite(features))


def test_build_dataset_matches_contract_and_writes_csv(tmp_path: Path):
    output = tmp_path / "sbox_dataset.csv"
    dataset = build_sbox_dataset(n_chaotic=2, n_aes=2, seed=7, output_path=output)
    feature_columns = [column for column in dataset if column.startswith("f_")]

    assert output.exists()
    assert len(dataset) == 4
    assert len(feature_columns) == 40
    assert set(dataset["source"]) == {"chaotic_swap", "aes_affine"}
    assert {"sbox_id", "nl", "du", "sac_max_err", "whs_cost", "composite_score"} <= set(dataset)
    assert np.all(np.isfinite(dataset[feature_columns + ["composite_score"]].to_numpy()))
    pd.testing.assert_frame_equal(pd.read_csv(output), dataset, check_dtype=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": -1},
        {"n": 1, "min_swaps": 2, "max_swaps": 1},
        {"n": 1, "skip": -1},
    ],
)
def test_chaotic_generator_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        generate_chaotic_swap_candidates(**kwargs)
