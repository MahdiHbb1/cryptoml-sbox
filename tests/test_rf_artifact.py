import json

import joblib
import numpy as np
import pandas as pd

from src.dataset_builder import FEATURE_NAMES, build_dataset
from src.train_rf import train_grouped_rf


def test_rf_artifact_preserves_predictions_and_split(tmp_path):
    from src.train_rf import save_grouped_rf

    rows = build_dataset(8, 3, 42, lambda box: {'composite_score': float(box[0])})
    model, metrics = train_grouped_rf(rows, n_estimators=3)
    save_grouped_rf(tmp_path, model, metrics, rows)
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['feature_names'] == list(FEATURE_NAMES)
    assert manifest['split_by_sbox'] == {row['sbox_id']: row['split'] for row in rows}
    frame = pd.DataFrame([rows[0]['features']], columns=FEATURE_NAMES)
    np.testing.assert_array_equal(joblib.load(tmp_path / 'rf.joblib').predict(frame), model.predict(frame))
