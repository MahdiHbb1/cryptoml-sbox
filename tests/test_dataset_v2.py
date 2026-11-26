import numpy as np
import subprocess
import sys


def test_dataset_reproducible_and_family_split_disjoint():
    from src.dataset_builder import build_dataset, FEATURE_NAMES

    def evaluate(box):
        return {'composite_score': float(box[0])}

    rows = build_dataset(20, 8, 42, evaluate)
    assert rows == build_dataset(20, 8, 42, evaluate)
    assert len(rows) == len({row['sbox_id'] for row in rows}) == 160
    assert {row['split'] for row in rows} == {'train', 'validation', 'test'}
    by_id = {row['sbox_id']: row for row in rows}
    for row in rows:
        assert set(row['features']) == set(FEATURE_NAMES)
        assert np.isfinite(list(row['features'].values())).all()
        if row['parent_id']:
            parent = by_id[row['parent_id']]
            assert (parent['family_id'], parent['split']) == (row['family_id'], row['split'])
    for family in {row['family_id'] for row in rows}:
        assert len({row['split'] for row in rows if row['family_id'] == family}) == 1


def test_write_dataset_bundle_matches_model_fingerprints_and_rejects_leakage(tmp_path):
    import json
    import pytest
    from src.dataset_builder import build_dataset, write_dataset_bundle

    rows = build_dataset(10, 4, 42, lambda box: {'composite_score': float(box[0])})
    manifest = write_dataset_bundle(
        rows, tmp_path / 'dataset.csv', tmp_path / 'split.json')
    assert manifest['n_samples_total'] == 40
    assert manifest['disjoint_family_verified'] and manifest['disjoint_hash_verified']
    saved = json.loads((tmp_path / 'split.json').read_text(encoding='utf-8'))
    assert saved['dataset_sha256'] == manifest['dataset_sha256']
    assert len(saved['train_ids']) + len(saved['val_ids']) + len(saved['test_ids']) == 40
    assert (tmp_path / 'dataset.csv').is_file()

    leaked = [dict(row) for row in rows]
    leaked[0] = {**leaked[0], 'split': 'test'}
    with pytest.raises(ValueError, match='family'):
        write_dataset_bundle(leaked, tmp_path / 'bad.csv', tmp_path / 'bad.json')


def test_v2_training_imports_do_not_load_legacy_generator_modules():
    subprocess.run(
        [
            sys.executable,
            '-c',
            (
                'import sys; import src.train_v2_models; '
                "assert 'src.aes_affine_generator' not in sys.modules; "
                "assert 'src.train_xgboost' not in sys.modules"
            ),
        ],
        check=True,
    )
