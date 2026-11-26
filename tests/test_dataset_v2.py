import numpy as np


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
