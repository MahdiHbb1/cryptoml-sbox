import numpy as np


def test_ranking_changes_shortlist_and_baseline_uses_same_budget():
    from src.ml_reconstruction import reconstruct

    def score(box):
        return float(box[0])

    def ascending(x):
        return np.arange(len(x))

    options = dict(seed=42, iterations=3, pool_size=8, top_k=1)
    first = reconstruct(score, ascending, **options)
    second = reconstruct(score, lambda x: -ascending(x), **options)
    random = reconstruct(score, None, **options)
    assert first['history'][0]['candidate_id'] != second['history'][0]['candidate_id']
    for result in (first, second, random):
        assert result['full_evaluations'] == 4
        np.testing.assert_array_equal(np.sort(result['sbox']), np.arange(256))
        scores = [row['best_score'] for row in result['history']]
        assert scores == sorted(scores)


def test_rf_uses_ordered_features_and_rejects_family_leakage():
    import pytest
    from src.dataset_builder import FEATURE_NAMES, build_dataset
    from src.train_rf import train_grouped_rf

    rows = build_dataset(8, 3, 42, lambda box: {'composite_score': float(box[0])})
    model, metrics = train_grouped_rf(rows, n_estimators=3)
    assert tuple(model.feature_names_in_) == FEATURE_NAMES
    assert metrics['n_test'] == sum(row['split'] == 'test' for row in rows)
    rows[0]['split'] = 'test' if rows[0]['split'] != 'test' else 'train'
    with pytest.raises(ValueError, match='family'):
        train_grouped_rf(rows, n_estimators=3)


def test_explicit_evaluation_cap_and_inverse():
    from src.ml_reconstruction import reconstruct

    calls = []

    def score(box):
        calls.append(box.copy())
        return float(box[0])

    result = reconstruct(score, iterations=20, pool_size=8, top_k=3, max_evaluations=5)
    assert len(calls) == result['full_evaluations'] == 5
    np.testing.assert_array_equal(result['initial_sbox'], calls[0])
    np.testing.assert_array_equal(result['inverse_sbox'][result['sbox']], np.arange(256))


def test_random_search_keeps_initial_and_respects_budget():
    from src.ml_reconstruction import reconstruct

    score = lambda box: float(box[0])
    initial = reconstruct(score, strategy='random_permutation', seed=7)
    search = reconstruct(score, strategy='random_search', seed=7, max_evaluations=7)
    np.testing.assert_array_equal(initial['sbox'], search['initial_sbox'])
    assert initial['full_evaluations'] == 1
    assert search['full_evaluations'] == 7
    assert search['best_score'] >= initial['best_score']


def test_restart_keeps_global_best_and_budget():
    from src.ml_reconstruction import reconstruct

    result = reconstruct(lambda box: 1.0, mode='fast', patience=1, restarts=1,
                         max_evaluations=5)
    assert result['restarts'] == 1
    assert result['full_evaluations'] == 5
    np.testing.assert_array_equal(result['sbox'], result['initial_sbox'])


def test_reconstruction_history_tracks_per_iteration_costs():
    from src.ml_reconstruction import reconstruct

    result = reconstruct(
        lambda box: float(box[0]),
        lambda frame: np.arange(len(frame), dtype=float),
        seed=3,
        iterations=2,
        pool_size=4,
        top_k=1,
        max_evaluations=3,
    )
    row = result['history'][0]
    assert row['operator_selected'] == 'single_swap'
    assert row['n_candidates_generated'] >= row['n_candidates_unique'] == 4
    assert row['full_evals_used'] == 1
    assert row['full_evals_cumulative'] == 2
    assert row['feature_time_sec'] >= 0
    assert row['inference_time_sec'] >= 0
    assert row['eval_time_sec'] >= 0
    assert row['iter_total_time_sec'] >= 0
