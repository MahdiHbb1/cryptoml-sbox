from src.v2_experiment import pool_metrics


def test_pool_metrics_handles_ties_and_reports_regret():
    assert pool_metrics([1, 3, 2], [1, 1, 1], 1) == {'spearman': None, 'regret': 2.0}
    assert pool_metrics([1, 3, 2], [1, 3, 2], 1) == {'spearman': 1.0, 'regret': 0.0}


def test_proxy_distinguishes_constant_from_variable_derivatives():
    import numpy as np
    from src.v2_experiment import differential_proxy

    identity = np.arange(256, dtype=np.uint8)
    shuffled = np.random.default_rng(42).permutation(identity)
    assert differential_proxy(identity) == 1.0
    assert differential_proxy(shuffled) < 1.0


def test_run_artifact_export_writes_inverse_metrics_and_history(tmp_path):
    import csv
    import json
    import numpy as np
    from src.ml_reconstruction import reconstruct
    from src.v2_experiment import write_run_artifacts

    result = reconstruct(
        lambda box: float(box[0]),
        seed=9,
        strategy='random_swap',
        iterations=2,
        max_evaluations=2,
    )
    write_run_artifacts(tmp_path, result, 9, 'random_swap', 'random_swap', 2, 2)
    directory = tmp_path / 'baselines' / 'random_swap' / 'seed_9'
    final = json.loads((directory / 'final_sbox.json').read_text(encoding='utf-8'))
    inverse = json.loads((directory / 'inverse_sbox.json').read_text(encoding='utf-8'))
    np.testing.assert_array_equal(
        np.asarray(inverse['sbox'])[np.asarray(final['sbox'])], np.arange(256))
    with (directory / 'reconstruction_history.csv').open(encoding='utf-8') as file:
        history = list(csv.DictReader(file))
    assert len(history) == len(result['history'])
    assert json.loads((directory / 'cryptographic_metrics.json').read_text(encoding='utf-8'))['metrics']
    assert json.loads((directory / 'run_metadata.json').read_text(encoding='utf-8'))['full_evaluations'] == 2
