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
