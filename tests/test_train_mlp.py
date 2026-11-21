import numpy as np

from src.train_mlp import train_and_evaluate_mlp


def test_mlp_trains_and_reports_metrics():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.1, size=200)
    model, m = train_and_evaluate_mlp(X, y, hidden_layer_sizes=(16,), max_iter=300)
    assert m["n_train"] == 160 and m["n_test"] == 40
    assert m["r2"] > 0.5
    assert model.predict(X[:3]).shape == (3,)