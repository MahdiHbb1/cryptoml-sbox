"""Ranking diagnostics on fully scored candidate pools."""

import numpy as np
from scipy.stats import spearmanr


def differential_proxy(box):
    """Maximum differential frequency for eight single-bit input masks."""
    indices = np.arange(256)
    return max(np.bincount(box ^ box[indices ^ (1 << bit)]).max()
               for bit in range(8)) / 256.0


def pool_metrics(actual, predicted, k):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if (actual.ndim != 1 or actual.shape != predicted.shape or not 1 <= k <= len(actual)
            or not np.isfinite(actual).all() or not np.isfinite(predicted).all()):
        raise ValueError('Expected aligned finite score vectors and valid k')
    rho = None
    if np.ptp(actual) and np.ptp(predicted):
        rho = float(spearmanr(actual, predicted).statistic)
    shortlist = np.argsort(-predicted, kind='stable')[:k]
    return {'spearman': rho, 'regret': float(actual.max() - actual[shortlist].max())}


def main():
    import argparse
    import json
    from collections import Counter
    from time import perf_counter

    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.dataset_builder import build_dataset, extract_features
    from src.ml_reconstruction import reconstruct
    from src.permutation_v2 import initialize, mutate
    from src.train_rf_v2 import evaluate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[100, 101, 102, 103, 104])
    args = parser.parse_args()
    score = lambda box: evaluate(box)['composite_score']
    start = perf_counter()
    rows = build_dataset(150, 10, 42, evaluate)
    report = {'counts': dict(Counter(row['split'] for row in rows)),
              'dataset_sec': perf_counter() - start, 'ablation': {}, 'paired': []}
    train = [r for r in rows if r['split'] == 'train']
    test = [r for r in rows if r['split'] == 'test']
    models = {}

    def frame(boxes, proxy):
        records = [extract_features(box) for box in boxes]
        if proxy:
            for record, box in zip(records, boxes):
                record['differential_proxy'] = differential_proxy(box)
        return pd.DataFrame(records)

    y_train = [r['composite_score'] for r in train]
    y_test = [r['composite_score'] for r in test]
    for proxy in (False, True):
        x_train, x_test = [frame([np.array(r['sbox'], dtype=np.uint8) for r in group], proxy)
                           for group in (train, test)]
        for name, model in [('rf', RandomForestRegressor(n_estimators=300, random_state=42)),
                            ('mlp', make_pipeline(StandardScaler(), MLPRegressor(
                                hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)))]:
            start = perf_counter()
            model.fit(x_train, y_train)
            train_sec = perf_counter() - start
            start = perf_counter()
            pred = model.predict(x_test)
            inference_sec = perf_counter() - start
            mse = mean_squared_error(y_test, pred)
            report['ablation'][f'{name}_{"proxy" if proxy else "cheap"}'] = {
                'mae': mean_absolute_error(y_test, pred), 'mse': mse, 'rmse': mse ** .5,
                'r2': r2_score(y_test, pred), 'train_sec': train_sec, 'inference_sec': inference_sec}
            models[name, proxy] = model
    report['pools'] = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed + 1000)
        parent = initialize(rng)
        pool = {}
        while len(pool) < 20:
            child = mutate(parent, 'single_swap', rng)
            pool[child.tobytes()] = child
        boxes = list(pool.values())
        actual = [score(box) for box in boxes]
        report['pools'].append({'seed': seed, **{
            f'{name}_{"proxy" if proxy else "cheap"}': pool_metrics(actual, model.predict(frame(boxes, proxy)), 3)
            for (name, proxy), model in models.items()}})
        for name, strategy, predict in [('rf', 'top_k', models['rf', False].predict),
                                        ('mlp', 'top_k', models['mlp', False].predict),
                                        ('random_top_k', 'top_k', None),
                                        ('random_permutation', 'random_permutation', None),
                                        ('random_search', 'random_search', None),
                                        ('random_swap', 'random_swap', None)]:
            start = perf_counter()
            result = reconstruct(score, predict, seed=seed, strategy=strategy, iterations=300,
                                 max_evaluations=151)
            report['paired'].append({'seed': seed, 'method': name, 'seconds': perf_counter() - start,
                                     **{key: result[key] for key in ('initial_score', 'best_score', 'full_evaluations')}})
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
