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


def write_run_artifacts(root, result, seed, method, strategy, budget, iterations):
    import csv
    import hashlib
    import json
    from pathlib import Path

    from src.train_rf_v2 import evaluate

    method_dir = {
        'rf': 'rf',
        'xgboost': 'xgboost',
        'mlp': 'mlp',
        'random_permutation': 'baselines/random_permutation',
        'random_search': 'baselines/random_search',
        'random_swap': 'baselines/random_swap',
        'random_top_k': 'baselines/random_topk',
    }[method]
    directory = Path(root) / method_dir / f'seed_{seed}'
    directory.mkdir(parents=True, exist_ok=True)

    def save_box(filename, box):
        box = np.asarray(box, dtype=np.uint8)
        artifact = {
            'schema_version': '2.0.0',
            'method': method,
            'seed': seed,
            'sbox': box.tolist(),
            'sha256': hashlib.sha256(box.tobytes()).hexdigest(),
        }
        (directory / filename).write_text(
            json.dumps(artifact, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    save_box('initial_sbox.json', result['initial_sbox'])
    save_box('final_sbox.json', result['sbox'])
    save_box('inverse_sbox.json', result['inverse_sbox'])
    with (directory / 'reconstruction_history.csv').open('w', newline='', encoding='utf-8') as output:
        history = result['history']
        fields = list(history[0]) if history else [
            'iteration', 'operator_selected', 'n_candidates_generated', 'n_candidates_unique',
            'predicted_best_score', 'actual_evaluated_best_score', 'current_score',
            'global_best_score', 'accepted', 'full_evals_used', 'full_evals_cumulative',
            'cache_hits_cumulative', 'cache_misses_cumulative', 'feature_time_sec',
            'inference_time_sec', 'eval_time_sec', 'iter_total_time_sec',
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
    final_metrics = evaluate(np.asarray(result['sbox'], dtype=np.uint8))
    (directory / 'cryptographic_metrics.json').write_text(
        json.dumps({
            'schema_version': '2.0.0',
            'method': method,
            'seed': seed,
            'sbox_sha256': hashlib.sha256(
                np.asarray(result['sbox'], dtype=np.uint8).tobytes()).hexdigest(),
            'metrics': final_metrics,
            'security_note': 'Empirical metrics are not evidence of cryptographic security.',
        }, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    (directory / 'run_metadata.json').write_text(
        json.dumps({
            'schema_version': '2.0.0',
            'method': method,
            'seed': seed,
            'strategy': strategy,
            'iterations': iterations,
            'max_evaluations': budget,
            'initial_score': result['initial_score'],
            'final_score': result['best_score'],
            'score_delta': result['best_score'] - result['initial_score'],
            'full_evaluations': result['full_evaluations'],
            'cache_hits': result['cache_hits'],
            'cache_misses': result['cache_misses'],
            'feature_time_sec': result['feature_time_sec'],
            'inference_time_sec': result['inference_time_sec'],
            'evaluation_time_sec': result['evaluation_time_sec'],
            'total_time_sec': result['total_time_sec'],
        }, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def main():
    import argparse
    import json
    from collections import Counter
    from time import perf_counter
    from pathlib import Path

    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor
    from scipy.stats import t as student_t

    from src.dataset_builder import build_dataset, extract_features
    from src.ml_reconstruction import load_trusted_model, reconstruct
    from src.permutation_v2 import initialize, mutate
    from src.train_xgboost_v2 import extract_crypto_cheap_features
    from src.train_rf_v2 import evaluate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[100, 101, 102, 103, 104])
    parser.add_argument('--output', type=Path, default=Path('results/v2/comparative_summary.json'))
    parser.add_argument('--models-dir', type=Path, default=Path('models'))
    parser.add_argument('--families', type=int, default=150)
    parser.add_argument('--family-size', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--budget', type=int, default=151)
    args = parser.parse_args()
    score = lambda box: evaluate(box)['composite_score']
    start = perf_counter()
    rows = build_dataset(args.families, args.family_size, 42, evaluate)
    import hashlib
    dataset_sha256 = hashlib.sha256(json.dumps(
        sorted(rows, key=lambda row: row['sbox_id']),
        sort_keys=True,
        allow_nan=False,
    ).encode()).hexdigest()
    split_sha256 = hashlib.sha256(json.dumps(sorted(
        (row['sbox_id'], row['family_id'], row['split']) for row in rows
    )).encode()).hexdigest()
    report = {'counts': dict(Counter(row['split'] for row in rows)),
              'dataset_sec': perf_counter() - start, 'ablation': {}, 'paired': [],
              'config': {'seeds': args.seeds, 'families': args.families,
                         'family_size': args.family_size, 'iterations': args.iterations,
                         'max_evaluations': args.budget,
                         'dataset_sha256': dataset_sha256,
                         'split_sha256': split_sha256}}
    train = [r for r in rows if r['split'] == 'train']
    test = [r for r in rows if r['split'] == 'test']

    def frame(boxes, feature_set='perm4'):
        if feature_set == 'cheap':
            return pd.DataFrame([
                extract_crypto_cheap_features(box, include_sac_fp=True)
                for box in boxes
            ])
        if feature_set == 'proxy':
            return pd.DataFrame([
                extract_crypto_cheap_features(box, include_sac_fp=False)
                for box in boxes
            ])
        return pd.DataFrame([extract_features(box) for box in boxes])

    y_train = [r['composite_score'] for r in train]
    y_test = [r['composite_score'] for r in test]
    train_boxes = [np.array(r['sbox'], dtype=np.uint8) for r in train]
    test_boxes = [np.array(r['sbox'], dtype=np.uint8) for r in test]
    for feature_set in ('perm4', 'proxy', 'cheap'):
        x_train = frame(train_boxes, feature_set)
        x_test = frame(test_boxes, feature_set)
        for name, model in [
            ('rf', RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)),
            ('xgboost', XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                     objective='reg:squarederror', random_state=42,
                                     n_jobs=1, verbosity=0)),
            ('mlp', make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42))),
        ]:
            start = perf_counter()
            model.fit(x_train, y_train)
            train_sec = perf_counter() - start
            start = perf_counter()
            pred = model.predict(x_test)
            inference_sec = perf_counter() - start
            mse = mean_squared_error(y_test, pred)
            ranking = pool_metrics(y_test, pred, min(5, len(y_test)))
            report['ablation'][f'{name}_{feature_set}'] = {
                'mae': mean_absolute_error(y_test, pred), 'mse': mse, 'rmse': mse ** .5,
                'r2': r2_score(y_test, pred), 'spearman': ranking['spearman'],
                'top_k_regret': ranking['regret'], 'train_sec': train_sec,
                'inference_sec': inference_sec,
                'n_features': int(x_train.shape[1]),
                'feature_importances': dict(zip(
                    x_train.columns, model.feature_importances_.tolist()))
                if hasattr(model, 'feature_importances_') else None}

    search_models = {}
    model_sources = {}
    x_search_train = frame(train_boxes, 'perm4')
    fallback_specs = {
        'rf': lambda: RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1),
        'xgboost': lambda: XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            objective='reg:squarederror', random_state=42, n_jobs=1, verbosity=0),
        'mlp': lambda: make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42)),
    }
    for name in ('rf', 'xgboost', 'mlp'):
        candidate_dir = args.models_dir / name
        loaded = None
        if (candidate_dir / 'manifest.json').is_file():
            try:
                model_obj, manifest = load_trusted_model(candidate_dir)
                if (manifest.get('dataset_sha256') == dataset_sha256
                        and manifest.get('split_sha256') == split_sha256):
                    loaded = model_obj
            except (OSError, ValueError, json.JSONDecodeError):
                loaded = None
        if loaded is not None:
            search_models[name] = loaded
            model_sources[name] = str(candidate_dir).replace('\\', '/')
        else:
            model = fallback_specs[name]()
            model.fit(x_search_train, y_train)
            search_models[name] = model
            model_sources[name] = 'fitted_inline'
    report['config']['model_sources'] = model_sources
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
            name: pool_metrics(actual, model.predict(frame(boxes, 'perm4')), 3)
            for name, model in search_models.items()}})
        for name, strategy, predict in [('rf', 'top_k', search_models['rf'].predict),
                                        ('xgboost', 'top_k', search_models['xgboost'].predict),
                                        ('mlp', 'top_k', search_models['mlp'].predict),
                                        ('random_top_k', 'top_k', None),
                                        ('random_permutation', 'random_permutation', None),
                                        ('random_search', 'random_search', None),
                                        ('random_swap', 'random_swap', None)]:
            start = perf_counter()
            result = reconstruct(score, predict, seed=seed, strategy=strategy,
                                 iterations=args.iterations, max_evaluations=args.budget)
            write_run_artifacts(
                args.output.parent, result, seed, name, strategy, args.budget, args.iterations)
            report['paired'].append({
                'seed': seed,
                'method': name,
                'seconds': perf_counter() - start,
                **{key: result[key] for key in (
                    'initial_score', 'best_score', 'full_evaluations', 'feature_time_sec',
                    'inference_time_sec', 'evaluation_time_sec', 'total_time_sec', 'history')},
                'initial_sbox': result['initial_sbox'].tolist(),
                'final_sbox': result['sbox'].tolist(),
                'inverse_sbox': result['inverse_sbox'].tolist(),
                'score_delta': result['best_score'] - result['initial_score'],
            })

    report['paired_summary'] = {}
    methods = sorted({row['method'] for row in report['paired']})
    for method in methods:
        records = [row for row in report['paired'] if row['method'] == method]
        deltas = np.asarray([row['score_delta'] for row in records], dtype=float)
        report['paired_summary'][method] = {
            'n_seeds': len(records),
            'mean_score_delta': float(deltas.mean()),
            'sd_score_delta': float(deltas.std(ddof=1)) if len(deltas) > 1 else None,
            'mean_full_evaluations': float(np.mean([row['full_evaluations'] for row in records])),
            'mean_total_time_sec': float(np.mean([row['total_time_sec'] for row in records])),
        }
    random_topk = {row['seed']: row['score_delta'] for row in report['paired']
                   if row['method'] == 'random_top_k'}
    report['paired_differences_vs_random_top_k'] = {}
    for method in methods:
        paired = [row['score_delta'] - random_topk[row['seed']] for row in report['paired']
                  if row['method'] == method and row['seed'] in random_topk]
        if not paired:
            continue
        differences = np.asarray(paired, dtype=float)
        margin = (float(student_t.ppf(0.975, len(differences) - 1)
                        * differences.std(ddof=1) / np.sqrt(len(differences)))
                  if len(differences) > 1 else None)
        report['paired_differences_vs_random_top_k'][method] = {
            'n_pairs': len(differences),
            'mean_score_delta_difference': float(differences.mean()),
            'sd_score_delta_difference': float(differences.std(ddof=1))
            if len(differences) > 1 else None,
            'ci95': [float(differences.mean() - margin), float(differences.mean() + margin)]
            if margin is not None else None,
        }
    payload = json.dumps(report, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + '\n', encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
