"""Iterative permutation search with optional model ranking."""

import numpy as np
import pandas as pd
import time

from src.dataset_builder import FEATURE_NAMES, extract_features
from src.permutation_v2 import initialize, mutate


def reconstruct(score, predict=None, *, seed=42, iterations=None, pool_size=20, top_k=3,
                max_evaluations=None, strategy="top_k", mode="standard",
                patience=None, restarts=0):
    started = time.perf_counter()
    presets = {'fast': 15, 'standard': 50, 'research': 200}
    if mode not in presets or (patience is not None and patience < 1) or restarts < 0:
        raise ValueError('Invalid mode, patience, or restarts')
    if iterations is None:
        iterations = presets[mode]
    if max_evaluations is not None and (not isinstance(max_evaluations, int) or max_evaluations < 1):
        raise ValueError('max_evaluations must be a positive integer')
    if not 1 <= top_k <= pool_size <= 32640 or iterations < 0:
        raise ValueError('Invalid search budget')
    if strategy not in ("top_k", "random_permutation", "random_search", "random_swap"):
        raise ValueError("Unknown strategy")
    if strategy != "top_k" and predict is not None:
        raise ValueError("Model ranking requires top_k strategy")
    if strategy == "random_permutation":
        iterations = 0
    elif strategy == "random_swap":
        pool_size = top_k = 1
    mutation_seed, ranking_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(mutation_seed)
    ranking_rng = np.random.default_rng(ranking_seed)
    initial = current = initialize(rng)
    cache = {}
    cache_hits = feature_time = inference_time = evaluation_time = 0

    def evaluate(box):
        nonlocal cache_hits, evaluation_time
        key = box.tobytes()
        if key in cache:
            cache_hits += 1
        else:
            evaluation_started = time.perf_counter()
            value = float(score(box.copy()))
            evaluation_time += time.perf_counter() - evaluation_started
            if not np.isfinite(value):
                raise ValueError('Score must be finite')
            cache[key] = value
        return cache[key]

    current_score = initial_score = evaluate(current)
    best, best_score = current, current_score
    stalled = restart_count = 0
    history = []
    for iteration in range(iterations):
        if max_evaluations is not None and len(cache) >= max_evaluations:
            break
        iteration_started = time.perf_counter()
        feature_before, inference_before = feature_time, inference_time
        evaluation_before, cache_hits_before = evaluation_time, cache_hits
        evaluations_before = len(cache)
        pool = {}
        candidates_generated = 0
        for _ in range(pool_size * 20):
            candidates_generated += 1
            candidate = (initialize(rng) if strategy == 'random_search'
                         else mutate(current, 'single_swap', rng))
            pool.setdefault(candidate.tobytes(), candidate)
            if len(pool) == pool_size:
                break
        if len(pool) != pool_size:
            raise RuntimeError('Could not fill unique candidate pool')
        candidates = list(pool.values())
        predictions = None
        if predict is None:
            indices = ranking_rng.choice(pool_size, top_k, replace=False)
        else:
            feature_started = time.perf_counter()
            features = [extract_features(box) for box in candidates]
            feature_time += time.perf_counter() - feature_started
            inference_started = time.perf_counter()
            predictions = np.asarray(predict(pd.DataFrame(features, columns=FEATURE_NAMES)))
            inference_time += time.perf_counter() - inference_started
            if predictions.shape != (pool_size,) or not np.isfinite(predictions).all():
                raise ValueError('Expected one finite prediction per candidate')
            indices = np.argsort(-predictions, kind='stable')[:top_k]
        ranked = []
        for i in indices:
            if max_evaluations is not None and len(cache) >= max_evaluations:
                break
            ranked.append((evaluate(candidates[i]), int(i)))
        if not ranked:
            break
        actual, index = max(ranked, key=lambda pair: pair[0])
        accepted = actual > current_score
        if accepted:
            current, current_score = candidates[index], actual
        if current_score > best_score:
            best, best_score = current, current_score
        stalled = 0 if accepted else stalled + 1
        history.append({
            'iteration': iteration,
            'operator_selected': 'random_permutation' if strategy == 'random_search' else 'single_swap',
            'n_candidates_generated': candidates_generated,
            'n_candidates_unique': len(pool),
            'candidate_id': candidates[index].tobytes().hex(),
            'predicted_score': None if predictions is None else float(predictions[index]),
            'predicted_best_score': None if predictions is None else float(predictions[index]),
            'actual_score': actual,
            'actual_evaluated_best_score': actual,
            'current_score': current_score,
            'best_score': best_score,
            'global_best_score': best_score,
            'accepted': accepted,
            'full_evals_used': len(ranked),
            'full_evals_cumulative': len(cache),
            'full_evaluations': len(cache),
            'cache_hits_cumulative': cache_hits,
            'cache_misses_cumulative': len(cache),
            'feature_time_sec': feature_time - feature_before,
            'inference_time_sec': inference_time - inference_before,
            'eval_time_sec': evaluation_time - evaluation_before,
            'iter_total_time_sec': time.perf_counter() - iteration_started,
            'evaluation_count_before': evaluations_before,
            'cache_hits_before': cache_hits_before,
        })
        if patience is not None and stalled >= patience:
            if restart_count >= restarts or (max_evaluations is not None and len(cache) >= max_evaluations):
                break
            current = initialize(rng)
            current_score = evaluate(current)
            if current_score > best_score:
                best, best_score = current, current_score
            restart_count += 1
            stalled = 0
    inverse = np.argsort(best).astype(np.uint8)
    return {'sbox': best, 'initial_sbox': initial, 'inverse_sbox': inverse,
            'initial_score': initial_score, 'best_score': best_score, 'restarts': restart_count,
            'history': history, 'full_evaluations': len(cache), 'cache_hits': cache_hits,
            'cache_misses': len(cache), 'feature_time_sec': feature_time,
            'inference_time_sec': inference_time, 'evaluation_time_sec': evaluation_time,
            'total_time_sec': time.perf_counter() - started}


def load_trusted_model(directory):
    """Load a versioned local artifact created by one of the grouped trainers."""
    import json
    from pathlib import Path

    import joblib
    from src.dataset_builder import FEATURE_SCHEMA_VERSION

    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('schema_version') != '2.0.0'
            or manifest.get('feature_schema_version') != FEATURE_SCHEMA_VERSION
            or tuple(manifest.get('feature_names', ())) != FEATURE_NAMES):
        raise ValueError('Incompatible feature schema')
    split_path = directory.parent.parent / 'datasets' / 'split_manifest.json'
    if split_path.is_file():
        split_manifest = json.loads(split_path.read_text(encoding='utf-8'))
        if (manifest.get('dataset_sha256') != split_manifest.get('dataset_sha256')
                or manifest.get('split_sha256') != split_manifest.get('split_sha256')):
            raise ValueError('Incompatible dataset or split fingerprint')
    kind = manifest.get('model_kind')
    if kind not in ('rf', 'mlp', 'xgboost'):
        raise ValueError('Unsupported model kind')
    model_path = directory / f'{kind}.joblib'
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = joblib.load(model_path)
    if (tuple(getattr(model, 'feature_names_in_', ())) != FEATURE_NAMES
            or getattr(model, 'n_features_in_', None) != len(FEATURE_NAMES)):
        raise ValueError('Incompatible model feature schema')
    if kind == 'rf':
        model.n_jobs = 1
    return model, manifest


def main():
    import argparse
    import json
    from pathlib import Path

    from src.train_rf_v2 import evaluate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', type=Path, help='Trusted local RF or MLP artifact directory; omit for random Top-K')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--strategy', choices=('top_k', 'random_permutation', 'random_search', 'random_swap'), default='top_k')
    parser.add_argument('--budget', type=int, default=151)
    parser.add_argument('--mode', choices=('fast', 'standard', 'research'), default='standard')
    parser.add_argument('--patience', type=int)
    parser.add_argument('--restarts', type=int, default=0)
    args = parser.parse_args()
    predict = None
    model_kind = None
    if args.model is not None:
        model, manifest = load_trusted_model(args.model)
        model_kind = manifest['model_kind']
        predict = model.predict
    result = reconstruct(lambda box: evaluate(box)['composite_score'], predict,
                         seed=args.seed, max_evaluations=args.budget, strategy=args.strategy, mode=args.mode,
                         patience=args.patience, restarts=args.restarts)
    for key in ('sbox', 'initial_sbox', 'inverse_sbox'):
        result[key] = result[key].tolist()
    result.update(model=str(args.model) if args.model else None, seed=args.seed,
                  max_evaluations=args.budget, strategy=args.strategy, model_kind=model_kind)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, allow_nan=False, indent=2)


if __name__ == '__main__':
    main()
