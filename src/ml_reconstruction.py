"""Iterative permutation search with optional model ranking."""

import numpy as np
import pandas as pd

from src.dataset_builder import FEATURE_NAMES, extract_features
from src.permutation_v2 import initialize, mutate


def reconstruct(score, predict=None, *, seed=42, iterations=None, pool_size=20, top_k=3,
                max_evaluations=None, strategy="top_k", mode="standard",
                patience=None, restarts=0):
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

    def evaluate(box):
        key = box.tobytes()
        if key not in cache:
            value = float(score(box.copy()))
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
        pool = {}
        for _ in range(pool_size * 20):
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
            features = [extract_features(box) for box in candidates]
            predictions = np.asarray(predict(pd.DataFrame(features, columns=FEATURE_NAMES)))
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
        history.append({'iteration': iteration, 'candidate_id': candidates[index].tobytes().hex(),
                        'predicted_score': None if predictions is None else float(predictions[index]),
                        'actual_score': actual, 'best_score': best_score,
                        'accepted': accepted, 'full_evaluations': len(cache)})
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
            'history': history, 'full_evaluations': len(cache)}


def main():
    import argparse
    import json
    from pathlib import Path

    import joblib

    from src.dataset_builder import FEATURE_SCHEMA_VERSION
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
    if args.model is not None:
        manifest = json.loads((args.model / 'manifest.json').read_text(encoding='utf-8'))
        if (manifest['feature_schema_version'] != FEATURE_SCHEMA_VERSION
                or tuple(manifest['feature_names']) != FEATURE_NAMES):
            parser.error('Incompatible feature schema')
        kind = manifest.get("model_kind", "rf")
        if kind not in ("rf", "mlp"):
            parser.error("Unsupported model kind")
        model = joblib.load(args.model / f"{kind}.joblib")
        if tuple(model.feature_names_in_) != FEATURE_NAMES:
            parser.error('Incompatible model feature schema')
        if kind == "rf":
            model.n_jobs = 1
        predict = model.predict
    result = reconstruct(lambda box: evaluate(box)['composite_score'], predict,
                         seed=args.seed, max_evaluations=args.budget, strategy=args.strategy, mode=args.mode,
                         patience=args.patience, restarts=args.restarts)
    for key in ('sbox', 'initial_sbox', 'inverse_sbox'):
        result[key] = result[key].tolist()
    result.update(model=str(args.model) if args.model else None, seed=args.seed,
                  max_evaluations=args.budget, strategy=args.strategy)
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(result, output, allow_nan=False, indent=2)


if __name__ == '__main__':
    main()
