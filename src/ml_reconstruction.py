"""Iterative permutation search with optional model ranking."""

import numpy as np
import pandas as pd

from src.dataset_builder import FEATURE_NAMES, extract_features
from src.permutation_v2 import initialize, mutate


def reconstruct(score, predict=None, *, seed=42, iterations=50, pool_size=20, top_k=3,
                max_evaluations=None):
    if max_evaluations is not None and (not isinstance(max_evaluations, int) or max_evaluations < 1):
        raise ValueError('max_evaluations must be a positive integer')
    if not 1 <= top_k <= pool_size <= 32640 or iterations < 0:
        raise ValueError('Invalid search budget')
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
    history = []
    for iteration in range(iterations):
        pool = {}
        for _ in range(pool_size * 20):
            candidate = mutate(current, 'single_swap', rng)
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
        history.append({'iteration': iteration, 'candidate_id': candidates[index].tobytes().hex(),
                        'predicted_score': None if predictions is None else float(predictions[index]),
                        'actual_score': actual, 'best_score': current_score,
                        'accepted': accepted, 'full_evaluations': len(cache)})
    inverse = np.argsort(current).astype(np.uint8)
    return {'sbox': current, 'initial_sbox': initial, 'inverse_sbox': inverse,
            'initial_score': initial_score, 'best_score': current_score,
            'history': history, 'full_evaluations': len(cache)}
