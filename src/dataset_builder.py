"""Generator kandidat Non-AES dan dataset bersama S-Box."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.composite_score import COMPOSITE_SCORE_VERSION, compute_composite_score
from src.permutation_v2 import initialize, mutate
from src.sbox_metrics import EVALUATOR_VERSION


FEATURE_SCHEMA_VERSION = 'permutation-v2.1'
FEATURE_NAMES = ('hamming_mean', 'hamming_std', 'displacement_mean', 'displacement_std')
_VALUES = np.arange(256, dtype=np.uint8)
_WEIGHTS = np.array([i.bit_count() for i in range(256)])
_OPERATORS = (('single_swap', 1), ('multi_swap', 2), ('three_cycle', 1), ('segment_reversal', 4))


def extract_features(box):
    if box.shape != (256,) or box.dtype != np.uint8 or np.unique(box).size != 256:
        raise ValueError('Expected a uint8 permutation of 0–255')
    hamming = _WEIGHTS[box ^ _VALUES]
    displacement = np.abs(box.astype(np.int16) - _VALUES)
    return dict(zip(FEATURE_NAMES, map(float, (hamming.mean(), hamming.std(), displacement.mean(), displacement.std()))))


def build_dataset(n_families, family_size, seed, evaluate):
    """Split whole families. Evaluator returns metrics including composite_score."""
    if n_families < 3 or family_size < 1:
        raise ValueError('Need at least three families and one row per family')
    split_rng, *family_seeds = np.random.SeedSequence(seed).spawn(n_families + 1)
    order = np.random.default_rng(split_rng).permutation(n_families)
    n_train = min(n_families - 2, max(1, round(0.70 * n_families)))
    n_val = min(n_families - n_train - 1, max(1, round(0.15 * n_families)))
    splits = {int(f): 'train' if i < n_train else 'validation' if i < n_train + n_val else 'test'
              for i, f in enumerate(order)}
    rows, seen = [], set()
    for family, sequence in enumerate(family_seeds):
        family_seed = int(sequence.generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(family_seed)
        parent, parent_id = None, None
        for iteration in range(family_size):
            operator, intensity = _OPERATORS[(iteration - 1) % len(_OPERATORS)] if iteration else ('initial', 0)
            for _ in range(100):
                box = initialize(rng) if parent is None else mutate(parent, operator, rng, intensity)
                box_id = sha256(box.tobytes()).hexdigest()
                if box_id not in seen:
                    break
            else:
                raise RuntimeError('Could not sample a unique permutation')
            seen.add(box_id)
            features = extract_features(box)
            metrics = dict(evaluate(box.copy()))
            score = float(metrics['composite_score'])
            if not np.isfinite(score):
                raise ValueError('Composite Score must be finite')
            rows.append({'sample_id': box_id, 'sbox_id': box_id, 'parent_id': parent_id, 'family_id': family,
                         'seed': family_seed, 'iteration': iteration, 'operator': operator,
                         'intensity': intensity, 'split': splits[family], 'sbox': box.tolist(),
                         'features': features, 'metrics': metrics, 'composite_score': score})
            parent, parent_id = box, box_id
    return rows


def write_dataset_bundle(rows, dataset_path='datasets/sbox_perm_dataset.csv',
                         manifest_path='datasets/split_manifest.json', seed=42):
    """Write reproducible dataset CSV and a verified, whole-family split manifest."""
    from collections import Counter

    if not rows:
        raise ValueError('Dataset must contain at least one row')
    families, hashes = {}, set()
    groups = {'train': [], 'validation': [], 'test': []}
    for row in rows:
        split = row['split']
        if split not in groups or families.setdefault(row['family_id'], split) != split:
            raise ValueError('Invalid split or family leakage')
        sbox_id = sha256(np.asarray(row['sbox'], dtype=np.uint8).tobytes()).hexdigest()
        if sbox_id != row['sbox_id'] or sbox_id in hashes:
            raise ValueError('Duplicate or mismatched S-Box fingerprint')
        hashes.add(sbox_id)
        groups[split].append(row['sbox_id'])
    if any(not groups[name] for name in groups):
        raise ValueError('Train, validation, and test splits must all be non-empty')

    counts = Counter(row['split'] for row in rows)
    split_by_sbox = {
        row['sbox_id']: (row['family_id'], row['split'])
        for row in rows
    }
    split_hash = sha256(json.dumps(sorted(
        (sbox_id, family_id, split)
        for sbox_id, (family_id, split) in split_by_sbox.items()
    )).encode()).hexdigest()
    dataset_hash = sha256(json.dumps(
        sorted(rows, key=lambda row: row['sbox_id']),
        sort_keys=True,
        allow_nan=False,
    ).encode()).hexdigest()
    n = len(rows)
    manifest = {
        'schema_version': '2.0.0',
        'split_seed': int(seed),
        'ratios_target': {'train': 0.70, 'val': 0.15, 'test': 0.15},
        'ratios_actual': {
            'train': counts['train'] / n,
            'val': counts['validation'] / n,
            'test': counts['test'] / n,
        },
        'dataset_path': str(dataset_path).replace('\\', '/'),
        'dataset_sha256': dataset_hash,
        'split_sha256': split_hash,
        'evaluator_version': EVALUATOR_VERSION,
        'composite_score_version': COMPOSITE_SCORE_VERSION,
        'feature_version': FEATURE_SCHEMA_VERSION,
        'n_samples_total': n,
        'n_families_total': len(families),
        'disjoint_family_verified': len(families) == len(set(families)),
        'disjoint_hash_verified': len(hashes) == n,
        'train_ids': sorted(groups['train']),
        'val_ids': sorted(groups['validation']),
        'test_ids': sorted(groups['test']),
    }
    dataset_file, manifest_file = Path(dataset_path), Path(manifest_path)
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for row in rows:
        records.append({
            'sample_id': row['sample_id'],
            'sbox_hash': row['sbox_id'],
            'parent_id': row['parent_id'],
            'family_id': row['family_id'],
            'seed': row['seed'],
            'operator': row['operator'],
            'mutation_intensity': row['intensity'],
            'iteration': row['iteration'],
            'split': row['split'],
            **row['features'],
            **row['metrics'],
            'sbox_json': json.dumps(row['sbox'], separators=(',', ':')),
        })
    pd.DataFrame(records).to_csv(dataset_file, index=False)
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    return manifest


def extract_fast_features(sbox: np.ndarray) -> np.ndarray:
    """Menghasilkan 40 fitur numerik cepat yang dipakai seluruh model ML."""
    from src.train_xgboost import extract_fast_features_default

    return extract_fast_features_default(sbox)


def _logistic_permutation(r: float, x0: float, skip: int) -> np.ndarray:
    value = x0
    for _ in range(skip):
        value = r * value * (1.0 - value)
    sequence = np.empty(256, dtype=np.float64)
    for index in range(256):
        value = r * value * (1.0 - value)
        sequence[index] = value
    return np.argsort(sequence, kind="stable").astype(np.uint8)


def generate_chaotic_swap_candidates(
    n: int,
    seed: int = 42,
    min_swaps: int = 1,
    max_swaps: int = 24,
    skip: int = 80,
) -> list[dict]:
    """Membangkitkan ``n`` S-Box unik dari logistic map lalu pertukaran elemen."""
    from src.aes_cipher_runner import validate_sbox

    if n < 0:
        raise ValueError("Jumlah kandidat tidak boleh negatif.")
    if not 0 <= min_swaps <= max_swaps:
        raise ValueError("Rentang jumlah swap tidak valid.")
    if skip < 0:
        raise ValueError("Jumlah iterasi skip tidak boleh negatif.")

    rng = np.random.default_rng(seed)
    candidates: list[dict] = []
    seen: set[bytes] = set()
    attempts = 0
    max_attempts = max(n * 20, 100)

    while len(candidates) < n and attempts < max_attempts:
        attempts += 1
        r = float(rng.uniform(3.85, 3.9999))
        x0 = float(rng.uniform(0.05, 0.95))
        sbox = _logistic_permutation(r, x0, skip)
        swap_count = int(rng.integers(min_swaps, max_swaps + 1))
        for _ in range(swap_count):
            left, right = rng.choice(256, size=2, replace=False)
            sbox[left], sbox[right] = sbox[right], sbox[left]
        sbox = validate_sbox(sbox)
        key = sbox.tobytes()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "sbox": sbox,
                "source": "chaotic_swap",
                "r": r,
                "x0": x0,
                "swaps": swap_count,
            }
        )

    if len(candidates) != n:
        raise RuntimeError(f"Hanya berhasil membuat {len(candidates)} dari {n} kandidat unik.")
    return candidates


def resolve_evaluator() -> Callable[[np.ndarray], dict]:
    try:
        from src.sbox_metrics import evaluate_sbox
    except ImportError:
        # ponytail: fallback sementara setara kontrak; otomatis naik ke evaluator Naila setelah modul di-merge.
        from src.train_xgboost import evaluate_sbox_default

        return evaluate_sbox_default
    return evaluate_sbox


def build_sbox_dataset(
    n_chaotic: int = 750,
    n_aes: int = 750,
    seed: int = 42,
    output_path: str | Path | None = Path("data/sbox_dataset.csv"),
    evaluate_fn: Callable[[np.ndarray], dict] | None = None,
) -> pd.DataFrame:
    """Membentuk dataset gabungan sesuai ``ARCHITECTURE_CONTRACT.md``."""
    from src.aes_affine_generator import generate_aes_affine_candidates
    from src.aes_cipher_runner import validate_sbox

    if n_chaotic < 0 or n_aes < 0 or n_chaotic + n_aes == 0:
        raise ValueError("Jumlah kandidat harus positif dan tidak boleh negatif.")

    candidates = generate_chaotic_swap_candidates(n_chaotic, seed=seed)
    candidates.extend(
        generate_aes_affine_candidates(
            n=n_aes,
            seed=seed + 1,
            avoid_fixed_points=False,
        )
    )
    evaluator = evaluate_fn or resolve_evaluator()
    rows: list[dict] = []

    for index, candidate in enumerate(candidates):
        sbox = validate_sbox(candidate["sbox"])
        features = extract_fast_features(sbox)
        metrics = evaluator(sbox)
        row = {
            "sbox_id": f"sbox_{index:06d}",
            "source": candidate["source"],
            **{f"f_{feature_index}": float(value) for feature_index, value in enumerate(features)},
            "nl": int(metrics["nl"]),
            "du": int(metrics["du"]),
            "sac_max_err": float(metrics["sac_max_err"]),
            "whs_cost": float(metrics["whs_cost"]),
            "composite_score": compute_composite_score(metrics),
        }
        rows.append(row)

    dataset = pd.DataFrame(rows)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(path, index=False)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Bangun dataset S-Box gabungan untuk model ML.")
    parser.add_argument("--n-chaotic", type=int, default=750)
    parser.add_argument("--n-aes", type=int, default=750)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/sbox_dataset.csv")
    args = parser.parse_args()
    dataset = build_sbox_dataset(
        n_chaotic=args.n_chaotic,
        n_aes=args.n_aes,
        seed=args.seed,
        output_path=args.output,
    )
    print(f"Tersimpan {len(dataset)} baris, {len([c for c in dataset if c.startswith('f_')])} fitur: {args.output}")


if __name__ == "__main__":
    main()
