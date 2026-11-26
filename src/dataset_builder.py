"""Generator kandidat Non-AES dan dataset bersama S-Box."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.aes_affine_generator import generate_aes_affine_candidates
from src.aes_cipher_runner import validate_sbox
from src.train_xgboost import (
    compute_composite_score,
    evaluate_sbox_default,
    extract_fast_features_default,
)


from src.permutation_v2 import initialize, mutate


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
            rows.append({'sbox_id': box_id, 'parent_id': parent_id, 'family_id': family,
                         'seed': family_seed, 'iteration': iteration, 'operator': operator,
                         'intensity': intensity, 'split': splits[family], 'sbox': box.tolist(),
                         'features': features, 'metrics': metrics, 'composite_score': score})
            parent, parent_id = box, box_id
    return rows


def extract_fast_features(sbox: np.ndarray) -> np.ndarray:
    """Menghasilkan 40 fitur numerik cepat yang dipakai seluruh model ML."""
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
