"""Generator kandidat Non-AES dan dataset bersama S-Box."""

from __future__ import annotations

import argparse
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


def _resolve_evaluator() -> Callable[[np.ndarray], dict]:
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
    evaluator = evaluate_fn or _resolve_evaluator()
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
