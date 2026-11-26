"""Generate the shared permutation-v2 dataset and verified split manifest."""

import argparse

from src.dataset_builder import build_dataset, write_dataset_bundle
from src.train_rf_v2 import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', type=int, default=150)
    parser.add_argument('--family-size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', default='datasets/sbox_perm_dataset.csv')
    parser.add_argument('--manifest', default='datasets/split_manifest.json')
    args = parser.parse_args()
    rows = build_dataset(args.families, args.family_size, args.seed, evaluate)
    manifest = write_dataset_bundle(rows, args.dataset, args.manifest, seed=args.seed)
    print(
        f"Saved {manifest['n_samples_total']} rows, "
        f"{manifest['n_families_total']} families; "
        f"dataset SHA-256 {manifest['dataset_sha256']}"
    )


if __name__ == '__main__':
    main()
