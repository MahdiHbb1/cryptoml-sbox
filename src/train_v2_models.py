"""Build the shared v2 dataset and train all models against its fixed split."""

import argparse
import json
from pathlib import Path

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset_builder import build_dataset, write_dataset_bundle
from src.train_rf import save_grouped_rf, train_grouped_rf
from src.train_rf_v2 import evaluate
from src.train_xgboost_v2 import save_grouped_xgboost, train_grouped_xgboost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', type=int, default=150)
    parser.add_argument('--family-size', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=Path('models'))
    parser.add_argument('--no-tune', action='store_true')
    args = parser.parse_args()

    rows = build_dataset(args.families, args.family_size, args.seed, evaluate)
    write_dataset_bundle(rows, seed=args.seed)

    rf, rf_metrics = train_grouped_rf(rows, n_estimators=300, seed=args.seed)
    save_grouped_rf(args.output_dir / 'rf', rf, rf_metrics, rows)

    mlp_estimator = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=args.seed),
    )
    mlp, mlp_metrics = train_grouped_rf(rows, seed=args.seed, estimator=mlp_estimator)
    save_grouped_rf(args.output_dir / 'mlp', mlp, mlp_metrics, rows, kind='mlp')

    xgboost, xgb_metrics = train_grouped_xgboost(
        rows, seed=args.seed, tune=not args.no_tune)
    save_grouped_xgboost(args.output_dir / 'xgboost', xgboost, xgb_metrics, rows)
    rf_manifest = json.loads(
        (args.output_dir / 'rf' / 'manifest.json').read_text(encoding='utf-8'))
    print(json.dumps({
        'dataset_sha256': rf_manifest['dataset_sha256'],
        'rf': rf_metrics,
        'mlp': mlp_metrics,
        'xgboost': xgb_metrics,
    }, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
