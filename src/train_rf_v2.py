"""Offline training CLI for the grouped permutation dataset."""

import argparse

from src.dataset_builder import build_dataset
from src.sbox_metrics import evaluate_sbox
from src.train_rf import save_grouped_rf, train_grouped_rf
from src.composite_score import compute_composite_score


def evaluate(box):
    metrics = evaluate_sbox(box)
    return dict(metrics, composite_score=compute_composite_score(metrics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    parser.add_argument('--families', type=int, default=150)
    parser.add_argument('--family-size', type=int, default=10)
    parser.add_argument('--trees', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', choices=('rf', 'mlp'), default='rf')
    args = parser.parse_args()
    estimator = None
    if args.model == 'mlp':
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(32, 16), max_iter=1000, random_state=args.seed))
    rows = build_dataset(args.families, args.family_size, args.seed, evaluate)
    model, metrics = train_grouped_rf(rows, n_estimators=args.trees, seed=args.seed, estimator=estimator)
    save_grouped_rf(args.output, model, metrics, rows, kind=args.model)


if __name__ == '__main__':
    main()
