"""Benchmark S-Box catalogues on image encryption and lossless decryption."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.mandrill_eval import (
    DEFAULT_IV,
    DEFAULT_KEY,
    evaluate_sbox_on_image,
    load_image,
)
from src.train_rf_v2 import evaluate

_BASELINES_DIR = Path(__file__).resolve().parents[1] / "data" / "baselines"


def _load_baseline_sbox(filename: str) -> np.ndarray:
    path = _BASELINES_DIR / filename
    return np.asarray(np.load(path, allow_pickle=False), dtype=np.uint8)


def evaluate_mandrill_catalog(sboxes, image_path, modes=("RGB",)):
    """Evaluate every supplied S-Box on the same image and cipher settings."""
    image_path = Path(image_path)
    report = {"image_path": str(image_path), "runs": []}
    for mode in modes:
        image = load_image(image_path, mode=mode)
        for label, sbox in sboxes.items():
            result = evaluate_sbox_on_image(image, sbox)
            report["runs"].append({"sbox": label, "image_mode": mode, **result})
    return report


def run_v2_mandrill_benchmark(
    comparative_path='results/v2/comparative_summary.json',
    image_path='data/mandril_color.jpg',
    output_path='results/v2/mandrill_v2_metrics.json',
    seed=100,
):
    """Compare fixed-seed v2 reconstructions and AES/K44 under identical settings."""
    comparative_path, image_path, output_path = map(
        Path, (comparative_path, image_path, output_path))
    payload = json.loads(comparative_path.read_text(encoding='utf-8'))
    sboxes = {
        'AES Standard (FIPS-197)': _load_baseline_sbox('aes_standard.npy'),
        'K44 baseline': _load_baseline_sbox('k44.npy'),
    }
    method_titles = {'rf': 'RF', 'xgboost': 'XGBoost', 'mlp': 'MLP'}
    selected = {}
    for method in ('rf', 'xgboost', 'mlp'):
        matches = [row for row in payload['paired']
                   if row['method'] == method and row['seed'] == seed]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {method} result for seed {seed}')
        result = matches[0]
        box = np.asarray(result['final_sbox'], dtype=np.uint8)
        inverse = np.asarray(result['inverse_sbox'], dtype=np.uint8)
        if (box.shape != (256,) or np.unique(box).size != 256
                or not np.array_equal(inverse[box], np.arange(256, dtype=np.uint8))):
            raise ValueError(f'Invalid {method} S-Box or inverse')
        label = f'S-Box {method_titles[method]} v2 (seed {seed})'
        sboxes[label] = box
        selected[method] = {
            'label': label,
            'seed': seed,
            'initial_score': result['initial_score'],
            'final_score': result['best_score'],
            'score_delta': result['score_delta'],
            'initial_sbox': result['initial_sbox'],
            'inverse_sbox': result['inverse_sbox'],
        }

    report = evaluate_mandrill_catalog(sboxes, image_path, modes=('RGB', 'GRAY'))
    report.update({
        'schema_version': '2.0.0',
        'cipher_mode': 'CBC2',
        'key_hex': DEFAULT_KEY.hex(),
        'iv_hex': DEFAULT_IV.hex(),
        'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest(),
        'image_modes': ['RGB', 'GRAY'],
        'selected_v2_runs': selected,
        'security_note': 'Empirical image metrics are not evidence of cryptographic security.',
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    for method, selection in selected.items():
        label = selection['label']
        box = sboxes[label]
        directory = output_path.parent / method / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        metrics = evaluate(box)
        artifact = {
            'schema_version': '2.0.0',
            'method': method,
            'seed': seed,
            'sbox': box.tolist(),
            'sha256': hashlib.sha256(box.tobytes()).hexdigest(),
            'metrics': metrics,
        }
        initial = np.asarray(selection['initial_sbox'], dtype=np.uint8)
        initial_artifact = {
            'schema_version': '2.0.0',
            'method': method,
            'seed': seed,
            'sbox': initial.tolist(),
            'sha256': hashlib.sha256(initial.tobytes()).hexdigest(),
        }
        inverse_artifact = {
            'schema_version': '2.0.0',
            'method': method,
            'seed': seed,
            'sbox': np.argsort(box).astype(np.uint8).tolist(),
            'sha256': hashlib.sha256(np.argsort(box).astype(np.uint8).tobytes()).hexdigest(),
        }
        for filename, value in (
            ('initial_sbox.json', initial_artifact),
            ('final_sbox.json', artifact),
            ('inverse_sbox.json', inverse_artifact),
            ('cryptographic_metrics.json', {
                'schema_version': '2.0.0',
                'method': method,
                'seed': seed,
                'sbox_sha256': hashlib.sha256(box.tobytes()).hexdigest(),
                'metrics': metrics,
                'security_note': 'Empirical S-Box metrics are not evidence of cryptographic security.',
            }),
        ):
            (directory / filename).write_text(
                json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparative', default='results/v2/comparative_summary.json')
    parser.add_argument('--image', default='data/mandril_color.jpg')
    parser.add_argument('--output', default='results/v2/mandrill_v2_metrics.json')
    parser.add_argument('--seed', type=int, default=100)
    args = parser.parse_args()
    report = run_v2_mandrill_benchmark(
        args.comparative, args.image, args.output, seed=args.seed)
    print(f"Saved {len(report['runs'])} image/method runs to {args.output}")


if __name__ == '__main__':
    main()
