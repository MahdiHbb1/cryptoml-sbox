import json
import subprocess
import sys


def test_offline_training_and_reconstruction(tmp_path):
    model = tmp_path / 'model'
    output = tmp_path / 'run.json'
    subprocess.run([sys.executable, '-m', 'src.train_rf_v2', str(model), '--families', '8', '--family-size', '2', '--trees', '3'], check=True)
    subprocess.run([sys.executable, '-m', 'src.ml_reconstruction', str(output), '--model', str(model), '--budget', '4'], check=True)
    run = json.loads(output.read_text())
    assert run['full_evaluations'] == 4
    assert sorted(run['sbox']) == list(range(256))
    assert run['model'] == str(model)
    assert run['history']


def test_grouped_mlp_can_drive_reconstruction(tmp_path):
    model = tmp_path / 'model'
    output = tmp_path / 'run.json'
    subprocess.run([sys.executable, '-m', 'src.train_rf_v2', str(model), '--model', 'mlp', '--families', '8', '--family-size', '2'], check=True)
    subprocess.run([sys.executable, '-m', 'src.ml_reconstruction', str(output), '--model', str(model), '--budget', '4'], check=True)
    assert json.loads(output.read_text())['full_evaluations'] == 4
    assert json.loads((model / 'manifest.json').read_text())['model_kind'] == 'mlp'
