# Issue #18 — Final v2 Results

## Reproducibility

Run from the repository root with the dependencies in `requirements.txt`:

```bash
python -m src.train_v2_models --output-dir models/reproduction
python -m src.v2_experiment --output results/v2/comparative_summary.json
python -m src.run_v2_benchmark --seed 100
python -m pytest -q
streamlit run app.py
```

Training created 1,500 unique S-Boxes from 150 whole families with seed 42. The split is 1,050 train, 220 validation, and 230 locked test rows (70.00% / 14.67% / 15.33%). Every model manifest has the same dataset SHA-256 `7d6d902568f0fffee059c3c3c764fa48caea4ad29ff5c5630d4c766daf5c3be9` and split SHA-256 `898d951927446223a5a0de4c24c014d2e4e60018e1912425a313827d163d42e5`. No family or permutation hash crosses splits.

The four model inputs are `hamming_mean`, `hamming_std`, `displacement_mean`, and `displacement_std`. XGBoost hyperparameters were selected using validation only; test metrics were calculated after selection. MLP uses a train-fitted `StandardScaler`. Saved bundles include model, schema, split/dataset fingerprints, versions, and metrics.

## Held-out model metrics

| Model | Val MAE / RMSE | Test MAE / MSE / RMSE | Test R² | Test Spearman | Mean per-family Spearman | Top-3 hit rate | Mean Top-3 regret | Train / inference (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF | 3.9861 / 5.0374 | 4.2052 / 24.7498 / 4.9749 | -0.1977 | 0.1010 | -0.0393 | 21.74% | 1.6285 | 5.672 / 0.045 |
| XGBoost | 3.5948 / 4.6870 | 4.0887 / 24.3820 / 4.9378 | -0.1799 | 0.0343 | -0.0457 | 21.74% | 1.9650 | 1.201 / 0.003 |
| MLP | 4.0308 / 5.1320 | 3.8210 / 22.0626 / 4.6971 | -0.0677 | 0.2840 | 0.0967 | 39.13% | 1.1631 | 4.621 / 0.003 |

The test contains 23 held-out families. RF/XGBoost feature importance is recorded in the model manifests: RF ranks displacement mean (0.2753) and Hamming standard deviation (0.2579) highest; XGBoost ranks Hamming standard deviation (0.2864) highest. MLP has no native tree-style feature importance. Despite the rank diagnostics, all test R² values are below zero: the regressors do not outperform a constant-mean predictor on held-out families. MLP reached its 1,000-iteration limit without convergence. Results are reported as observed; no model-quality improvement is claimed.

## Paired reconstruction and ablation

Five paired seeds (100–104) used identical initial permutations per seed, actual-score acceptance, and at most 151 full evaluations. RF, XGBoost, MLP, random swap, and random Top-K use single-swap candidates; random search samples independent permutations. The former methods each used 151 evaluations per seed. The initial-permutation baseline evaluates only its starting point (one evaluation), so its score delta is not a budget-matched search result. All run histories, S-Boxes, inverse S-Boxes, cryptographic metrics, timing counters, and metadata are saved under `results/v2/<method>/seed_<seed>/`.

| Method | Mean score delta | Sample SD | Mean full evaluations | Mean wall time (s) |
| --- | ---: | ---: | ---: | ---: |
| RF | 10.3776 | 5.4633 | 151 | 4.480 |
| XGBoost | 9.8689 | 3.2066 | 151 | 2.939 |
| MLP | 9.8607 | 2.0710 | 151 | 2.967 |
| Random Top-K | 10.1946 | 3.4037 | 151 | 2.397 |
| Random search | 9.6149 | 3.3454 | 151 | 2.778 |
| Random swap | 11.1546 | 3.1220 | 151 | 3.745 |
| Initial permutation | 0.0000 | 0.0000 | 1 | 0.059 |

Paired score-delta differences versus random Top-K (mean and 95% t-interval across five seeds):

| Method | Mean difference | 95% interval |
| --- | ---: | ---: |
| RF | +0.1830 | [-5.2184, 5.5844] |
| XGBoost | -0.3258 | [-1.5502, 0.8987] |
| MLP | -0.3339 | [-2.9098, 2.2421] |
| Random search | -0.5798 | [-1.5598, 0.4003] |
| Random swap | +0.9599 | [-1.7041, 3.6239] |

Every model interval includes zero; five seeds do not establish that learned ranking improves reconstruction. The report includes per-seed initial/final scores and convergence histories, plus feature, inference, evaluation, and total times.

The held-out ablation compares inexpensive features (including SAC/fixed-point features) against the differential-proxy set without SAC/fixed-point features:

| Model | Set | MAE | RMSE | R² | Spearman | Top-5 regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RF | Cheap | 4.1033 | 4.8437 | -0.1353 | 0.0973 | 3.7327 |
| RF | Proxy | 4.5931 | 5.5337 | -0.4818 | -0.1180 | 4.7171 |
| XGBoost | Cheap | 4.3039 | 5.1447 | -0.2808 | 0.0982 | 3.2978 |
| XGBoost | Proxy | 4.5392 | 5.4873 | -0.4571 | -0.0387 | 4.7054 |
| MLP | Cheap | 5.7441 | 7.5251 | -1.7403 | 0.0942 | 6.0604 |
| MLP | Proxy | 4.6072 | 5.6967 | -0.5704 | -0.0831 | 4.4690 |

Across five independently generated candidate pools, mean Top-3 regrets were 1.9069 (RF), 1.8479 (XGBoost), and 1.5222 (MLP); mean Spearman correlations were -0.0058, 0.0422, and -0.0556 respectively. These small-pool diagnostics are not security measurements.

## Mandrill v2 benchmark

The fixed-seed-100 reconstructions were compared with standard AES and K44 on the same 512×512 Mandrill image in RGB and grayscale. All runs used CBC2, the same key/IV, and a one-bit plaintext change for NPCR/UACI. The report stores image SHA-256, cipher metadata, per-run timing, horizontal/vertical/diagonal pixel correlation, and inverse/decryption checks.

| Mode | S-Box | Cipher entropy | χ² | NPCR (%) | UACI (%) | Mean absolute pixel correlation | Lossless |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| RGB | AES standard S-Box | 7.99976 | 257.13 | 99.6136 | 33.4650 | 0.000265 | Yes |
| RGB | K44 | 7.99976 | 259.91 | 99.6072 | 33.4807 | 0.000344 | Yes |
| RGB | RF v2 | 7.99972 | 307.71 | 99.6232 | 33.4897 | 0.001047 | Yes |
| RGB | XGBoost v2 | 7.99974 | 279.58 | 99.6043 | 33.4563 | 0.000744 | Yes |
| RGB | MLP v2 | 7.99978 | 234.83 | 99.6189 | 33.4538 | 0.000877 | Yes |
| Gray | AES standard S-Box | 7.99939 | 220.04 | 99.5907 | 33.4784 | 0.001264 | Yes |
| Gray | K44 | 7.99934 | 239.35 | 99.5995 | 33.4546 | 0.001983 | Yes |
| Gray | RF v2 | 7.99937 | 228.70 | 99.6201 | 33.4715 | 0.001197 | Yes |
| Gray | XGBoost v2 | 7.99928 | 260.62 | 99.6185 | 33.4604 | 0.002793 | Yes |
| Gray | MLP v2 | 7.99926 | 270.32 | 99.6162 | 33.4955 | 0.002522 | Yes |

All 10 image runs decrypted byte-for-byte. Custom S-Boxes are labelled as modified AES; the AES-standard S-Box row also uses the project's modified CBC2 mode. Entropy, histogram, NPCR/UACI, and correlation are empirical image metrics, not evidence of cryptographic security.

## Dashboard and verification

The six Streamlit pages load the persisted dataset, model, reconstruction, crypto-evaluation, Mandrill, and paired-comparison artifacts without training on app startup. Live reconstruction is opt-in and uses the schema-checked saved XGBoost bundle. Missing/incompatible artifacts are shown explicitly. Legacy v1 results remain available and separate from v2.

Verification on the development checkout: `181 passed` (`python -m pytest -q`; 247 dependency/convergence warnings), all six pages plus a live reconstruction via Streamlit `AppTest`, and the XGBoost train/save/load CLI smoke test. See the PR checks for the final commit's CI status.
