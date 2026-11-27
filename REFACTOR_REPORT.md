# Issue #18 — Final v2 Results (`CryptoML-SBox v2`)

## Reproducibility

Run from the repository root with the dependencies in `requirements.txt`:

```bash
python -m src.train_v2_models --output-dir models/reproduction
python -m src.v2_experiment --output results/v2/comparative_summary.json
python -m src.run_v2_benchmark --seed 100
python -m pytest -q
streamlit run app.py
```

Training created **1,500 unique polynomial-free S-Boxes** from **150 whole families** with seed `42` (zero irreducible polynomials, zero GF(2⁸) multiplicative inverses, zero affine matrices, and zero chaotic maps on the proposed S-Box path). The family-disjoint split is **1,050 train, 220 validation, and 230 locked test rows** (`70.00% / 14.67% / 15.33%`). Every saved model manifest shares identical fingerprints:

- `dataset_sha256`: `7d6d902568f0fffee059c3c3c764fa48caea4ad29ff5c5630d4c766daf5c3be9`
- `split_sha256`: `898d951927446223a5a0de4c24c014d2e4e60018e1912425a313827d163d42e5`

No family or permutation hash crosses splits (`disjoint_family_verified: true`, `disjoint_hash_verified: true`).

The four production model inputs (`FEATURE_SCHEMA_VERSION = "permutation-v2.1"`) are `hamming_mean`, `hamming_std`, `displacement_mean`, and `displacement_std` — preserving Yafi's (`@twizzcode`, Issue #17) and Naila's (`@NAILAAR22`, Issue #16) authentic grouped dataset, RF, and MLP artifacts intact. XGBoost hyperparameters were selected using validation only; test metrics were computed once after selection. MLP uses a train-fitted `StandardScaler`. Saved bundles under `models/{rf,xgboost,mlp}/` include model weights, feature schema, dataset/split fingerprints, package versions, and held-out metrics.

## Held-out model metrics (`permutation-v2.1`)

| Model | Val MAE / RMSE | Test MAE / MSE / RMSE | Test R² | Test Spearman | Mean per-family Spearman | Top-3 hit rate | Mean Top-3 regret | Train / inference (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF | 3.9861 / 5.0374 | 4.2052 / 24.7498 / 4.9749 | -0.1977 | +0.1010 | -0.0393 | 21.74% | 1.6285 | 5.672 / 0.045 |
| XGBoost | 3.5948 / 4.6870 | 4.0887 / 24.3820 / 4.9378 | -0.1799 | +0.0343 | -0.0457 | 21.74% | 1.9650 | 1.201 / 0.003 |
| MLP | 4.0308 / 5.1320 | 3.8210 / 22.0626 / 4.6971 | -0.0677 | +0.2840 | +0.0967 | 39.13% | 1.1631 | 4.621 / 0.003 |

The locked test split contains 23 held-out families. RF ranks `displacement_mean` (0.2753) and `hamming_std` (0.2579) highest; XGBoost ranks `hamming_std` (0.2864) highest. MLP has no native tree feature importance and reached its 1,000-iteration limit without convergence. All test $R^2$ values are below zero: the four permutation-distance features alone do not outperform a constant-mean predictor on unseen families. Results are reported honestly as observed; no fabricated accuracy improvement is claimed.

## Paired reconstruction and 3-way feature ablation

Five paired seeds (`100–104`) used identical initial permutations per seed, actual-score acceptance (`evaluate_sbox`), and a matched cap of **151 full evaluations** per search method. Paired ML runs loaded the verified saved artifacts directly (`models/rf`, `models/xgboost`, `models/mlp`). RF, XGBoost, MLP, random swap, and random Top-K use single-swap candidates; random search samples independent permutations. The initial-permutation baseline evaluates only its starting point (`1` evaluation), so its score delta is `0.0000`. All run histories, S-Boxes, inverse S-Boxes, cryptographic metrics, timing counters, and metadata are saved under `results/v2/<method>/seed_<seed>/`.

| Method | Mean score delta | Sample SD | Mean full evaluations | Mean wall time (s) |
| --- | ---: | ---: | ---: | ---: |
| XGBoost (saved `models/xgboost`) | 11.4357 | 1.7348 | 151 | 2.074 |
| Random swap | 11.1546 | 3.1220 | 151 | 2.184 |
| RF (saved `models/rf`) | 10.3776 | 5.4633 | 151 | 3.052 |
| Random Top-K | 10.1946 | 3.4037 | 151 | 1.930 |
| MLP (saved `models/mlp`) | 9.8607 | 2.0710 | 151 | 2.032 |
| Random search | 9.6149 | 3.3454 | 151 | 2.266 |
| Initial permutation | 0.0000 | 0.0000 | 1 | 0.023 |

Paired score-delta differences versus `Random Top-K` (mean and 95% $t$-interval across five seeds):

| Method | Mean difference | 95% interval |
| --- | ---: | ---: |
| XGBoost | +1.2411 | [-1.4264, +3.9086] |
| Random swap | +0.9599 | [-1.7041, +3.6239] |
| RF | +0.1830 | [-5.2184, +5.5844] |
| MLP | -0.3339 | [-2.9098, +2.2421] |
| Random search | -0.5798 | [-1.5598, +0.4003] |

Although the saved validation-tuned XGBoost model achieved the highest mean score gain (`+11.4357`, `+1.2411` above Random Top-K), every 95% confidence interval crosses zero; five seeds with the 4-feature `permutation-v2.1` schema do not establish a statistically significant advantage over random selection.

### 3-way held-out feature ablation (diagnostic & leakage audit)

To explain why 4 permutation-distance features yield $R^2 < 0$ while transparently auditing cheap feature overlap with Composite Score components, `src/v2_experiment.py` evaluates three polynomial-free feature sets on the identical locked test split (`perm4` = 4 permutation features; `proxy` = 5 features adding single-bit differential proxy without SAC/FP; `cheap` = 8 features adding exact 8×8 `sac_mean`, `sac_max_err`, and `fixed_points`):

| Model | Feature set | n_features | MAE | RMSE | R² | Spearman ρ | Top-5 regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RF | `perm4` | 4 | 4.2052 | 4.9749 | -0.1977 | +0.1010 | 4.7171 |
| RF | `proxy` (no SAC/FP) | 5 | 4.5931 | 5.5337 | -0.4818 | -0.1180 | 4.7171 |
| RF | `cheap` (+8×8 SAC & FP) | 8 | 3.1196 | 3.8962 | **+0.2654** | **+0.6036** | **0.0000** |
| XGBoost | `perm4` | 4 | 4.1222 | 4.9678 | -0.1943 | +0.1448 | 2.5377 |
| XGBoost | `proxy` (no SAC/FP) | 5 | 4.5392 | 5.4873 | -0.4571 | -0.0387 | 4.7054 |
| XGBoost | `cheap` (+8×8 SAC & FP) | 8 | 3.4069 | 4.1618 | **+0.1618** | **+0.4812** | 2.5377 |
| MLP | `perm4` | 4 | 3.8210 | 4.6971 | -0.0677 | +0.2840 | 6.0604 |
| MLP | `proxy` (no SAC/FP) | 5 | 4.6072 | 5.6967 | -0.5704 | -0.0831 | 4.4690 |
| MLP | `cheap` (+8×8 SAC & FP) | 8 | 3.8855 | 4.8005 | -0.1152 | **+0.5126** | 6.0604 |

This controlled ablation proves two key findings: (1) permutation-distance statistics and a single-bit differential proxy alone carry almost no predictive signal for full S-Box quality; (2) inexpensive $O(8 \times 256)$ SAC matrix and fixed-point statistics immediately lift test ranking correlation to $\rho = +0.48\text{ to }+0.60$ (RF $R^2 = +0.2654$, Top-5 regret `0.0000`) while remaining transparent about their direct overlap with the SAC/FP terms of Composite Score.

Across five independent 20-candidate single-swap pools (using the saved `permutation-v2.1` models), mean Top-3 regrets were `1.9069` (RF), `0.5104` (XGBoost), and `1.5222` (MLP); mean pool Spearman correlations were `-0.0058`, `+0.1418`, and `-0.0556` respectively.

### Fixed-seed 100 cryptographic metrics (Metrik 1)

| S-Box | NL | BIC-NL | SAC Mean | SAC Max Err | BIC-SAC | DU | LAP | FP | Composite Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AES Standard (FIPS-197) | 112 | 112 | 0.50488 | 0.06250 | 0.50460 | 4 | 0.0625 | 0 | 124.14 |
| K44 baseline | 112 | 112 | 0.49731 | 0.06250 | 0.50237 | 4 | 0.0625 | 0 | 124.28 |
| S-Box RF v2 (seed 100) | 98 | 100 | 0.50244 | 0.07812 | 0.50286 | 10 | 0.1172 | 2 | 89.79 |
| S-Box XGBoost v2 (seed 100) | 96 | 100 | 0.49976 | 0.07812 | 0.50537 | 10 | 0.1250 | 2 | 87.80 |
| S-Box MLP v2 (seed 100) | 96 | 100 | 0.50024 | 0.09375 | 0.50146 | 10 | 0.1250 | 2 | 86.05 |

## Mandrill v2 benchmark

The fixed-seed-100 reconstructions were compared against standard AES and K44 (loaded from `data/baselines/`) on the same 512×512 Mandrill image (`data/mandril_color.jpg`) in RGB and grayscale. All runs used `CBC2`, identical key/IV, and a one-bit plaintext flip at pixel `(511, 511)` for NPCR/UACI/NBCR. The report (`results/v2/mandrill_v2_metrics.json`) records image SHA-256, cipher variant metadata, single-encryption timing, horizontal/vertical/diagonal pixel correlation, and byte-for-byte lossless decryption checks.

| Mode | S-Box | Cipher entropy | χ² (p-value) | NPCR (%) | UACI (%) | Mean |r| cipher | Lossless |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| RGB | AES Standard (FIPS-197) | 7.99976 | 257.13 (0.4508) | 99.6136 | 33.4650 | 0.000265 | Yes |
| RGB | K44 baseline | 7.99976 | 259.91 (0.4031) | 99.6072 | 33.4807 | 0.000344 | Yes |
| RGB | S-Box RF v2 (seed 100) | 7.99972 | 307.71 (0.0132) | 99.6232 | 33.4897 | 0.001047 | Yes |
| RGB | S-Box XGBoost v2 (seed 100) | 7.99978 | 238.72 (0.7602) | 99.6220 | 33.5024 | 0.000952 | Yes |
| RGB | S-Box MLP v2 (seed 100) | 7.99978 | 234.83 (0.8127) | 99.6189 | 33.4538 | 0.000877 | Yes |
| Gray | AES Standard (FIPS-197) | 7.99939 | 220.04 (0.9447) | 99.5907 | 33.4784 | 0.001264 | Yes |
| Gray | K44 baseline | 7.99934 | 239.35 (0.7511) | 99.5995 | 33.4546 | 0.001983 | Yes |
| Gray | S-Box RF v2 (seed 100) | 7.99937 | 228.70 (0.8805) | 99.6201 | 33.4715 | 0.001197 | Yes |
| Gray | S-Box XGBoost v2 (seed 100) | 7.99938 | 224.08 (0.9191) | 99.5983 | 33.4148 | 0.000789 | Yes |
| Gray | S-Box MLP v2 (seed 100) | 7.99926 | 270.32 (0.2437) | 99.6162 | 33.4955 | 0.002522 | Yes |

All **10/10 runs** decrypted byte-for-byte identically and collapsed adjacent pixel correlation from `0.8485` (RGB plain) / `0.7838` (Gray plain) to `< 0.003`. Note transparently that `RGB S-Box RF v2 (seed 100)` recorded $\chi^2 = 307.71$ ($p = 0.0132 < 0.05$, above the 293.25 critical value), while the remaining 9/10 runs passed $p > 0.05$. Custom S-Boxes are explicitly labeled as **AES-128 Modifikasi** (`SubBytes`, `InvSubBytes`, `KeyExpansion.SubWord`); standard AES `MixColumns`/`InvMixColumns` remain intact, and the AES-standard S-Box row also uses the project's modified `CBC2` mode. Entropy, histogram, NPCR/UACI/NBCR, and pixel correlation are empirical image metrics, not formal cryptographic security proofs.

## Dashboard and verification

The six Streamlit pages (`Overview`, `Dataset/Training`, `Reconstruction`, `Crypto Evaluation`, `Image Encryption`, `Comparative Analysis`) load persisted dataset, model, reconstruction, crypto-evaluation, Mandrill v2 (including pixel correlation), and paired-comparison artifacts without retraining on startup. Live reconstruction is opt-in, cached, and supports selecting any verified saved model (**XGBoost**, **Random Forest**, or **MLP**). Missing or schema-mismatched artifacts surface explicit error messages. Legacy v1 (`GF(2⁸)` + chaotic) results and the interactive Archify diagram (`assets/cryptoml_sbox_archify.html`) remain accessible and strictly separated from v2.

Verification on the final checkout: `181 passed` (`python -m pytest -q`), Streamlit `AppTest` + live Chrome browser QA across all six pages (including opt-in live reconstruction for XGBoost and RF), `impeccable detect` clean (`0` issues), and CLI round-trip verification.
