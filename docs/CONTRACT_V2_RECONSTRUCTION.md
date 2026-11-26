# Kontrak Antarmuka & Artefak v2 — Machine Learning-Guided S-Box Reconstruction

Dokumen ini melengkapi `docs/ARCHITECTURE_CONTRACT.md` (v1) sebagai **kontrak beku v2** agar modul dan artefak dari **Naila (#16)**, **Yafi (#17)**, dan **Mahdi (#18)** saling terhubung tanpa ketidakcocokan format atau kebocoran data.

> **Status Versi:** `schema_version: "2.0.0"`  
> **Prinsip Kompatibilitas:** Artefak v1 (`results/sbox_*.npy`, `results/*_metrics.json`, `results/mandrill_metrics.json`) tidak dihapus; seluruh artefak metode baru disimpan terpisah pada `models/`, `datasets/`, dan `results/v2/`.

---

## 1. Format Baku S-Box & Hard Constraint Bijectivity

- **Representasi Memori:** `numpy.ndarray`, `shape == (256,)`, `dtype == np.uint8`, merupakan permutasi eksak `{0, 1, ..., 255}`.
- **Representasi JSON (`initial_sbox.json`, `final_sbox.json`, `inverse_sbox.json`):**
  ```json
  {
    "schema_version": "2.0.0",
    "method": "xgboost",
    "seed": 42,
    "sbox": [0, 1, "... 256 int 0..255 ..."],
    "sha256": "<hex sha256 dari bytes(sbox)>"
  }
  ```
- **Representasi `.npy` (opsional untuk utilitas Mandrill):** 1D `np.uint8` `(256,)`.
- **Hard Constraint:** Setiap operator mutasi dan loader wajib menegakkan bijektivitas (`validate_sbox`), serta **tidak memutasi array parent secara in-place**.

---

## 2. Kontrak Evaluator & Composite Score (Issue #16 — Naila)

### 2.1 `src/sbox_metrics.py`
- Menyediakan konstanta `EVALUATOR_VERSION = "2.0.0"`.
- Fungsi utama: `evaluate_sbox(sbox: np.ndarray, *, strict_bijective: bool = True) -> dict`.
  - Mengembalikan kunci wajib: `bijective`, `balanced`, `nl`, `sac_mean`, `sac_max_err`, `bic_nl`, `bic_sac`, `du`, `dap`, `lap`, `fixed_points`, `whs_cost`.
  - **Konvensi LAP:** `lap = max_walsh / 512.0` menyatakan **bias maksimum** $|P - 0.5|$ (AES Standar & K44 = `0.0625`, ekuivalen probabilitas $0.5 \pm 0.0625$).

### 2.2 `src/composite_score.py`
- Dipisahkan dari `src/train_xgboost.py` sebagai sumber kebenaran tunggal:
  ```python
  COMPOSITE_SCORE_VERSION = "1.0.0"  # rumus identik dengan v1 agar kompatibel & dapat diaudit

  def compute_composite_score(metrics: dict) -> float:
      return float(
          metrics["nl"]
          + 0.25 * metrics["bic_nl"]
          - 2.0 * metrics["du"]
          - 120.0 * metrics["sac_max_err"]
          - 40.0 * abs(metrics["sac_mean"] - 0.5)
          - 30.0 * abs(metrics["bic_sac"] - 0.5)
          - 1.5 * metrics["fixed_points"]
          - 0.05 * metrics["whs_cost"]
      )
  ```
- `src/train_xgboost.py` dan `src/web_helpers.py` akan me-re-export fungsi ini setelah #16 di-merge.

---

## 3. Kontrak Dataset Permutasi & Family Split Manifest (Issue #17 — Yafi)

### 3.1 Dataset CSV (`datasets/sbox_perm_dataset.csv`)
CSV dibentuk oleh `python -m src.train_v2_models` dari satu dataset reproducible. Metadata non-fitur: `sample_id`, `sbox_hash`, `parent_id`, `family_id`, `seed`, `operator`, `mutation_intensity`, `iteration`, `split`, dan `sbox_json`. Fitur v2 aktif adalah `hamming_mean`, `hamming_std`, `displacement_mean`, dan `displacement_std`. Evaluasi penuh menjadi label/metadata, bukan input model: `nl`, `sac_mean`, `sac_max_err`, `bic_nl`, `bic_sac`, `du`, `dap`, `lap`, `fixed_points`, `whs_cost`, dan `composite_score`.

Eksperimen ablation tersimpan di `results/v2/comparative_summary.json`: fitur murah termasuk SAC/fixed-point dibanding proxy diferensial tanpa kedua fitur tersebut. Fitur model produksi mengikuti urutan `FEATURE_NAMES` di `src.dataset_builder`.

### 3.2 Split Manifest (`datasets/split_manifest.json`)
File ini **di-commit ke Git** agar RF (#17), MLP (#16), dan XGBoost (#18) memakai pembagian data **100% identik**:
```json
{
  "schema_version": "2.0.0",
  "split_seed": 42,
  "ratios_target": {"train": 0.70, "val": 0.15, "test": 0.15},
  "ratios_actual": {"train": 0.70, "val": 0.1467, "test": 0.1533},
  "dataset_path": "datasets/sbox_perm_dataset.csv",
  "dataset_sha256": "<sha256>",
  "split_sha256": "<sha256>",
  "evaluator_version": "2.0.0",
  "composite_score_version": "2.0.0",
  "feature_version": "permutation-v2.1",
  "n_samples_total": 1500,
  "n_families_total": 150,
  "disjoint_family_verified": true,
  "disjoint_hash_verified": true,
  "train_ids": ["..."],
  "val_ids": ["..."],
  "test_ids": ["..."],
  "split_sha256": "<sha256 dari payload train/val/test ids>"
}
```

---

## 4. Kontrak Artefak Model Terlatih (`models/`)

Setiap model disimpan sebagai bundle mandiri yang dapat dimuat tanpa retraining pada Phase B:
- `models/rf/rf.joblib` (Yafi #17)
- `models/mlp/mlp.joblib` (Naila #16 — wajib berupa satu `Pipeline` dengan `StandardScaler` yang di-fit **hanya pada train**)
- `models/xgboost/xgboost.joblib` (Mahdi #18)
- Setiap bundle berisi `manifest.json`; schema fitur aktif `permutation-v2.1` berisi fitur permutasi:
  ```json
  {
    "schema_version": "2.0.0",
    "feature_schema_version": "permutation-v2.1",
    "feature_names": ["hamming_mean", "hamming_std", "displacement_mean", "displacement_std"],
    "n_features": 4
  }
  ```
- Manifest model mencatat:
  ```json
  {
    "schema_version": "2.0.0",
    "model_kind": "xgboost",
    "feature_schema_version": "permutation-v2.1",
    "feature_names": ["hamming_mean", "hamming_std", "displacement_mean", "displacement_std"],
    "n_features": 4,
    "dataset_sha256": "<dataset fingerprint>",
    "split_sha256": "<family split fingerprint>",
    "model_parameters": {},
    "metrics": {},
    "python_version": "...",
    "sklearn_version": "...",
    "xgboost_version": "..."
  }
  ```
- **Aturan Loader Ketat:** Saat engine rekonstruksi atau Streamlit memuat model, loader memverifikasi `schema_version`, `feature_schema_version`, `feature_names`, `n_features`, dan jumlah fitur model (`n_features_in_`). Jika tidak cocok atau file tidak ada, tampilkan/lempar error eksplisit — **dilarang diam-diam melanjutkan atau membuat prediksi palsu**.

---

## 5. Kontrak Output Rekonstruksi & Eksperimen Komparatif (`results/v2/`)

Struktur direktori keluaran:
```text
results/v2/
├── rf/
│   └── seed_42/
│       ├── initial_sbox.json
│       ├── final_sbox.json
│       ├── inverse_sbox.json
│       ├── reconstruction_history.csv
│       ├── cryptographic_metrics.json
│       └── run_metadata.json
├── xgboost/seed_<seed>/...
├── mlp/seed_<seed>/...
├── baselines/
│   ├── random_search/seed_<seed>/...
│   ├── random_swap/seed_<seed>/...
│   └── random_topk/seed_<seed>/...
├── comparative_summary.json
└── mandrill_v2_metrics.json
```

### Kolom Wajib `reconstruction_history.csv`
`iteration, operator_selected, n_candidates_generated, n_candidates_unique, predicted_best_score, actual_evaluated_best_score, current_score, global_best_score, accepted, full_evals_used, full_evals_cumulative, cache_hits_cumulative, cache_misses_cumulative, feature_time_sec, inference_time_sec, eval_time_sec, iter_total_time_sec`

### Aturan Perbandingan Adil (Multi-Seed Paired Protocol)
- Untuk mengisolasi efek ranking model ML (`rf`, `xgboost`, `mlp`) terhadap `random_topk` / `random_swap` / `random_search`:
  - Gunakan daftar `seed` identik (mis. `[42, 101, 202, 303, 404]`) sehingga **permutasi awal identik per seed**.
  - Samakan himpunan operator mutasi, aturan acceptance (selalu berdasarkan **skor aktual** hasil `evaluate_sbox`), stopping rule (`patience`), dan **budget maksimum evaluasi penuh (`max_full_evals`)**.
  - Satu run hanya boleh menggunakan **satu model** (tidak mencampur RF+XGB+MLP dalam satu run).
  - Laporkan waktu total, waktu ekstraksi fitur, waktu inferensi, dan waktu evaluasi penuh secara terpisah.

---

## 6. Kontrak Tambahan Evaluasi Citra Mandrill (Issue #18 — Mahdi)

- `src.image_metrics.evaluate_image_security(plain_arr, cipher1_arr, cipher2_arr)` tetap mempertahankan tepat **6 kunci kontrak v1** (`entropy_plain`, `entropy_cipher`, `hist_chi2`, `hist_p_value`, `npcr`, `uaci`) agar tidak memutus kompatibilitas.
- Fungsi tambahan pada `src/image_metrics.py`:
  - `pixel_correlation(img_arr: np.ndarray) -> dict[str, float]` mengembalikan `{"horizontal": float, "vertical": float, "diagonal": float, "mean_abs": float}` (rata-rata korelasi Pearson seluruh pasangan piksel bertetangga per kanal; nilai ideal cipher ≈ `0.0`).
- Field tambahan pada setiap entri `runs[]` di `src/mandrill_eval.py`:
  - `encrypt_single_sec`: waktu enkripsi 1 citra (`cipher1`) saja (selain `encrypt_sec` lama).
  - `pixel_correlation`: `{"plain": {...}, "cipher": {...}}`.
  - `cipher_variant`: `{"is_standard_sbox": bool, "cipher_mode": str, "sbox_applied_to": [...], "uses_standard_mixcolumns_gf256": True, "cipher_label": str}`.
