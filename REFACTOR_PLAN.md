# REFACTOR_PLAN.md — CryptoML-SBox v2: Polynomial-Free Machine Learning-Guided S-Box Reconstruction

**Status:** Seluruh Fase 0–6 Selesai & Terverifikasi (Lihat hasil aktual di `REFACTOR_REPORT.md`)  
**Baseline `main`:** Commit `7d1e3a4` (tag `v1-legacy-gf256`) | **Live Deployment:** `https://cryptoml-sbox.streamlit.app/`  
**Koordinasi GitHub Issues:**
- **Issue #16 (`@NAILAAR22`):** Validasi metrik S-Box, pemisahan Composite Score, dan retraining MLP.
- **Issue #17 (`@twizzcode`):** Dataset permutasi polynomial-free, split keluarga 70/15/15, Random Forest, dan engine rekonstruksi bersama + baseline non-ML.
- **Issue #18 (`@MahdiHbb1`):** Audit Tahap 0 (`REFACTOR_PLAN.md`), kontrak v2, XGBoost, eksperimen komparatif multi-seed, audit/upgrade Mandrill, dan integrasi Streamlit 6 halaman.

> **Prinsip Ilmiah Utama:**  
> Model Machine Learning (`Random Forest`, `XGBoost`, `MLP`) **tidak** mengklaim membangkitkan 256 nilai S-Box secara langsung, melainkan berperan sebagai **surrogate scorer / guiding model** untuk mengarahkan rekonstruksi permutasi bijektif secara iteratif tanpa polinomial irreducible, tanpa invers multiplikatif $\mathrm{GF}(2^8)$, tanpa transformasi affine, dan tanpa *chaotic map* pada jalur generator S-Box usulan.  
> Composite Score dan metrik citra (Entropy, Chi-Square, Korelasi Piksel, NPCR/NBCR, UACI) **bukan bukti keamanan kriptografi formal**; keduanya adalah indikator empiris kualitas kriptografi dan difusi citra.

---

## 1. Kondisi Project Saat Ini (Audit Baseline `main` `7d1e3a4`)

### 1.1 Verifikasi Baseline Tes & Lingkungan
Hasil eksekusi bersih pada `main` (`7d1e3a4`), Windows (Python `3.12.3`, `pytest 8.4.2`):
- **Command:** `python -m pytest -v`
- **Hasil:** **`102 passed, 247 warnings in 42.36s`** (0 failed, 0 skipped).
- **Dependensi aktual terpasang:** `numpy 1.26.1`, `pandas 2.1.2`, `scipy 1.16.2`, `scikit-learn 1.3.2`, `xgboost 3.1.1`, `Pillow 12.3.0`, `matplotlib 3.10.7`, `streamlit 1.50.0`, `pytest 8.4.2`, `joblib 1.3.2` (sebelumnya transitif), `PyYAML 6.0.3` (sebelumnya transitif).
- **Deployment aktif:** `https://cryptoml-sbox.streamlit.app/` melacak branch `main` secara otomatis. Oleh karena itu, seluruh pengembangan v2 wajib dilakukan di branch fitur terpisah agar deployment live tidak pernah rusak atau menampilkan halaman kosong sebelum artefak v2 tersedia.

### 1.2 Struktur Repositori Saat Ini
| Direktori / Berkas | Isi Aktual (`7d1e3a4`) | Peran v1 |
| :--- | :--- | :--- |
| `src/` (11 modul) | `aes_affine_generator.py`, `aes_cipher_runner.py`, `dataset_builder.py`, `sbox_metrics.py`, `train_xgboost.py`, `train_rf.py`, `train_mlp.py`, `image_metrics.py`, `mandrill_eval.py`, `visualize_mandrill.py`, `web_helpers.py` | Pipeline v1 berbasis screening kandidat `aes_affine` + `chaotic_swap` |
| `tests/` (10 file) | 102 unit test mencakup seluruh modul `src/` | Seluruh tes v1 lulus |
| `results/` | `sbox_{xgb,rf,mlp}.npy`, `{xgb,rf,mlp}_metrics.json`, `mandrill_metrics.json`, `experiment_log.md`, `figures/*.png` (3 gambar) | Artefak v1 (**Legacy** — terpilih dari pool `aes_affine`) |
| `data/` | `mandril_color.jpg` (512×512 RGB); `data/sbox_dataset.csv` di-ignore Git | Citra uji Mandrill |
| `notebooks/` | `Mahdi_XGBoost_SBox_Colab_Kaggle.ipynb` | Notebook re-run v1 (masih memakai `aes_affine` + `chaotic_swap`) |
| `docs/` | `ARCHITECTURE_CONTRACT.md`, `PEMBAGIAN_JOBDESK.md`, `mandrill_analysis.md`, `LAPORAN_PROYEK_AKHIR_CRYPTOML_SBOX.docx` | Kontrak v1 & dokumentasi Mandrill |
| `assets/` | `cryptoml_sbox_archify.html` | Diagram arsitektur interaktif v1 |
| `app.py` | Dashboard Streamlit 3 halaman (1.607 baris) | Ringkasan, Archify, Simulasi Langsung |
| `models/` | **Belum ada** | Belum ada model ML terlatih yang disimpan |

---

## 2. Audit Wajib 16 Komponen & Permasalahan Teridentifikasi

### 2.1 Peta Penggunaan Polinomial Irreducible, $\mathrm{GF}(2^8)$, Affine, dan Chaotic Map

Pemisahan tegas antara **jalur yang DILARANG di generator usulan** vs **jalur yang DIPERTAHANKAN untuk cipher AES & baseline**:

| Lokasi File & Baris | Konstruksi Matematis | Status di v2 | Alasan |
| :--- | :--- | :--- | :--- |
| `src/aes_affine_generator.py` L36–39, L59–73, L118–140, L155–216 | 30 polinomial irreducible `IRREDUCIBLE_POLYS_GF256`, `gf256_inverse_table`, `random_invertible_matrix`, `build_aes_affine_sbox`, `generate_aes_affine_candidates` | **DIISOLASI (Baseline & Legacy Only)** | Tidak boleh dipanggil oleh generator, dataset, training, atau rekonstruksi S-Box usulan. `build_standard_aes_sbox()` & `build_k44_sbox()` dipertahankan khusus sebagai baseline pembanding. |
| `src/dataset_builder.py` L12, L26–84, L107–114 | `_logistic_permutation` (chaotic logistic map), `generate_chaotic_swap_candidates`, `generate_aes_affine_candidates` | **DIGANTI di jalur v2** | Jalur v2 memakai permutasi acak + mutasi bijektif + lintasan optimasi non-ML (Issue #17). |
| `src/train_xgboost.py` L27, L170–243 | `_chaotic_logistic_sbox`, `generate_candidate_pool`, `build_fallback_training_data` (`aes_affine` + `chaotic`) | **DIGANTI di jalur v2** | Fallback training dan candidate pool v1 memakai `aes_affine` dan chaotic map; v2 wajib menggunakan dataset permutasi & engine rekonstruksi bersama. |
| `src/train_rf.py` L15–20, L59–68 | `generate_candidate_pool` (`chaotic_swap` + `aes_affine`) | **DIGANTI di jalur v2** | Sama — dipisahkan ke training & reconstruction engine v2 (Issue #17). |
| `src/train_mlp.py` L19, L99 | Mengimpor `generate_candidate_pool` dari `train_rf` | **DIGANTI di jalur v2** | Sama — dipisahkan ke pipeline MLP v2 (Issue #16). |
| `src/web_helpers.py` L22–26, L238–246 & `app.py` L23, L1003–1023 | `generate_live_affine_sbox`, `IRREDUCIBLE_POLYS_GF256` di Tab Simulasi | **DIPISAHKAN / DIGANTI** | Tab Simulasi utama v2 harus menjalankan live ML-guided reconstruction (polynomial-free); generator affine legacy dihapus dari jalur usulan atau diberi label eksplisit "Legacy/Baseline". |
| `src/aes_cipher_runner.py` L17–24, L70–89, L126–154 | `AES_IRREDUCIBLE_POLY` (`0x11B`) dan `gf256_mul` untuk tabel `_MUL_NP[2,3,9,11,13,14]` (`MixColumns` / `InvMixColumns`) | **DIPERTAHANKAN (WAJIB)** | Ini adalah aritmetika **cipher AES-128** untuk enkripsi/dekripsi citra Mandrill, **bukan** generator S-Box usulan. Direkomendasikan menyalin/memindahkan helper `gf256_mul` mandiri agar `aes_cipher_runner.py` tidak bergantung pada modul generator affine. |

### 2.2 Audit `src/aes_cipher_runner.py` (Pemakaian Custom S-Box)
- **SubBytes:** memakai custom `sbox` (`_encrypt_blocks` L95–96, `_encrypt_single_block_fast` L129–161).
- **InvSubBytes:** memakai `build_inverse_sbox(sb)` (`_decrypt_blocks` L102–103).
- **KeyExpansion (`SubWord`):** memakai custom `sbox` pada `expand_key` L64 (`t = [int(sb[x]) for x in t]`).
- **Mode Operasi:** `ECB`, `CBC`, dan `CBC2` (CBC 2 lintasan maju-mundur dengan `iv_rev = iv ^ 0xA5`).
- **Dekripsi Lossless:** Terverifikasi byte-for-byte identik (`np.array_equal`) dengan zero-padding yang dipotong kembali ke `orig_shape` (`L258–279`).
- **Temuan Wajib Label:** Karena custom S-Box mengubah `SubBytes`, `InvSubBytes`, dan `KeyExpansion`, serta `CBC2` adalah modifikasi 2 lintasan, setiap pengujian dengan S-Box non-FIPS-197 wajib diberi label eksplisit **"AES-128 Modifikasi (Custom S-Box)"** — tidak boleh diklaim sebagai AES-128 standar FIPS-197.

### 2.3 Audit Evaluator Metrik 1 & Composite Score (`src/sbox_metrics.py` & `src/train_xgboost.py`)
- **Duplikasi Evaluator:** `src/sbox_metrics.py::evaluate_sbox` dan `src/train_xgboost.py::evaluate_sbox_default` menghasilkan angka identik, tetapi masih terduplikasi di dua tempat.
- **Coupling Composite Score:** `compute_composite_score` saat ini berada di `src/train_xgboost.py` L156–167 dan diimpor oleh `dataset_builder.py` serta `web_helpers.py`. Pada v2 (Issue #16), harus diekstrak ke modul mandiri `src/composite_score.py` dengan rumus tetap:
  $$\text{CompositeScore} = \mathrm{NL} + 0.25\,\mathrm{BIC\_NL} - 2\,\mathrm{DU} - 120\,\mathrm{SAC\_max\_err} - 40\,|\mathrm{SAC\_mean}-0.5| - 30\,|\mathrm{BIC\_SAC}-0.5| - 1.5\,\mathrm{FP} - 0.05\,\mathrm{WHS\_cost}$$
- **Konvensi LAP:** `lap = max_w / 512.0` (`sbox_metrics.py` L40) adalah **bias aproksimasi linier maksimum** $|P - 0.5|$ (untuk AES/K44: $\max|W|=32 \Rightarrow \mathrm{LAP} = 32/512 = 0.0625$, ekuivalen probabilitas $0.5 \pm 0.0625$). Wajib didokumentasikan eksplisit di Issue #16 dan laporan.
- **Bijectivity sebagai Hard Constraint:** `sbox_metrics.py` mengembalikan `"bijective": bool` tanpa melempar error jika input bukan permutasi; pada jalur rekonstruksi v2, bijectivity wajib menjadi *hard constraint* yang ditegakkan oleh operator mutasi dan validator.

### 2.4 Audit Dataset, Fitur Input, Target, dan Potensi Data Leakage
1. **Overlap Fitur v1 (`f_0..f_39`) dengan Komponen Target:**
   - `f_0..f_15`: 16 nilai NL parsial (16 mask tetap dari 255).
   - `f_16..f_31`: 16 nilai WHS parsial (L2 terhadap 16).
   - `f_32..f_34`: `min`, `mean`, `std` dari 16 NL parsial.
   - **`f_35` = `sac_mean` eksak**, **`f_36` = `sac_max_err` eksak**, **`f_38` = `fixed_points` eksak**, `f_37` = `std(sac_mat)`, `f_39` = `opposite_fixed_points`.
   - **Audit Leakage:** `f_35`, `f_36`, dan `f_38` adalah suku eksak dari `Composite Score` (`-120*sac_max_err - 40*|sac_mean-0.5| - 1.5*fp`). Meskipun biaya hitungnya sangat murah (~0.1 ms, sehingga tidak tergolong *future label leakage* seperti memasukkan `nl`/`du`/`bic_nl`/`whs_cost` penuh), **eksperimen ablation wajib membandingkan fitur murah lengkap vs fitur proxy tanpa suku SAC/FP eksak** (Issue #16, #17, #18) agar kontribusi prediksi benar-benar transparan.
   - **Perbedaan Signifikan di v1 vs v2:** Di v1, `R² ≈ 0.93` sebagian besar muncul karena dataset mencampur dua klaster ekstrem (`chaotic_swap` skor ~60–90 vs `aes_affine` skor ~120–126) sehingga model sangat mudah memisahkan klaster besar. Pada v2 (murni ruang permutasi tanpa `aes_affine`), rentang skor lebih sempit sehingga **Spearman rank correlation** dan **Top-K retrieval per pool** menjadi metrik evaluasi utama di samping MAE/RMSE/$R^2$.
2. **Risiko Kebocoran Split Keluarga (*Family Leakage*):**
   - Di v1, ketiga trainer (`train_xgboost.py`, `train_rf.py`, `train_mlp.py`) menggunakan `train_test_split(..., test_size=0.2, random_state=42)` secara acak tanpa set validasi terpisah (XGBoost bahkan melakukan 3-fold CV pada train tanpa holdout validasi resmi).
   - Di v2, karena dataset berisi permutasi awal beserta turunan mutasinya (*parent/child* dan lintasan optimasi), split acak per baris akan membocorkan permutasi yang hampir identik antara train dan test.
   - **Mitigasi Wajib:** Deduplikasi eksak berdasarkan byte permutasi (`sbox.tobytes()`) **sebelum** split, lalu gunakan `GroupShuffleSplit` berbasis `family_id` dengan rasio target **70% train / 15% validation / 15% test** (`seed=42`), dan simpan dalam `datasets/split_manifest.json` bersama SHA-256 fingerprint.
3. **Konflik `.gitignore` untuk Persistensi Model:**
   - **Temuan Kritis:** `.gitignore` L18–20 saat ini mengabaikan `*.joblib`, `*.pkl`, dan `*.h5` secara global! Jika tidak ditambahkan pengecualian `!models/*.joblib`, model terlatih untuk Phase B (Reconstruction) tidak dapat di-commit ke repositori untuk deployment Streamlit Cloud.

### 2.5 Audit Evaluasi Citra Mandrill (`src/image_metrics.py`, `src/mandrill_eval.py`, `src/visualize_mandrill.py`)
- **Sudah Ada & Teruji (47 unit test):** `shannon_entropy`, `histogram_chi_square`, `npcr_uaci` (rata-rata per kanal), `bit_change_rate` (NBCR), dekripsi lossless, visualisasi 3 figur publikasi (`mandrill_entropy_chi2.png`, `mandrill_diffusion.png`, `mandrill_encryption_xgb.png`).
- **Kesenjangan yang Ditemukan (Akan Diperbaiki Langsung di Tahap 0/Mandrill):**
  1. **Belum ada Korelasi Piksel Bertetangga (Horizontal, Vertikal, Diagonal)** pada `src/image_metrics.py` dan `src/mandrill_eval.py` (diminta eksplisit di Issue #18).
  2. **`encrypt_sec` pada `src/mandrill_eval.py` L156–159** menjumlahkan waktu enkripsi `cipher1` + `cipher2` sekaligus; perlu menambahkan `encrypt_single_sec` agar waktu enkripsi satu citra tercatat akurat tanpa merusak kunci lama `encrypt_sec`.
  3. **Belum ada field `cipher_variant`** yang secara eksplisit melabeli pemakaian S-Box pada `SubBytes`, `InvSubBytes`, `KeyExpansion` dan status `AES-128 Modifikasi`.

### 2.6 Audit Dashboard Streamlit (`app.py`, `src/web_helpers.py`)
- Saat ini `app.py` memiliki 3 halaman (`Ringkasan Proyek`, `Diagram Alur & Arsitektur`, `Simulasi Langsung`) yang terikat pada artefak v1 (`results/sbox_*.npy`, `results/*_metrics.json`) dan memakai generator affine `GF(2^8)` di tab simulasi.
- Target v2 (Issue #18): mempertahankan identitas visual yang sudah rapi/profesional (tipografi, palet teal `#0F5C5C` / slate `#465664` / amber `#8A4307`), namun merestrukturisasi menjadi **6 halaman**:
  1. `Overview`
  2. `Dataset & Training`
  3. `ML-Guided Reconstruction` (initial vs final, mutasi, prediksi vs aktual, kurva konvergensi, live reconstruction ringan ter-cache dari model tersimpan)
  4. `Cryptographic Evaluation`
  5. `Image Encryption (Mandrill)`
  6. `Comparative Analysis` (perbandingan berpasangan multi-seed + pemisahan tegas Legacy v1 vs v2)
- **Aturan Anti-Data-Palsu:** Jika artefak v2 belum tersedia atau `schema_version`/fingerprint tidak cocok, UI wajib menampilkan status eksplisit ("Artefak belum tersedia / Schema tidak cocok"), **tidak pernah** mengarang angka simulasi atau melatih ulang model saat app dimuat.

### 2.7 Profiling Komputasi & Bottleneck (Hasil Pengukuran Aktual)

Diukur langsung pada laptop pengembangan lokal (Windows, Python `3.12.3`, CPU tunggal):

| Operasi | Hasil Terukur | Estimasi Beban Kerja v2 | Strategi Eksekusi |
| :--- | :---: | :--- | :--- |
| `src.sbox_metrics.evaluate_sbox` (`n=50`) | **10.680 ms/S-Box** (~93.6/s) | 1.500 sampel ≈ **16s**; 5.000 sampel ≈ **53s**; 10.000 sampel ≈ **107s** | Aman dijalankan lokal hingga 5.000 sampel; 10.000+ sampel sangat cepat di Kaggle/Colab |
| `extract_fast_features` (`n=50`) | **1.146 ms/S-Box** (~873/s) | 9.3× lebih cepat dari evaluasi penuh; 50 kandidat/iterasi ≈ **57 ms/iterasi** | Cukup ringan untuk live reconstruction mode `fast` (50 iterasi × 20 kandidat ≈ ~2–3s di Streamlit!) |
| 1 Run Rekonstruksi (`fast`: 50×20, top-3) | ~150 eval penuh + 1.000 fitur | ≈ **2.7 detik/run** (lokal) | Ideal untuk live demo Streamlit & smoke test |
| 1 Run Rekonstruksi (`standard`: 200×50, top-5) | ~1.000 eval penuh + 10.000 fitur | ≈ **22 detik/run** (lokal) | Cocok untuk eksperimen pengembangan multi-seed |
| 1 Run Rekonstruksi (`research`: 500×100, top-5) | ~2.500 eval penuh + 50.000 fitur | ≈ **85 detik/run** → 6 metode × 5 seed = 30 run ≈ **42 menit** lokal | Disarankan dijalankan di Kaggle/Colab (atau paralel per seed) dan hasilnya disimpan sebagai precomputed artifacts |
| AES-128 `CBC2` `128×128×3` | **0.400s** enc / **0.051s** dec | Cepat untuk uji unit & live preview ringan | Default preview cepat web |
| AES-128 `CBC2` `256×256×3` | **1.786s** enc / **0.188s** dec | Cukup cepat untuk simulasi interaktif | Default simulasi interaktif Streamlit |
| AES-128 `CBC2` `512×512×3` (Mandrill penuh) | **6.282s** enc / **0.494s** dec | 1 S-Box (2× enc + 1× dec) ≈ **13s**; 5–8 S-Box RGB+GRAY ≈ **2–3 menit** | Precomputed ke JSON + tetap tersedia untuk CLI `src.mandrill_eval` |

---

## 3. Rencana Disposisi File (Keep / Modify / Archive / New)

> Tidak ada file legacy yang dihapus sebelum seluruh dependensi dan hasil v1 dipisahkan secara aman.

### 3.1 File Dipertahankan (*Retained*)
| File | Keterangan |
| :--- | :--- |
| `src/aes_cipher_runner.py` | Dipertahankan untuk enkripsi/dekripsi AES-128 (`ECB`/`CBC`/`CBC2`) dan `validate_sbox`/`build_inverse_sbox`; hanya ditambah helper metadata `describe_cipher_variant` + (opsional) inlining `gf256_mul` agar tidak mengimpor modul generator affine. |
| `src/image_metrics.py` | Dipertahankan utuh (6 kunci kontrak `evaluate_image_security` tetap tidak berubah); ditambah fungsi `pixel_correlation` (H/V/D). |
| `src/visualize_mandrill.py` | Dipertahankan; nanti diperluas agar juga dapat memplot perbandingan Mandrill v2 (baseline + RF/XGBoost/MLP rekonstruksi). |
| `data/mandril_color.jpg` | Citra uji standar dipertahankan tanpa modifikasi. |
| `.streamlit/config.toml` | Identitas visual tema dipertahankan. |
| `results/sbox_{xgb,rf,mlp}.npy`, `results/{xgb,rf,mlp}_metrics.json`, `results/mandrill_metrics.json`, `results/figures/*.png` | Dipertahankan sebagai artefak **Legacy v1** (diberi label jelas di dokumentasi & UI agar tidak tercampur dengan `results/v2/`). |

### 3.2 File Dimodifikasi (*Modified*)
| File | Pemilik Issue | Perubahan Terencana |
| :--- | :---: | :--- |
| `src/sbox_metrics.py` | #16 (Naila) | Tambah `EVALUATOR_VERSION`, validasi input ketat, dokumentasi konvensi LAP (bias vs probabilitas), verifikasi referensi AES/K44. |
| `src/mandrill_eval.py` | #18 (Mahdi) | Tambah `pixel_correlation` (plain & cipher H/V/D), `encrypt_single_sec`, `cipher_variant`, dan dukungan evaluasi katalog S-Box v2. |
| `src/train_xgboost.py` | #18 (Mahdi) | Dipertahankan kompatibilitas legacy sementara; training v2 membaca `datasets/split_manifest.json` + menyimpan model ke `models/xgb_model.joblib` + `model_metadata.json` + evaluasi Spearman/Top-K/ablation. |
| `src/train_rf.py` | #17 (Yafi) | Diperbarui/didampingi pipeline v2 tanpa generator legacy, menyimpan `models/rf_model.joblib`. |
| `src/train_mlp.py` | #16 (Naila) | Diperbarui/didampingi pipeline v2 (`StandardScaler` + `MLPRegressor` dalam satu pipeline, fit train-only), menyimpan `models/mlp_model.joblib`. |
| `src/web_helpers.py` & `app.py` | #18 (Mahdi) | Direstrukturisasi ke 6 halaman v2 dengan pemisahan tegas Legacy v1 vs v2, status eksplisit jika artefak belum ada, dan live reconstruction ringan ter-cache. |
| `.gitignore` & `requirements.txt` | #18 (Mahdi) | Izinkan `!models/*.joblib` & `!datasets/split_manifest.json` di `.gitignore`; pin `joblib>=1.3.0` & `pyyaml>=6.0` di `requirements.txt`. |
| `README.md`, `notebooks/Mahdi_XGBoost_SBox_Colab_Kaggle.ipynb`, `assets/cryptoml_sbox_archify.html` | #18 (Mahdi) | Diperbarui setelah fase eksperimen v2 selesai dengan alur dan angka aktual. |

### 3.3 File Diisolasi / Diarsipkan (*Legacy-Isolated*)
| File | Status | Catatan |
| :--- | :--- | :--- |
| `src/aes_affine_generator.py` | **Legacy / Baseline-Only** | Hanya `build_standard_aes_sbox()` & `build_k44_sbox()` yang boleh dipakai sebagai pembanding. Tes anti-leakage wajib memverifikasi pipeline v2 tidak mengimpor atau memanggil `generate_aes_affine_candidates`. |
| `src/dataset_builder.py` (v1 chaotic/affine) | **Legacy** | Dipertahankan untuk reproduksi v1 sementara; generator dataset v2 tidak memakai logistic map maupun `aes_affine`. |

### 3.4 File Baru (*New*)
| File / Direktori | Pemilik Issue | Fungsi |
| :--- | :---: | :--- |
| `REFACTOR_PLAN.md` | #18 (Tahap 0) | Dokumen audit & rencana refaktor menyeluruh (dokumen ini). |
| `docs/CONTRACT_V2_RECONSTRUCTION.md` | #18 (Tahap 0) | Kontrak beku schema dataset, split manifest, model artifact, dan output rekonstruksi v2. |
| `config/training.yaml` & `config/reconstruction.yaml` | #18 (Tahap 0) | Konfigurasi terpusat mode `fast`/`standard`/`research`, seed, dan hiperparameter. |
| `src/composite_score.py` | #16 (Naila) | Rumus Composite Score mandiri (dipisah dari `train_xgboost.py`). |
| `src/permutation_generator.py` & `src/feature_extractor.py` *(atau modul v2 di `src/`)* | #17 (Yafi) | Inisialisasi permutasi acak + 4 operator mutasi bijektif + ekstraksi fitur berversi + dataset builder polynomial-free. |
| `src/reconstruction_engine.py` | #17 (Yafi) | Engine rekonstruksi iteratif bersama (1 model/run, dedup, Top-K eval penuh, cache hit/miss, baseline random search / random swap / random Top-K). |
| `src/model_registry.py` *(atau loader bersama)* | #17 / #18 | Validasi & pemuatan aman `models/{rf,xgb,mlp}_model.joblib` + cek `feature_schema.json` & fingerprint. |
| `src/compare_experiments.py` | #18 (Mahdi) | Runner eksperimen komparatif multi-seed berpasangan + uji signifikansi/ringkasan mean±SD + Mandrill v2. |
| `models/` | #16, #17, #18 | `rf_model.joblib`, `xgb_model.joblib`, `mlp_model.joblib`, `feature_schema.json`, `model_metadata.json`. |
| `datasets/split_manifest.json` | #17 (Yafi) | Manifest pembagian keluarga 70/15/15 (`seed=42`) & SHA-256 fingerprint. |
| `results/v2/` | #16, #17, #18 | Artefak terstruktur per metode & seed (`final_sbox.json`, `inverse_sbox.json`, `reconstruction_history.csv`, `cryptographic_metrics.json`, `comparative_summary.json`, `mandrill_v2_metrics.json`). |
| `REFACTOR_REPORT.md` | #18 (Tahap akhir) | Laporan akhir hasil aktual v2, keterbatasan, dan perintah reproduksi. |

---

## 4. Peta Dependensi Antar-Modul & Urutan Kerja Tim

```text
[Tahap 0 — Mahdi #18 (UNBLOCKED, SELESAI DI BRANCH INI)]
  REFACTOR_PLAN.md + docs/CONTRACT_V2_RECONSTRUCTION.md + config/*.yaml
  + Mandrill pixel_correlation & cipher_variant audit
        │
        ├──► [Tahap 1 — Naila #16 (UNBLOCKED)]
        │      src/sbox_metrics.py (EVALUATOR_VERSION + LAP doc)
        │      src/composite_score.py (terpisah dari trainer)
        │      Validasi AES/K44 + tes kesetaraan skor lama/baru
        │            │
        │            ▼
        ├──► [Tahap 2 — Yafi #17 (mulai paralel untuk mutasi/fitur; dataset final setelah Tahap 1)]
        │      Permutasi acak + 4 mutasi bijektif (polynomial-free)
        │      Ekstraktor fitur berversi + dataset permutasi (1.500 -> 5.000/10.000)
        │      datasets/split_manifest.json (family split 70/15/15, seed=42)
        │      Engine rekonstruksi bersama + baseline non-ML (Random Search / Random Swap / Random Top-K)
        │            │
        │            ├─────────────────────────┬─────────────────────────┐
        │            ▼                         ▼                         ▼
        │    [Tahap 3a — Yafi #17]    [Tahap 3b — Naila #16]    [Tahap 3c — Mahdi #18]
        │     Retraining RF v2         Retraining MLP v2         Retraining XGBoost v2
        │     models/rf_model.joblib   models/mlp_model.joblib   models/xgb_model.joblib
        │            │                         │                         │
        │            └─────────────────────────┴─────────────────────────┘
        │                                      │
        │                                      ▼
        └──► [Tahap 4–6 — Mahdi #18 (Integrasi & Rilis)]
               Eksperimen berpasangan multi-seed (6 metode × K seed, budget full-eval setara)
               Evaluasi Mandrill v2 (AES, K44, RF, XGBoost, MLP + korelasi piksel + lossless)
               Dashboard Streamlit 6 halaman (precomputed + live cached reconstruction)
               Verifikasi penuh pytest + smoke Streamlit + README.md + REFACTOR_REPORT.md
```

---

## 5. Daftar Risiko Teknis & Ilmiah + Mitigasinya

| Risiko | Dampak | Mitigasi |
| :--- | :--- | :--- |
| **Rentang skor sempit di ruang permutasi murni** (perubahan 1 swap sering menggeser Composite Score sangat kecil) | $R^2$ regresi jauh lebih rendah daripada v1 (`0.93` di v1 berasal dari kontras `chaotic` vs `aes_affine`), sehingga model sulit memeringkat kandidat tetangga. | Sertakan lintasan optimasi non-ML & variasi intensitas mutasi di dataset latih (#17); evaluasi model dengan **Spearman $\rho$** dan **Top-K retrieval per pool**, bukan $R^2$ saja. |
| **Overlap fitur `f_35/f_36/f_38` dengan target** | Dosen/penguji mempertanyakan apakah model hanya "menghafal" SAC & FP. | Wajib laporkan **ablation study** (fitur murah lengkap vs fitur proxy parsial tanpa SAC/FP eksak) pada RF, XGBoost, dan MLP. |
| **Ekspektasi NL tidak menyamai AES (112)** | Tanpa struktur aljabar $\mathrm{GF}(2^8)$, permutasi hasil optimasi heuristik umumnya mencapai NL ~102–108 (di bawah AES 112). | Terapkan kejujuran ilmiah eksplisit: **tidak wajib mengalahkan AES standar**; klaim utama adalah bukti empiris apakah panduan ML mengungguli baseline non-ML (*initial permutation*, *random search*, *random Top-K*) pada budget evaluasi setara. |
| **Kebocoran keluarga antar split** | Metrik evaluasi model menjadi terlalu optimistis dan gagal saat rekonstruksi pada permutasi baru. | Dedup byte permutasi sebelum split + `GroupShuffleSplit(family_id)` + unit test penegak disjoint family & hash. |
| **Streamlit Cloud rusak saat refaktor berjalan** | Live app `https://cryptoml-sbox.streamlit.app/` tidak dapat diakses saat presentasi/review dosen. | Semua kerja v2 berjalan di branch fitur; `main` hanya menerima perubahan non-breaking sampai artefak v2 lengkap dan lolos smoke test. |
| **`.gitignore` memblokir `*.joblib` & ukuran file model besar** | Model gagal di-push ke GitHub atau memperlambat startup Streamlit Cloud. | Tambahkan `!models/*.joblib` di `.gitignore`, simpan dengan `joblib.dump(..., compress=3)`, dan jaga ukuran model RF/XGB/MLP tetap kompak (target < 15 MB total). |

---

## 6. Acceptance Criteria Per Fase

| Fase | Pemilik | Acceptance Criteria Terverifikasi |
| :--- | :---: | :--- |
| **Tahap 0 — Audit & Kontrak** | Mahdi (#18) | `REFACTOR_PLAN.md`, `docs/CONTRACT_V2_RECONSTRUCTION.md`, `config/training.yaml`, `config/reconstruction.yaml` tersedia; baseline `102/102` pytest tercatat; `pixel_correlation` + `cipher_variant` teruji di `src/image_metrics.py` & `src/mandrill_eval.py`; semua tes lulus tanpa memecah live app. |
| **Tahap 1 — Evaluator & Composite Score** | Naila (#16) | `src/composite_score.py` terpisah dari trainer; `EVALUATOR_VERSION` & dokumentasi LAP tercatat; AES Standar & K44 tervalidasi; bijectivity ditegakkan; skor lama == skor baru pada fixture identik. |
| **Tahap 2 — Dataset Permutasi & Engine Bersama** | Yafi (#17) | 4 operator mutasi mempertahankan bijectivity tanpa mutasi parent; jalur usulan 0 pemanggilan `aes_affine_generator` / chaotic map; `datasets/split_manifest.json` 70/15/15 bebas kebocoran family/hash; engine rekonstruksi deterministik (global best monoton tidak turun) + baseline Random Search/Swap/Top-K + smoke artifacts nyata. |
| **Tahap 3 — Training 3 Model (RF, MLP, XGBoost)** | Yafi, Naila, Mahdi | Ketiga model memakai manifest split & schema fitur identik; tuning hanya pada train/val (test terkunci); MAE/MSE/RMSE/$R^2$/Spearman/Top-K/ablation/waktu tercatat; artefak `.joblib` + `feature_schema.json` + `model_metadata.json` lolos uji round-trip & penolakan schema tidak cocok. |
| **Tahap 4 — Eksperimen Komparatif Multi-Seed** | Mahdi (#18) | Perbandingan berpasangan multi-seed (`Initial Permutation`, `Random Search`, `Random Swap/Top-K`, `RF`, `XGBoost`, `MLP`) dengan titik awal, operator, stopping rule, dan budget full-eval setara; melaporkan mean±SD, per-seed, selisih berpasangan, kurva konvergensi, dan waktu terpisah (termasuk hasil negatif bila ada). |
| **Tahap 5 — Evaluasi Citra Mandrill v2** | Mahdi (#18) | AES Standar, K44, RF v2, XGBoost v2, MLP v2 dievaluasi setara pada Mandrill RGB & Grayscale; Entropy, Histogram $\chi^2$, Korelasi Piksel (H/V/D), NPCR/NBCR, UACI, waktu enkripsi/dekripsi, label `AES-128 Modifikasi`, dan dekripsi 100% byte-for-byte lossless terverifikasi. |
| **Tahap 6 — Dashboard Streamlit 6 Halaman & Rilis** | Mahdi (#18) | 6 halaman aktif membaca artefak v2 nyata tanpa training saat startup; live reconstruction ringan ter-cache; status eksplisit bila artefak/schema tidak cocok; Legacy v1 dipisahkan tegas; `pytest` penuh + smoke Streamlit lulus; `README.md` & `REFACTOR_REPORT.md` terbarui. |
