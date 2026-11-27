# CryptoML-SBox

Proyek akhir mata kuliah **Kriptografi**: konstruksi dan seleksi **Substitution-Box (S-Box)** menggunakan **Machine Learning** (`XGBoost`, `Random Forest`, dan `MLP`) serta evaluasi keamanannya terhadap enkripsi citra digital **Mandrill** menggunakan **AES-128 CBC**.

## Tim & Pembagian Peran

| Anggota | GitHub | Fokus Modul | Branch Kerja |
| :--- | :--- | :--- | :--- |
| **Mahdi** | [@MahdiHbb1](https://github.com/MahdiHbb1) | Model **XGBoost** + Generator Modifikasi AES (`GF(2^8)`) + `aes_cipher_runner.py` + Integrasi | `feat/aes-and-xgboost` |
| **Yafi** | [@twizzcode](https://github.com/twizzcode) | Generator Non-AES (*Chaotic + Swap*) + Fitur Cepat + Dataset Latih + Model **Random Forest** | `feat/dataset-and-rf` |
| **Naila** | [@NAILAAR22](https://github.com/NAILAAR22) | Evaluator **Metrik 1** (`sbox_metrics.py`) + Model **MLP** + Validasi AES/K44 | `feat/metrics-and-mlp` |
| **Mas Rizqi** | [@rizqinp2](https://github.com/rizqinp2) | Evaluasi Citra Mandrill **Metrik 2 & 3** (`image_metrics.py`) + Visualisasi Akhir | `feat/mandrill-evaluation` |

Detail lengkap alur kerja dan kontrak antarmuka modul ada di:
- [docs/PEMBAGIAN_JOBDESK.md](docs/PEMBAGIAN_JOBDESK.md)
- [docs/ARCHITECTURE_CONTRACT.md](docs/ARCHITECTURE_CONTRACT.md)
- [docs/mandrill_analysis.md](docs/mandrill_analysis.md) — narasi Bab 2.6, 3.5.3, 4.3, dan 5.1 (Metrik 2 & 3)

---

## Struktur Direktori

```text
cryptoml-sbox/
├── README.md
├── requirements.txt
├── app.py                         # Aplikasi web interaktif Streamlit (6 halaman)
├── .streamlit/config.toml         # Konfigurasi tema web Streamlit
├── assets/
│   └── cryptoml_sbox_archify.html # Diagram arsitektur & alur sistem interaktif (Archify)
├── data/
│   └── mandril_color.jpg          # Citra uji standar Mandrill (512x512 RGB)
├── docs/
│   ├── ARCHITECTURE_CONTRACT.md   # Kontrak format data & tanda tangan fungsi antar modul
│   └── PEMBAGIAN_JOBDESK.md       # Rincian tugas per anggota dan urutan fase kerja
├── notebooks/
│   └── Mahdi_XGBoost_SBox_Colab_Kaggle.ipynb
├── src/                           # Modul Python tiap anggota tim
├── tests/                         # Unit test validasi AES Standar, K44, XGBoost, & web helper
└── results/
    ├── sbox_xgb.npy               # S-Box terpilih XGBoost (0x1E7, C=0x07)
    ├── sbox_rf.npy                # S-Box terpilih Random Forest (Yafi)
    ├── sbox_mlp.npy               # S-Box terpilih MLP (Naila)
    ├── xgb_metrics.json           # Log metrik pelatihan & penyaringan Kaggle (apple-to-apple)
    ├── rf_metrics.json            # Log metrik Random Forest & feature importance
    ├── mlp_metrics.json           # Log metrik MLPRegressor
    ├── mandrill_metrics.json      # Hasil Metrik 2 & 3 (5 S-Box × RGB/Grayscale)
    ├── figures/                   # Grafik Metrik 2 & 3 (PNG) untuk laporan
    └── experiment_log.md          # Tabel pelacakan metrik ML, Metrik 1, Metrik 2 & 3
```

---

## Cara Menjalankan Aplikasi Web (Streamlit)

```bash
git clone https://github.com/MahdiHbb1/cryptoml-sbox.git
cd cryptoml-sbox
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi web memiliki 6 halaman:
1. **Overview** — Ringkasan pipeline v2 bebas polinomial, pratinjau enkripsi Mandrill v2, perbandingan 5 S-Box fokus (AES, K44, RF/XGBoost/MLP v2 seed 100), kontribusi tim, dan akses Diagram Archify.
2. **Dataset/Training** — Ringkasan manifest split keluarga 70/15/15 (SHA-256 fingerprint + verifikasi anti-leakage), metrik hold-out 3 model tersimpan, dan *feature importance*; tidak melatih model saat aplikasi dimuat.
3. **Reconstruction** — Kurva konvergensi pra-hitung multi-seed dan simulasi rekonstruksi ringan *opt-in* ter-cache untuk memilih model **XGBoost**, **Random Forest**, atau **MLP** tersimpan yang lolos validasi schema.
4. **Crypto Evaluation** — Pemisahan tegas antara tabel/grafik **Hasil v2 Polynomial-Free**, **Arsip Legacy v1 (GF(2⁸))**, dan inspeksi detail grid 16×16 + heatmap SAC 8×8.
5. **Image Encryption** — Tabel & grafik Korelasi Piksel benchmark **Mandrill v2** (10 run RGB & Grayscale berlabel *AES-128 Modifikasi*), simulasi interaktif citra (Mandrill/upload), dan simulasi teks AES-128.
6. **Comparative Analysis** — Grafik & tabel perbandingan berpasangan 5 seed (CI 95% vs Random Top-K), studi ablasi 3-arah (`perm4` vs `proxy` vs `cheap` SAC 8×8 + FP), serta rincian waktu komputasi per seed.

Arsitektur Archify tetap tersedia pada `assets/cryptoml_sbox_archify.html` dan dapat dibuka/diunduh dari halaman *Overview*. Halaman menampilkan status artefak eksplisit, bukan angka contoh saat hasil v2 belum tersedia.

### Reproduksi Pipeline v2 (Issue #16, #17, #18)

```bash
python -m src.train_v2_models --output-dir models/reproduction
python -m src.v2_experiment --output results/v2/comparative_summary.json
python -m src.run_v2_benchmark --seed 100
python -m pytest -q
```

Pipeline v2 membangun dataset 150 keluarga (1.500 permutasi unik bebas polinomial) dengan split keluarga 70/15/15 (`seed=42`), lalu melatih RF, XGBoost (tuning pada validation), dan MLP pada dataset/fingerprint yang sama. Eksperimen komparatif memuat artefak tersimpan `models/{rf,xgboost,mlp}`, menjalankan lima seed berpasangan (`100–104`) dengan batas 151 evaluasi penuh per run serta studi ablasi 3-arah; evaluasi Mandrill v2 menguji seed 100 (AES/K44 + 3 model v2 pada RGB & Grayscale) lengkap dengan korelasi piksel H/V/D dan verifikasi dekripsi 100% *lossless*. Detail angka aktual dan keterbatasan ilmiah tercantum di [REFACTOR_REPORT.md](REFACTOR_REPORT.md). Semua metrik empiris bukan bukti keamanan kriptografi formal.

### Cara Mulai Pengembangan di Laptop Masing-Masing

### Membangun Dataset Bersama (Yafi — Issue #3)

```bash
python -m src.dataset_builder --n-chaotic 750 --n-aes 750
```

Perintah menghasilkan `data/sbox_dataset.csv` berisi 1.500 kandidat bijektif dari jalur
`chaotic_swap` dan `aes_affine`, 40 fitur cepat, serta target metrik untuk model ML.
File CSV diabaikan Git karena dapat dibangun ulang dengan `--seed 42`.

### Melatih Random Forest (Yafi — Issue #5)

```bash
python -m src.train_rf --n-screen 2000 --top-k 25
```

Perintah menyimpan pemenang ke `results/sbox_rf.npy` dan metrik model beserta
peringkat *feature importance* ke `results/rf_metrics.json`.

### Evaluasi Citra Mandrill — Metrik 2 & 3 (Mas Rizqi)

```bash
python -m src.mandrill_eval
```

Perintah mengevaluasi 5 S-Box (AES Standar, K44, XGBoost, Random Forest, MLP)
pada `data/mandril_color.jpg` dalam mode RGB dan grayscale memakai AES-128
`CBC2`, membalik 1 bit pada 1 piksel plaintext (default piksel terakhir), lalu
menyimpan Entropy, Histogram χ², NPCR, UACI, NBCR, serta status dekripsi
*lossless* ke `results/mandrill_metrics.json`. Tabel Markdown hasilnya dicetak
ke terminal untuk ditempel ke `results/experiment_log.md`.

Perhitungan Metrik 2 & 3 berada di `src/image_metrics.py`
(`evaluate_image_security`, sesuai Kontrak §5), sehingga aplikasi Streamlit
otomatis memakai implementasi yang sama lewat `src/web_helpers.py`. Opsi CLI:
`--modes RGB GRAY`, `--flip-x/--flip-y/--channel/--bit`,
`--cipher-mode CBC2|CBC|ECB`, dan `--out <path.json>`.

Grafik untuk laporan dibuat dari JSON tersebut:

```bash
python -m src.visualize_mandrill
```

Menghasilkan `results/figures/mandrill_entropy_chi2.png` (entropy plaintext vs
cipher dan uji Chi-square histogram cipher) serta
`results/figures/mandrill_diffusion.png` (selisih NPCR, UACI, dan NBCR terhadap
nilai ideal). Opsi CLI: `--json`, `--out-dir`, `--format png|svg|both`,
`--dpi`, dan `--show`.

Gambar 4.2 (plaintext, cipher, hasil dekripsi, dan histogram RGB) dibuat dengan:

```bash
python -m src.visualize_mandrill --showcase
```

Perintah ini menambah `results/figures/mandrill_encryption_xgb.png`; S-Box dan
citra dapat diganti lewat `--sbox` dan `--image`.

Narasi lengkap Bab Pengujian Citra Mandrill (persamaan Metrik 2 & 3, protokol
uji, Tabel 4.3, pembahasan, dan kesimpulan) tersedia di
[docs/mandrill_analysis.md](docs/mandrill_analysis.md).

### Aturan Git & Kolaborasi
1. **Jangan push langsung ke `main`** untuk kode yang belum selesai. Selalu buat/pindah ke branch masing-masing:
   ```bash
   git checkout -b feat/nama-fitur
   ```
2. Tarik update terbaru dari `main` sebelum mulai mengoding:
   ```bash
   git pull origin main
   ```
3. Commit kecil dan jelas (`feat: ...`, `fix: ...`, `docs: ...`), lalu *push* dan buka **Pull Request (PR)** di GitHub dengan menautkan nomor Issue (mis. `Closes #1`).
4. Catat setiap hasil eksperimen penting di `results/experiment_log.md`.

---

## Ringkasan Metrik Evaluasi

- **Metrik 1 (S-Box):** Bijectivity & Balance, Nonlinearity (255 kombinasi linier), SAC, BIC (`BIC-NL` & `BIC-SAC`), DAP (`DU`), dan LAP.
- **Metrik 2 (Keacakan Citra):** Information Entropy dan Histogram (+ uji Chi-square).
- **Metrik 3 (Performa Enkripsi Citra):** NPCR *(atau NBCR sesuai konfirmasi dosen)* dan UACI pada citra Mandrill (dibandingkan antara dua *cipher image* yang berbeda 1 bit di 1 piksel pada kunci yang sama).
