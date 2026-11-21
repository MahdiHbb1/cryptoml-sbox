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

---

## Struktur Direktori

```text
cryptoml-sbox/
├── README.md
├── requirements.txt
├── app.py                         # Aplikasi web interaktif Streamlit (3 halaman)
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

Aplikasi web terdiri dari 3 halaman utama:
1. **Ringkasan Proyek** — Penjelasan sederhana untuk umum/dosen, pratinjau enkripsi Mandrill langsung, tabel & grafik perbandingan Metrik 1 (otomatis mendeteksi `sbox_rf.npy` & `sbox_mlp.npy` saat di-merge), serta progres 4 anggota tim.
2. **Diagram Alur & Arsitektur (Archify)** — Diagram node interaktif dengan 3 *Guided Views*, referensi langsung ke baris kode GitHub, toggle *Light/Dark*, *Live/Still*, serta ekspor PNG/SVG.
3. **Simulasi Langsung** — Inspeksi & pembangkit S-Box GF(2^8) baru (tabel 16×16 + heatmap SAC 8×8), simulasi enkripsi-dekripsi citra Mandrill/unggah sendiri (Entropy, Histogram, NPCR, UACI, *lossless*), serta simulasi enkripsi-dekripsi teks AES-128.

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
