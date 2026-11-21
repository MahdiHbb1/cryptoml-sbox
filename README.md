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
├── data/
│   └── mandril_color.jpg          # Citra uji standar Mandrill (512x512 RGB)
├── docs/
│   ├── ARCHITECTURE_CONTRACT.md   # Kontrak format data & tanda tangan fungsi antar modul
│   └── PEMBAGIAN_JOBDESK.md       # Rincian tugas per anggota dan urutan fase kerja
├── src/                           # Modul Python tiap anggota tim
├── tests/                         # Unit test validasi AES Standar, K44, & lossless cipher
└── results/
    └── experiment_log.md          # Tabel pelacakan metrik ML, Metrik 1, Metrik 2 & 3
```

---

## Cara Mulai di Laptop Masing-Masing

```bash
git clone https://github.com/MahdiHbb1/cryptoml-sbox.git
cd cryptoml-sbox
pip install -r requirements.txt
```

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
