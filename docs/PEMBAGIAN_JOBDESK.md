# Pembagian Tugas Tim (`CryptoML-SBox`)

## 1. Alur Ringkas
1. Fungsi pembuat kandidat S-box bijektif (`0–255` unik) dibuat sebagai modul bersama oleh **Yafi** (jalur Non-AES *chaotic + swap*) dan **Mahdi** (jalur Modifikasi AES polinomial $GF(2^8)$ + matriks *affine*).
2. **Mahdi (`XGBoost`)**, **Yafi (`Random Forest`)**, dan **Naila (`MLP`)** masing-masing melatih **1 model ML** di atas dataset bersama (`sbox_dataset.csv`) dan menyaring keluar **1 tabel S-box terbaik** versi modelnya (`sbox_xgb.npy`, `sbox_rf.npy`, `sbox_mlp.npy`).
3. **Naila** memvalidasi penuh seluruh S-box pemenang pada **Metrik 1**, lalu **Mas Rizqi** menjalankan uji enkripsi & dekripsi citra **Mandrill** (`data/mandril_color.jpg`) pada **Metrik 2 & Metrik 3** memakai runner AES-CBC dari **Mahdi**.

---

## 2. Pembagian Per Anggota

### 1. Mahdi (`@MahdiHbb1`) — *XGBoost + Modifikasi AES & Integrasi*
- **Model ML:** `src/train_xgboost.py` $\to$ hasil `results/sbox_xgb.npy` + metrik $R^2$/MAE/RMSE.
- **Kriptografi:** generator variasi polinomial *irreducible* $GF(2^8)$ & matriks *affine* invertibel + `src/aes_cipher_runner.py` (AES-128 CBC custom S-box & dekripsi *lossless*).
- **Integrasi & Laporan:** menggabungkan seluruh modul tim + Bab Pendahuluan, Metodologi, Modifikasi AES, dan Hasil XGBoost.

### 2. Yafi (`@twizzcode`) — *Random Forest + Generator Dataset & Fitur*
- **Dataset & Fitur:** generator kandidat Non-AES (*chaotic map* + *swap*) & `extract_fast_features` untuk menyusun `data/sbox_dataset.csv` bersama fungsi skor dari Naila.
- **Model ML:** `src/train_rf.py` $\to$ hasil `results/sbox_rf.npy` + metrik $R^2$/MAE/RMSE & *feature importance*.
- **Laporan:** Bab Desain Dataset/Fitur dan Hasil Random Forest.

### 3. Naila (`@NAILAAR22`) — *MLP (`MLPRegressor`) + Evaluator Metrik 1*
- **Evaluator Metrik 1:** `src/sbox_metrics.py` (`Bijectivity`, `Balance`, `NL` 255-mask, `SAC`, `BIC-NL`, `BIC-SAC`, `DU/DAP`, `LAP`) + validasi ke **AES Standar** & **K44**.
- **Model ML:** `src/train_mlp.py` (`StandardScaler` + `MLPRegressor`) $\to$ hasil `results/sbox_mlp.npy` + metrik $R^2$/MAE/RMSE.
- **Laporan:** Bab Hasil MLP dan Analisis Perbandingan Metrik 1 S-box.

### 4. Mas Rizqi (`@rizqinp2`) — *Penguji Citra Mandrill (Metrik 2 & 3) + Visualisasi*
- **Penguji Citra:** `src/image_metrics.py` pada `data/mandril_color.jpg` (RGB 24-bit & grayscale 8-bit + skenario ubah 1 bit di 1 piksel) untuk **5 S-box** (AES Standar, K44, XGBoost, RF, MLP).
- **Metrik 2 & 3:** Entropy, Histogram (+ Chi-square), NPCR/NBCR & UACI antar dua *cipher image*, serta cek dekripsi *lossless*.
- **Visualisasi & Laporan:** grafik perbandingan akhir + Bab Pengujian Citra Mandrill dan Kesimpulan.

---

## 3. Urutan Fase Kerja
1. **Fase 1 (Paralel):** Naila selesaikan `sbox_metrics.py`; Mahdi siapkan `aes_cipher_runner.py` + generator AES *affine*; Yafi siapkan generator *chaotic* + ekstraksi fitur; Mas Rizqi siapkan `image_metrics.py` dan tes awal pakai AES Standar.
2. **Fase 2 (Training ML):** Yafi gabungkan `sbox_dataset.csv` memakai fungsi Naila $\to$ Mahdi, Yafi, dan Naila melatih model masing-masing (`XGBoost`, `Random Forest`, `MLP`) dan menghasilkan S-box terbaik.
3. **Fase 3 (Evaluasi & Laporan):** Naila rekap Metrik 1 seluruh S-box; Mas Rizqi rekap Metrik 2 & 3 di Mandrill; tim menyusun laporan dan slide presentasi.
