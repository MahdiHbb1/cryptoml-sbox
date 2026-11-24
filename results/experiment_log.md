# Log Eksperimen (`CryptoML-SBox`)

Catat setiap hasil eksperimen tim di tabel ini agar progres terekam rapi di GitHub.

## 1. Perbandingan Model Machine Learning (Prediksi Skor S-Box)

| Tanggal | Model | Anggota | Fitur & Jumlah Data | $R^2$ (Test) | MAE | RMSE | Waktu Latih (s) | Catatan |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 2026-11-21 | XGBoost *(Kaggle tuned)* | Mahdi | 40 fitur cepat, `n=1500` (`data/sbox_dataset.csv`, 1200 train / 300 test), saring `2.000` kandidat (`top-25`, `2,90s`) | 0.9338 | 9.7038 | 17.8486 | 7.31 | `RandomizedSearchCV` (`n_estimators=150, max_depth=3, lr=0.05, subsample=1.0, colsample=1.0`) — *apple-to-apple* di dataset bersama |
| 2026-11-21 | Random Forest | Yafi | 40 fitur cepat, `n=1500` (1200 train / 300 test), saring `2.000` kandidat (`top-25`, `4,86s`) | 0.9352 | 9.5576 | 17.6609 | 1.20 | `RandomForestRegressor(n_estimators=300, random_state=42)`; fitur terpenting: `f_33`, `f_12`, `f_31`, `f_36`, `f_29` |
| 2026-11-21 | MLP (`StandardScaler` + target scaling) | Naila | 40 fitur cepat, `n=1500` (1200 train / 300 test), saring 2.000 kandidat (top-25, 4,59s) | 0.9245 | 10.3825 | 19.0665 | 2.31 | `MLPRegressor(128,64)`, relu, adam, alpha=1e-3, early stopping (n_iter=60) |

---

## 2. Evaluasi Metrik 1 (Kualitas Kriptografi S-Box)

| S-Box | Bijective | NL | SAC (Mean) | SAC Max Err | BIC-NL | BIC-SAC | DU | DAP | LAP | Fixed Points |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AES Standar *(Baseline)* | Yes | 112 | 0.50488 | 0.06250 | 112 | 0.50460 | 4 | 0.015625 | 0.0625 | 0 |
| K44 Alamsyah *(Baseline)* | Yes | 112 | 0.50073 | 0.06250 | 112 | 0.50237 | 4 | 0.015625 | 0.0625 | 1 |
| S-Box XGBoost (Mahdi, `0x1E7`, `C=0x07`) | Yes | 112 | 0.49951 | 0.04688 | 112 | 0.50146 | 4 | 0.015625 | 0.0625 | 0 |
| S-Box Random Forest (Yafi) | Yes | 112 | 0.49976 | 0.04688 | 112 | 0.50028 | 4 | 0.015625 | 0.0625 | 0 |
| S-Box MLP (Naila, `aes_affine`) | Yes | 112 | 0.49951 | 0.04688 | 112 | 0.49951 | 4 | 0.015625 | 0.0625 | 0 |

---

## 3. Evaluasi Metrik 2 & 3 pada Citra Mandrill (`data/mandril_color.jpg`)

Hasil `python -m src.mandrill_eval` (2026-11-22), tersimpan lengkap di `results/mandrill_metrics.json`:

- **Citra:** Mandrill RGB 512×512×3 dan grayscale 512×512.
- **Mode AES:** `CBC2` (dua lintasan) — kunci `CryptoML-SBox26!` (`43727970746f4d4c2d53426f78323621`), IV `MandrillCBC2-IV!` (`4d616e6472696c6c434243322d495621`).
- **Uji 1 bit:** piksel terakhir `(511, 511)` kanal 0 bit 0 (LSB) dibalik; kedua cipher memakai kunci & IV identik.
- **Histogram $\chi^2$:** 256 bin `cipher1` vs distribusi uniform, derajat kebebasan 255 (p besar = histogram rata).
- **NPCR/UACI:** rata-rata per kanal; acuan ideal 99.6094% dan 33.4635%.

| S-Box | Mode Citra | Entropy Plain | Entropy Cipher | Histogram $\chi^2$ (p-val) | NPCR (%) | UACI (%) | NBCR (%) | Lossless Decrypt |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AES Standar *(Baseline)* | RGB | 7.76244 | 7.99976 | 257.13 (p = 0.4508) | 99.6136 | 33.4650 | 49.9714 | Yes |
| K44 Alamsyah *(Baseline)* | RGB | 7.76244 | 7.99976 | 259.91 (p = 0.4031) | 99.6072 | 33.4807 | 49.9954 | Yes |
| S-Box XGBoost (Mahdi) | RGB | 7.76244 | 7.99977 | 255.72 (p = 0.4755) | 99.6113 | 33.4518 | 49.9877 | Yes |
| S-Box Random Forest (Yafi) | RGB | 7.76244 | 7.99974 | 286.49 (p = 0.0853) | 99.6212 | 33.4691 | 50.0388 | Yes |
| S-Box MLP (Naila) | RGB | 7.76244 | 7.99977 | 252.04 (p = 0.5406) | 99.6080 | 33.4302 | 49.9875 | Yes |
| AES Standar *(Baseline)* | Grayscale | 7.35832 | 7.99939 | 220.04 (p = 0.9447) | 99.5907 | 33.4784 | 50.0364 | Yes |
| K44 Alamsyah *(Baseline)* | Grayscale | 7.35832 | 7.99934 | 239.35 (p = 0.7511) | 99.5995 | 33.4546 | 49.9318 | Yes |
| S-Box XGBoost (Mahdi) | Grayscale | 7.35832 | 7.99930 | 255.60 (p = 0.4777) | 99.6197 | 33.4497 | 49.9960 | Yes |
| S-Box Random Forest (Yafi) | Grayscale | 7.35832 | 7.99940 | 216.82 (p = 0.9602) | 99.6017 | 33.4192 | 49.9779 | Yes |
| S-Box MLP (Naila) | Grayscale | 7.35832 | 7.99927 | 265.15 (p = 0.3181) | 99.5934 | 33.5018 | 50.0257 | Yes |

**Catatan hasil:**

1. Seluruh S-Box memenuhi acuan difusi: NPCR 99.5907–99.6212%, UACI 33.4192–33.5018%, entropy cipher ≥ 7.99927, dan $\chi^2$ p-value ≥ 0.0853 (histogram cipher tidak menolak distribusi uniform). Dekripsi `lossless` pada semua kombinasi.
2. Selisih antar S-Box sangat kecil (NPCR ≤ 0.031 pp, UACI ≤ 0.069 pp) sehingga tidak ada S-Box yang unggul sistematis pada citra Mandrill; Metrik 1 tetap menjadi pembeda utama kualitas S-Box.
3. Entropy plaintext RGB 7.76244 cocok dengan pengukuran CBC2 Kaggle sebelumnya (7.76244), sedangkan NPCR/UACI berbeda tipis karena kunci, IV, dan posisi piksel uji berbeda. Baris di atas adalah hasil runner repo yang dapat direproduksi ulang.

**Grafik laporan:** `python -m src.visualize_mandrill` → `results/figures/mandrill_entropy_chi2.png` (entropy plaintext vs cipher + uji $\chi^2$ terhadap batas kritis 293.25) dan `results/figures/mandrill_diffusion.png` (selisih NPCR, UACI, NBCR terhadap nilai ideal).

**Gambar 4.2:** `python -m src.visualize_mandrill --showcase` → `results/figures/mandrill_encryption_xgb.png` (plaintext, cipher, dekripsi lossless, dan histogram RGB dengan $\chi^2$ per kanal: R 255.08 / G 261.94 / B 254.30).

---

## 3.1 Pengaruh Mode Operasi pada Difusi (RGB 512×512)

Pengukuran terpisah dengan kunci, IV, dan piksel uji identik (piksel terakhir `(511, 511)`), khusus citra RGB:

| S-Box | NPCR ECB (%) | NPCR CBC (%) | NPCR CBC2 (%) | UACI ECB (%) | UACI CBC (%) | UACI CBC2 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| AES Standar | 0.0020 | 0.0020 | 99.6136 | 0.0006 | 0.0008 | 33.4650 |
| K44 Alamsyah | 0.0020 | 0.0020 | 99.6072 | 0.0007 | 0.0010 | 33.4807 |
| S-Box XGBoost | 0.0020 | 0.0020 | 99.6113 | 0.0007 | 0.0007 | 33.4518 |
| S-Box Random Forest | 0.0020 | 0.0020 | 99.6212 | 0.0007 | 0.0006 | 33.4691 |
| S-Box MLP | 0.0020 | 0.0020 | 99.6080 | 0.0005 | 0.0006 | 33.4302 |

Catatan: perubahan 1 bit di blok terakhir hanya mengubah 1 blok ciphertext pada ECB maupun CBC satu lintasan (16 dari 786.432 byte = 0,00203%), sehingga NPCR/UACI gagal acuan. Menariknya entropy cipher ECB/CBC tetap 7.99974–7.99979 bit dan $\chi^2$ 223.15–284.48 (p = 0.0990–0.9256), jadi Metrik 2 saja tidak cukup mendeteksi kegagalan difusi. Perintah: `python -m src.mandrill_eval --modes RGB --cipher-mode ECB` dan `--cipher-mode CBC`; JSON hasil pengukuran belum diarsipkan di repo.
