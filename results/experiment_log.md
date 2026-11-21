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

| S-Box | Mode Citra | Entropy Plain | Entropy Cipher | Histogram $\chi^2$ (p-val) | NPCR (%) | UACI (%) | Lossless Decrypt |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| AES Standar | RGB / Gray | - | - | - | - | - | - |
| K44 Alamsyah | RGB / Gray | - | - | - | - | - | - |
| S-Box XGBoost *(Kaggle CBC2)* | RGB 512×512 | 7.76244 | 7.99978 | Menunggu `image_metrics.py` | 99.6166 | 33.4645 | Yes |
| S-Box Random Forest | RGB / Gray | - | - | - | - | - | - |
| S-Box MLP | RGB / Gray | - | - | - | - | - | - |
