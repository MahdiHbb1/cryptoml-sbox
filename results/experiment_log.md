# Log Eksperimen (`CryptoML-SBox`)

Catat setiap hasil eksperimen tim di tabel ini agar progres terekam rapi di GitHub.

## 1. Perbandingan Model Machine Learning (Prediksi Skor S-Box)

| Tanggal | Model | Anggota | Fitur & Jumlah Data | $R^2$ (Test) | MAE | RMSE | Waktu Latih (s) | Catatan |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 2026-11-21 | XGBoost *(Kaggle tuned)* | Mahdi | 40 fitur cepat, `n=1500` (1200 train / 300 test), saring `15.000` kandidat (`top-40`, `17,94s`) | 0.9149 | 8.7212 | 16.8364 | 6.99 | `RandomizedSearchCV` (`n_estimators=150, max_depth=6, lr=0.03, subsample=0.8, colsample=0.8`); siap diulang begitu `sbox_dataset.csv` final dari Yafi tersedia |
| - | Random Forest | Yafi | - | - | - | - | - | Menunggu `train_rf.py` |
| - | MLP (`StandardScaler`) | Naila | - | - | - | - | - | Menunggu `train_mlp.py` |

---

## 2. Evaluasi Metrik 1 (Kualitas Kriptografi S-Box)

| S-Box | Bijective | NL | SAC (Mean) | SAC Max Err | BIC-NL | BIC-SAC | DU | DAP | LAP | Fixed Points |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AES Standar *(Baseline)* | Yes | 112 | 0.50488 | 0.06250 | 112 | 0.50460 | 4 | 0.015625 | 0.0625 | 0 |
| K44 Alamsyah *(Baseline)* | Yes | 112 | 0.50073 | 0.06250 | 112 | 0.50237 | 4 | 0.015625 | 0.0625 | 1 |
| S-Box XGBoost (Mahdi, `0x13F`, `C=0x02`) | Yes | 112 | 0.50024 | 0.04688 | 112 | 0.50077 | 4 | 0.015625 | 0.0625 | 0 |
| S-Box Random Forest (Yafi) | - | - | - | - | - | - | - | - | - | - |
| S-Box MLP (Naila) | - | - | - | - | - | - | - | - | - | - |

---

## 3. Evaluasi Metrik 2 & 3 pada Citra Mandrill (`data/mandril_color.jpg`)

| S-Box | Mode Citra | Entropy Plain | Entropy Cipher | Histogram $\chi^2$ (p-val) | NPCR (%) | UACI (%) | Lossless Decrypt |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| AES Standar | RGB / Gray | - | - | - | - | - | - |
| K44 Alamsyah | RGB / Gray | - | - | - | - | - | - |
| S-Box XGBoost *(Kaggle CBC2)* | RGB 512×512 | 7.76244 | 7.99973 | Menunggu `image_metrics.py` | 99.6215 | 33.4303 | Yes |
| S-Box Random Forest | RGB / Gray | - | - | - | - | - | - |
| S-Box MLP | RGB / Gray | - | - | - | - | - | - |
