# Log Eksperimen (`CryptoML-SBox`)

Catat setiap hasil eksperimen tim di tabel ini agar progres terekam rapi di GitHub.

## 1. Perbandingan Model Machine Learning (Prediksi Skor S-Box)

| Tanggal | Model | Anggota | Fitur & Jumlah Data | $R^2$ (Test) | MAE | RMSE | Waktu Latih (s) | Catatan |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 2026-11-21 | XGBoost *(baseline)* | Mahdi | 40 fitur cepat, `n=200` (160 train / 40 test) | 0.9185 | 8.7501 | 15.9863 | 0.18 | Fallback sintetis; siap diulang dengan `sbox_dataset.csv` / Colab (`--tune`) |
| - | Random Forest | Yafi | - | - | - | - | - | Menunggu `train_rf.py` |
| - | MLP (`StandardScaler`) | Naila | - | - | - | - | - | Menunggu `train_mlp.py` |

---

## 2. Evaluasi Metrik 1 (Kualitas Kriptografi S-Box)

| S-Box | Bijective | NL | SAC (Mean) | SAC Max Err | BIC-NL | BIC-SAC | DU | DAP | LAP | Fixed Points |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AES Standar *(Baseline)* | Yes | 112 | 0.5049 | 0.0625 | 112 | 0.5046 | 4 | 0.015625 | 0.0625 | 0 |
| K44 Alamsyah *(Baseline)* | Yes | 112 | 0.5007 | 0.0625 | 112 | 0.5024 | 4 | 0.015625 | 0.0625 | 1 |
| S-Box XGBoost (Mahdi, `0x19F`, `C=0x06`) | Yes | 112 | 0.5015 | 0.0469 | 112 | 0.5004 | 4 | 0.015625 | 0.0625 | 0 |
| S-Box Random Forest (Yafi) | - | - | - | - | - | - | - | - | - | - |
| S-Box MLP (Naila) | - | - | - | - | - | - | - | - | - | - |

---

## 3. Evaluasi Metrik 2 & 3 pada Citra Mandrill (`data/mandril_color.jpg`)

| S-Box | Mode Citra | Entropy Plain | Entropy Cipher | Histogram $\chi^2$ (p-val) | NPCR (%) | UACI (%) | Lossless Decrypt |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| AES Standar | RGB / Gray | - | - | - | - | - | - |
| K44 Alamsyah | RGB / Gray | - | - | - | - | - | - |
| S-Box XGBoost *(smoke check CBC2)* | RGB 512×512 | 7.76244 | 7.99974 | Menunggu `image_metrics.py` | 99.6189 | 33.4582 | Yes |
| S-Box Random Forest | RGB / Gray | - | - | - | - | - | - |
| S-Box MLP | RGB / Gray | - | - | - | - | - | - |
