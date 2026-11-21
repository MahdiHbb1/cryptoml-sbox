# Log Eksperimen (`CryptoML-SBox`)

Catat setiap hasil eksperimen tim di tabel ini agar progres terekam rapi di GitHub.

## 1. Perbandingan Model Machine Learning (Prediksi Skor S-Box)

| Tanggal | Model | Anggota | Fitur & Jumlah Data | $R^2$ (Test) | MAE | RMSE | Waktu Latih (s) | Catatan |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| - | XGBoost | Mahdi | - | - | - | - | - | Menunggu `sbox_dataset.csv` |
| - | Random Forest | Yafi | - | - | - | - | - | Menunggu `sbox_dataset.csv` |
| - | MLP (`StandardScaler`) | Naila | - | - | - | - | - | Menunggu `sbox_dataset.csv` |

---

## 2. Evaluasi Metrik 1 (Kualitas Kriptografi S-Box)

| S-Box | Bijective | NL | SAC (Mean) | SAC Max Err | BIC-NL | BIC-SAC | DU | DAP | LAP | Fixed Points |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AES Standar *(Baseline)* | Yes | 112 | 0.5049 | - | 112 | 0.5046 | 4 | 0.015625 | 0.0625 | 0 |
| K44 Alamsyah *(Baseline)* | Yes | 112 | 0.5007 | - | 112 | 0.5024 | 4 | 0.015625 | 0.0625 | 1 |
| S-Box XGBoost (Mahdi) | - | - | - | - | - | - | - | - | - | - |
| S-Box Random Forest (Yafi) | - | - | - | - | - | - | - | - | - | - |
| S-Box MLP (Naila) | - | - | - | - | - | - | - | - | - | - |

---

## 3. Evaluasi Metrik 2 & 3 pada Citra Mandrill (`data/mandril_color.jpg`)

| S-Box | Mode Citra | Entropy Plain | Entropy Cipher | Histogram $\chi^2$ (p-val) | NPCR (%) | UACI (%) | Lossless Decrypt |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| AES Standar | RGB / Gray | - | - | - | - | - | - |
| K44 Alamsyah | RGB / Gray | - | - | - | - | - | - |
| S-Box XGBoost | RGB / Gray | - | - | - | - | - | - |
| S-Box Random Forest | RGB / Gray | - | - | - | - | - | - |
| S-Box MLP | RGB / Gray | - | - | - | - | - | - |
