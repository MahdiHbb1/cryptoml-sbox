# Kontrak Antarmuka Modul (`Architecture Contract`)

Dokumen ini memastikan kode yang dibuat oleh **Mahdi**, **Yafi**, **Naila**, dan **Mas Rizqi** langsung cocok saat digabungkan tanpa error format data.

---

## 1. Format Baku S-Box
Setiap S-box disimpan dan dipertukarkan dalam format:
- **Memori (Python):** `numpy.ndarray` 1D dengan `dtype=np.uint8`, `shape=(256,)`, berisi permutasi unik `0..255`.
- **File (`.npy`):** `results/sbox_aes_std.npy`, `results/sbox_k44.npy`, `results/sbox_xgb.npy`, `results/sbox_rf.npy`, `results/sbox_mlp.npy`.
- **Fungsi cek cepat bijektivitas:** `sbox.shape == (256,)` dan `len(set(sbox.tolist())) == 256`.

---

## 2. Kontrak `src/sbox_metrics.py` (Naila)

```python
def evaluate_sbox(sbox: np.ndarray) -> dict:
    """
    Mengembalikan dictionary dengan key standar:
    - "bijective": bool
    - "balanced": bool
    - "nl": int              # Nonlinearity minimum seluruh 255 kombinasi linier
    - "sac_mean": float      # Rata-rata matriks SAC 8x8 (ideal ~0.5)
    - "sac_max_err": float   # max(|P_ij - 0.5|) pada seluruh 64 entri SAC
    - "bic_nl": int          # Minimum NL seluruh 28 pasangan XOR bit output
    - "bic_sac": float       # Rata-rata SAC pasangan XOR bit output (konvensi ~0.5)
    - "du": int              # Differential Uniformity maksimum DDT[1:, :]
    - "dap": float           # du / 256.0
    - "lap": float           # max_walsh / 512.0 (bias maksimum |P - 0.5|)
    - "fixed_points": int    # Jumlah x dengan S(x) == x
    - "whs_cost": float      # Biaya spektral kontinu untuk skor target ML
    """
```

**Validasi wajib (`tests/test_sbox_metrics.py`):**
- AES Standar harus menghasilkan: `bijective=True`, `nl=112`, `du=4`, `dap=0.015625`, `lap=0.0625`, `sac_mean≈0.5049`.
- K44 (Alamsyah et al.) harus menghasilkan: `bijective=True`, `nl=112`, `du=4`, `dap=0.015625`, `lap=0.0625`, `sac_mean≈0.50073`.

---

## 3. Kontrak `src/dataset_builder.py` (Yafi + Mahdi)

- Menghasilkan fitur cepat S-box: `extract_fast_features(sbox: np.ndarray) -> np.ndarray` (1D array fitur numerik untuk input `X`).
- Menghasilkan file `data/sbox_dataset.csv` dengan kolom:
  - `sbox_id`, `source` (`"chaotic_swap"` atau `"aes_affine"`), kolom fitur `f_0 ... f_k`, serta target `nl`, `du`, `sac_max_err`, `whs_cost`, dan `composite_score`.
- Skrip masing-masing model (`train_xgboost.py`, `train_rf.py`, `train_mlp.py`) membaca `data/sbox_dataset.csv` dengan **split train/test dan `random_state=42` yang sama** agar perbandingan $R^2$, MAE, dan RMSE adil.

---

## 4. Kontrak `src/aes_cipher_runner.py` (Mahdi)

```python
def encrypt_image_array(img_arr: np.ndarray, sbox: np.ndarray, key: bytes, iv: bytes) -> np.ndarray:
    """Enkripsi array citra uint8 memakai AES-128 CBC dua lintasan/difusi penuh dengan custom S-box."""

def decrypt_image_array(cipher_arr: np.ndarray, sbox: np.ndarray, key: bytes, iv: bytes, orig_shape: tuple) -> np.ndarray:
    """Dekripsi kembali ke dimensi asli; wajib menghasilkan array yang identik (lossless)."""
```

---

## 5. Kontrak `src/image_metrics.py` (Mas Rizqi)

```python
def evaluate_image_security(plain_arr: np.ndarray, cipher1_arr: np.ndarray, cipher2_arr: np.ndarray) -> dict:
    """
    cipher1_arr = enkripsi(plain_arr, key, iv)
    cipher2_arr = enkripsi(plain_1bit_flipped, key, iv) dengan key & iv identik.
    Mengembalikan:
    - "entropy_plain": float
    - "entropy_cipher": float
    - "hist_chi2": float
    - "hist_p_value": float
    - "npcr": float   # persentase piksel berbeda antara cipher1_arr dan cipher2_arr
    - "uaci": float   # rata-rata selisih intensitas (%) antara cipher1_arr dan cipher2_arr
    """
```
