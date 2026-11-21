"""Helper modular untuk aplikasi web Streamlit CryptoML-SBox.

Menyediakan katalog S-Box (otomatis mendeteksi hasil tim di ``results/``),
evaluasi Metrik 1 + matriks SAC, pemformatan grid 16x16, parser upload S-Box,
simulasi enkripsi-dekripsi teks (PKCS#7) dan citra (Metrik 2 & 3).
"""

from __future__ import annotations

import io
import json
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import chisquare

from src.aes_affine_generator import (
    build_k44_sbox,
    build_standard_aes_sbox,
    generate_aes_affine_candidates,
)
from src.aes_cipher_runner import (
    aes_decrypt,
    aes_encrypt,
    decrypt_image_array,
    encrypt_image_array,
    validate_sbox,
)
from src.train_xgboost import (
    _resolve_team_functions,
    _sac_matrix,
    compute_composite_score,
)


def _resolve_image_metrics_fn():
    """Otomatis memakai src.image_metrics.evaluate_image_security (Mas Rizqi) bila tersedia."""
    try:
        from src.image_metrics import evaluate_image_security as img_eval_fn  # type: ignore
    except ImportError:
        img_eval_fn = None
    return img_eval_fn


def _load_json_if_exists(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_available_sboxes(results_dir: str | Path = Path("results")) -> dict[str, dict]:
    """Memuat baseline (AES Standar, K44) dan seluruh S-Box hasil model ML yang tersedia."""
    res_dir = Path(results_dir)
    catalog: dict[str, dict] = {
        "AES Standar (FIPS-197)": {
            "sbox": build_standard_aes_sbox(),
            "owner": "Baseline FIPS-197",
            "status": "Baseline",
            "poly": "0x11B",
            "constant": "0x63",
            "notes": "Standar Rijndael / NIST FIPS-197 (2001).",
        },
        "K44 Alamsyah (2024/2025)": {
            "sbox": build_k44_sbox(),
            "owner": "Referensi Dosen (UNNES)",
            "status": "Baseline",
            "poly": "0x11B",
            "constant": "0x63",
            "notes": "S-Box44 berbasis matriks sirkulan K44 (Alamsyah et al., Nonlinear Dynamics).",
        },
    }

    xgb_meta = _load_json_if_exists(res_dir / "xgb_metrics.json") or {}
    rf_meta = _load_json_if_exists(res_dir / "rf_metrics.json") or {}
    mlp_meta = _load_json_if_exists(res_dir / "mlp_metrics.json") or {}

    xgb_r2 = xgb_meta.get("ml_metrics", {}).get("r2")
    xgb_n = xgb_meta.get("n_screen_candidates", 2000)
    rf_r2 = rf_meta.get("ml_metrics", {}).get("r2")
    mlp_r2 = mlp_meta.get("ml_metrics", {}).get("r2")

    dynamic_entries = [
        (
            "S-Box XGBoost (Mahdi)",
            res_dir / "sbox_xgb.npy",
            "Mahdi (@MahdiHbb1)",
            xgb_meta.get("selected_sbox_poly", "0x1E7"),
            xgb_meta.get("selected_sbox_constant", "0x07"),
            (
                f"Terpilih dari penyaringan {xgb_n:,} kandidat via XGBoost (R²={xgb_r2:.4f})."
                if xgb_r2 is not None
                else "S-Box terpilih hasil model XGBoost (tim Mahdi)."
            ),
        ),
        (
            "S-Box Random Forest (Yafi)",
            res_dir / "sbox_rf.npy",
            "Yafi (@twizzcode)",
            None,
            None,
            (
                f"Terpilih dari penyaringan {rf_meta.get('n_screen_candidates', 2000):,} kandidat via Random Forest (R²={rf_r2:.4f})."
                if rf_r2 is not None
                else "S-Box terpilih hasil model Random Forest (tim Yafi)."
            ),
        ),
        (
            "S-Box MLP (Naila)",
            res_dir / "sbox_mlp.npy",
            "Naila (@NAILAAR22)",
            None,
            None,
            (
                f"Terpilih dari penyaringan {mlp_meta.get('n_screen_candidates', 2000):,} kandidat via MLPRegressor (R²={mlp_r2:.4f})."
                if mlp_r2 is not None
                else "S-Box terpilih hasil model MLPRegressor (tim Naila)."
            ),
        ),
    ]

    for name, npy_path, owner, poly, constant, notes in dynamic_entries:
        if not npy_path.exists():
            continue
        arr = validate_sbox(np.load(npy_path, allow_pickle=False))
        catalog[name] = {
            "sbox": arr,
            "owner": owner,
            "status": "Model ML",
            "poly": poly,
            "constant": constant,
            "notes": notes,
        }
    return catalog


def compute_sac_matrix(sbox: np.ndarray) -> np.ndarray:
    """Menghitung matriks Strict Avalanche Criterion (8x8)."""
    sb = validate_sbox(sbox).astype(np.int32)
    return _sac_matrix(sb)


def evaluate_sbox_full(sbox: np.ndarray) -> dict:
    """Evaluasi Metrik 1 lengkap (otomatis memakai modul Naila bila tersedia) + skor gabungan."""
    sb = validate_sbox(sbox)
    _, eval_fn = _resolve_team_functions()
    metrics = dict(eval_fn(sb))
    x = np.arange(256, dtype=np.uint8)
    metrics["opposite_fixed_points"] = int(np.sum(sb == (x ^ np.uint8(0xFF))))
    metrics["composite_score"] = compute_composite_score(metrics)
    return metrics


def format_sbox_grid(sbox: np.ndarray, fmt: str = "hex") -> pd.DataFrame:
    """Membentuk tabel 16x16 S-Box berindeks baris 0x0_..0xF_ dan kolom _0.._F."""
    sb = validate_sbox(sbox).reshape(16, 16)
    rows = [f"0x{r:X}_" for r in range(16)]
    cols = [f"_{c:X}" for c in range(16)]
    if fmt == "hex":
        data = [[f"{int(v):02X}" for v in row] for row in sb]
        return pd.DataFrame(data, index=rows, columns=cols)
    if fmt == "dec":
        return pd.DataFrame(sb.astype(int), index=rows, columns=cols)
    raise ValueError(f"Format tidak dikenal: {fmt!r}. Gunakan 'hex' atau 'dec'.")


def parse_uploaded_sbox(raw_bytes: bytes, filename: str) -> np.ndarray:
    """Membaca unggahan S-Box (.npy, .csv, atau .txt) dan memvalidasi bijektivitasnya."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".npy":
        arr = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
        return validate_sbox(np.asarray(arr).reshape(-1))

    text = raw_bytes.decode("utf-8", errors="strict")
    tokens = [tok for tok in re.split(r"[\s,;]+", text.strip()) if tok]
    values: list[int] = []
    for tok in tokens:
        if tok.lower().startswith("0x"):
            values.append(int(tok, 16))
        elif re.fullmatch(r"\d+", tok):
            values.append(int(tok, 10))
        elif re.fullmatch(r"[0-9a-fA-F]{1,2}", tok):
            values.append(int(tok, 16))
        else:
            continue
    if len(values) != 256:
        raise ValueError(
            f"File harus berisi tepat 256 nilai S-Box (ditemukan {len(values)} nilai valid)."
        )
    return validate_sbox(np.array(values, dtype=np.int64))


def generate_live_affine_sbox(seed: int, poly: Optional[int] = None) -> dict:
    """Membangkitkan 1 S-Box acak baru jalur Modifikasi AES GF(2^8) bebas fixed point."""
    polys = (poly,) if poly is not None else None
    candidates = generate_aes_affine_candidates(
        n=1, seed=int(seed), polys=polys, avoid_fixed_points=True
    )
    if not candidates:
        raise ValueError("Gagal membangkitkan kandidat S-Box GF(2^8) untuk parameter tersebut.")
    return candidates[0]


def normalize_16byte_secret(raw_value: str, mode: str = "utf8") -> bytes:
    """Mengonversi teks UTF-8 atau string hex (32 karakter) menjadi tepat 16 byte."""
    cleaned = raw_value.strip()
    if mode == "hex":
        compact = re.sub(r"[\s:-]+", "", cleaned)
        if len(compact) != 32:
            raise ValueError("Format hex harus berisi tepat 32 karakter heksadesimal (16 byte).")
        try:
            return bytes.fromhex(compact)
        except ValueError as exc:
            raise ValueError("Karakter hex tidak valid (gunakan hanya 0-9 dan A-F).") from exc
    if mode == "utf8":
        encoded = raw_value.encode("utf-8")
        if len(encoded) >= 16:
            return encoded[:16]
        return encoded.ljust(16, b"\x00")
    raise ValueError(f"Mode kunci tidak dikenal: {mode!r}. Pilih 'utf8' atau 'hex'.")


def _pkcs7_pad(raw: bytes, block_size: int = 16) -> bytes:
    pad_len = block_size - (len(raw) % block_size)
    return raw + bytes([pad_len] * pad_len)


def _pkcs7_unpad(padded: bytes, block_size: int = 16) -> bytes:
    if not padded or len(padded) % block_size != 0:
        raise ValueError("Panjang ciphertext/plaintext ter-pad tidak valid.")
    pad_len = padded[-1]
    if pad_len < 1 or pad_len > block_size:
        raise ValueError("Padding PKCS#7 tidak valid — pastikan kunci, IV, mode, dan S-Box sama.")
    if any(b != pad_len for b in padded[-pad_len:]):
        raise ValueError("Padding PKCS#7 rusak — dekripsi tidak cocok.")
    return padded[:-pad_len]


def encrypt_text_aes(
    plaintext: str,
    sbox: np.ndarray,
    key: bytes,
    iv: bytes,
    mode: str = "CBC2",
) -> bytes:
    """Enkripsi teks UTF-8 menggunakan AES-128 custom S-Box dengan padding PKCS#7."""
    padded = _pkcs7_pad(plaintext.encode("utf-8"), 16)
    arr = np.frombuffer(padded, dtype=np.uint8)
    cipher = aes_encrypt(arr, key=key, sbox=sbox, mode=mode, iv=iv)
    return cipher.tobytes()


def decrypt_text_aes(
    ciphertext: bytes,
    sbox: np.ndarray,
    key: bytes,
    iv: bytes,
    mode: str = "CBC2",
) -> str:
    """Dekripsi bytes ciphertext menjadi teks UTF-8 asli."""
    if not ciphertext or len(ciphertext) % 16 != 0:
        raise ValueError("Panjang ciphertext harus positif dan kelipatan 16 byte.")
    arr = np.frombuffer(ciphertext, dtype=np.uint8)
    plain_padded = aes_decrypt(arr, key=key, sbox=sbox, mode=mode, iv=iv).tobytes()
    raw = _pkcs7_unpad(plain_padded, 16)
    return raw.decode("utf-8")


def _shannon_entropy(arr: np.ndarray) -> float:
    counts = np.bincount(arr.reshape(-1), minlength=256).astype(np.float64)
    total = float(counts.sum())
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def compute_image_security_metrics(
    plain_arr: np.ndarray,
    cipher1_arr: np.ndarray,
    cipher2_arr: np.ndarray,
) -> dict:
    """Menghitung Metrik 2 (Entropy, Histogram Chi-Square) & Metrik 3 (NPCR, UACI)."""
    team_fn = _resolve_image_metrics_fn()
    if team_fn is not None:
        return dict(team_fn(plain_arr, cipher1_arr, cipher2_arr))

    p = np.asarray(plain_arr, dtype=np.uint8)
    c1 = np.asarray(cipher1_arr, dtype=np.uint8)
    c2 = np.asarray(cipher2_arr, dtype=np.uint8)

    entropy_plain = _shannon_entropy(p)
    entropy_cipher = _shannon_entropy(c1)

    hist_counts = np.bincount(c1.reshape(-1), minlength=256).astype(np.float64)
    expected = np.full(256, hist_counts.sum() / 256.0, dtype=np.float64)
    chi2_stat, p_val = chisquare(hist_counts, f_exp=expected)

    # Per-channel mean NPCR (konvensi standar Wu et al. 2011 / RGB ideal ~99.6094%)
    npcr = float(np.mean(c1 != c2) * 100.0)
    uaci = float(
        np.mean(np.abs(c1.astype(np.float64) - c2.astype(np.float64)) / 255.0) * 100.0
    )

    return {
        "entropy_plain": entropy_plain,
        "entropy_cipher": entropy_cipher,
        "hist_chi2": float(chi2_stat),
        "hist_p_value": float(p_val),
        "npcr": npcr,
        "uaci": uaci,
    }


def _resize_array_if_needed(img_arr: np.ndarray, max_dim: Optional[int]) -> np.ndarray:
    if max_dim is None or max_dim <= 0:
        return np.asarray(img_arr, dtype=np.uint8)
    h, w = img_arr.shape[:2]
    if max(h, w) <= max_dim:
        return np.asarray(img_arr, dtype=np.uint8)
    scale = max_dim / float(max(h, w))
    new_w = max(16, int(round(w * scale)))
    new_h = max(16, int(round(h * scale)))
    pil_img = Image.fromarray(np.asarray(img_arr, dtype=np.uint8))
    resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


def run_image_simulation(
    img_arr: np.ndarray,
    sbox: np.ndarray,
    key: bytes,
    iv: bytes,
    mode: str = "CBC2",
    max_dim: Optional[int] = None,
) -> dict:
    """Menjalankan simulasi enkripsi, uji difusi 1-bit (NPCR/UACI), dan dekripsi lossless."""
    plain = _resize_array_if_needed(img_arr, max_dim=max_dim)
    sb = validate_sbox(sbox)

    t0 = time.perf_counter()
    cipher1 = encrypt_image_array(plain, sbox=sb, key=key, iv=iv, mode=mode)
    encrypt_sec = time.perf_counter() - t0

    plain_flipped = plain.copy()
    if plain_flipped.ndim == 3:
        plain_flipped[-1, -1, 0] ^= np.uint8(1)
    else:
        plain_flipped[-1, -1] ^= np.uint8(1)
    cipher2 = encrypt_image_array(plain_flipped, sbox=sb, key=key, iv=iv, mode=mode)

    t1 = time.perf_counter()
    decrypted = decrypt_image_array(
        cipher1, sbox=sb, key=key, iv=iv, orig_shape=plain.shape, mode=mode
    )
    decrypt_sec = time.perf_counter() - t1

    c1_vis = cipher1[: plain.size].reshape(plain.shape) if cipher1.shape != plain.shape else cipher1
    c2_vis = cipher2[: plain.size].reshape(plain.shape) if cipher2.shape != plain.shape else cipher2

    metrics = compute_image_security_metrics(plain, c1_vis, c2_vis)
    return {
        "plain": plain,
        "cipher": c1_vis,
        "decrypted": decrypted,
        "lossless": bool(np.array_equal(plain, decrypted)),
        "encrypt_sec": float(encrypt_sec),
        "decrypt_sec": float(decrypt_sec),
        "metrics": metrics,
    }
