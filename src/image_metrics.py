"""Evaluator Metrik 2 (keacakan citra) & Metrik 3 (performa enkripsi) untuk Mandrill.

Modul ini adalah implementasi Kontrak §5 ``docs/ARCHITECTURE_CONTRACT.md``:

    evaluate_image_security(plain_arr, cipher1_arr, cipher2_arr) -> dict

``src.web_helpers.compute_image_security_metrics`` otomatis memakai fungsi ini
begitu modul tersedia, sehingga angka di aplikasi Streamlit, berkas JSON
eksperimen, dan tabel laporan akhir selalu berasal dari satu sumber hitung.

Konvensi yang disepakati tim:

- **Entropy**    : Shannon entropy 8-bit dari seluruh elemen citra (semua kanal
                   digabung), nilai ideal 8.0 bit.
- **Chi-square** : histogram 256 bin ``cipher1_arr`` diuji terhadap distribusi
                   uniform (derajat kebebasan 255). ``hist_chi2`` kecil dan
                   ``hist_p_value`` besar berarti histogram cipher rata.
- **NPCR**       : rata-rata per kanal dari persentase piksel yang berubah,
                   nilai ideal 99.6094% (konvensi Wu et al., 2011).
- **UACI**       : rata-rata per kanal dari selisih intensitas ternormalisasi,
                   nilai ideal 33.4635%.
- **NBCR**       : rasio bit yang berubah (Hamming distance) antar dua cipher
                   image, nilai ideal 50%. Disediakan sebagai metrik pendukung
                   selama status NPCR/NBCR masih dikonfirmasi ke dosen.

Contoh::

    from src.image_metrics import evaluate_image_security

    hasil = evaluate_image_security(plain, cipher_asli, cipher_varian)
    print(hasil["npcr"], hasil["uaci"])
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chisquare

HIST_BINS = 256
BITS_PER_BYTE = 8
MAX_INTENSITY = 255.0

# Nilai acuan teoretis untuk pelaporan, bukan hasil perhitungan.
ENTROPY_IDEAL = 8.0
NPCR_IDEAL = 99.609375
UACI_IDEAL = 33.4635
NBCR_IDEAL = 50.0
CORRELATION_IDEAL = 0.0


def _as_image(arr: np.ndarray, name: str) -> np.ndarray:
    """Memvalidasi dan menormalkan citra menjadi ``np.uint8`` 2D/3D."""
    img = np.asarray(arr)
    if img.ndim not in (2, 3):
        raise ValueError(f"{name} harus berupa citra 2D (grayscale) atau 3D (berkanal), diterima {img.shape}.")
    if img.ndim == 3 and img.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} harus memiliki 1, 3, atau 4 kanal, diterima {img.shape[2]}.")
    if img.size == 0:
        raise ValueError(f"{name} tidak boleh kosong.")
    if img.dtype != np.uint8:
        if not np.issubdtype(img.dtype, np.integer):
            raise TypeError(f"{name} harus bertipe integer/uint8, diterima {img.dtype}.")
        if int(img.min()) < 0 or int(img.max()) > 255:
            raise ValueError(f"{name} harus berisi nilai 0..255.")
        img = img.astype(np.uint8)
    return img


def _per_channel(img: np.ndarray) -> list[np.ndarray]:
    """Memecah citra menjadi daftar kanal 2D (grayscale menghasilkan 1 kanal)."""
    if img.ndim == 2:
        return [img]
    return [img[:, :, ch] for ch in range(img.shape[2])]


def _entropy_1d(values: np.ndarray) -> float:
    counts = np.bincount(values.reshape(-1), minlength=HIST_BINS).astype(np.float64)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def shannon_entropy(img_arr: np.ndarray, per_channel: bool = False):
    """Shannon entropy 8-bit citra.

    Args:
        img_arr: Citra 2D atau 3D bertipe ``np.uint8``.
        per_channel: Bila ``True`` dan citra berkanal, kembalikan ``np.ndarray``
            berisi entropy tiap kanal; bila ``False``, seluruh kanal digabung.

    Returns:
        ``float`` bila ``per_channel=False``, selain itu ``np.ndarray`` per kanal.
    """
    img = _as_image(img_arr, "img_arr")
    if per_channel and img.ndim == 3:
        return np.array([_entropy_1d(ch) for ch in _per_channel(img)], dtype=np.float64)
    return _entropy_1d(img)


def histogram_chi_square(img_arr: np.ndarray) -> tuple[float, float]:
    """Uji Chi-square histogram 256 bin terhadap distribusi uniform.

    Nilai p kecil (umumnya < 0.05) menandakan histogram cipher menyimpang dari
    uniform; cipher image berkualitas tinggi menghasilkan p yang besar.

    Returns:
        Tuple ``(chi2_statistic, p_value)`` dengan derajat kebebasan 255.
    """
    img = _as_image(img_arr, "img_arr")
    counts = np.bincount(img.reshape(-1), minlength=HIST_BINS).astype(np.float64)
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("Citra kosong tidak dapat diuji Chi-square.")
    expected = np.full(HIST_BINS, total / HIST_BINS, dtype=np.float64)
    statistic, p_value = chisquare(counts, f_exp=expected)
    return float(statistic), float(p_value)


def _check_pair(cipher1_arr: np.ndarray, cipher2_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c1 = _as_image(cipher1_arr, "cipher1_arr")
    c2 = _as_image(cipher2_arr, "cipher2_arr")
    if c1.shape != c2.shape:
        raise ValueError(f"Dimensi kedua cipher image harus identik, diterima {c1.shape} dan {c2.shape}.")
    return c1, c2


def npcr_uaci(cipher1_arr: np.ndarray, cipher2_arr: np.ndarray) -> tuple[float, float]:
    """Menghitung NPCR dan UACI (%) dengan rata-rata per kanal.

    Untuk citra RGB, NPCR tiap kanal dihitung terpisah lalu dirata-ratakan agar
    nilainya dapat dibandingkan langsung dengan nilai ideal 99.6094%, bukan
    dirata-ratakan sebagai satu himpunan piksel.
    """
    c1, c2 = _check_pair(cipher1_arr, cipher2_arr)
    npcr_scores: list[float] = []
    uaci_scores: list[float] = []
    for ch1, ch2 in zip(_per_channel(c1), _per_channel(c2)):
        npcr_scores.append(float(np.mean(ch1 != ch2)) * 100.0)
        selisih = np.abs(ch1.astype(np.float64) - ch2.astype(np.float64))
        uaci_scores.append(float(np.mean(selisih / MAX_INTENSITY)) * 100.0)
    return float(np.mean(npcr_scores)), float(np.mean(uaci_scores))


def bit_change_rate(cipher1_arr: np.ndarray, cipher2_arr: np.ndarray) -> float:
    """NBCR (%) — persentase bit yang berbeda antara dua cipher image."""
    c1, c2 = _check_pair(cipher1_arr, cipher2_arr)
    xor_result = np.bitwise_xor(c1, c2)
    changed_bits = int(np.unpackbits(xor_result.reshape(-1)).sum())
    total_bits = c1.size * BITS_PER_BYTE
    if total_bits == 0:
        return 0.0
    return float(changed_bits / total_bits) * 100.0


def _pearson_adjacent(a: np.ndarray, b: np.ndarray) -> float:
    """Koefisien korelasi Pearson deterministik antar seluruh pasangan piksel."""
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    dx = x - x.mean()
    dy = y - y.mean()
    denom = float(np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)))
    if denom <= 0.0:
        return 0.0
    return float(np.clip(np.dot(dx, dy) / denom, -1.0, 1.0))


def pixel_correlation(img_arr: np.ndarray) -> dict[str, float]:
    """Menghitung korelasi piksel bertetangga (horizontal, vertikal, diagonal).

    Dihitung secara deterministik atas seluruh pasangan piksel bertetangga per kanal
    lalu dirata-ratakan lintas kanal (citra grayscale memiliki 1 kanal). Citra konstan
    dengan variansi nol mengembalikan 0.0 agar nilai selalu berhingga dan aman untuk JSON.

    Returns:
        ``dict`` dengan kunci ``horizontal``, ``vertical``, ``diagonal``, dan ``mean_abs``.
    """
    img = _as_image(img_arr, "img_arr")
    height, width = img.shape[:2]
    if height < 2 or width < 2:
        raise ValueError(
            f"Citra harus berukuran minimal 2x2 untuk korelasi piksel, diterima {img.shape}."
        )

    h_vals: list[float] = []
    v_vals: list[float] = []
    d_vals: list[float] = []
    for ch in _per_channel(img):
        h_vals.append(_pearson_adjacent(ch[:, :-1], ch[:, 1:]))
        v_vals.append(_pearson_adjacent(ch[:-1, :], ch[1:, :]))
        d_vals.append(_pearson_adjacent(ch[:-1, :-1], ch[1:, 1:]))

    horiz = float(np.mean(h_vals))
    vert = float(np.mean(v_vals))
    diag = float(np.mean(d_vals))
    mean_abs = float((abs(horiz) + abs(vert) + abs(diag)) / 3.0)
    return {
        "horizontal": horiz,
        "vertical": vert,
        "diagonal": diag,
        "mean_abs": mean_abs,
    }


def evaluate_image_security(
    plain_arr: np.ndarray,
    cipher1_arr: np.ndarray,
    cipher2_arr: np.ndarray,
) -> dict:
    """Menghitung Metrik 2 & 3 sesuai Kontrak §5.

    Args:
        plain_arr: Citra plaintext (dipakai untuk entropy plaintext).
        cipher1_arr: Hasil enkripsi ``plain_arr`` dengan kunci & IV tertentu.
        cipher2_arr: Hasil enkripsi plaintext yang berbeda 1 bit pada 1 piksel,
            memakai kunci & IV yang sama persis.

    Returns:
        ``dict`` dengan kunci: ``entropy_plain``, ``entropy_cipher``,
        ``hist_chi2``, ``hist_p_value``, ``npcr``, ``uaci``.
    """
    plain = _as_image(plain_arr, "plain_arr")
    c1, c2 = _check_pair(cipher1_arr, cipher2_arr)

    chi2_stat, p_value = histogram_chi_square(c1)
    npcr, uaci = npcr_uaci(c1, c2)

    return {
        "entropy_plain": shannon_entropy(plain),
        "entropy_cipher": shannon_entropy(c1),
        "hist_chi2": chi2_stat,
        "hist_p_value": p_value,
        "npcr": npcr,
        "uaci": uaci,
    }
