"""Runner enkripsi & dekripsi AES-128 dengan dukungan custom S-Box.

Diadaptasi dari implementasi AES-128 dari nol (FIPS-197) agar dapat
menerima sembarang S-box bijektif (256 byte) serta mendukung mode:
- ``ECB``  : enkripsi per blok independen (untuk verifikasi vektor uji FIPS-197).
- ``CBC``  : Cipher Block Chaining standar 1 lintasan.
- ``CBC2`` : CBC 2 lintasan (maju lalu mundur) agar perubahan 1 bit di piksel
             mana pun — termasuk piksel terakhir — menyebar penuh ke seluruh citra.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.aes_affine_generator import AES_IRREDUCIBLE_POLY, build_standard_aes_sbox, gf256_mul

_MUL_NP = {
    m: np.array([gf256_mul(x, m, AES_IRREDUCIBLE_POLY) for x in range(256)], dtype=np.uint8)
    for m in (2, 3, 9, 11, 13, 14)
}
_MUL2 = _MUL_NP[2].tolist()
_MUL3 = _MUL_NP[3].tolist()
_RCON = (0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36)

# Indeks state AES-128 urut kolom: idx = 4*kolom + baris
_SHIFT = np.array([4 * ((c + r) % 4) + r for c in range(4) for r in range(4)], dtype=np.intp)
_INV_SHIFT = np.array([4 * ((c - r) % 4) + r for c in range(4) for r in range(4)], dtype=np.intp)


def validate_sbox(sbox: np.ndarray) -> np.ndarray:
    """Memastikan S-box bertipe uint8, panjang 256, dan bijektif 0..255."""
    arr = np.asarray(sbox)
    if arr.shape != (256,):
        raise ValueError(f"S-box harus berbentuk (256,), diterima {arr.shape}.")
    if arr.dtype != np.uint8:
        if np.min(arr) < 0 or np.max(arr) > 255:
            raise ValueError("Nilai elemen S-box harus berada dalam rentang 0..255.")
        arr = arr.astype(np.uint8)
    if len(set(arr.tolist())) != 256:
        raise ValueError("S-box tidak bijective: seluruh nilai 0..255 harus muncul tepat satu kali.")
    return arr


def build_inverse_sbox(sbox: np.ndarray) -> np.ndarray:
    """Membentuk Inverse S-Box dari S-Box bijektif."""
    valid = validate_sbox(sbox)
    inv = np.empty(256, dtype=np.uint8)
    inv[valid] = np.arange(256, dtype=np.uint8)
    return inv


def expand_key(key: bytes, sbox: np.ndarray) -> np.ndarray:
    """Ekspansi kunci AES-128 (16 byte -> 11 round key ukuran 16 byte)."""
    if not isinstance(key, (bytes, bytearray)) or len(key) != 16:
        raise ValueError("Kunci AES-128 harus berupa bytes dengan panjang tepat 16 byte.")
    sb = validate_sbox(sbox)
    w = [list(key[4 * i : 4 * i + 4]) for i in range(4)]
    for i in range(4, 44):
        t = w[i - 1][:]
        if i % 4 == 0:
            t = t[1:] + t[:1]
            t = [int(sb[x]) for x in t]
            t[0] ^= _RCON[i // 4 - 1]
        w.append([a ^ b for a, b in zip(w[i - 4], t)])
    return np.array([sum(w[4 * r : 4 * r + 4], []) for r in range(11)], dtype=np.uint8)


def _mix_columns(state: np.ndarray) -> np.ndarray:
    a = state.reshape(-1, 4, 4)
    s0, s1, s2, s3 = a[:, :, 0], a[:, :, 1], a[:, :, 2], a[:, :, 3]
    out = np.empty_like(a)
    out[:, :, 0] = _MUL_NP[2][s0] ^ _MUL_NP[3][s1] ^ s2 ^ s3
    out[:, :, 1] = s0 ^ _MUL_NP[2][s1] ^ _MUL_NP[3][s2] ^ s3
    out[:, :, 2] = s0 ^ s1 ^ _MUL_NP[2][s2] ^ _MUL_NP[3][s3]
    out[:, :, 3] = _MUL_NP[3][s0] ^ s1 ^ s2 ^ _MUL_NP[2][s3]
    return out.reshape(-1, 16)


def _inv_mix_columns(state: np.ndarray) -> np.ndarray:
    a = state.reshape(-1, 4, 4)
    s0, s1, s2, s3 = a[:, :, 0], a[:, :, 1], a[:, :, 2], a[:, :, 3]
    out = np.empty_like(a)
    out[:, :, 0] = _MUL_NP[14][s0] ^ _MUL_NP[11][s1] ^ _MUL_NP[13][s2] ^ _MUL_NP[9][s3]
    out[:, :, 1] = _MUL_NP[9][s0] ^ _MUL_NP[14][s1] ^ _MUL_NP[11][s2] ^ _MUL_NP[13][s3]
    out[:, :, 2] = _MUL_NP[13][s0] ^ _MUL_NP[9][s1] ^ _MUL_NP[14][s2] ^ _MUL_NP[11][s3]
    out[:, :, 3] = _MUL_NP[11][s0] ^ _MUL_NP[13][s1] ^ _MUL_NP[9][s2] ^ _MUL_NP[14][s3]
    return out.reshape(-1, 16)


def _encrypt_blocks(blocks: np.ndarray, rk: np.ndarray, sbox: np.ndarray) -> np.ndarray:
    s = blocks ^ rk[0]
    for r in range(1, 10):
        s = _mix_columns(sbox[s][:, _SHIFT]) ^ rk[r]
    return sbox[s][:, _SHIFT] ^ rk[10]


def _decrypt_blocks(blocks: np.ndarray, rk: np.ndarray, inv_sbox: np.ndarray) -> np.ndarray:
    s = blocks ^ rk[10]
    for r in range(9, 0, -1):
        s = _inv_mix_columns(inv_sbox[s[:, _INV_SHIFT]] ^ rk[r])
    return inv_sbox[s[:, _INV_SHIFT]] ^ rk[0]


def _encrypt_single_block_fast(block: list[int], rk_list: list[list[int]], sb: list[int]) -> list[int]:
    """Enkripsi 1 blok 16-byte tanpa alokasi NumPy agar loop CBC jauh lebih cepat."""
    rk0 = rk_list[0]
    s0 = block[0] ^ rk0[0]
    s1 = block[1] ^ rk0[1]
    s2 = block[2] ^ rk0[2]
    s3 = block[3] ^ rk0[3]
    s4 = block[4] ^ rk0[4]
    s5 = block[5] ^ rk0[5]
    s6 = block[6] ^ rk0[6]
    s7 = block[7] ^ rk0[7]
    s8 = block[8] ^ rk0[8]
    s9 = block[9] ^ rk0[9]
    s10 = block[10] ^ rk0[10]
    s11 = block[11] ^ rk0[11]
    s12 = block[12] ^ rk0[12]
    s13 = block[13] ^ rk0[13]
    s14 = block[14] ^ rk0[14]
    s15 = block[15] ^ rk0[15]

    m2, m3 = _MUL2, _MUL3
    for r in range(1, 10):
        # SubBytes + ShiftRows
        t0, t1, t2, t3 = sb[s0], sb[s5], sb[s10], sb[s15]
        t4, t5, t6, t7 = sb[s4], sb[s9], sb[s14], sb[s3]
        t8, t9, t10, t11 = sb[s8], sb[s13], sb[s2], sb[s7]
        t12, t13, t14, t15 = sb[s12], sb[s1], sb[s6], sb[s11]
        rk = rk_list[r]
        # MixColumns + AddRoundKey
        s0 = m2[t0] ^ m3[t1] ^ t2 ^ t3 ^ rk[0]
        s1 = t0 ^ m2[t1] ^ m3[t2] ^ t3 ^ rk[1]
        s2 = t0 ^ t1 ^ m2[t2] ^ m3[t3] ^ rk[2]
        s3 = m3[t0] ^ t1 ^ t2 ^ m2[t3] ^ rk[3]

        s4 = m2[t4] ^ m3[t5] ^ t6 ^ t7 ^ rk[4]
        s5 = t4 ^ m2[t5] ^ m3[t6] ^ t7 ^ rk[5]
        s6 = t4 ^ t5 ^ m2[t6] ^ m3[t7] ^ rk[6]
        s7 = m3[t4] ^ t5 ^ t6 ^ m2[t7] ^ rk[7]

        s8 = m2[t8] ^ m3[t9] ^ t10 ^ t11 ^ rk[8]
        s9 = t8 ^ m2[t9] ^ m3[t10] ^ t11 ^ rk[9]
        s10 = t8 ^ t9 ^ m2[t10] ^ m3[t11] ^ rk[10]
        s11 = m3[t8] ^ t9 ^ t10 ^ m2[t11] ^ rk[11]

        s12 = m2[t12] ^ m3[t13] ^ t14 ^ t15 ^ rk[12]
        s13 = t12 ^ m2[t13] ^ m3[t14] ^ t15 ^ rk[13]
        s14 = t12 ^ t13 ^ m2[t14] ^ m3[t15] ^ rk[14]
        s15 = m3[t12] ^ t13 ^ t14 ^ m2[t15] ^ rk[15]

    rk10 = rk_list[10]
    return [
        sb[s0] ^ rk10[0], sb[s5] ^ rk10[1], sb[s10] ^ rk10[2], sb[s15] ^ rk10[3],
        sb[s4] ^ rk10[4], sb[s9] ^ rk10[5], sb[s14] ^ rk10[6], sb[s3] ^ rk10[7],
        sb[s8] ^ rk10[8], sb[s13] ^ rk10[9], sb[s2] ^ rk10[10], sb[s7] ^ rk10[11],
        sb[s12] ^ rk10[12], sb[s1] ^ rk10[13], sb[s6] ^ rk10[14], sb[s11] ^ rk10[15],
    ]


def _cbc_encrypt_pass(blocks: np.ndarray, rk: np.ndarray, sbox: np.ndarray, iv_bytes: bytes) -> np.ndarray:
    rk_list = rk.tolist()
    sb_list = sbox.tolist()
    prev = list(iv_bytes)
    blocks_list = blocks.tolist()
    out_list = [None] * len(blocks_list)
    for idx, blk in enumerate(blocks_list):
        xored = [a ^ b for a, b in zip(blk, prev)]
        prev = _encrypt_single_block_fast(xored, rk_list, sb_list)
        out_list[idx] = prev
    return np.array(out_list, dtype=np.uint8)


def _cbc_decrypt_pass(blocks: np.ndarray, rk: np.ndarray, inv_sbox: np.ndarray, iv_bytes: bytes) -> np.ndarray:
    iv_arr = np.frombuffer(iv_bytes, dtype=np.uint8)[None, :]
    prev = np.vstack([iv_arr, blocks[:-1]])
    return _decrypt_blocks(blocks, rk, inv_sbox) ^ prev


def _validate_iv(iv: Optional[bytes]) -> bytes:
    if iv is None or not isinstance(iv, (bytes, bytearray)) or len(iv) != 16:
        raise ValueError("Mode CBC/CBC2 memerlukan IV sepanjang tepat 16 byte.")
    return bytes(iv)


def aes_encrypt(
    data: np.ndarray,
    key: bytes,
    sbox: Optional[np.ndarray] = None,
    mode: str = "CBC2",
    iv: Optional[bytes] = None,
) -> np.ndarray:
    """Enkripsi array 1D uint8 berkelipatan 16 byte menggunakan AES-128 custom S-Box."""
    arr = np.asarray(data, dtype=np.uint8).reshape(-1)
    if arr.size == 0 or arr.size % 16 != 0:
        raise ValueError("Panjang data harus positif dan merupakan kelipatan 16 byte.")
    sb = build_standard_aes_sbox() if sbox is None else validate_sbox(sbox)
    rk = expand_key(key, sb)
    blocks = arr.reshape(-1, 16)

    if mode == "ECB":
        return _encrypt_blocks(blocks, rk, sb).reshape(-1)
    if mode == "CBC":
        iv_b = _validate_iv(iv)
        return _cbc_encrypt_pass(blocks, rk, sb, iv_b).reshape(-1)
    if mode == "CBC2":
        iv_b = _validate_iv(iv)
        iv_rev = bytes(b ^ 0xA5 for b in iv_b)
        pass1 = _cbc_encrypt_pass(blocks, rk, sb, iv_b)
        pass2_rev = _cbc_encrypt_pass(pass1[::-1], rk, sb, iv_rev)
        return pass2_rev[::-1].reshape(-1)
    raise ValueError(f"Mode tidak dikenal: {mode!r}. Pilih 'ECB', 'CBC', atau 'CBC2'.")


def aes_decrypt(
    data: np.ndarray,
    key: bytes,
    sbox: Optional[np.ndarray] = None,
    mode: str = "CBC2",
    iv: Optional[bytes] = None,
) -> np.ndarray:
    """Dekripsi array 1D uint8 berkelipatan 16 byte menggunakan AES-128 custom S-Box."""
    arr = np.asarray(data, dtype=np.uint8).reshape(-1)
    if arr.size == 0 or arr.size % 16 != 0:
        raise ValueError("Panjang data harus positif dan merupakan kelipatan 16 byte.")
    sb = build_standard_aes_sbox() if sbox is None else validate_sbox(sbox)
    inv_sb = build_inverse_sbox(sb)
    rk = expand_key(key, sb)
    blocks = arr.reshape(-1, 16)

    if mode == "ECB":
        return _decrypt_blocks(blocks, rk, inv_sb).reshape(-1)
    if mode == "CBC":
        iv_b = _validate_iv(iv)
        return _cbc_decrypt_pass(blocks, rk, inv_sb, iv_b).reshape(-1)
    if mode == "CBC2":
        iv_b = _validate_iv(iv)
        iv_rev = bytes(b ^ 0xA5 for b in iv_b)
        pass1_rev = _cbc_decrypt_pass(blocks[::-1], rk, inv_sb, iv_rev)
        pass1 = pass1_rev[::-1]
        return _cbc_decrypt_pass(pass1, rk, inv_sb, iv_b).reshape(-1)
    raise ValueError(f"Mode tidak dikenal: {mode!r}. Pilih 'ECB', 'CBC', atau 'CBC2'.")


def encrypt_image_array(
    img_arr: np.ndarray,
    sbox: np.ndarray,
    key: bytes,
    iv: bytes,
    mode: str = "CBC2",
) -> np.ndarray:
    """Enkripsi array citra uint8 memakai AES-128 custom S-Box (default CBC2 difusi penuh)."""
    arr = np.asarray(img_arr, dtype=np.uint8)
    flat = arr.reshape(-1)
    padded_len = ((flat.size + 15) // 16) * 16
    padded = np.zeros(padded_len, dtype=np.uint8)
    padded[: flat.size] = flat
    cipher = aes_encrypt(padded, key=key, sbox=sbox, mode=mode, iv=iv)
    if padded_len == flat.size:
        return cipher.reshape(arr.shape)
    return cipher


def decrypt_image_array(
    cipher_arr: np.ndarray,
    sbox: np.ndarray,
    key: bytes,
    iv: bytes,
    orig_shape: tuple[int, ...],
    mode: str = "CBC2",
) -> np.ndarray:
    """Dekripsi kembali ke dimensi asli `orig_shape` secara lossless."""
    flat_cipher = np.asarray(cipher_arr, dtype=np.uint8).reshape(-1)
    plain_padded = aes_decrypt(flat_cipher, key=key, sbox=sbox, mode=mode, iv=iv)
    orig_size = int(np.prod(orig_shape))
    return plain_padded[:orig_size].reshape(orig_shape)
