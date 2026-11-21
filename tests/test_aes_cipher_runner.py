import numpy as np
import pytest

from src.aes_affine_generator import build_k44_sbox, build_standard_aes_sbox
from src.aes_cipher_runner import (
    aes_decrypt,
    aes_encrypt,
    build_inverse_sbox,
    decrypt_image_array,
    encrypt_image_array,
)


def test_fips197_appendix_b_and_c1_vectors_with_standard_sbox():
    sbox = build_standard_aes_sbox()

    # FIPS-197 Appendix B
    key_b = bytes.fromhex("2b7e151628aed2a6abf7158809cf4f3c")
    pt_b = np.frombuffer(bytes.fromhex("3243f6a8885a308d313198a2e0370734"), dtype=np.uint8)
    ct_b = aes_encrypt(pt_b, key_b, sbox=sbox, mode="ECB")
    assert bytes(ct_b).hex() == "3925841d02dc09fbdc118597196a0b32"
    assert np.array_equal(aes_decrypt(ct_b, key_b, sbox=sbox, mode="ECB"), pt_b)

    # FIPS-197 Appendix C.1
    key_c = bytes(range(16))
    pt_c = np.frombuffer(bytes.fromhex("00112233445566778899aabbccddeeff"), dtype=np.uint8)
    ct_c = aes_encrypt(pt_c, key_c, sbox=sbox, mode="ECB")
    assert bytes(ct_c).hex() == "69c4e0d86a7b0430d8cdb78070b4c55a"
    assert np.array_equal(aes_decrypt(ct_c, key_c, sbox=sbox, mode="ECB"), pt_c)


def test_inverse_sbox_inverts_custom_bijective_sbox_and_rejects_non_bijective():
    k44 = build_k44_sbox()
    inv_k44 = build_inverse_sbox(k44)
    assert np.array_equal(inv_k44[k44], np.arange(256, dtype=np.uint8))
    assert np.array_equal(k44[inv_k44], np.arange(256, dtype=np.uint8))

    bad_sbox = np.zeros(256, dtype=np.uint8)
    with pytest.raises(ValueError, match="bijective"):
        build_inverse_sbox(bad_sbox)


@pytest.mark.parametrize("mode", ["ECB", "CBC", "CBC2"])
@pytest.mark.parametrize("shape", [(32, 32, 3), (33, 47)])
def test_image_array_roundtrip_is_lossless_for_custom_sbox(mode, shape):
    rng = np.random.default_rng(2026)
    plain = rng.integers(0, 256, size=shape, dtype=np.uint8)
    k44 = build_k44_sbox()
    key = bytes.fromhex("00112233445566778899aabbccddeeff")
    iv = bytes.fromhex("ffeeddccbbaa99887766554433221100")

    cipher = encrypt_image_array(plain, sbox=k44, key=key, iv=iv, mode=mode)
    assert cipher.dtype == np.uint8
    assert not np.array_equal(cipher[: plain.size].reshape(shape), plain)

    recovered = decrypt_image_array(cipher, sbox=k44, key=key, iv=iv, orig_shape=shape, mode=mode)
    assert recovered.shape == shape
    assert recovered.dtype == np.uint8
    assert np.array_equal(recovered, plain)


def test_cbc2_full_diffusion_for_single_bit_flip_even_at_last_pixel():
    rng = np.random.default_rng(99)
    plain1 = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    plain2 = plain1.copy()
    # Flip 1 LSB at the very last pixel (where 1-pass CBC would fail NPCR!)
    plain2[-1, -1, -1] ^= np.uint8(1)

    k44 = build_k44_sbox()
    key = b"KriptografiML128"
    iv = b"InitVector123456"

    c1 = encrypt_image_array(plain1, sbox=k44, key=key, iv=iv, mode="CBC2")
    c2 = encrypt_image_array(plain2, sbox=k44, key=key, iv=iv, mode="CBC2")

    npcr = float(np.mean(c1 != c2) * 100.0)
    uaci = float(np.mean(np.abs(c1.astype(np.int32) - c2.astype(np.int32)) / 255.0) * 100.0)
    assert npcr > 98.5
    assert 30.0 < uaci < 37.0


def test_rejects_invalid_key_or_missing_iv():
    sbox = build_standard_aes_sbox()
    data = np.zeros(16, dtype=np.uint8)
    with pytest.raises(ValueError, match="16 byte"):
        aes_encrypt(data, b"short_key", sbox=sbox, mode="ECB")
    with pytest.raises(ValueError, match="IV"):
        aes_encrypt(data, b"1234567890123456", sbox=sbox, mode="CBC", iv=None)
