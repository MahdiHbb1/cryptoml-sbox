import numpy as np
import pytest

from src.image_metrics import (
    bit_change_rate,
    evaluate_image_security,
    histogram_chi_square,
    npcr_uaci,
    shannon_entropy,
)

CONTRACT_KEYS = {
    "entropy_plain",
    "entropy_cipher",
    "hist_chi2",
    "hist_p_value",
    "npcr",
    "uaci",
}


def test_evaluate_image_security_mengembalikan_kunci_kontrak():
    rng = np.random.default_rng(42)
    plain = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    cipher1 = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    cipher2 = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)

    hasil = evaluate_image_security(plain, cipher1, cipher2)

    assert set(hasil) == CONTRACT_KEYS
    for nilainya in hasil.values():
        assert isinstance(nilainya, float)
    assert 0.0 <= hasil["entropy_plain"] <= 8.0
    assert 0.0 <= hasil["entropy_cipher"] <= 8.0
    assert hasil["hist_chi2"] >= 0.0
    assert 0.0 <= hasil["hist_p_value"] <= 1.0
    assert 0.0 <= hasil["npcr"] <= 100.0
    assert 0.0 <= hasil["uaci"] <= 100.0


def test_cipher_identik_menghasilkan_metrik_nol():
    plain = np.full((16, 16), 128, dtype=np.uint8)
    hasil = evaluate_image_security(plain, plain, plain.copy())

    assert hasil["entropy_plain"] == pytest.approx(0.0, abs=1e-12)
    assert hasil["entropy_cipher"] == pytest.approx(0.0, abs=1e-12)
    assert hasil["npcr"] == pytest.approx(0.0, abs=1e-12)
    assert hasil["uaci"] == pytest.approx(0.0, abs=1e-12)


def test_npcr_rata_rata_per_kanal_bukan_gabungan_piksel():
    c1 = np.zeros((4, 4, 3), dtype=np.uint8)
    c2 = c1.copy()
    c2[:, :, 0] = 1  # hanya 1 dari 3 kanal berubah -> (100 + 0 + 0) / 3

    npcr, uaci = npcr_uaci(c1, c2)

    assert npcr == pytest.approx(100.0 / 3.0, abs=1e-9)
    assert uaci == pytest.approx((100.0 / 255.0) / 3.0, abs=1e-9)


def test_uaci_kontras_penuh_mencapai_100_persen():
    c1 = np.zeros((8, 8), dtype=np.uint8)
    c2 = np.full((8, 8), 255, dtype=np.uint8)

    npcr, uaci = npcr_uaci(c1, c2)

    assert npcr == pytest.approx(100.0)
    assert uaci == pytest.approx(100.0)


def test_entropy_histogram_uniform_penuh_adalah_8_bit():
    img = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    assert shannon_entropy(img) == pytest.approx(8.0, abs=1e-12)


def test_entropy_citra_konstan_adalah_nol():
    img = np.full((32, 32, 3), 7, dtype=np.uint8)
    assert shannon_entropy(img) == pytest.approx(0.0, abs=1e-12)


def test_entropy_per_kanal_mengembalikan_satu_nilai_per_kanal():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :, 1] = np.tile(np.arange(4, dtype=np.uint8), (4, 1))
    img[:, :, 2] = np.arange(16, dtype=np.uint8).reshape(4, 4)

    per_kanal = shannon_entropy(img, per_channel=True)

    assert per_kanal.shape == (3,)
    assert per_kanal[0] == pytest.approx(0.0, abs=1e-12)
    assert per_kanal[1] == pytest.approx(2.0, abs=1e-12)
    assert per_kanal[2] == pytest.approx(4.0, abs=1e-12)
    assert shannon_entropy(img) < 4.0


def test_chi_square_histogram_sempurna_uniform_mendekati_nol():
    img = np.tile(np.arange(256, dtype=np.uint8), (4, 1))

    stat, p_value = histogram_chi_square(img)

    assert stat == pytest.approx(0.0, abs=1e-9)
    assert p_value == pytest.approx(1.0, abs=1e-9)


def test_chi_square_menolak_citra_kosong():
    with pytest.raises(ValueError):
        histogram_chi_square(np.zeros((0, 0), dtype=np.uint8))


def test_nbcr_menghitung_hamming_distance_bit():
    c1 = np.zeros((2, 2), dtype=np.uint8)
    c2 = c1.copy()
    c2[0, 0] = 0b0000_0111  # 3 bit berbeda dari 32 bit total

    assert bit_change_rate(c1, c2) == pytest.approx(3 / 32 * 100)


def test_dimensi_cipher_berbeda_ditolak():
    c1 = np.zeros((8, 8), dtype=np.uint8)
    c2 = np.zeros((8, 16), dtype=np.uint8)

    with pytest.raises(ValueError):
        npcr_uaci(c1, c2)
    with pytest.raises(ValueError):
        bit_change_rate(c1, c2)
    with pytest.raises(ValueError):
        evaluate_image_security(c1, c1, c2)


def test_input_bukan_uint8_dinormalkan_dan_float_ditolak():
    img_int = np.array([[0, 1], [2, 255]], dtype=np.int64)
    assert shannon_entropy(img_int) > 0.0

    with pytest.raises(TypeError):
        shannon_entropy(np.zeros((4, 4), dtype=np.float32))


def test_citra_berdimensi_tidak_valid_ditolak():
    with pytest.raises(ValueError):
        shannon_entropy(np.zeros(16, dtype=np.uint8))
    with pytest.raises(ValueError):
        shannon_entropy(np.zeros((4, 4, 2), dtype=np.uint8))
