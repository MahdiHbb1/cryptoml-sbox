import json

import numpy as np
import pytest
from PIL import Image

from src.aes_affine_generator import build_k44_sbox, build_standard_aes_sbox
from src.aes_cipher_runner import describe_cipher_variant
from src.mandrill_eval import (
    DEFAULT_KEY,
    DEFAULT_IV,
    collect_sboxes,
    evaluate_sbox_on_image,
    format_markdown_table,
    load_image,
    resolve_flip_position,
    run_mandrill_evaluation,
    ubah_satu_bit,
    write_report,
)

CONTRACT_KEYS = {
    "entropy_plain",
    "entropy_cipher",
    "hist_chi2",
    "hist_p_value",
    "npcr",
    "uaci",
}


def _citra_acak(shape=(16, 16, 3), seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, shape, dtype=np.uint8)


def _simpan_citra(tmp_path, arr, name="mandril_uji.png"):
    path = tmp_path / name
    Image.fromarray(arr).save(path)
    return path


def _folder_results_dengan_xgb(tmp_path):
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    np.save(res_dir / "sbox_xgb.npy", build_standard_aes_sbox())
    return res_dir


def test_kunci_dan_iv_default_berukuran_16_byte():
    assert len(DEFAULT_KEY) == 16
    assert len(DEFAULT_IV) == 16


def test_ubah_satu_bit_hanya_mengubah_satu_bit_dan_tidak_mutasi_asli():
    img = np.zeros((8, 8, 3), dtype=np.uint8)

    hasil = ubah_satu_bit(img, x=3, y=2, channel=2, bit_pos=0)

    assert hasil[2, 3, 2] == 1
    assert img[2, 3, 2] == 0  # citra asli tidak ikut berubah
    assert int(hasil.sum()) == 1

    hasil_msb = ubah_satu_bit(img, x=3, y=2, channel=2, bit_pos=7)
    assert hasil_msb[2, 3, 2] == 128


def test_ubah_satu_bit_grayscale_mengabaikan_kanal():
    img = np.zeros((4, 4), dtype=np.uint8)
    hasil = ubah_satu_bit(img, x=1, y=1, channel=2, bit_pos=4)
    assert hasil[1, 1] == 16


def test_ubah_satu_bit_menolak_koordinat_dan_bit_tidak_valid():
    img = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        ubah_satu_bit(img, x=4, y=0)
    with pytest.raises(ValueError):
        ubah_satu_bit(img, x=0, y=-1)
    with pytest.raises(ValueError):
        ubah_satu_bit(img, x=0, y=0, bit_pos=8)
    with pytest.raises(ValueError):
        ubah_satu_bit(np.zeros((4, 4, 3), dtype=np.uint8), x=0, y=0, channel=3)


def test_resolve_flip_position_default_piksel_terakhir():
    assert resolve_flip_position((10, 20, 3)) == (19, 9)
    assert resolve_flip_position((10, 20), flip_x=4, flip_y=5) == (4, 5)
    with pytest.raises(ValueError):
        resolve_flip_position((10, 20), flip_x=20, flip_y=0)


def test_evaluate_sbox_cbc2_lossless_dan_difusi_tinggi():
    plain = _citra_acak()
    sbox = build_standard_aes_sbox()

    hasil = evaluate_sbox_on_image(plain, sbox, cipher_mode="CBC2")

    assert hasil["lossless"] is True
    assert set(hasil["metrics"]) == CONTRACT_KEYS
    assert hasil["metrics"]["npcr"] > 99.0
    assert 25.0 < hasil["metrics"]["uaci"] < 42.0
    assert hasil["plain_shape"] == [16, 16, 3]
    assert hasil["cipher_shape"] == [16, 16, 3]
    assert hasil["flip"]["x"] == 15 and hasil["flip"]["y"] == 15
    assert hasil["flip"]["pixel_plain"] != hasil["flip"]["pixel_flipped"]
    assert hasil["encrypt_sec"] >= 0.0 and hasil["decrypt_sec"] >= 0.0


def test_evaluate_sbox_grayscale_lossless_dan_flip_mengikuti_koordinat():
    plain = _citra_acak(shape=(16, 16), seed=11)

    hasil = evaluate_sbox_on_image(
        plain, build_standard_aes_sbox(), flip_x=4, flip_y=6, flip_channel=2, flip_bit=0
    )

    assert hasil["lossless"] is True
    assert hasil["flip"]["channel"] == 0  # grayscale hanya punya satu kanal
    assert (hasil["flip"]["x"], hasil["flip"]["y"]) == (4, 6)
    assert hasil["metrics"]["npcr"] > 95.0


def test_collect_sboxes_melaporkan_berkas_yang_belum_ada(tmp_path):
    res_dir = _folder_results_dengan_xgb(tmp_path)

    catalog, warnings = collect_sboxes(res_dir)

    assert list(catalog) == ["AES Standar (FIPS-197)", "K44 Alamsyah", "S-Box XGBoost"]
    assert any("sbox_rf.npy" in pesan for pesan in warnings)
    assert any("sbox_mlp.npy" in pesan for pesan in warnings)
    assert catalog["S-Box XGBoost"].shape == (256,)


def test_collect_sboxes_menolak_berkas_npy_tidak_valid(tmp_path):
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    np.save(res_dir / "sbox_rf.npy", np.zeros(256, dtype=np.uint8))  # tidak bijektif

    catalog, warnings = collect_sboxes(res_dir)

    assert "S-Box Random Forest" not in catalog
    assert any("sbox_rf.npy" in pesan for pesan in warnings)


def test_load_image_rgb_dan_grayscale(tmp_path):
    arr = _citra_acak()
    path = _simpan_citra(tmp_path, arr)

    rgb = load_image(path, mode="RGB")
    grayscale = load_image(path, mode="GRAY")

    assert rgb.shape == (16, 16, 3)
    assert grayscale.shape == (16, 16)
    with pytest.raises(ValueError):
        load_image(path, mode="CMYK")
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "tidak_ada.png")


def test_run_mandrill_evaluation_menghasilkan_laporan_lengkap(tmp_path):
    arr = _citra_acak()
    path = _simpan_citra(tmp_path, arr)
    res_dir = _folder_results_dengan_xgb(tmp_path)

    laporan = run_mandrill_evaluation(image_path=path, results_dir=res_dir)

    assert set(laporan["images"]) == {"RGB", "GRAY"}
    assert len(laporan["runs"]) == 3 * 2  # 2 baseline + XGB, dikali 2 mode citra
    assert all(run["lossless"] for run in laporan["runs"])
    assert all(set(run["metrics"]) == CONTRACT_KEYS for run in laporan["runs"])
    assert laporan["cipher_mode"] == "CBC2"
    assert laporan["key_hex"] == DEFAULT_KEY.hex()
    assert laporan["iv_hex"] == DEFAULT_IV.hex()
    assert laporan["ideal"]["npcr"] == pytest.approx(99.609375)
    assert laporan["warnings"]

    tabel = format_markdown_table(laporan)
    baris_hasil = [baris for baris in tabel.splitlines() if baris.endswith("| Yes |")]
    assert len(baris_hasil) == 6  # 3 S-Box x 2 mode citra
    assert tabel.splitlines()[0].startswith("| S-Box |")
    assert "Grayscale" in tabel


def test_write_report_menyimpan_json_yang_dapat_dibaca(tmp_path):
    arr = _citra_acak(shape=(16, 16), seed=3)
    path = _simpan_citra(tmp_path, arr, name="gray.png")
    res_dir = _folder_results_dengan_xgb(tmp_path)
    laporan = run_mandrill_evaluation(image_path=path, results_dir=res_dir, modes=["GRAY"])

    out = write_report(laporan, tmp_path / "keluaran" / "mandrill_metrics.json")
    isi = json.loads(out.read_text(encoding="utf-8"))

    assert isi["runs"][0]["image_mode"] == "GRAY"
    assert isi["image_path"] == path.as_posix()
    assert len(isi["runs"]) == 3


def test_evaluate_sbox_mencatat_korelasi_piksel_waktu_tunggal_dan_label_aes_modifikasi():
    y, x = np.meshgrid(np.linspace(0, 255, 32), np.linspace(0, 255, 32), indexing="ij")
    base = np.clip(np.rint(0.6 * x + 0.4 * y), 0, 255).astype(np.uint8)
    plain = np.stack([base, np.roll(base, 3, axis=1), 255 - base], axis=2)

    custom_sbox = np.random.default_rng(123).permutation(256).astype(np.uint8)
    hasil = evaluate_sbox_on_image(plain, custom_sbox, cipher_mode="CBC2")

    assert hasil["lossless"] is True
    assert set(hasil["metrics"]) == CONTRACT_KEYS
    assert 0.0 <= hasil["encrypt_single_sec"] <= hasil["encrypt_sec"]

    corr = hasil["pixel_correlation"]
    assert set(corr) == {"plain", "cipher"}
    assert set(corr["plain"]) == {"horizontal", "vertical", "diagonal", "mean_abs"}
    assert set(corr["cipher"]) == {"horizontal", "vertical", "diagonal", "mean_abs"}
    assert corr["plain"]["mean_abs"] > 0.90
    assert corr["cipher"]["mean_abs"] < 0.25

    variant = hasil["cipher_variant"]
    assert variant["is_standard_sbox"] is False
    assert "Modifikasi" in variant["cipher_label"]
    assert variant["sbox_applied_to"] == ["SubBytes", "InvSubBytes", "KeyExpansion.SubWord"]
    assert variant["uses_standard_mixcolumns_gf256"] is True


def test_describe_cipher_variant_membedakan_aes_standar_dan_modifikasi():
    aes_sb = build_standard_aes_sbox()
    k44_sb = build_k44_sbox()

    std_cbc = describe_cipher_variant(aes_sb, mode="CBC")
    std_cbc2 = describe_cipher_variant(aes_sb, mode="CBC2")
    mod_cbc2 = describe_cipher_variant(k44_sb, mode="CBC2")

    assert std_cbc["is_standard_sbox"] is True
    assert "Standar" in std_cbc["cipher_label"]
    assert "Modifikasi" not in std_cbc["cipher_label"]

    assert std_cbc2["is_standard_sbox"] is True
    assert "CBC2" in std_cbc2["cipher_label"]
    assert "Modifikasi" in std_cbc2["cipher_label"]

    assert mod_cbc2["is_standard_sbox"] is False
    assert "Custom S-Box" in mod_cbc2["cipher_label"]
    assert "Modifikasi" in mod_cbc2["cipher_label"]
