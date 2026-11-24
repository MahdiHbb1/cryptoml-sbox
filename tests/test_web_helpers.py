"""Unit test untuk helper aplikasi web Streamlit CryptoML-SBox."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from src.aes_affine_generator import build_k44_sbox, build_standard_aes_sbox
from src.web_helpers import (
    compute_image_security_metrics,
    compute_sac_matrix,
    decrypt_text_aes,
    encrypt_text_aes,
    evaluate_sbox_full,
    format_sbox_grid,
    list_available_sboxes,
    normalize_16byte_secret,
    parse_uploaded_sbox,
    run_image_simulation,
)


def test_list_available_sboxes_includes_baselines_and_saved_xgb(tmp_path: Path):
    sbox_xgb = build_k44_sbox()
    np.save(tmp_path / "sbox_xgb.npy", sbox_xgb)

    catalog = list_available_sboxes(results_dir=tmp_path)
    assert "AES Standar (FIPS-197)" in catalog
    assert "K44 Alamsyah (2024/2025)" in catalog
    assert "S-Box XGBoost (Mahdi)" in catalog
    for entry in catalog.values():
        assert entry["sbox"].shape == (256,)
        assert entry["sbox"].dtype == np.uint8
        assert len(set(entry["sbox"].tolist())) == 256


def test_evaluate_sbox_full_and_sac_matrix_match_standard_aes():
    aes = build_standard_aes_sbox()
    metrics = evaluate_sbox_full(aes)
    sac = compute_sac_matrix(aes)

    assert metrics["bijective"] is True
    assert metrics["balanced"] is True
    assert metrics["nl"] == 112
    assert metrics["du"] == 4
    assert metrics["fixed_points"] == 0
    assert metrics["opposite_fixed_points"] == 0
    assert "composite_score" in metrics
    assert sac.shape == (8, 8)
    assert np.isclose(float(np.mean(sac)), metrics["sac_mean"], atol=1e-9)


def test_format_sbox_grid_supports_hex_and_decimal():
    aes = build_standard_aes_sbox()
    df_hex = format_sbox_grid(aes, fmt="hex")
    df_dec = format_sbox_grid(aes, fmt="dec")

    assert df_hex.shape == (16, 16)
    assert df_dec.shape == (16, 16)
    assert df_hex.iloc[0, 0] == "63"
    assert int(df_dec.iloc[0, 0]) == 99


def test_parse_uploaded_sbox_reads_npy_and_csv_and_rejects_invalid():
    aes = build_standard_aes_sbox()

    buf = io.BytesIO()
    np.save(buf, aes)
    parsed_npy = parse_uploaded_sbox(buf.getvalue(), "custom.npy")
    assert np.array_equal(parsed_npy, aes)

    csv_bytes = ",".join(f"0x{int(v):02X}" for v in aes).encode("utf-8")
    parsed_csv = parse_uploaded_sbox(csv_bytes, "custom.csv")
    assert np.array_equal(parsed_csv, aes)

    bad_csv = ",".join(["0"] * 256).encode("utf-8")
    with pytest.raises(ValueError, match="bijective"):
        parse_uploaded_sbox(bad_csv, "bad.csv")


def test_normalize_16byte_secret_supports_utf8_and_hex():
    k_utf8 = normalize_16byte_secret("CryptoML-2026", mode="utf8")
    assert len(k_utf8) == 16

    k_hex = normalize_16byte_secret("000102030405060708090a0b0c0d0e0f", mode="hex")
    assert k_hex == bytes(range(16))

    with pytest.raises(ValueError):
        normalize_16byte_secret("not-hex!", mode="hex")


@pytest.mark.parametrize("mode", ["ECB", "CBC", "CBC2"])
@pytest.mark.parametrize(
    "text",
    [
        "",
        "Halo Kriptografi UNNES!",
        "Uji S-Box XGBoost (0x13F, C=0x02) — blok tepat 16 byte & UTF-8 panjang.",
    ],
)
def test_encrypt_decrypt_text_roundtrip_is_exact(mode: str, text: str):
    sbox = build_k44_sbox()
    key = normalize_16byte_secret("KunciRahasia2026", mode="utf8")
    iv = normalize_16byte_secret("InitVector2026!!", mode="utf8")

    cipher = encrypt_text_aes(text, sbox=sbox, key=key, iv=iv, mode=mode)
    assert len(cipher) >= 16 and len(cipher) % 16 == 0
    recovered = decrypt_text_aes(cipher, sbox=sbox, key=key, iv=iv, mode=mode)
    assert recovered == text


def test_run_image_simulation_cbc2_is_lossless_with_high_diffusion():
    rng = np.random.default_rng(7)
    plain = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    sbox = build_standard_aes_sbox()
    key = normalize_16byte_secret("KunciMandrill128", mode="utf8")
    iv = normalize_16byte_secret("IVMandrill2026!!", mode="utf8")

    result = run_image_simulation(plain, sbox=sbox, key=key, iv=iv, mode="CBC2")
    assert result["lossless"] is True
    assert np.array_equal(result["plain"], result["decrypted"])
    assert result["cipher"].shape == plain.shape
    metrics = result["metrics"]
    assert 7.5 < metrics["entropy_cipher"] <= 8.0
    assert metrics["npcr"] > 95.0
    assert 25.0 < metrics["uaci"] < 42.0
    assert metrics["hist_chi2"] >= 0.0
    assert 0.0 <= metrics["hist_p_value"] <= 1.0


def test_compute_image_security_metrics_handles_identical_images():
    plain = np.full((16, 16), 128, dtype=np.uint8)
    m = compute_image_security_metrics(plain, plain, plain)
    assert m["entropy_plain"] == pytest.approx(0.0, abs=1e-9)
    assert m["npcr"] == pytest.approx(0.0, abs=1e-9)
    assert m["uaci"] == pytest.approx(0.0, abs=1e-9)


def test_web_helpers_memakai_modul_image_metrics_sesuai_kontrak():
    from src.image_metrics import evaluate_image_security
    from src.web_helpers import _resolve_image_metrics_fn

    assert _resolve_image_metrics_fn() is evaluate_image_security


def test_compute_image_security_metrics_uses_per_channel_mean_npcr_for_rgb():
    c1 = np.zeros((4, 4, 3), dtype=np.uint8)
    c2 = c1.copy()
    c2[:, :, 0] = 1  # Hanya 1 dari 3 kanal yang berubah -> NPCR rata-rata per kanal = 33.3333%
    m = compute_image_security_metrics(c1, c1, c2)
    assert m["npcr"] == pytest.approx(100.0 / 3.0, abs=1e-6)


def test_load_mandrill_benchmark_table_reads_all_10_runs():
    from src.web_helpers import load_mandrill_benchmark_table

    raw, df = load_mandrill_benchmark_table("results/mandrill_metrics.json")
    assert raw["cipher_mode"] == "CBC2"
    assert len(df) == 10
    assert set(df["Mode"]) == {"RGB", "Grayscale"}
    assert (df["Lossless"] == "Ya").all()


def test_run_image_simulation_includes_nbcr_metric():
    rng = np.random.default_rng(7)
    plain = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
    sim = run_image_simulation(
        plain,
        sbox=build_standard_aes_sbox(),
        key=b"CryptoML-SBox26!",
        iv=b"MandrillCBC2-IV!",
        mode="CBC2",
    )
    assert sim["lossless"] is True
    assert 40.0 <= sim["nbcr"] <= 60.0
