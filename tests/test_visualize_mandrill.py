import json

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image

from src.aes_affine_generator import build_standard_aes_sbox
from src.visualize_mandrill import (
    DEFAULT_JSON_PATH,
    build_encryption_figure,
    chi_square_critical,
    figure_diffusion,
    figure_entropy_chi2,
    load_report,
    main,
    render_encryption_showcase,
    save_encryption_showcase,
    save_figures,
    sbox_label_from_path,
    sbox_slug_from_path,
    summary_line,
)

IDEAL = {"entropy": 8.0, "npcr": 99.609375, "uaci": 33.4635, "nbcr": 50.0}


def _citra_acak_visualisasi(shape=(8, 8, 3), seed=5):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def _simpan_citra_visualisasi(tmp_path, array, name="citra_uji.png"):
    path = tmp_path / name
    Image.fromarray(array).save(path)
    return path


def _tulis_laporan(path, modes=("GRAY",), sboxes=("S-Box MLP",)):
    runs = []
    for mode in modes:
        for index, sbox in enumerate(sboxes):
            runs.append(
                {
                    "sbox": sbox,
                    "image_mode": mode,
                    "plain_shape": [8, 8],
                    "cipher_shape": [8, 8],
                    "flip": {"x": 7, "y": 7, "channel": 0, "bit": 0, "pixel_plain": 1, "pixel_flipped": 0},
                    "metrics": {
                        "entropy_plain": 7.0,
                        "entropy_cipher": 7.99,
                        "hist_chi2": 250.0 + index,
                        "hist_p_value": 0.5,
                        "npcr": 99.6,
                        "uaci": 33.46,
                    },
                    "nbcr": 50.0,
                    "lossless": True,
                    "encrypt_sec": 0.01,
                    "decrypt_sec": 0.01,
                }
            )
    data = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "image_path": "data/uji.jpg",
        "cipher_mode": "CBC2",
        "key_hex": "00" * 16,
        "iv_hex": "11" * 16,
        "ideal": IDEAL,
        "sboxes": list(sboxes),
        "images": {mode: {"shape": [8, 8]} for mode in modes},
        "warnings": [],
        "runs": runs,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_report_membaca_laporan_hasil_evaluasi():
    report = load_report()

    assert DEFAULT_JSON_PATH.exists()
    assert len(report.runs) == 10
    assert report.modes == ("RGB", "GRAY")
    assert report.sbox_order == (
        "AES Standar (FIPS-197)",
        "K44 Alamsyah",
        "S-Box XGBoost",
        "S-Box Random Forest",
        "S-Box MLP",
    )
    assert report.ideal["npcr"] == pytest.approx(99.609375)
    assert report.cipher_mode == "CBC2"
    assert report.image_path == "data/mandril_color.jpg"
    assert report.image_shapes["RGB"] == (512, 512, 3)

    run = report.by_mode("GRAY")[0]
    assert run.mode_label == "Grayscale"
    assert run.label == "AES Standar"
    assert run.lossless is True
    assert run.deviation_pp("npcr", report.ideal["npcr"]) == pytest.approx(run.npcr - 99.609375)


def test_load_report_menolak_berkas_yang_belum_ada(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        load_report(tmp_path / "tidak_ada.json")
    assert "python -m src.mandrill_eval" in str(excinfo.value)


def test_load_report_menolak_json_tidak_valid(tmp_path):
    path = tmp_path / "rusak.json"
    path.write_text("{bukan json", encoding="utf-8")

    with pytest.raises(ValueError):
        load_report(path)


def test_load_report_menolak_kunci_wajib_yang_hilang(tmp_path):
    path = _tulis_laporan(tmp_path / "laporan.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["runs"][0]["metrics"]["npcr"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_report(path)
    assert "npcr" in str(excinfo.value)


def test_load_report_menolak_ideal_tidak_lengkap(tmp_path):
    path = _tulis_laporan(tmp_path / "laporan.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    del data["ideal"]["uaci"]
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_report(path)
    assert "uaci" in str(excinfo.value)


def test_chi_square_critical_sesuai_derajat_kebebasan_255():
    assert chi_square_critical() == pytest.approx(293.2478, abs=0.01)


def test_figure_entropy_chi2_punya_dua_panel_berjudul_benar():
    report = load_report()
    fig = figure_entropy_chi2(report, dpi=72)
    try:
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 2
        judul = [ax.get_title(loc="left") for ax in fig.axes]
        assert any("Entropy" in teks for teks in judul)
        assert any("Uniformitas" in teks for teks in judul)
        assert fig.axes[0].get_ylabel() == "Entropy Shannon (bit)"
    finally:
        plt.close(fig)


def test_figure_diffusion_punya_tiga_panel_metrik():
    report = load_report()
    fig = figure_diffusion(report, dpi=72)
    try:
        assert len(fig.axes) == 3
        judul = " ".join(ax.get_title(loc="left") for ax in fig.axes)
        assert "NPCR" in judul and "UACI" in judul and "NBCR" in judul
        assert "99.6094%" in judul
        assert "0 = ideal" in fig.axes[0].get_xlabel()
    finally:
        plt.close(fig)


def test_figure_bisa_dibuat_untuk_laporan_satu_mode_satu_sbox(tmp_path):
    report = load_report(_tulis_laporan(tmp_path / "satu.json"))

    fig_metrik2 = figure_entropy_chi2(report, dpi=72)
    fig_metrik3 = figure_diffusion(report, dpi=72)
    try:
        assert report.modes == ("GRAY",)
        assert len(fig_metrik2.axes) == 2
        assert len(fig_metrik3.axes) == 3
    finally:
        plt.close(fig_metrik2)
        plt.close(fig_metrik3)


def test_save_figures_menulis_png_ke_folder_tujuan(tmp_path):
    report = load_report(_tulis_laporan(tmp_path / "laporan.json"))

    paths = save_figures(report, out_dir=tmp_path / "figures", dpi=72)

    assert [path.name for path in paths] == ["mandrill_entropy_chi2.png", "mandrill_diffusion.png"]
    for path in paths:
        assert path.exists() and path.stat().st_size > 0


def test_save_figures_mendukung_svg_dan_menolak_format_lain(tmp_path):
    report = load_report(_tulis_laporan(tmp_path / "laporan.json"))

    svg_paths = save_figures(report, out_dir=tmp_path / "figures", formats=("svg",), dpi=72)
    assert all(path.suffix == ".svg" for path in svg_paths)
    assert all(path.exists() for path in svg_paths)

    with pytest.raises(ValueError):
        save_figures(report, out_dir=tmp_path / "figures", formats=("jpg",), dpi=72)


def test_summary_line_memuat_rentang_dan_status_lossless():
    report = load_report()
    line = summary_line(report)

    assert "NPCR 99.5907–99.6212%" in line
    assert "UACI 33.4192–33.5018%" in line
    assert "χ² maks 286.49" in line
    assert "lossless 10/10" in line


def test_main_menulis_gambar_dan_mengembalikan_nol(tmp_path):
    exit_code = main(["--json", str(DEFAULT_JSON_PATH), "--out-dir", str(tmp_path), "--dpi", "72"])

    assert exit_code == 0
    assert (tmp_path / "mandrill_entropy_chi2.png").exists()
    assert (tmp_path / "mandrill_diffusion.png").exists()


def test_sbox_label_dan_slug_dari_nama_berkas():
    assert sbox_label_from_path("results/sbox_xgb.npy") == "S-Box XGBoost"
    assert sbox_label_from_path("results/sbox_rf.npy") == "S-Box Random Forest"
    assert sbox_label_from_path("results/kustom.npy") == "S-Box kustom"
    assert sbox_slug_from_path("results/sbox_xgb.npy") == "xgb"
    assert sbox_slug_from_path("results/kustom.npy") == "kustom"


def test_build_encryption_figure_punya_tiga_gambar_dan_tiga_histogram():
    plain = _citra_acak_visualisasi((8, 8, 3))
    cipher = _citra_acak_visualisasi((8, 8, 3), seed=21)

    fig = build_encryption_figure(plain, cipher, plain.copy(), sbox_label="S-Box XGBoost", dpi=72)
    try:
        assert len(fig.axes) == 6
        judul = [ax.get_title(loc="left") for ax in fig.axes]
        assert any("Plaintext" in teks for teks in judul)
        assert any("Cipher image" in teks for teks in judul)
        assert any("Hasil dekripsi" in teks for teks in judul)
        assert sum("χ² cipher" in teks for teks in judul) == 3
        assert all(len(ax.patches) > 0 for ax in fig.axes[3:])
        assert any("identik byte-per-byte" in teks.get_text() for teks in fig.texts)
    finally:
        plt.close(fig)


def test_build_encryption_figure_menandai_dekripsi_tidak_identik():
    plain = _citra_acak_visualisasi((8, 8, 3))
    decrypted = plain.copy()
    decrypted[0, 0, 0] = np.uint8((int(decrypted[0, 0, 0]) + 1) % 256)

    fig = build_encryption_figure(plain, plain, decrypted, sbox_label="S-Box Uji", dpi=72)
    try:
        teks = " ".join(item.get_text() for item in fig.texts)
        assert "np.array_equal = False" in teks
        assert "maks |plaintext − dekripsi| = 1" in teks
    finally:
        plt.close(fig)


def test_build_encryption_figure_menolak_citra_grayscale():
    plain = _citra_acak_visualisasi((8, 8))

    with pytest.raises(ValueError):
        build_encryption_figure(plain, plain, plain, sbox_label="S-Box Uji", dpi=72)


def test_render_encryption_showcase_lossless_pada_citra_kecil(tmp_path):
    image_path = _simpan_citra_visualisasi(tmp_path, _citra_acak_visualisasi((16, 16, 3)))
    sbox_path = tmp_path / "sbox_xgb.npy"
    np.save(sbox_path, build_standard_aes_sbox())

    fig, info = render_encryption_showcase(image_path=image_path, sbox_path=sbox_path, dpi=72)
    try:
        assert info["lossless"] is True
        assert info["max_abs_diff"] == 0
        assert info["sbox_label"] == "S-Box XGBoost"
        assert info["cipher_mode"] == "CBC2"
        assert len(fig.axes) == 6
    finally:
        plt.close(fig)


def test_render_encryption_showcase_menolak_sbox_tidak_ada(tmp_path):
    image_path = _simpan_citra_visualisasi(tmp_path, _citra_acak_visualisasi((16, 16, 3)))

    with pytest.raises(FileNotFoundError):
        render_encryption_showcase(image_path=image_path, sbox_path=tmp_path / "tidak_ada.npy")


def test_save_encryption_showcase_menulis_png_bernama_sbox(tmp_path):
    image_path = _simpan_citra_visualisasi(tmp_path, _citra_acak_visualisasi((16, 16, 3)))
    sbox_path = tmp_path / "sbox_xgb.npy"
    np.save(sbox_path, build_standard_aes_sbox())

    paths = save_encryption_showcase(
        out_dir=tmp_path / "figures",
        image_path=image_path,
        sbox_path=sbox_path,
        dpi=72,
    )

    assert [path.name for path in paths] == ["mandrill_encryption_xgb.png"]
    assert paths[0].exists() and paths[0].stat().st_size > 0


def test_main_dengan_showcase_menulis_gambar_keempat(tmp_path):
    image_path = _simpan_citra_visualisasi(tmp_path, _citra_acak_visualisasi((16, 16, 3)))
    sbox_path = tmp_path / "sbox_xgb.npy"
    np.save(sbox_path, build_standard_aes_sbox())

    exit_code = main(
        [
            "--json",
            str(DEFAULT_JSON_PATH),
            "--out-dir",
            str(tmp_path / "figures"),
            "--dpi",
            "72",
            "--showcase",
            "--image",
            str(image_path),
            "--sbox",
            str(sbox_path),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "figures" / "mandrill_encryption_xgb.png").exists()
