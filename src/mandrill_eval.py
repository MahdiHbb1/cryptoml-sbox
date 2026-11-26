"""Runner evaluasi citra Mandrill untuk Metrik 2 & 3 (5 S-Box x RGB/Grayscale).

Modul ini memasangkan Kontrak §4 (``src/aes_cipher_runner.py``) dengan Kontrak §5
(``src/image_metrics.py``) menjadi satu tabel eksperimen
Pemakaian::

    python -m src.mandrill_eval
    python -m src.mandrill_eval --modes RGB GRAY --flip-x 120 --flip-y 85 --bit 0

Keluaran default: ``results/mandrill_metrics.json`` dan tabel Markdown di stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from PIL import Image

from src.aes_affine_generator import build_k44_sbox, build_standard_aes_sbox
from src.aes_cipher_runner import (
    decrypt_image_array,
    describe_cipher_variant,
    encrypt_image_array,
    validate_sbox,
)
from src.image_metrics import (
    CORRELATION_IDEAL,
    ENTROPY_IDEAL,
    NBCR_IDEAL,
    NPCR_IDEAL,
    UACI_IDEAL,
    bit_change_rate,
    evaluate_image_security,
    pixel_correlation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_PATH = REPO_ROOT / "data" / "mandril_color.jpg"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / "mandrill_metrics.json"

# Kunci & IV demo tetap agar angka laporan dapat direproduksi ulang.
DEFAULT_KEY = b"CryptoML-SBox26!"
DEFAULT_IV = b"MandrillCBC2-IV!"

CIPHER_MODE = "CBC2"
GRAYSCALE_MODE = "GRAY"
IMAGE_MODES = ("RGB", GRAYSCALE_MODE)

#: Urutan S-Box pada laporan: baseline lebih dulu, lalu hasil model ML.
SBOX_SOURCES: tuple[tuple[str, Optional[str]], ...] = (
    ("AES Standar (FIPS-197)", None),
    ("K44 Alamsyah", None),
    ("S-Box XGBoost", "sbox_xgb.npy"),
    ("S-Box Random Forest", "sbox_rf.npy"),
    ("S-Box MLP", "sbox_mlp.npy"),
)

_PIL_MODES = {"RGB": "RGB", GRAYSCALE_MODE: "L"}


def _display_path(path: str | Path) -> str:
    """Path relatif terhadap root repo bila memungkinkan, agar JSON portabel antar-laptop."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_image(path: str | Path, mode: str = "RGB") -> np.ndarray:
    """Memuat citra uji sebagai array ``np.uint8`` (RGB 3 kanal atau grayscale 2D)."""
    if mode not in _PIL_MODES:
        raise ValueError(f"Mode citra tidak dikenal: {mode!r}. Pilih 'RGB' atau 'GRAY'.")
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Citra uji tidak ditemukan: {image_path}")
    with Image.open(image_path) as img:
        return np.asarray(img.convert(_PIL_MODES[mode]), dtype=np.uint8)


def ubah_satu_bit(
    image: np.ndarray,
    x: int,
    y: int,
    channel: int = 0,
    bit_pos: int = 0,
) -> np.ndarray:
    """Mengubah 1 bit pada piksel ``(x, y)`` memakai XOR tanpa merusak citra asli.

    Args:
        image: Citra ``np.uint8`` 2D (grayscale) atau 3D (BGR/RGB).
        x: Koordinat kolom (lebar).
        y: Koordinat baris (tinggi).
        channel: Indeks kanal untuk citra berwarna; diabaikan pada grayscale.
        bit_pos: Posisi bit yang diubah, 0 = LSB dan 7 = MSB.

    Returns:
        Salinan citra dengan tepat satu bit terbalik.
    """
    arr = np.asarray(image, dtype=np.uint8)
    if not 0 <= int(bit_pos) <= 7:
        raise ValueError(f"bit_pos harus 0..7, diterima {bit_pos}.")
    height, width = arr.shape[:2]
    if not 0 <= int(y) < height or not 0 <= int(x) < width:
        raise ValueError(f"Koordinat ({x}, {y}) di luar batas citra {width}x{height}.")

    out = arr.copy()
    mask = np.uint8(1 << int(bit_pos))
    if out.ndim == 3:
        if not 0 <= int(channel) < out.shape[2]:
            raise ValueError(f"Kanal {channel} di luar jumlah kanal {out.shape[2]}.")
        out[int(y), int(x), int(channel)] ^= mask
    else:
        out[int(y), int(x)] ^= mask
    return out


def resolve_flip_position(
    shape: Sequence[int],
    flip_x: Optional[int] = None,
    flip_y: Optional[int] = None,
) -> tuple[int, int]:
    """Menentukan koordinat piksel yang diubah; ``None`` berarti piksel terakhir."""
    height, width = int(shape[0]), int(shape[1])
    x = width - 1 if flip_x is None else int(flip_x)
    y = height - 1 if flip_y is None else int(flip_y)
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError(f"Koordinat flip ({x}, {y}) di luar batas citra {width}x{height}.")
    return x, y


def evaluate_sbox_on_image(
    plain: np.ndarray,
    sbox: np.ndarray,
    key: bytes = DEFAULT_KEY,
    iv: bytes = DEFAULT_IV,
    cipher_mode: str = CIPHER_MODE,
    flip_x: Optional[int] = None,
    flip_y: Optional[int] = None,
    flip_channel: int = 0,
    flip_bit: int = 0,
) -> dict:
    """Menguji satu S-Box pada satu citra: enkripsi, uji 1-bit, dekripsi lossless.

    Returns:
        ``dict`` berisi ``metrics`` (Kontrak §5), ``nbcr``, ``lossless``,
        detail piksel yang diubah, dan durasi enkripsi/dekripsi.
    """
    plain_arr = np.asarray(plain, dtype=np.uint8)
    sb = validate_sbox(sbox)
    x, y = resolve_flip_position(plain_arr.shape, flip_x=flip_x, flip_y=flip_y)
    channel = 0 if plain_arr.ndim == 2 else int(flip_channel)
    flipped = ubah_satu_bit(plain_arr, x=x, y=y, channel=channel, bit_pos=flip_bit)

    start_single = time.perf_counter()
    cipher1 = encrypt_image_array(plain_arr, sbox=sb, key=key, iv=iv, mode=cipher_mode)
    encrypt_single_sec = time.perf_counter() - start_single

    start_second = time.perf_counter()
    cipher2 = encrypt_image_array(flipped, sbox=sb, key=key, iv=iv, mode=cipher_mode)
    encrypt_sec = encrypt_single_sec + (time.perf_counter() - start_second)

    start = time.perf_counter()
    decrypted = decrypt_image_array(
        cipher1, sbox=sb, key=key, iv=iv, orig_shape=plain_arr.shape, mode=cipher_mode
    )
    decrypt_sec = time.perf_counter() - start

    cipher1_arr = np.asarray(cipher1, dtype=np.uint8)
    cipher1_for_corr = (
        cipher1_arr[: plain_arr.size].reshape(plain_arr.shape)
        if cipher1_arr.shape != plain_arr.shape
        else cipher1_arr
    )

    metrics = evaluate_image_security(plain_arr, cipher1, cipher2)
    corr_plain = pixel_correlation(plain_arr)
    corr_cipher = pixel_correlation(cipher1_for_corr)
    nilai_asli = int(plain_arr[y, x, channel]) if plain_arr.ndim == 3 else int(plain_arr[y, x])
    nilai_baru = int(flipped[y, x, channel]) if flipped.ndim == 3 else int(flipped[y, x])

    return {
        "plain_shape": list(plain_arr.shape),
        "cipher_shape": list(cipher1_arr.shape),
        "cipher_variant": describe_cipher_variant(sb, mode=cipher_mode),
        "flip": {
            "x": x,
            "y": y,
            "channel": channel,
            "bit": int(flip_bit),
            "pixel_plain": nilai_asli,
            "pixel_flipped": nilai_baru,
        },
        "metrics": metrics,
        "pixel_correlation": {
            "plain": corr_plain,
            "cipher": corr_cipher,
        },
        "nbcr": bit_change_rate(cipher1, cipher2),
        "lossless": bool(np.array_equal(plain_arr, decrypted)),
        "encrypt_single_sec": float(encrypt_single_sec),
        "encrypt_sec": float(encrypt_sec),
        "decrypt_sec": float(decrypt_sec),
    }


def collect_sboxes(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """Mengumpulkan S-Box baseline + hasil model ML yang tersedia di ``results/``.

    Berkas ``.npy`` yang belum ada (mis. model belum di-merge) dilewati dan
    dicatat pada daftar peringatan, sehingga runner tetap dapat dijalankan.
    """
    res_dir = Path(results_dir)
    catalog: dict[str, np.ndarray] = {
        "AES Standar (FIPS-197)": build_standard_aes_sbox(),
        "K44 Alamsyah": build_k44_sbox(),
    }
    warnings: list[str] = []
    for label, filename in SBOX_SOURCES[2:]:
        path = res_dir / str(filename)
        if not path.exists():
            warnings.append(f"{filename} tidak ditemukan di {res_dir.as_posix()}; {label} dilewati.")
            continue
        try:
            sbox = validate_sbox(np.load(path))
        except (OSError, ValueError) as exc:
            warnings.append(f"{filename} bukan S-Box valid ({exc}); {label} dilewati.")
            continue
        catalog[label] = sbox
    return catalog, warnings


def run_mandrill_evaluation(
    image_path: str | Path = DEFAULT_IMAGE_PATH,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    modes: Sequence[str] = IMAGE_MODES,
    cipher_mode: str = CIPHER_MODE,
    key: bytes = DEFAULT_KEY,
    iv: bytes = DEFAULT_IV,
    flip_x: Optional[int] = None,
    flip_y: Optional[int] = None,
    flip_channel: int = 0,
    flip_bit: int = 0,
    progress: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Menjalankan seluruh kombinasi S-Box x mode citra dan mengembalikan laporan."""
    image_file = Path(image_path)
    sboxes, warnings = collect_sboxes(results_dir)
    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "image_path": _display_path(image_file),
        "cipher_mode": cipher_mode,
        "key_hex": key.hex(),
        "iv_hex": iv.hex(),
        "ideal": {
            "entropy": ENTROPY_IDEAL,
            "npcr": NPCR_IDEAL,
            "uaci": UACI_IDEAL,
            "nbcr": NBCR_IDEAL,
            "correlation": CORRELATION_IDEAL,
        },
        "sboxes": list(sboxes),
        "images": {},
        "warnings": warnings,
        "runs": [],
    }

    for mode in modes:
        if mode not in _PIL_MODES:
            raise ValueError(f"Mode citra tidak dikenal: {mode!r}. Pilih 'RGB' atau 'GRAY'.")
        plain = load_image(image_file, mode=mode)
        report["images"][mode] = {"shape": list(plain.shape)}
        for label, sbox in sboxes.items():
            result = evaluate_sbox_on_image(
                plain,
                sbox,
                key=key,
                iv=iv,
                cipher_mode=cipher_mode,
                flip_x=flip_x,
                flip_y=flip_y,
                flip_channel=flip_channel,
                flip_bit=flip_bit,
            )
            run = {"sbox": label, "image_mode": mode, **result}
            report["runs"].append(run)
            if progress is not None:
                progress(run)
    return report


def write_report(report: dict, output_path: str | Path = DEFAULT_OUTPUT_PATH) -> Path:
    """Menyimpan laporan evaluasi sebagai JSON rapi (UTF-8, indentasi 2)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def format_markdown_table(report: dict) -> str:
    """Merender tabel Markdown siap tempel untuk ``results/experiment_log.md``."""
    header = (
        "| S-Box | Mode Citra | Entropy Plain | Entropy Cipher | "
        "Histogram Chi2 (p-val) | NPCR (%) | UACI (%) | NBCR (%) | Lossless Decrypt |"
    )
    separator = "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    rows = [header, separator]
    for run in report.get("runs", []):
        metrics = run["metrics"]
        rows.append(
            "| {sbox} | {mode} | {ep:.5f} | {ec:.5f} | {chi2:.2f} (p = {p:.4f}) | "
            "{npcr:.4f} | {uaci:.4f} | {nbcr:.4f} | {lossless} |".format(
                sbox=run["sbox"],
                mode="Grayscale" if run["image_mode"] == GRAYSCALE_MODE else run["image_mode"],
                ep=metrics["entropy_plain"],
                ec=metrics["entropy_cipher"],
                chi2=metrics["hist_chi2"],
                p=metrics["hist_p_value"],
                npcr=metrics["npcr"],
                uaci=metrics["uaci"],
                nbcr=run["nbcr"],
                lossless="Yes" if run["lossless"] else "No",
            )
        )
    return "\n".join(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.mandrill_eval",
        description="Evaluasi Metrik 2 & 3 pada citra Mandrill untuk seluruh S-Box tim.",
    )
    parser.add_argument("--image", default=str(DEFAULT_IMAGE_PATH), help="Path citra uji Mandrill.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Folder berkas .npy S-Box.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH), help="Path keluaran JSON.")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=list(IMAGE_MODES),
        default=list(IMAGE_MODES),
        help="Mode citra yang diuji (default: RGB GRAY).",
    )
    parser.add_argument(
        "--cipher-mode",
        default=CIPHER_MODE,
        choices=("CBC2", "CBC", "ECB"),
        help="Mode AES pada runner (default: CBC2 agar difusi penuh).",
    )
    parser.add_argument("--flip-x", type=int, default=None, help="Kolom piksel yang diubah (default: terakhir).")
    parser.add_argument("--flip-y", type=int, default=None, help="Baris piksel yang diubah (default: terakhir).")
    parser.add_argument("--channel", type=int, default=0, help="Kanal piksel yang diubah (0..2 untuk RGB).")
    parser.add_argument("--bit", type=int, default=0, help="Posisi bit yang diubah (0 = LSB, 7 = MSB).")
    parser.add_argument("--key-hex", default=DEFAULT_KEY.hex(), help="Kunci AES-128 16 byte dalam hex.")
    parser.add_argument("--iv-hex", default=DEFAULT_IV.hex(), help="IV 16 byte dalam hex.")
    parser.add_argument("--no-json", action="store_true", help="Jangan menulis berkas JSON.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:  # Agar simbol seperti "x" pada tabel tidak memicu UnicodeEncodeError di konsol Windows.
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover - hanya untuk konsol lama
        pass

    args = build_arg_parser().parse_args(argv)
    key = bytes.fromhex(args.key_hex)
    iv = bytes.fromhex(args.iv_hex)
    if len(key) != 16:
        raise SystemExit("--key-hex harus tepat 16 byte (32 digit hex).")
    if args.cipher_mode != "ECB" and len(iv) != 16:
        raise SystemExit("--iv-hex harus tepat 16 byte (32 digit hex).")

    def _progress(run: dict) -> None:
        metrics = run["metrics"]
        print(
            f"  {run['sbox']:<26} {run['image_mode']:<5} "
            f"NPCR={metrics['npcr']:7.4f}% UACI={metrics['uaci']:7.4f}% "
            f"lossless={'Ya' if run['lossless'] else 'Tidak'}",
            flush=True,
        )

    report = run_mandrill_evaluation(
        image_path=args.image,
        results_dir=args.results_dir,
        modes=args.modes,
        cipher_mode=args.cipher_mode,
        key=key,
        iv=iv,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        flip_channel=args.channel,
        flip_bit=args.bit,
        progress=_progress,
    )

    if not args.no_json:
        path = write_report(report, args.out)
        print(f"\nLaporan JSON tersimpan di: {path.as_posix()}")

    flip = report["runs"][0]["flip"] if report["runs"] else {}
    print(
        f"\nMode AES={report['cipher_mode']} | piksel diubah=({flip.get('x')}, {flip.get('y')}) "
        f"kanal {flip.get('channel')} bit {flip.get('bit')} | "
        f"key={report['key_hex']} iv={report['iv_hex']}\n"
    )
    print(format_markdown_table(report))

    for warning in report["warnings"]:
        print(f"[peringatan] {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
