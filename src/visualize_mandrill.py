"""Visualisasi Metrik 2 & 3 citra Mandrill dari ``results/mandrill_metrics.json``.

Modul ini membaca keluaran :mod:`src.mandrill_eval` lalu menghasilkan dua gambar
Matplotlib yang siap dipakai di Bab Pengujian Citra Mandrill:

1. ``mandrill_entropy_chi2.png`` — entropy Shannon plaintext vs cipher, serta
   statistik Chi-square histogram cipher terhadap batas kritis (df = 255,
   alpha = 0.05).
2. ``mandrill_diffusion.png`` — selisih NPCR, UACI, dan NBCR terhadap nilai
   ideal dalam poin persentase, karena nilai antar S-Box hanya berbeda pada
   angka desimal ketiga sehingga sulit dibaca pada skala 0–100%.

Warna mengikuti tema aplikasi Streamlit (``.streamlit/config.toml`` dan blok CSS
``app.py``): teal ``#0F5C5C`` untuk cipher, slate ``#465664``/``#90A2B2`` untuk
pembanding, dan amber ``#8A4307`` khusus garis acuan (ideal/kritis).

Pemakaian::

    python -m src.visualize_mandrill
    python -m src.visualize_mandrill --format both --dpi 300 --show
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from scipy.stats import chi2 as chi2_distribution

from src.aes_cipher_runner import decrypt_image_array, encrypt_image_array, validate_sbox
from src.image_metrics import HIST_BINS, histogram_chi_square
from src.mandrill_eval import CIPHER_MODE, DEFAULT_IMAGE_PATH, DEFAULT_IV, DEFAULT_KEY, load_image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON_PATH = REPO_ROOT / "results" / "mandrill_metrics.json"
DEFAULT_FIGURE_DIR = REPO_ROOT / "results" / "figures"
DEFAULT_SBOX_PATH = REPO_ROOT / "results" / "sbox_xgb.npy"

CHANNEL_LABELS = ("Kanal R", "Kanal G", "Kanal B")
SBOX_SHORT_NAMES = {"sbox_xgb": "XGBoost", "sbox_rf": "Random Forest", "sbox_mlp": "MLP"}
SBOX_SLUGS = {"sbox_xgb": "xgb", "sbox_rf": "rf", "sbox_mlp": "mlp"}

ALPHA = 0.05
FIGURE_DPI = 200

# Token warna dari tema aplikasi (lihat .streamlit/config.toml dan app.py).
SURFACE = "#FFFFFF"
INK = "#141D24"
INK_SOFT = "#344450"
INK_MUTED = "#465664"
GRID = "#E1E8EE"
BORDER = "#C5D1DC"
ACCENT = "#0F5C5C"
SLATE = "#465664"
PLAINTEXT_FILL = "#B3C1CC"
REFERENCE = "#8A4307"

MODE_LABELS = {"RGB": "RGB", "GRAY": "Grayscale"}
MODE_COLORS = {"RGB": ACCENT, "GRAY": SLATE}

#: Label pendek agar sumbu Y terbaca pada lebar gambar laporan.
SHORT_LABELS = {
    "AES Standar (FIPS-197)": "AES Standar",
    "K44 Alamsyah": "K44 Alamsyah",
    "S-Box XGBoost": "XGBoost",
    "S-Box Random Forest": "Random Forest",
    "S-Box MLP": "MLP",
}

DEFAULT_FORMATS = ("png",)


@dataclass(frozen=True)
class RunRecord:
    """Satu kombinasi S-Box x mode citra dari ``mandrill_metrics.json``."""

    sbox: str
    image_mode: str
    entropy_plain: float
    entropy_cipher: float
    hist_chi2: float
    hist_p_value: float
    npcr: float
    uaci: float
    nbcr: float
    lossless: bool
    plain_shape: tuple[int, ...]
    flip_x: int
    flip_y: int
    flip_channel: int
    flip_bit: int

    @property
    def label(self) -> str:
        return SHORT_LABELS.get(self.sbox, self.sbox)

    @property
    def mode_label(self) -> str:
        return MODE_LABELS.get(self.image_mode, self.image_mode)

    def value(self, metric: str) -> float:
        """Nilai metrik ('npcr', 'uaci', 'nbcr', 'entropy_cipher', 'hist_chi2', ...)."""
        return float(getattr(self, metric))

    def deviation_pp(self, metric: str, ideal: float) -> float:
        """Selisih metrik terhadap nilai ideal dalam poin persentase."""
        return self.value(metric) - float(ideal)


@dataclass(frozen=True)
class MandrillReport:
    """Laporan evaluasi Mandrill yang sudah divalidasi dan siap divisualisasikan."""

    runs: tuple[RunRecord, ...]
    ideal: dict[str, float]
    cipher_mode: str
    image_path: str
    key_hex: str
    iv_hex: str
    generated_at: str
    image_shapes: dict[str, tuple[int, ...]]
    warnings: tuple[str, ...]

    @property
    def modes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(run.image_mode for run in self.runs))

    @property
    def sbox_order(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(run.sbox for run in self.runs))

    def by_mode(self, mode: str) -> tuple[RunRecord, ...]:
        return tuple(run for run in self.runs if run.image_mode == mode)

    def value_range(self, metric: str) -> tuple[float, float]:
        values = [run.value(metric) for run in self.runs]
        return min(values), max(values)


def _require(mapping, key: str, where: str):
    if not isinstance(mapping, dict):
        raise ValueError(f"{where} harus berupa objek JSON.")
    if key not in mapping:
        raise ValueError(f"{where} tidak memiliki kunci wajib '{key}'.")
    return mapping[key]


def _as_float(value, where: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} harus berupa angka, diterima {value!r}.") from exc


def _as_int(value, where: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} harus berupa bilangan bulat, diterima {value!r}.") from exc


def _parse_run(item, index: int) -> RunRecord:
    where = f"runs[{index}]"
    metrics = _require(item, "metrics", where)
    flip = _require(item, "flip", where)
    shape = tuple(_as_int(v, f"{where}.plain_shape") for v in _require(item, "plain_shape", where))
    metrics_where = f"{where}.metrics"
    return RunRecord(
        sbox=str(_require(item, "sbox", where)),
        image_mode=str(_require(item, "image_mode", where)),
        entropy_plain=_as_float(_require(metrics, "entropy_plain", metrics_where), f"{metrics_where}.entropy_plain"),
        entropy_cipher=_as_float(_require(metrics, "entropy_cipher", metrics_where), f"{metrics_where}.entropy_cipher"),
        hist_chi2=_as_float(_require(metrics, "hist_chi2", metrics_where), f"{metrics_where}.hist_chi2"),
        hist_p_value=_as_float(_require(metrics, "hist_p_value", metrics_where), f"{metrics_where}.hist_p_value"),
        npcr=_as_float(_require(metrics, "npcr", metrics_where), f"{metrics_where}.npcr"),
        uaci=_as_float(_require(metrics, "uaci", metrics_where), f"{metrics_where}.uaci"),
        nbcr=_as_float(_require(item, "nbcr", where), f"{where}.nbcr"),
        lossless=bool(_require(item, "lossless", where)),
        plain_shape=shape,
        flip_x=_as_int(_require(flip, "x", f"{where}.flip"), f"{where}.flip.x"),
        flip_y=_as_int(_require(flip, "y", f"{where}.flip"), f"{where}.flip.y"),
        flip_channel=_as_int(_require(flip, "channel", f"{where}.flip"), f"{where}.flip.channel"),
        flip_bit=_as_int(_require(flip, "bit", f"{where}.flip"), f"{where}.flip.bit"),
    )


def load_report(path: str | Path = DEFAULT_JSON_PATH) -> MandrillReport:
    """Membaca dan memvalidasi ``mandrill_metrics.json``.

    Raises:
        FileNotFoundError: Berkas laporan belum ada.
        ValueError: JSON tidak valid atau kehilangan kunci wajib.
    """
    report_path = Path(path)
    if not report_path.exists():
        raise FileNotFoundError(
            f"Laporan metrik tidak ditemukan: {report_path.as_posix()}. "
            "Jalankan `python -m src.mandrill_eval` lebih dulu."
        )
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Berkas {report_path.as_posix()} bukan JSON valid: {exc}") from exc

    runs_data = _require(data, "runs", "laporan")
    if not isinstance(runs_data, list) or not runs_data:
        raise ValueError("laporan['runs'] harus berupa daftar yang tidak kosong.")

    ideal_data = _require(data, "ideal", "laporan")
    ideal = {
        key: _as_float(_require(ideal_data, key, "laporan['ideal']"), f"laporan['ideal'].{key}")
        for key in ("entropy", "npcr", "uaci", "nbcr")
    }

    images_data = data.get("images") or {}
    image_shapes = {
        mode: tuple(_as_int(v, f"laporan['images']['{mode}']['shape']") for v in payload.get("shape", []))
        for mode, payload in images_data.items()
        if isinstance(payload, dict) and payload.get("shape")
    }

    return MandrillReport(
        runs=tuple(_parse_run(item, idx) for idx, item in enumerate(runs_data)),
        ideal=ideal,
        cipher_mode=str(data.get("cipher_mode", "CBC2")),
        image_path=str(data.get("image_path", "data/mandril_color.jpg")),
        key_hex=str(data.get("key_hex", "")),
        iv_hex=str(data.get("iv_hex", "")),
        generated_at=str(data.get("generated_at", "")),
        image_shapes=image_shapes,
        warnings=tuple(str(w) for w in data.get("warnings") or ()),
    )


@lru_cache(maxsize=1)
def chi_square_critical(alpha: float = ALPHA, df: int = HIST_BINS - 1) -> float:
    """Nilai kritis Chi-square untuk histogram 256 bin (df = 255)."""
    return float(chi2_distribution.ppf(1.0 - alpha, df))


def _nice_step(span: float) -> float:
    """Langkah tick yang enak dibaca untuk rentang sumbu ``span``."""
    if span <= 0:
        return 0.01
    target = span / 4.0
    for candidate in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0):
        if candidate >= target:
            return candidate
    return float(np.ceil(target))


def _style_axes(ax: Axes, grid_axis: str = "x") -> None:
    """Gaya axes yang sama dengan grafik Metrik 1 di ``app.py``."""
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BORDER)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.6, color=GRID)
    ax.set_axisbelow(True)


def _panel_title(ax: Axes, title: str) -> None:
    ax.set_title(title, fontsize=10.2, fontweight="bold", color=INK, loc="left", pad=10)


def _add_header(fig: Figure, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.965, title, fontsize=13.5, fontweight="bold", color=INK, ha="left", va="top")
    fig.text(0.012, 0.905, subtitle, fontsize=9.2, color=INK_MUTED, ha="left", va="top")


def _add_text_footer(fig: Figure, text: str, y: float = 0.016) -> None:
    fig.text(0.012, y, text, fontsize=7.6, color=INK_MUTED, ha="left", va="bottom", family="monospace")


def _add_footer(fig: Figure, report: MandrillReport) -> None:
    first = report.runs[0]
    flip = f"piksel ({first.flip_x}, {first.flip_y}) kanal {first.flip_channel} bit {first.flip_bit}"
    key = report.key_hex or "-"
    iv = report.iv_hex or "-"
    config = (
        f"Citra {report.image_path}  |  AES-128 {report.cipher_mode}  |  ubah 1 bit: {flip}\n"
        f"Kunci {key}  |  IV {iv}  |  {len(report.runs)} run "
        f"({len(report.sbox_order)} S-Box x {len(report.modes)} mode)  |  {report.generated_at}"
    )
    _add_text_footer(fig, config)


def _add_panel_captions(fig: Figure, items: Sequence[tuple[Axes, str]], y: float = 0.098) -> None:
    """Menaruh catatan kecil di bawah setiap panel, sejajar sisi kiri axes."""
    for ax, text in items:
        if not text:
            continue
        fig.text(ax.get_position().x0, y, text, fontsize=7.9, color=INK_MUTED, ha="left", va="center")


def _mode_caption(report: MandrillReport) -> str:
    parts = []
    for mode in report.modes:
        runs = report.by_mode(mode)
        shape = report.image_shapes.get(mode, runs[0].plain_shape)
        channels = "3 kanal" if len(shape) == 3 else "1 kanal"
        parts.append(f"{MODE_LABELS.get(mode, mode)} {shape[0]}x{shape[1]} ({channels})")
    return "  |  ".join(parts)


def _draw_entropy_panel(ax: Axes, report: MandrillReport) -> str:
    modes = report.modes
    positions = np.arange(len(modes), dtype=float)
    width = 0.32
    plain = [float(np.mean([run.entropy_plain for run in report.by_mode(mode)])) for mode in modes]
    cipher = [float(np.mean([run.entropy_cipher for run in report.by_mode(mode)])) for mode in modes]

    ax.bar(positions - width / 2 - 0.012, plain, width, color=PLAINTEXT_FILL, edgecolor="none", label="Plaintext")
    ax.bar(positions + width / 2 + 0.012, cipher, width, color=ACCENT, edgecolor="none", label="Cipher")

    for pos, value in zip(positions - width / 2 - 0.012, plain):
        ax.text(pos, value + 0.04, f"{value:.5f}", ha="center", va="bottom", fontsize=8.4, family="monospace", color=INK)
    for pos, value in zip(positions + width / 2 + 0.012, cipher):
        ax.text(pos, value + 0.04, f"{value:.5f}", ha="center", va="bottom", fontsize=8.4, family="monospace", color=INK)

    ax.axhline(report.ideal["entropy"], color=REFERENCE, linestyle="--", linewidth=1.0)
    ax.annotate(
        f"Ideal {report.ideal['entropy']:.1f} bit",
        xy=(1.0, report.ideal["entropy"]),
        xycoords=("axes fraction", "data"),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.4,
        color=REFERENCE,
        annotation_clip=False,
    )

    floor = min(plain) - 0.45
    ax.set_ylim(floor, report.ideal["entropy"] + 0.12)
    ax.set_xticks(positions)
    ax.set_xticklabels([MODE_LABELS.get(mode, mode) for mode in modes], fontsize=9.5, color=INK)
    ax.set_xlim(-0.55, len(modes) - 0.45)
    ax.set_ylabel("Entropy Shannon (bit)", fontsize=9, color=INK_SOFT, labelpad=6)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.005),
        frameon=False,
        fontsize=8.6,
        ncol=2,
        handlelength=1.1,
        columnspacing=1.2,
    )
    _style_axes(ax, grid_axis="y")
    _panel_title(ax, "Entropy plaintext vs cipher")

    ranges = []
    for mode in modes:
        values = [run.entropy_cipher for run in report.by_mode(mode)]
        ranges.append(f"{MODE_LABELS.get(mode, mode)}: {min(values):.5f}–{max(values):.5f}")
    return "Rentang entropy cipher 5 S-Box  →  " + "  |  ".join(ranges)


def _draw_chi_square_panel(ax: Axes, report: MandrillReport) -> str:
    sboxes = report.sbox_order
    modes = report.modes
    critical = chi_square_critical()
    height = 0.32
    y_base = np.arange(len(sboxes), dtype=float)

    values_by_mode: dict[str, dict[str, RunRecord]] = {
        mode: {run.sbox: run for run in report.by_mode(mode)} for mode in modes
    }
    all_values = [run.hist_chi2 for run in report.runs]
    upper = max(max(all_values), critical) * 1.26
    worst_exceeds = False

    for idx, mode in enumerate(modes):
        lookup = values_by_mode[mode]
        offsets = y_base + (idx - (len(modes) - 1) / 2) * height
        values = [lookup[sbox].hist_chi2 if sbox in lookup else np.nan for sbox in sboxes]
        bars = ax.barh(
            offsets,
            values,
            height=height,
            color=MODE_COLORS.get(mode, ACCENT),
            edgecolor="none",
            label=MODE_LABELS.get(mode, mode),
        )
        for bar, value, sbox in zip(bars, values, sboxes):
            if np.isnan(value):
                continue
            run = lookup[sbox]
            worst_exceeds = worst_exceeds or value >= critical
            ax.text(
                value - upper * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"p = {run.hist_p_value:.3f}",
                ha="right",
                va="center",
                fontsize=8.1,
                family="monospace",
                color="#FFFFFF",
            )

    ax.axvline(critical, color=REFERENCE, linestyle="--", linewidth=1.0)
    ax.text(
        critical + upper * 0.012,
        -0.72,
        f"χ² kritis {critical:.1f}\n(α = {ALPHA:g}, df = {HIST_BINS - 1})",
        ha="left",
        va="center",
        fontsize=8.0,
        color=REFERENCE,
        linespacing=1.35,
    )

    ax.set_yticks(y_base)
    ax.set_yticklabels([SHORT_LABELS.get(sbox, sbox) for sbox in sboxes], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, upper)
    ax.set_ylim(len(sboxes) - 0.35, -1.05)
    ax.set_xlabel("Statistik χ² histogram 256 bin  (lebih kecil lebih rata)", fontsize=8.8, color=INK_SOFT, labelpad=6)
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.005),
        frameon=False,
        fontsize=8.6,
        ncol=2,
        handlelength=1.1,
        columnspacing=1.2,
    )
    _style_axes(ax, grid_axis="x")
    _panel_title(ax, "Uniformitas histogram cipher")

    return (
        "Seluruh χ² berada di bawah batas kritis → histogram cipher tidak menolak distribusi uniform."
        if not worst_exceeds
        else "Terdapat χ² yang melewati batas kritis → histogram cipher menyimpang dari uniform."
    )


def figure_entropy_chi2(report: MandrillReport, figsize=(13.0, 5.4), dpi: int = FIGURE_DPI) -> Figure:
    """Gambar Metrik 2: entropy plaintext/cipher dan uji Chi-square histogram cipher."""
    fig, (ax_entropy, ax_chi2) = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=dpi,
        gridspec_kw={"width_ratios": (1.0, 1.34)},
    )
    fig.patch.set_facecolor(SURFACE)
    caption_entropy = _draw_entropy_panel(ax_entropy, report)
    caption_chi2 = _draw_chi_square_panel(ax_chi2, report)
    _add_header(
        fig,
        "Metrik 2 — Keacakan Cipher Image Mandrill",
        f"Entropy Shannon 8-bit dan uji Chi-square histogram cipher  |  {_mode_caption(report)}",
    )
    _add_footer(fig, report)
    fig.tight_layout(rect=(0.008, 0.175, 0.995, 0.862), w_pad=3.6)
    _add_panel_captions(fig, ((ax_entropy, caption_entropy), (ax_chi2, caption_chi2)))
    return fig


def _draw_deviation_panel(
    ax: Axes,
    report: MandrillReport,
    metric: str,
    title: str,
    ideal: float,
) -> None:
    sboxes = report.sbox_order
    modes = report.modes
    height = 0.30
    y_base = np.arange(len(sboxes), dtype=float)
    values_by_mode = {mode: {run.sbox: run for run in report.by_mode(mode)} for mode in modes}

    lookup_values = {
        mode: [values_by_mode[mode][sbox].value(metric) if sbox in values_by_mode[mode] else np.nan for sbox in sboxes]
        for mode in modes
    }
    deviations = [value - ideal for mode in modes for value in lookup_values[mode] if not np.isnan(value)]
    span = max(max(abs(deviation) for deviation in deviations), 1e-3)
    padding = span * 0.95
    label_pad = span * 0.07

    for idx, mode in enumerate(modes):
        offsets = y_base + (idx - (len(modes) - 1) / 2) * height
        deltas = [value - ideal for value in lookup_values[mode]]
        bars = ax.barh(
            offsets,
            deltas,
            height=height,
            color=MODE_COLORS.get(mode, ACCENT),
            edgecolor="none",
            label=MODE_LABELS.get(mode, mode),
        )
        for bar, delta, value in zip(bars, deltas, lookup_values[mode]):
            if np.isnan(value):
                continue
            sign = 1.0 if delta >= 0 else -1.0
            ax.text(
                delta + sign * label_pad,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.4f}%",
                ha="left" if delta >= 0 else "right",
                va="center",
                fontsize=8.2,
                family="monospace",
                color=INK,
            )

    ax.axvline(0, color="#8FA3B2", linewidth=1.2)
    ax.set_xlim(-span - padding, span + padding)
    ax.set_yticks(y_base)
    ax.set_yticklabels([SHORT_LABELS.get(sbox, sbox) for sbox in sboxes], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_ylim(len(sboxes) - 0.4, -0.7)
    ax.xaxis.set_major_locator(MultipleLocator(_nice_step(span + padding)))
    ax.set_xlabel(
        "Selisih terhadap ideal (poin persentase)  ·  0 = ideal", fontsize=8.8, color=INK_SOFT, labelpad=6
    )
    _style_axes(ax, grid_axis="x")
    _panel_title(ax, f"{title}  ·  ideal {ideal:.4f}%")


def figure_diffusion(report: MandrillReport, figsize=(13.0, 5.2), dpi: int = FIGURE_DPI) -> Figure:
    """Gambar Metrik 3: selisih NPCR, UACI, dan NBCR terhadap nilai ideal."""
    panels = (
        (metric, title, report.ideal[key])
        for metric, title, key in (
            ("npcr", "NPCR", "npcr"),
            ("uaci", "UACI", "uaci"),
            ("nbcr", "NBCR", "nbcr"),
        )
    )
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (metric, title, ideal) in zip(axes, panels):
        _draw_deviation_panel(ax, report, metric, title, ideal)
    _add_header(
        fig,
        "Metrik 3 — Difusi 1-Bit pada Citra Mandrill",
        f"Selisih NPCR, UACI, dan NBCR dari nilai ideal; batang ke kanan berarti di atas ideal  |  {_mode_caption(report)}",
    )
    _add_footer(fig, report)
    legends = [
        Patch(facecolor=MODE_COLORS.get(mode, ACCENT), edgecolor="none", label=MODE_LABELS.get(mode, mode))
        for mode in report.modes
    ]
    fig.legend(
        handles=legends,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.955),
        ncol=len(legends),
        frameon=False,
        fontsize=8.8,
        handlelength=1.1,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0.008, 0.125, 0.995, 0.865), w_pad=2.6)
    return fig


def sbox_label_from_path(sbox_path: str | Path) -> str:
    """Label ringkas S-Box dari nama berkas, mis. ``sbox_xgb.npy`` → "S-Box XGBoost"."""
    stem = Path(sbox_path).stem
    return f"S-Box {SBOX_SHORT_NAMES.get(stem, stem)}"


def sbox_slug_from_path(sbox_path: str | Path) -> str:
    """Slug nama berkas untuk penamaan gambar, mis. ``sbox_xgb.npy`` → ``xgb``."""
    stem = Path(sbox_path).stem
    return SBOX_SLUGS.get(stem, stem.replace("sbox_", "").lower().replace(" ", "_"))


def _percentage_histogram(channel: np.ndarray) -> np.ndarray:
    """Histogram 256 bin ternormalisasi (persen piksel) untuk satu kanal."""
    counts = np.bincount(np.asarray(channel, dtype=np.uint8).reshape(-1), minlength=HIST_BINS).astype(np.float64)
    total = float(counts.sum())
    return counts / total * 100.0 if total > 0 else counts


def _shape_text(image: np.ndarray) -> str:
    return " × ".join(str(dimension) for dimension in image.shape)


def build_encryption_figure(
    plain: np.ndarray,
    cipher: np.ndarray,
    decrypted: np.ndarray,
    *,
    sbox_label: str,
    cipher_mode: str = CIPHER_MODE,
    key: bytes = DEFAULT_KEY,
    iv: bytes = DEFAULT_IV,
    encrypt_sec: Optional[float] = None,
    decrypt_sec: Optional[float] = None,
    figsize=(13.0, 8.6),
    dpi: int = FIGURE_DPI,
) -> Figure:
    """Membentuk Gambar 4.2: plaintext, cipher, hasil dekripsi, dan histogram RGB.

    Baris atas menampilkan tiga citra berukuran sama; baris bawah menampilkan
    histogram kanal R, G, dan B yang membandingkan plaintext (batang) dengan
    cipher image (garis) beserta statistik Chi-square kanal tersebut.
    """
    plain_arr = np.asarray(plain, dtype=np.uint8)
    if plain_arr.ndim != 3 or plain_arr.shape[2] != 3:
        raise ValueError(f"Gambar 4.2 membutuhkan citra RGB 3 kanal, diterima bentuk {plain_arr.shape}.")

    flattened_cipher = np.asarray(cipher, dtype=np.uint8).reshape(-1)
    if flattened_cipher.size < plain_arr.size:
        raise ValueError("Cipher image lebih pendek daripada plaintext.")
    cipher_view = flattened_cipher[: plain_arr.size].reshape(plain_arr.shape)

    decrypted_arr = np.asarray(decrypted, dtype=np.uint8)
    if decrypted_arr.size != plain_arr.size:
        raise ValueError("Hasil dekripsi harus berukuran sama dengan plaintext.")
    decrypted_arr = decrypted_arr.reshape(plain_arr.shape)

    difference = np.abs(plain_arr.astype(np.int16) - decrypted_arr.astype(np.int16))
    max_diff = int(difference.max()) if difference.size else 0
    lossless = bool(np.array_equal(plain_arr, decrypted_arr))

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=SURFACE)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1.0, 0.60),
        left=0.022,
        right=0.988,
        top=0.842,
        bottom=0.118,
        hspace=0.34,
        wspace=0.13,
    )

    panels = (
        (plain_arr, "Plaintext Mandrill", f"citra asli · {_shape_text(plain_arr)}"),
        (
            cipher_view,
            f"Cipher image · AES-128 {cipher_mode}",
            f"{sbox_label} · {_shape_text(cipher_view)}",
        ),
        (
            decrypted_arr,
            "Hasil dekripsi",
            f"{'identik byte-per-byte' if lossless else 'TIDAK identik'} · maks |Δ| = {max_diff}",
        ),
    )
    for column, (image, title, caption) in enumerate(panels):
        ax = fig.add_subplot(grid[0, column])
        ax.imshow(image, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        _panel_title(ax, title)
        ax.text(
            0.0,
            -0.035,
            caption,
            transform=ax.transAxes,
            fontsize=8.1,
            color=INK_MUTED,
            ha="left",
            va="top",
        )

    bins = np.arange(HIST_BINS)
    plain_histograms = [_percentage_histogram(plain_arr[:, :, channel]) for channel in range(3)]
    cipher_histograms = [_percentage_histogram(cipher_view[:, :, channel]) for channel in range(3)]
    histogram_top = max(max(float(np.max(hist)) for hist in plain_histograms), 0.5) * 1.22

    for column in range(3):
        ax = fig.add_subplot(grid[1, column])
        plain_hist = plain_histograms[column]
        cipher_hist = cipher_histograms[column]
        chi2_stat, p_value = histogram_chi_square(cipher_view[:, :, column])

        ax.bar(bins, plain_hist, width=1.0, color=PLAINTEXT_FILL, edgecolor="none", label="Plaintext")
        ax.plot(bins, cipher_hist, color=ACCENT, linewidth=1.2, label="Cipher")
        ax.set_xlim(0, HIST_BINS - 1)
        ax.set_ylim(0, histogram_top)
        ax.xaxis.set_major_locator(MultipleLocator(64))
        ax.set_title(
            f"{CHANNEL_LABELS[column]}  ·  χ² cipher = {chi2_stat:.2f} (p = {p_value:.4f})",
            fontsize=9.4,
            fontweight="bold",
            color=INK,
            loc="left",
            pad=8,
        )
        if column == 0:
            ax.set_ylabel("Frekuensi (% piksel kanal)", fontsize=8.6, color=INK_SOFT, labelpad=5)
            ax.legend(loc="upper right", frameon=False, fontsize=8.3)
        if column == 1:
            ax.set_xlabel("Intensitas piksel (0–255)", fontsize=8.8, color=INK_SOFT, labelpad=5)
        _style_axes(ax, grid_axis="y")

    _add_header(
        fig,
        "Enkripsi–Dekripsi Citra Mandrill (AES-128 CBC2)",
        (
            f"{sbox_label}  |  RGB {plain_arr.shape[0]}×{plain_arr.shape[1]}  |  "
            f"dekripsi {'identik byte-per-byte' if lossless else 'TIDAK identik'}"
        ),
    )
    timing = ""
    if encrypt_sec is not None and decrypt_sec is not None:
        timing = f"  ·  enkripsi {encrypt_sec:.2f} s  ·  dekripsi {decrypt_sec:.2f} s"
    _add_text_footer(
        fig,
        (
            f"Kunci {key.hex()}  |  IV {iv.hex()}  |  mode {cipher_mode}{timing}\n"
            f"maks |plaintext − dekripsi| = {max_diff}  |  np.array_equal = {lossless}  |  "
            "histogram: plaintext (batang) vs cipher (garis), sumbu Y = persen piksel per kanal"
        ),
    )
    return fig


def render_encryption_showcase(
    image_path: str | Path = DEFAULT_IMAGE_PATH,
    sbox_path: str | Path = DEFAULT_SBOX_PATH,
    key: bytes = DEFAULT_KEY,
    iv: bytes = DEFAULT_IV,
    cipher_mode: str = CIPHER_MODE,
    sbox_label: Optional[str] = None,
    figsize=(13.0, 8.6),
    dpi: int = FIGURE_DPI,
) -> tuple[Figure, dict]:
    """Mengenkripsi-dekripsi satu S-Box lalu membentuk Gambar 4.2 beserta ringkasannya."""
    plain = load_image(image_path, mode="RGB")
    sbox_file = Path(sbox_path)
    if not sbox_file.exists():
        raise FileNotFoundError(f"Berkas S-Box tidak ditemukan: {sbox_file.as_posix()}")
    sbox = validate_sbox(np.load(sbox_file))

    start = time.perf_counter()
    cipher = encrypt_image_array(plain, sbox=sbox, key=key, iv=iv, mode=cipher_mode)
    encrypt_sec = time.perf_counter() - start

    start = time.perf_counter()
    decrypted = decrypt_image_array(
        cipher, sbox=sbox, key=key, iv=iv, orig_shape=plain.shape, mode=cipher_mode
    )
    decrypt_sec = time.perf_counter() - start

    label = sbox_label or sbox_label_from_path(sbox_file)
    figure = build_encryption_figure(
        plain,
        cipher,
        decrypted,
        sbox_label=label,
        cipher_mode=cipher_mode,
        key=key,
        iv=iv,
        encrypt_sec=encrypt_sec,
        decrypt_sec=decrypt_sec,
        figsize=figsize,
        dpi=dpi,
    )
    info = {
        "sbox_label": label,
        "sbox_path": sbox_file.as_posix(),
        "image_path": Path(image_path).as_posix(),
        "cipher_mode": cipher_mode,
        "lossless": bool(np.array_equal(plain, np.asarray(decrypted))),
        "max_abs_diff": int(
            np.max(np.abs(plain.astype(np.int16) - np.asarray(decrypted).astype(np.int16)))
        ),
        "encrypt_sec": float(encrypt_sec),
        "decrypt_sec": float(decrypt_sec),
    }
    return figure, info


def save_encryption_showcase(
    out_dir: str | Path = DEFAULT_FIGURE_DIR,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = FIGURE_DPI,
    **config,
) -> list[Path]:
    """Menyimpan Gambar 4.2 ke ``out_dir``; nama berkas mengikuti S-Box yang dipakai."""
    extensions = _validate_formats(formats)
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    figure, info = render_encryption_showcase(dpi=dpi, **config)
    name = f"mandrill_encryption_{sbox_slug_from_path(info['sbox_path'])}"
    written: list[Path] = []
    try:
        for extension in extensions:
            path = target_dir / f"{name}.{extension}"
            figure.savefig(
                path,
                dpi=dpi,
                facecolor=figure.get_facecolor(),
                bbox_inches="tight",
                pad_inches=0.22,
            )
            written.append(path)
    finally:
        plt.close(figure)
    return written


#: Pasangan nama berkas dan pembuat gambar yang disimpan oleh :func:`save_figures`.
FIGURE_BUILDERS = (
    ("mandrill_entropy_chi2", figure_entropy_chi2),
    ("mandrill_diffusion", figure_diffusion),
)


def _validate_formats(formats: Sequence[str]) -> list[str]:
    """Menormalkan daftar ekstensi gambar dan menolak format yang tidak didukung."""
    extensions = list(dict.fromkeys(ext.lower() for ext in formats))
    unknown = [ext for ext in extensions if ext not in {"png", "svg", "pdf"}]
    if unknown:
        raise ValueError(f"Format gambar tidak didukung: {unknown}. Pilih 'png', 'svg', atau 'pdf'.")
    return extensions


def save_figures(
    report: MandrillReport,
    out_dir: str | Path = DEFAULT_FIGURE_DIR,
    formats: Sequence[str] = DEFAULT_FORMATS,
    dpi: int = FIGURE_DPI,
) -> list[Path]:
    """Menyimpan seluruh gambar ke ``out_dir`` dan mengembalikan daftar path-nya."""
    extensions = _validate_formats(formats)
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, builder in FIGURE_BUILDERS:
        fig = builder(report, dpi=dpi)
        try:
            for extension in extensions:
                path = target_dir / f"{name}.{extension}"
                fig.savefig(
                    path,
                    dpi=dpi,
                    facecolor=fig.get_facecolor(),
                    bbox_inches="tight",
                    pad_inches=0.22,
                )
                written.append(path)
        finally:
            plt.close(fig)
    return written


def summary_line(report: MandrillReport) -> str:
    """Ringkasan satu baris untuk konsol dan laporan."""
    npcr_min, npcr_max = report.value_range("npcr")
    uaci_min, uaci_max = report.value_range("uaci")
    chi2_max = max(run.hist_chi2 for run in report.runs)
    critical = chi_square_critical()
    return (
        f"NPCR {npcr_min:.4f}–{npcr_max:.4f}% (ideal {report.ideal['npcr']:.4f}%)  |  "
        f"UACI {uaci_min:.4f}–{uaci_max:.4f}% (ideal {report.ideal['uaci']:.4f}%)  |  "
        f"χ² maks {chi2_max:.2f} vs kritis {critical:.2f}  |  "
        f"lossless {sum(1 for run in report.runs if run.lossless)}/{len(report.runs)}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.visualize_mandrill",
        description=(
            "Membuat grafik Matplotlib dari results/mandrill_metrics.json, "
            "ditambah Gambar 4.2 (plaintext, cipher, dekripsi, histogram RGB) bila --showcase dipakai."
        ),
    )
    parser.add_argument("--json", default=str(DEFAULT_JSON_PATH), help="Path laporan metrik JSON.")
    parser.add_argument("--out-dir", default=str(DEFAULT_FIGURE_DIR), help="Folder keluaran gambar.")
    parser.add_argument(
        "--format",
        dest="formats",
        choices=("png", "svg", "pdf", "both"),
        default="png",
        help="Format gambar keluaran (default: png; 'both' = png + svg).",
    )
    parser.add_argument("--dpi", type=int, default=FIGURE_DPI, help="Resolusi raster (default: 200).")
    parser.add_argument(
        "--showcase",
        action="store_true",
        help="Juga buat Gambar 4.2: plaintext, cipher, dekripsi, dan histogram RGB satu S-Box.",
    )
    parser.add_argument(
        "--image",
        default=str(DEFAULT_IMAGE_PATH),
        help="Citra RGB untuk Gambar 4.2 (default: data/mandril_color.jpg).",
    )
    parser.add_argument(
        "--sbox",
        default=str(DEFAULT_SBOX_PATH),
        help="Berkas .npy S-Box untuk Gambar 4.2 (default: results/sbox_xgb.npy).",
    )
    parser.add_argument("--key-hex", default=DEFAULT_KEY.hex(), help="Kunci AES-128 16 byte (hex) untuk Gambar 4.2.")
    parser.add_argument("--iv-hex", default=DEFAULT_IV.hex(), help="IV 16 byte (hex) untuk Gambar 4.2.")
    parser.add_argument("--show", action="store_true", help="Tampilkan jendela matplotlib setelah menyimpan.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:  # Simbol seperti χ² dan α tidak tersedia pada cp1252 (konsol Windows).
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover - hanya untuk konsol lama
        pass

    args = build_arg_parser().parse_args(argv)
    formats = ("png", "svg") if args.formats == "both" else (args.formats,)

    report = load_report(args.json)
    print(f"Laporan : {Path(args.json).as_posix()} ({len(report.runs)} run, {len(report.sbox_order)} S-Box)")
    print(f"Ringkas : {summary_line(report)}")

    written = save_figures(report, out_dir=args.out_dir, formats=formats, dpi=args.dpi)
    for path in written:
        print(f"Gambar  : {path.as_posix()}")

    if args.showcase:
        key = bytes.fromhex(args.key_hex)
        iv = bytes.fromhex(args.iv_hex)
        if len(key) != 16 or len(iv) != 16:
            raise SystemExit("--key-hex dan --iv-hex harus tepat 16 byte (32 digit hex).")
        print(f"Gambar 4.2: mengenkripsi {Path(args.image).name} dengan {Path(args.sbox).name} ...", flush=True)
        showcase_paths = save_encryption_showcase(
            out_dir=args.out_dir,
            formats=formats,
            dpi=args.dpi,
            image_path=args.image,
            sbox_path=args.sbox,
            key=key,
            iv=iv,
        )
        for path in showcase_paths:
            print(f"Gambar  : {path.as_posix()}")

    for warning in report.warnings:
        print(f"[peringatan] {warning}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
