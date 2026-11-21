"""Aplikasi Web Interaktif CryptoML-SBox (Streamlit).

Terdiri dari 3 halaman utama:
1. Ringkasan Proyek      — Penjelasan lugas agar siapa pun langsung memahami sistem & hasil.
2. Diagram & Arsitektur  — Diagram node interaktif Archify (alur kerja, pengembangan, hasil).
3. Simulasi Langsung     — Inspeksi/generator S-Box, enkripsi citra Mandrill/upload, & teks.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.aes_affine_generator import IRREDUCIBLE_POLYS_GF256
from src.web_helpers import (
    compute_sac_matrix,
    decrypt_text_aes,
    encrypt_text_aes,
    evaluate_sbox_full,
    format_sbox_grid,
    generate_live_affine_sbox,
    list_available_sboxes,
    normalize_16byte_secret,
    parse_uploaded_sbox,
    run_image_simulation,
)

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
MANDRILL_PATH = REPO_ROOT / "data" / "mandril_color.jpg"
ARCHIFY_HTML_PATH = REPO_ROOT / "assets" / "cryptoml_sbox_archify.html"
XGB_METRICS_PATH = RESULTS_DIR / "xgb_metrics.json"

PAGES = [
    "Ringkasan Proyek",
    "Diagram Alur & Arsitektur",
    "Simulasi Langsung",
]

st.set_page_config(
    page_title="CryptoML-SBox — Konstruksi S-Box AES Berbasis Machine Learning",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Source+Sans+3:wght@400;500;600;700&display=swap');

:root {
  --bg: #F7F8FA;
  --surface: #FFFFFF;
  --surface-subtle: #EDF1F4;
  --ink: #141D24;
  --ink-muted: #465664;
  --accent: #0F5C5C;
  --accent-hover: #0B4747;
  --accent-tint: #E9F3F3;
  --amber: #8A4307;
  --amber-tint: #FDF3E7;
  --border: #D5DEE6;
  --shadow-sm: 0 1px 2px rgba(20, 29, 36, 0.04), 0 6px 18px rgba(20, 29, 36, 0.04);
}

::selection {
  background: #0F5C5C;
  color: #FFFFFF;
}

::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: var(--surface-subtle);
}

::-webkit-scrollbar-thumb {
  background: #B3C1CC;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #8FA3B2;
}

html, body, [class*="css"] {
  font-family: 'Source Sans 3', -apple-system, BlinkMacSystemFont, sans-serif;
  color: var(--ink);
}

.stApp {
  background-color: var(--bg);
}

.block-container {
  max-width: 1220px;
  padding-top: 4.35rem;
  padding-bottom: 3.5rem;
}

header[data-testid="stHeader"] {
  background: rgba(247, 248, 250, 0.92);
  border-bottom: 1px solid #E5ECF1;
}

h1, h2, h3, .display-title {
  font-family: 'Bricolage Grotesque', sans-serif !important;
  color: var(--ink) !important;
  letter-spacing: -0.025em;
}

code, pre, .mono-val {
  font-family: 'JetBrains Mono', monospace !important;
  font-variant-numeric: tabular-nums;
}

table, .stDataFrame, [data-testid="stMetricValue"] {
  font-variant-numeric: tabular-nums;
}

/* Header & top navigation bar */
.brand-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  padding-bottom: 0.9rem;
  margin-bottom: 1.1rem;
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}

.brand-mark {
  display: inline-flex;
  align-items: center;
  gap: 0.65rem;
}

.brand-name {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-size: 1.2rem;
  font-weight: 700;
  color: var(--ink);
  letter-spacing: -0.02em;
}

.brand-sub {
  font-size: 0.92rem;
  color: var(--ink-muted);
}

.nav-radio-wrap div[role="radiogroup"] {
  display: flex;
  flex-direction: row;
  gap: 0.45rem;
  background: var(--surface-subtle);
  padding: 0.3rem;
  border-radius: 10px;
  border: 1px solid var(--border);
  width: fit-content;
}

.nav-radio-wrap div[role="radiogroup"] > label {
  background: transparent;
  padding: 0.48rem 1.05rem;
  border-radius: 7px;
  margin: 0 !important;
  cursor: pointer;
  transition: background-color 140ms ease, color 140ms ease, box-shadow 140ms ease;
}

.nav-radio-wrap div[role="radiogroup"] > label > div:first-child {
  display: none;
}

.nav-radio-wrap div[role="radiogroup"] > label p {
  font-size: 0.96rem;
  font-weight: 600;
  color: var(--ink-muted);
  margin: 0;
}

.nav-radio-wrap div[role="radiogroup"] > label:has(input:checked) {
  background: var(--surface);
  box-shadow: 0 1px 3px rgba(20, 29, 36, 0.1);
}

.nav-radio-wrap div[role="radiogroup"] > label:has(input:checked) p {
  color: var(--accent);
}

/* Editorial lead & proof strip */
.lead-title {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-size: clamp(2rem, 3.2vw, 2.75rem);
  line-height: 1.1;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--ink);
  margin: 0 0 0.85rem 0;
}

.lead-copy {
  font-size: 1.12rem;
  line-height: 1.62;
  color: #2A3843;
  max-width: 62ch;
  margin: 0 0 1.25rem 0;
}

.section-heading {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-size: 1.55rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin: 2.2rem 0 0.5rem 0;
}

.section-sub {
  font-size: 1rem;
  color: var(--ink-muted);
  max-width: 70ch;
  margin: 0 0 1.2rem 0;
  line-height: 1.55;
}

.proof-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.15rem 1.25rem;
  box-shadow: var(--shadow-sm);
}

.stat-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1.15rem;
  margin-top: 1.35rem;
  padding: 1.05rem 1.25rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  box-shadow: var(--shadow-sm);
}

@media (max-width: 820px) {
  .stat-strip {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

.stat-item {
  padding-right: 0.5rem;
}

.stat-item-label {
  font-size: 0.84rem;
  color: var(--ink-muted);
  font-weight: 600;
  margin-bottom: 0.2rem;
}

.stat-item-value {
  font-family: 'JetBrains Mono', monospace;
  font-size: 1.35rem;
  font-weight: 600;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
  line-height: 1.2;
}

.stat-item-note {
  font-size: 0.82rem;
  color: var(--ink-muted);
  margin-top: 0.2rem;
}

/* Ruled 3-step editorial strip (not boxed cards) */
.step-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1.8rem;
  margin-top: 0.75rem;
}

@media (max-width: 820px) {
  .step-row {
    grid-template-columns: 1fr;
    gap: 1.2rem;
  }
}

.step-col {
  border-top: 2px solid var(--ink);
  padding-top: 0.9rem;
}

.step-head {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin-bottom: 0.45rem;
}

.step-title {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-size: 1.12rem;
  font-weight: 600;
  color: var(--ink);
  margin: 0;
}

.step-body {
  font-size: 0.97rem;
  line-height: 1.58;
  color: #344450;
  margin: 0;
}

/* Status pills */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.2rem 0.6rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
  white-space: nowrap;
}

.pill-done {
  background: var(--accent-tint);
  color: var(--accent);
}

.pill-wip {
  background: var(--amber-tint);
  color: var(--amber);
}

.pill-base {
  background: var(--surface-subtle);
  color: #364552;
}

/* Team ledger table */
.team-table {
  width: 100%;
  border-collapse: collapse;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  font-size: 0.95rem;
}

.team-table th {
  text-align: left;
  padding: 0.72rem 0.95rem;
  background: var(--surface-subtle);
  color: var(--ink-muted);
  font-weight: 600;
  font-size: 0.84rem;
  border-bottom: 1px solid var(--border);
}

.team-table td {
  padding: 0.82rem 0.95rem;
  border-bottom: 1px solid #E8EEF3;
  vertical-align: top;
}

.team-table tr:last-child td {
  border-bottom: none;
}

.team-member {
  font-weight: 700;
  color: var(--ink);
}

.team-handle {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.82rem;
  color: var(--ink-muted);
}

/* Archify frame */
.archify-frame {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.35rem;
  box-shadow: var(--shadow-sm);
}

.guide-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  margin-bottom: 0.9rem;
  font-size: 0.93rem;
  color: #2D3D4A;
}

/* Metric verification banner */
.verify-ok {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  background: var(--accent-tint);
  color: var(--accent);
  border: 1px solid #B7D8D8;
  border-radius: 10px;
  padding: 0.72rem 1rem;
  font-weight: 600;
  font-size: 0.96rem;
  margin: 0.8rem 0 1rem 0;
}

.verify-warn {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  background: var(--amber-tint);
  color: var(--amber);
  border: 1px solid #EBD0AE;
  border-radius: 10px;
  padding: 0.72rem 1rem;
  font-weight: 600;
  font-size: 0.96rem;
  margin: 0.8rem 0 1rem 0;
}

/* Focus rings & buttons */
button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible {
  outline: 2px solid var(--accent) !important;
  outline-offset: 2px !important;
}

.stButton > button {
  white-space: nowrap;
}

.stButton > button[kind="primary"] {
  background-color: var(--accent);
  border-color: var(--accent);
  color: #FFFFFF;
  font-weight: 600;
}

.stButton > button[kind="primary"]:hover {
  background-color: var(--accent-hover);
  border-color: var(--accent-hover);
}
</style>
"""

SVG_BRAND = (
    '<svg width="26" height="26" viewBox="0 0 24 24" fill="none" '
    'stroke="#0F5C5C" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="3" y="3" width="7" height="7" rx="1.5"/>'
    '<rect x="14" y="3" width="7" height="7" rx="1.5"/>'
    '<rect x="3" y="14" width="7" height="7" rx="1.5"/>'
    '<path d="M14 17.5h7M17.5 14v7"/></svg>'
)

SVG_GRID = (
    '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" '
    'stroke="#0F5C5C" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M3 3h18v18H3zM9 3v18M15 3v18M3 9h18M3 15h18"/></svg>'
)

SVG_SLIDERS = (
    '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" '
    'stroke="#0F5C5C" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/></svg>'
)

SVG_SHIELD = (
    '<svg width="19" height="19" viewBox="0 0 24 24" fill="none" '
    'stroke="#0F5C5C" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>'
)

SVG_CHECK = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>'
)


@st.cache_data(show_spinner=False)
def load_catalog_and_metrics() -> tuple[dict[str, dict], pd.DataFrame]:
    catalog = list_available_sboxes(RESULTS_DIR)
    rows = []
    for name, info in catalog.items():
        m = evaluate_sbox_full(info["sbox"])
        rows.append(
            {
                "S-Box": name,
                "Kategori": info["status"],
                "Poli": info.get("poly") or "-",
                "Bij": "Ya" if m["bijective"] else "Tidak",
                "NL": int(m["nl"]),
                "SAC": round(float(m["sac_mean"]), 5),
                "SAC Max Err": round(float(m["sac_max_err"]), 5),
                "BIC-NL": int(m["bic_nl"]),
                "BIC-SAC": round(float(m["bic_sac"]), 5),
                "DU": int(m["du"]),
                "DAP": round(float(m["dap"]), 6),
                "LAP": round(float(m["lap"]), 4),
                "FP": int(m["fixed_points"]),
                "Skor": round(float(m["composite_score"]), 2),
            }
        )
    return catalog, pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_xgb_summary() -> dict:
    if XGB_METRICS_PATH.exists():
        return json.loads(XGB_METRICS_PATH.read_text(encoding="utf-8"))
    return {}


@st.cache_data(show_spinner=False)
def load_mandrill_array() -> np.ndarray:
    img = Image.open(MANDRILL_PATH).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


@st.cache_data(show_spinner=False)
def build_hero_mandrill_preview(sbox_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Enkripsi preview cepat Mandrill 160x160 untuk demonstrasi langsung di halaman depan."""
    sbox = np.frombuffer(sbox_bytes, dtype=np.uint8)
    plain = load_mandrill_array()
    key = normalize_16byte_secret("CryptoML-SBox26!", mode="utf8")
    iv = normalize_16byte_secret("UNNES-Kripto2026", mode="utf8")
    sim = run_image_simulation(plain, sbox=sbox, key=key, iv=iv, mode="CBC2", max_dim=160)
    return sim["plain"], sim["cipher"]


@st.cache_data(show_spinner=False)
def load_archify_html() -> str:
    if ARCHIFY_HTML_PATH.exists():
        return ARCHIFY_HTML_PATH.read_text(encoding="utf-8")
    return ""


def _fig_to_clean_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C5D1DC")
    ax.spines["bottom"].set_color("#C5D1DC")
    ax.tick_params(colors="#344450", labelsize=9)


def render_comparison_chart(df_metrics: pd.DataFrame):
    labels = [name.replace(" (FIPS-197)", "").replace(" (2024/2025)", "") for name in df_metrics["S-Box"]]
    sac_err = df_metrics["SAC Max Err"].to_numpy(dtype=float)
    bic_dev = np.abs(df_metrics["BIC-SAC"].to_numpy(dtype=float) - 0.5)

    colors = [
        "#0F5C5C" if "XGBoost" in lbl or "Random Forest" in lbl or "MLP" in lbl else "#90A2B2"
        for lbl in labels
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(labels))
    for ax, vals, title, fmt in [
        (axes[0], sac_err, "Deviasi Maksimum SAC  |P - 0.5|  (lebih kecil lebih baik)", "{:.5f}"),
        (axes[1], bic_dev, "Jarak BIC-SAC ke Ideal 0.5  (lebih kecil lebih baik)", "{:.5f}"),
    ]:
        ax.set_facecolor("#FFFFFF")
        bars = ax.barh(y_pos, vals, color=colors, height=0.52, edgecolor="none")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9.5, color="#141D24")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, fontweight="bold", color="#141D24", loc="left", pad=10)
        ax.grid(axis="x", linestyle="--", linewidth=0.6, color="#E1E8EE")
        ax.set_axisbelow(True)
        _fig_to_clean_style(ax)
        max_v = max(float(np.max(vals)), 1e-6)
        ax.set_xlim(0, max_v * 1.28)
        for bar, val in zip(bars, vals):
            ax.text(
                val + max_v * 0.025,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(val),
                va="center",
                ha="left",
                fontsize=8.8,
                family="monospace",
                color="#141D24",
            )

    fig.tight_layout(w_pad=3.2)
    return fig


def render_sac_heatmap(sac_mat: np.ndarray):
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    vmax = max(0.065, float(np.max(np.abs(sac_mat - 0.5))))
    im = ax.imshow(sac_mat, cmap="BrBG", vmin=0.5 - vmax, vmax=0.5 + vmax, aspect="equal")
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels([f"b{i}" for i in range(8)], fontsize=8.5, family="monospace")
    ax.set_yticklabels([f"in{i}" for i in range(8)], fontsize=8.5, family="monospace")
    ax.set_xlabel("Bit Output", fontsize=9.5, color="#344450", labelpad=6)
    ax.set_ylabel("Bit Input Dibalik", fontsize=9.5, color="#344450", labelpad=6)
    ax.set_title("Matriks SAC 8x8 (Ideal = 0.500)", fontsize=10.5, fontweight="bold", color="#141D24", pad=10)

    for r in range(8):
        for c in range(8):
            val = sac_mat[r, c]
            txt_color = "#FFFFFF" if abs(val - 0.5) > vmax * 0.58 else "#141D24"
            ax.text(
                c,
                r,
                f"{val:.3f}",
                ha="center",
                va="center",
                fontsize=7.3,
                family="monospace",
                color=txt_color,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


def render_histogram_comparison(plain: np.ndarray, cipher: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.1), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    bins = np.arange(257)

    for ax, arr, title in [
        (axes[0], plain, "Histogram Citra Asli (Pola Tidak Merata)"),
        (axes[1], cipher, "Histogram Citra Terenkripsi (Distribusi Seragam)"),
    ]:
        ax.set_facecolor("#FFFFFF")
        if arr.ndim == 3:
            for ch, col, lbl in [(0, "#C0392B", "R"), (1, "#1E8449", "G"), (2, "#2471A3", "B")]:
                counts, _ = np.histogram(arr[:, :, ch].ravel(), bins=bins)
                ax.plot(np.arange(256), counts, color=col, linewidth=1.0, alpha=0.85, label=lbl)
            ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        else:
            counts, _ = np.histogram(arr.ravel(), bins=bins)
            ax.fill_between(np.arange(256), counts, color="#0F5C5C", alpha=0.35)
            ax.plot(np.arange(256), counts, color="#0F5C5C", linewidth=1.1)

        ax.set_xlim(0, 255)
        ax.set_xlabel("Intensitas Piksel (0-255)", fontsize=8.8, color="#344450")
        ax.set_ylabel("Frekuensi", fontsize=8.8, color="#344450")
        ax.set_title(title, fontsize=10, fontweight="bold", color="#141D24", loc="left", pad=8)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#E1E8EE")
        _fig_to_clean_style(ax)

    fig.tight_layout(w_pad=2.4)
    return fig


def _array_to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _set_active_page(page_name: str) -> None:
    st.session_state["active_page"] = page_name


def render_page_overview(catalog: dict[str, dict], df_metrics: pd.DataFrame):
    xgb_summary = load_xgb_summary()
    xgb_entry = catalog.get("S-Box XGBoost (Mahdi)") or next(iter(catalog.values()))
    preview_plain, preview_cipher = build_hero_mandrill_preview(xgb_entry["sbox"].tobytes())

    lead_left, lead_right = st.columns([1.15, 1.0], gap="large")
    with lead_left:
        st.markdown(
            '<h1 class="lead-title">Merancang S-Box AES Lebih Kuat dengan Machine Learning</h1>'
            '<p class="lead-copy">'
            'Di dalam enkripsi <strong>AES-128</strong>, <strong>S-Box</strong> adalah tabel pengacak '
            '256 angka yang menjaga data tetap rahasia. Alih-alih menguji kandidat satu per satu secara lambat, '
            'sistem kami melatih <strong>XGBoost</strong>, <strong>Random Forest</strong>, dan <strong>MLP</strong> '
            'untuk menyaring <strong>15.000 kandidat S-Box</strong> secara cepat lalu membuktikan ketahanannya '
            'langsung pada enkripsi citra <strong>Mandrill</strong>.'
            '</p>',
            unsafe_allow_html=True,
        )
        btn_col1, btn_col2 = st.columns([1.0, 1.0], gap="small")
        with btn_col1:
            st.button(
                "Coba Simulasi Langsung",
                type="primary",
                use_container_width=True,
                on_click=_set_active_page,
                args=(PAGES[2],),
            )
        with btn_col2:
            st.button(
                "Lihat Diagram Arsitektur",
                use_container_width=True,
                on_click=_set_active_page,
                args=(PAGES[1],),
            )

    with lead_right:
        st.markdown('<div class="proof-panel">', unsafe_allow_html=True)
        img_c1, img_c2 = st.columns(2, gap="small")
        with img_c1:
            st.image(preview_plain, caption="Mandrill Asli (RGB)", use_container_width=True)
        with img_c2:
            st.image(
                preview_cipher,
                caption="Terenkripsi AES-CBC2 + S-Box XGBoost",
                use_container_width=True,
            )
        st.markdown(
            '<div style="font-size:0.88rem; color:#465664; margin-top:0.35rem; line-height:1.5;">'
            'Pratinjau di atas dienkripsi secara nyata oleh mesin <code class="mono-val">aes_cipher_runner.py</code> '
            'memakai S-Box terpilih XGBoost (<code class="mono-val">poly=0x13F, C=0x02, Fixed Points=0</code>). '
            'Dekripsi mengembalikan setiap piksel identik 100%.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="stat-strip">'
        '<div class="stat-item"><div class="stat-item-label">Nonlinearity (NL)</div>'
        '<div class="stat-item-value">112</div>'
        '<div class="stat-item-note">Batas maksimal keluarga AES (DU = 4, LAP = 0.0625)</div></div>'
        '<div class="stat-item"><div class="stat-item-label">SAC Mean &amp; Max Error</div>'
        '<div class="stat-item-value">0.50024</div>'
        '<div class="stat-item-note">Max Err 0.04688 (25% lebih ketat dari AES &amp; K44)</div></div>'
        '<div class="stat-item"><div class="stat-item-label">Entropi Citra Mandrill</div>'
        '<div class="stat-item-value">7.99973</div>'
        '<div class="stat-item-note">Dari 7.76244 menuju ideal 8.000 bit (512×512 CBC2)</div></div>'
        '<div class="stat-item"><div class="stat-item-label">Difusi NPCR &amp; UACI</div>'
        '<div class="stat-item-value">99.62%</div>'
        '<div class="stat-item-note">UACI 33.43% · Dekripsi 100% Lossless</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # 3-step plain-language explainer
    st.markdown(
        '<h2 class="section-heading">Bagaimana Sistem Ini Bekerja dalam 3 Langkah</h2>'
        '<p class="section-sub">Dirancang agar mudah dipahami tanpa latar belakang kriptografi mendalam, '
        'sekaligus tetap ketat terhadap seluruh metrik uji standar.</p>'
        '<div class="step-row">'
        f'<div class="step-col"><div class="step-head">{SVG_GRID}'
        '<h3 class="step-title">1. Bangkitkan Ribuan Kandidat</h3></div>'
        '<p class="step-body">Program menghasilkan kandidat S-Box bijektif (angka 0–255 unik tanpa duplikat) '
        'dari dua jalur: variasi 30 polinomial <em>irreducible</em> GF(2⁸) + matriks <em>affine</em> invertibel, '
        'serta jalur Non-AES berbasis <em>chaotic map</em>.</p></div>'
        f'<div class="step-col"><div class="step-head">{SVG_SLIDERS}'
        '<h3 class="step-title">2. Saring Cepat dengan Machine Learning</h3></div>'
        '<p class="step-body">Setiap S-Box diekstrak menjadi 40 fitur cepat (~0,7 ms/S-Box). '
        'Model <strong>XGBoost</strong> (R²=0,9149), <strong>Random Forest</strong>, dan <strong>MLP</strong> '
        'memprediksi kualitas kriptografinya lalu menyaring 15.000 kandidat hingga tersisa Top-40 untuk diuji penuh.</p></div>'
        f'<div class="step-col"><div class="step-head">{SVG_SHIELD}'
        '<h3 class="step-title">3. Buktikan pada AES-128 &amp; Mandrill</h3></div>'
        '<p class="step-body">S-Box pemenang dipasang langsung ke AES-128 mode <code>CBC2</code> (difusi 2 arah) '
        'untuk mengenkripsi citra Mandrill 512×512. Hasil diukur melalui Entropi, Histogram, NPCR, UACI, '
        'dan verifikasi dekripsi <em>lossless</em>.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Live Benchmark Comparison Table & Chart
    st.markdown(
        '<h2 class="section-heading">Perbandingan Kualitas S-Box (Metrik 1)</h2>'
        '<p class="section-sub">Tabel dan grafik berikut dihitung langsung dari artefak <code>.npy</code> '
        'di repositori. Saat S-Box Random Forest (Yafi) dan MLP (Naila) di-merge, tabel ini otomatis bertambah.</p>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        df_metrics,
        use_container_width=True,
        hide_index=True,
        height=38 * len(df_metrics) + 42,
    )

    chart_fig = render_comparison_chart(df_metrics)
    st.pyplot(chart_fig, use_container_width=True)
    plt.close(chart_fig)

    if xgb_summary:
        ml = xgb_summary.get("ml_metrics", {})
        st.caption(
            f"Log Pelatihan Kaggle (XGBoost Tuned): R² = {ml.get('r2', 0):.4f} | "
            f"MAE = {ml.get('mae', 0):.4f} | RMSE = {ml.get('rmse', 0):.4f} | "
            f"Waktu latih = {ml.get('train_time_sec', 0):.2f} dtk | "
            f"Penyaringan {xgb_summary.get('n_screen_candidates', 0):,} kandidat = "
            f"{xgb_summary.get('screen_time_sec', 0):.2f} dtk."
        )

    # Team ledger
    has_rf = (RESULTS_DIR / "sbox_rf.npy").exists()
    has_mlp = (RESULTS_DIR / "sbox_mlp.npy").exists()
    rf_pill = '<span class="pill pill-done">Selesai (.npy)</span>' if has_rf else '<span class="pill pill-wip">Tahap Pengembangan</span>'
    mlp_pill = '<span class="pill pill-done">Selesai (.npy)</span>' if has_mlp else '<span class="pill pill-wip">Tahap Pengembangan</span>'
    has_img_mod = (REPO_ROOT / "src" / "image_metrics.py").exists()
    rizqi_pill = '<span class="pill pill-done">Modul Siap</span>' if has_img_mod else '<span class="pill pill-wip">Tahap Pengembangan</span>'

    st.markdown(
        '<h2 class="section-heading">Pembagian Pengembangan Tim (4 Anggota)</h2>'
        '<p class="section-sub">Setiap anggota memegang satu jalur tanggung jawab independen '
        'dengan kontrak antarmuka seragam di <code>docs/ARCHITECTURE_CONTRACT.md</code>.</p>'
        '<table class="team-table">'
        '<thead><tr><th>Anggota</th><th>Model ML &amp; Modul Kriptografi</th><th>Keluaran Utama</th><th>Status</th></tr></thead>'
        '<tbody>'
        '<tr><td><div class="team-member">Mahdi</div><div class="team-handle">@MahdiHbb1</div></td>'
        '<td>Model <strong>XGBoost</strong>, Generator GF(2⁸) (30 polinomial + GL(8,F₂)), Runner AES-128 CBC2, Integrasi &amp; Web Streamlit</td>'
        '<td><code>sbox_xgb.npy</code>, <code>xgb_metrics.json</code>, 39 unit test</td>'
        '<td><span class="pill pill-done">Merged ke main</span></td></tr>'
        '<tr><td><div class="team-member">Yafi</div><div class="team-handle">@twizzcode</div></td>'
        '<td>Model <strong>Random Forest</strong>, Generator Non-AES (Chaotic + Swap), Ekstraksi Fitur Cepat &amp; Dataset Bersama</td>'
        '<td><code>sbox_dataset.csv</code>, <code>train_rf.py</code>, <code>sbox_rf.npy</code></td>'
        f'<td>{rf_pill}</td></tr>'
        '<tr><td><div class="team-member">Naila</div><div class="team-handle">@NAILAAR22</div></td>'
        '<td>Model <strong>MLP</strong> (StandardScaler + MLPRegressor), Evaluator Metrik 1 (255-mask NL, SAC, BIC, DU, LAP) &amp; Validasi AES/K44</td>'
        '<td><code>sbox_metrics.py</code>, <code>train_mlp.py</code>, <code>sbox_mlp.npy</code></td>'
        f'<td>{mlp_pill}</td></tr>'
        '<tr><td><div class="team-member">Mas Rizqi</div><div class="team-handle">@rizqinp2</div></td>'
        '<td>Evaluasi Keamanan Citra Mandrill (Metrik 2 &amp; 3: Entropy, Histogram χ², NPCR, UACI) untuk 5 S-Box &amp; Visualisasi Laporan</td>'
        '<td><code>image_metrics.py</code>, grafik perbandingan akhir</td>'
        f'<td>{rizqi_pill}</td></tr>'
        '</tbody></table>',
        unsafe_allow_html=True,
    )


def render_page_architecture():
    st.markdown(
        '<h1 class="lead-title" style="font-size:2.1rem; margin-bottom:0.45rem;">'
        'Diagram Alur Kerja Sistem, Pengembangan Tim &amp; Hasil</h1>'
        '<p class="section-sub" style="margin-bottom:1rem;">'
        'Visualisasi interaktif berbasis <strong>Archify</strong> yang terikat langsung ke baris kode repositori GitHub. '
        'Gunakan menu <strong>Views</strong> di dalam diagram untuk menyorot jalur sistem, pembagian 4 anggota, atau hasil terverifikasi.'
        '</p>',
        unsafe_allow_html=True,
    )

    archify_html = load_archify_html()
    if not archify_html:
        st.error(
            "File diagram Archify tidak ditemukan di `assets/cryptoml_sbox_archify.html`. "
            "Pastikan repositori sudah menarik commit terbaru."
        )
        return

    bar_left, bar_right = st.columns([3.2, 1.0], vertical_alignment="center")
    with bar_left:
        st.markdown(
            '<div class="guide-bar">'
            '<span><strong>Tips Eksplorasi:</strong> Klik <strong>node</strong> untuk membuka tautan kode GitHub · '
            'Pilih <strong>Views (1–3)</strong> untuk fokus tahap tertentu · Gunakan <strong>Light/Dark</strong>, '
            '<strong>Live/Still</strong>, atau <strong>Export PNG/SVG</strong>.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    with bar_right:
        st.download_button(
            "Unduh Diagram (.html)",
            data=archify_html.encode("utf-8"),
            file_name="cryptoml_sbox_architecture_archify.html",
            mime="text/html",
            use_container_width=True,
        )

    st.markdown('<div class="archify-frame">', unsafe_allow_html=True)
    components.html(archify_html, height=1060, scrolling=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<h2 class="section-heading">Ringkasan 3 Fase Pengembangan &amp; Kontrak Integrasi</h2>'
        '<div class="step-row">'
        '<div class="step-col"><h3 class="step-title">Fase 1 — Fondasi Kriptografi</h3>'
        '<p class="step-body">Membangun generator GF(2⁸) (Mahdi), generator Non-AES chaotic (Yafi), '
        'evaluator Metrik 1 <code>sbox_metrics.py</code> (Naila), runner <code>aes_cipher_runner.py</code> (Mahdi), '
        'dan penguji citra <code>image_metrics.py</code> (Mas Rizqi).</p></div>'
        '<div class="step-col"><h3 class="step-title">Fase 2 — Pelatihan 3 Model ML</h3>'
        '<p class="step-body">Seluruh model (<strong>XGBoost</strong>, <strong>Random Forest</strong>, '
        '<strong>MLP</strong>) memakai dataset dan <code>random_state=42</code> yang sama agar perbandingan '
        'R², MAE, RMSE, dan S-Box pemenang <code>uint8(256)</code> bersifat adil dan objektif.</p></div>'
        '<div class="step-col"><h3 class="step-title">Fase 3 — Uji Mandrill &amp; Komparasi</h3>'
        '<p class="step-body">Kelima S-Box (AES Standar, K44, XGBoost, RF, MLP) diuji serentak pada '
        'citra Mandrill 512×512 mode <code>CBC2</code> untuk mengukur Entropy, Histogram χ², NPCR, UACI, '
        'dan verifikasi dekripsi lossless.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_sim_sbox_tab(catalog: dict[str, dict]):
    st.markdown(
        '<p class="section-sub" style="margin-top:0.2rem;">'
        'Inspeksi S-Box hasil proyek, bangkitkan kandidat baru GF(2⁸) secara langsung, '
        'atau unggah file S-Box Anda sendiri (<code>.npy</code> / <code>.csv</code> / <code>.txt</code>) '
        'untuk diuji seluruh Metrik 1 dan matriks SAC-nya.</p>',
        unsafe_allow_html=True,
    )

    source_mode = st.radio(
        "Sumber S-Box",
        [
            "Katalog Proyek (AES, K44, ML)",
            "Bangkitkan S-Box GF(2^8) Baru",
            "Unggah File S-Box (.npy / .csv)",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    selected_name = ""
    selected_sbox: np.ndarray | None = None
    selected_meta = ""

    if source_mode == "Katalog Proyek (AES, K44, ML)":
        names = list(catalog.keys())
        default_idx = names.index("S-Box XGBoost (Mahdi)") if "S-Box XGBoost (Mahdi)" in names else 0
        selected_name = st.selectbox("Pilih S-Box untuk diinspeksi", names, index=default_idx)
        entry = catalog[selected_name]
        selected_sbox = entry["sbox"]
        selected_meta = (
            f"Polinomial: {entry.get('poly') or '-'} · Konstanta: {entry.get('constant') or '-'} · "
            f"{entry.get('notes', '')}"
        )

    elif source_mode == "Bangkitkan S-Box GF(2^8) Baru":
        g1, g2, g3 = st.columns([1.4, 1.0, 1.0], vertical_alignment="bottom")
        poly_options = ["Acak (30 Polinomial)"] + [f"0x{p:03X}" for p in IRREDUCIBLE_POLYS_GF256]
        with g1:
            poly_choice = st.selectbox("Polinomial Irreducible GF(2⁸)", poly_options)
        with g2:
            seed_val = int(st.number_input("Seed Acak", min_value=0, max_value=999999, value=2026, step=1))
        with g3:
            gen_clicked = st.button("Bangkitkan S-Box Baru", type="primary", use_container_width=True)

        if gen_clicked or "live_gen_sbox" not in st.session_state:
            poly_int = None if poly_choice.startswith("Acak") else int(poly_choice, 16)
            st.session_state["live_gen_sbox"] = generate_live_affine_sbox(seed=seed_val, poly=poly_int)

        gen = st.session_state["live_gen_sbox"]
        selected_name = f"S-Box GF(2⁸) Live (0x{gen['poly']:03X}, C=0x{gen['constant']:02X})"
        selected_sbox = gen["sbox"]
        selected_meta = (
            f"Dibangkitkan langsung dari matriks rank-8 GL(8, F₂), polinomial 0x{gen['poly']:03X}, "
            f"konstanta bebas fixed point C=0x{gen['constant']:02X}."
        )

    else:
        uploaded = st.file_uploader(
            "Unggah S-Box 256 elemen (.npy, .csv, atau .txt)",
            type=["npy", "csv", "txt"],
        )
        if uploaded is not None:
            try:
                selected_sbox = parse_uploaded_sbox(uploaded.getvalue(), uploaded.name)
                selected_name = f"S-Box Unggahan ({uploaded.name})"
                selected_meta = "Tervalidasi bijektif (seluruh nilai 0..255 muncul tepat satu kali)."
            except ValueError as err:
                st.error(f"Gagal memuat S-Box: {err}")
                return
        else:
            st.info("Unggah file `.npy` (array 256 byte) atau `.csv`/`.txt` berisi 256 angka desimal/hex (0–255).")
            return

    assert selected_sbox is not None
    st.caption(selected_meta)

    metrics = evaluate_sbox_full(selected_sbox)
    aes_ref = evaluate_sbox_full(catalog["AES Standar (FIPS-197)"]["sbox"])
    sac_mat = compute_sac_matrix(selected_sbox)

    # Metric 1 highlights
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Nonlinearity", f"{metrics['nl']}", delta=f"{metrics['nl'] - aes_ref['nl']:+d} vs AES")
    m2.metric(
        "SAC Mean",
        f"{metrics['sac_mean']:.5f}",
        delta=f"{abs(metrics['sac_mean'] - 0.5) - abs(aes_ref['sac_mean'] - 0.5):+.5f} err",
        delta_color="inverse",
    )
    m3.metric(
        "SAC Max Err",
        f"{metrics['sac_max_err']:.5f}",
        delta=f"{metrics['sac_max_err'] - aes_ref['sac_max_err']:+.5f}",
        delta_color="inverse",
    )
    m4.metric(
        "BIC-SAC",
        f"{metrics['bic_sac']:.5f}",
        delta=f"{abs(metrics['bic_sac'] - 0.5) - abs(aes_ref['bic_sac'] - 0.5):+.5f} err",
        delta_color="inverse",
    )
    m5.metric("DU / DAP", f"{metrics['du']} ({metrics['dap']:.4f})", delta=f"LAP {metrics['lap']:.4f}", delta_color="off")
    m6.metric(
        "Fixed Points",
        f"{metrics['fixed_points']}",
        delta=f"Opp-FP: {metrics['opposite_fixed_points']}",
        delta_color="off",
    )

    grid_col, heat_col = st.columns([1.35, 1.0], gap="large")
    with grid_col:
        top_g1, top_g2 = st.columns([1.5, 1.0], vertical_alignment="center")
        with top_g1:
            st.markdown(f"**Tabel S-Box 16×16 — {selected_name}**")
        with top_g2:
            fmt_choice = st.radio(
                "Format tabel",
                ["Heksadesimal", "Desimal"],
                horizontal=True,
                label_visibility="collapsed",
            )
        grid_df = format_sbox_grid(selected_sbox, fmt="hex" if fmt_choice == "Heksadesimal" else "dec")
        st.dataframe(grid_df, use_container_width=True, height=430)

        dl1, dl2 = st.columns(2)
        npy_buf = io.BytesIO()
        np.save(npy_buf, selected_sbox)
        csv_text = grid_df.to_csv()
        with dl1:
            st.download_button(
                "Unduh S-Box (.npy)",
                data=npy_buf.getvalue(),
                file_name="sbox_selected.npy",
                mime="application/octet-stream",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "Unduh Tabel 16x16 (.csv)",
                data=csv_text.encode("utf-8"),
                file_name="sbox_16x16.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with heat_col:
        sac_fig = render_sac_heatmap(sac_mat)
        st.pyplot(sac_fig, use_container_width=True)
        plt.close(sac_fig)
        st.caption(
            f"Bijective: {'Ya' if metrics['bijective'] else 'Tidak'} · "
            f"Balanced: {'Ya' if metrics['balanced'] else 'Tidak'} · "
            f"BIC-NL: {metrics['bic_nl']} · Skor Gabungan: {metrics['composite_score']:.3f}"
        )


def render_sim_image_tab(catalog: dict[str, dict]):
    st.markdown(
        '<p class="section-sub" style="margin-top:0.2rem;">'
        'Uji langsung enkripsi &amp; dekripsi citra <strong>Mandrill</strong> atau foto Anda sendiri '
        'memakai AES-128 custom S-Box. Sistem otomatis menghitung Entropy, Histogram, NPCR, UACI '
        '(perubahan 1 bit pada piksel terakhir), dan verifikasi lossless.</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([1.35, 1.0, 1.05, 1.1])
    names = list(catalog.keys())
    default_idx = names.index("S-Box XGBoost (Mahdi)") if "S-Box XGBoost (Mahdi)" in names else 0
    with c1:
        sbox_name = st.selectbox("Pilih S-Box", names, index=default_idx, key="img_sbox_name")
    with c2:
        color_mode = st.selectbox("Kanal Warna", ["RGB (24-bit)", "Grayscale (8-bit)"], key="img_color")
    with c3:
        aes_mode_label = st.selectbox(
            "Mode Operasi AES",
            ["CBC2 (Difusi Penuh)", "CBC (1 Lintasan)", "ECB (Blok Independen)"],
            key="img_mode",
        )
    with c4:
        res_label = st.selectbox(
            "Resolusi Simulasi",
            ["256×256 (Cepat ~0.8 dtk)", "128×128 (Instan)", "512×512 (Penuh ~3.2 dtk)"],
            key="img_res",
        )

    img_src_col, key_col, iv_col = st.columns([1.4, 1.1, 1.1])
    with img_src_col:
        img_source = st.radio(
            "Sumber Citra",
            ["Mandrill Standar (data/mandril_color.jpg)", "Unggah Gambar Sendiri"],
            horizontal=True,
        )
    with key_col:
        key_text = st.text_input("Kunci AES-128 (16 karakter UTF-8)", value="CryptoML-SBox26!", key="img_key")
    with iv_col:
        iv_text = st.text_input("Initialization Vector / IV (16 kar)", value="UNNES-Kripto2026", key="img_iv")

    uploaded_img = None
    if img_source == "Unggah Gambar Sendiri":
        uploaded_img = st.file_uploader("Pilih file gambar (PNG / JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_img is None:
            st.info("Unggah gambar JPG/PNG untuk memulai simulasi, atau pilih Mandrill Standar.")
            return

    run_btn = st.button("Jalankan Simulasi Enkripsi & Dekripsi Citra", type="primary")
    if run_btn:
        try:
            key_b = normalize_16byte_secret(key_text, mode="utf8")
            iv_b = normalize_16byte_secret(iv_text, mode="utf8")
            if uploaded_img is not None:
                pil_img = Image.open(uploaded_img)
            else:
                pil_img = Image.open(MANDRILL_PATH)
            pil_img = pil_img.convert("L" if color_mode.startswith("Grayscale") else "RGB")
            img_arr = np.asarray(pil_img, dtype=np.uint8)

            aes_mode = aes_mode_label.split()[0]
            max_dim = 128 if res_label.startswith("128") else (256 if res_label.startswith("256") else 512)

            with st.spinner("Menjalankan AES-128 custom S-Box, uji difusi 1-bit, dan dekripsi lossless..."):
                sim = run_image_simulation(
                    img_arr=img_arr,
                    sbox=catalog[sbox_name]["sbox"],
                    key=key_b,
                    iv=iv_b,
                    mode=aes_mode,
                    max_dim=max_dim,
                )
            st.session_state["last_img_sim"] = {
                "sim": sim,
                "sbox_name": sbox_name,
                "mode": aes_mode,
            }
        except Exception as exc:
            st.error(f"Simulasi gagal: {exc}")
            return

    if "last_img_sim" not in st.session_state:
        st.caption("Klik tombol di atas untuk menjalankan simulasi nyata pada citra.")
        return

    payload = st.session_state["last_img_sim"]
    sim = payload["sim"]
    metrics = sim["metrics"]
    mode_used = payload["mode"]

    if sim["lossless"]:
        st.markdown(
            f'<div class="verify-ok">{SVG_CHECK}'
            f'<span>Dekripsi 100% Lossless ({payload["sbox_name"]} · Mode {mode_used} · '
            f'Resolusi {sim["plain"].shape[1]}×{sim["plain"].shape[0]} · '
            f'Enkripsi {sim["encrypt_sec"]:.2f} dtk / Dekripsi {sim["decrypt_sec"]:.2f} dtk)</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="verify-warn"><span>Peringatan: hasil dekripsi tidak identik dengan citra asli.</span></div>',
            unsafe_allow_html=True,
        )

    if mode_used == "ECB":
        st.markdown(
            '<div class="verify-warn"><span>Catatan edukasi: Mode ECB mengenkripsi setiap blok 16-byte secara terpisah, '
            'sehingga perubahan 1 bit di piksel terakhir hanya memengaruhi 1 blok (NPCR rendah). '
            'Bandingkan dengan mode CBC2 untuk melihat difusi penuh ~99,6%.</span></div>',
            unsafe_allow_html=True,
        )

    p1, p2, p3 = st.columns(3, gap="medium")
    with p1:
        st.image(sim["plain"], caption="Citra Asli (Plaintext)", use_container_width=True)
    with p2:
        st.image(sim["cipher"], caption=f"Terenkripsi (AES-{mode_used})", use_container_width=True)
    with p3:
        st.image(sim["decrypted"], caption="Hasil Dekripsi (Identik 100%)", use_container_width=True)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Entropy Plain", f"{metrics['entropy_plain']:.5f}")
    k2.metric(
        "Entropy Cipher",
        f"{metrics['entropy_cipher']:.5f}",
        delta=f"{metrics['entropy_cipher'] - metrics['entropy_plain']:+.5f}",
    )
    k3.metric("NPCR (1-bit flip)", f"{metrics['npcr']:.4f}%", delta="Ideal ~99.61%", delta_color="off")
    k4.metric("UACI (1-bit flip)", f"{metrics['uaci']:.4f}%", delta="Ideal ~33.46%", delta_color="off")
    k5.metric("Histogram χ²", f"{metrics['hist_chi2']:.1f}", delta=f"p = {metrics['hist_p_value']:.4f}", delta_color="off")

    hist_fig = render_histogram_comparison(sim["plain"], sim["cipher"])
    st.pyplot(hist_fig, use_container_width=True)
    plt.close(hist_fig)

    d1, d2, _ = st.columns([1.0, 1.0, 2.0])
    with d1:
        st.download_button(
            "Unduh Citra Terenkripsi (.png)",
            data=_array_to_png_bytes(sim["cipher"]),
            file_name=f"cipher_{mode_used.lower()}.png",
            mime="image/png",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Unduh Hasil Dekripsi (.png)",
            data=_array_to_png_bytes(sim["decrypted"]),
            file_name="decrypted_lossless.png",
            mime="image/png",
            use_container_width=True,
        )


def render_sim_text_tab(catalog: dict[str, dict]):
    st.markdown(
        '<p class="section-sub" style="margin-top:0.2rem;">'
        'Buktikan bahwa S-Box hasil Machine Learning dapat dipasang langsung (<em>drop-in replacement</em>) '
        'pada enkripsi dan dekripsi pesan teks AES-128. Coba ubah satu huruf kunci saat dekripsi '
        'untuk membuktikan sensitivitas kunci.</p>',
        unsafe_allow_html=True,
    )

    names = list(catalog.keys())
    default_idx = names.index("S-Box XGBoost (Mahdi)") if "S-Box XGBoost (Mahdi)" in names else 0

    t1, t2, t3, t4 = st.columns([1.4, 0.9, 1.1, 1.1])
    with t1:
        sbox_name = st.selectbox("S-Box AES-128", names, index=default_idx, key="txt_sbox")
    with t2:
        mode = st.selectbox("Mode AES", ["CBC2", "CBC", "ECB"], key="txt_mode")
    with t3:
        key_str = st.text_input("Kunci (16 kar UTF-8)", value="KunciRahasia2026", key="txt_key")
    with t4:
        iv_str = st.text_input("IV (16 kar UTF-8)", value="VectorAwal2026!!", key="txt_iv")

    enc_col, dec_col = st.columns(2, gap="large")
    with enc_col:
        st.markdown("**Enkripsi Pesan Teks**")
        plain_input = st.text_area(
            "Pesan Asli (Plaintext)",
            value="Proyek Akhir Kriptografi UNNES: S-Box XGBoost (0x13F, NL=112, SAC=0.50024) siap mengamankan data.",
            height=135,
        )
        if st.button("Enkripsi Teks", type="primary", use_container_width=True):
            try:
                key_b = normalize_16byte_secret(key_str, mode="utf8")
                iv_b = normalize_16byte_secret(iv_str, mode="utf8")
                cipher_bytes = encrypt_text_aes(
                    plain_input,
                    sbox=catalog[sbox_name]["sbox"],
                    key=key_b,
                    iv=iv_b,
                    mode=mode,
                )
                st.session_state["last_cipher_hex"] = cipher_bytes.hex().upper()
                st.session_state["last_cipher_b64"] = base64.b64encode(cipher_bytes).decode("ascii")
            except Exception as exc:
                st.error(f"Enkripsi gagal: {exc}")

        if "last_cipher_hex" in st.session_state:
            st.markdown("Ciphertext (Hex):")
            st.code(st.session_state["last_cipher_hex"], language=None)
            st.markdown("Ciphertext (Base64):")
            st.code(st.session_state["last_cipher_b64"], language=None)

    with dec_col:
        st.markdown("**Dekripsi Ciphertext**")
        cipher_hex_in = st.text_area(
            "Masukkan Ciphertext (format Hex)",
            value=st.session_state.get("last_cipher_hex", ""),
            height=135,
            placeholder="Jalankan enkripsi di sisi kiri atau tempel string heksadesimal di sini...",
        )
        if st.button("Dekripsi Ciphertext", use_container_width=True):
            try:
                compact_hex = "".join(cipher_hex_in.split())
                if not compact_hex:
                    raise ValueError("Ciphertext Hex masih kosong. Jalankan enkripsi terlebih dahulu.")
                raw_cipher = bytes.fromhex(compact_hex)
                key_b = normalize_16byte_secret(key_str, mode="utf8")
                iv_b = normalize_16byte_secret(iv_str, mode="utf8")
                recovered = decrypt_text_aes(
                    raw_cipher,
                    sbox=catalog[sbox_name]["sbox"],
                    key=key_b,
                    iv=iv_b,
                    mode=mode,
                )
                st.markdown(
                    f'<div class="verify-ok">{SVG_CHECK}<span>Dekripsi berhasil &amp; padding PKCS#7 valid.</span></div>',
                    unsafe_allow_html=True,
                )
                st.text_area("Hasil Dekripsi", value=recovered, height=110, disabled=True)
            except Exception as exc:
                st.error(
                    f"Dekripsi ditolak: {exc}. Pastikan S-Box, mode, kunci, dan IV sama persis dengan saat enkripsi."
                )


def render_page_simulation(catalog: dict[str, dict]):
    st.markdown(
        '<h1 class="lead-title" style="font-size:2.1rem; margin-bottom:0.45rem;">'
        'Simulasi Langsung S-Box &amp; Enkripsi AES-128</h1>',
        unsafe_allow_html=True,
    )
    tab_sbox, tab_img, tab_txt = st.tabs(
        [
            "Inspeksi & Generator S-Box",
            "Enkripsi Citra (Mandrill / Upload)",
            "Enkripsi Teks AES-128",
        ]
    )
    with tab_sbox:
        render_sim_sbox_tab(catalog)
    with tab_img:
        render_sim_image_tab(catalog)
    with tab_txt:
        render_sim_text_tab(catalog)


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = PAGES[0]

    catalog, df_metrics = load_catalog_and_metrics()

    # Compact sidebar with live repository status
    with st.sidebar:
        st.markdown("### Status Repositori")
        st.caption(f"S-Box terdeteksi di `results/`: **{len(catalog)}** ({len(catalog) - 2} model ML + 2 baseline)")
        for name, info in catalog.items():
            st.markdown(f"- **{name}** (`{info['status']}`)")
        st.divider()
        st.caption(
            "Aplikasi ringan: menggunakan caching memori dan pratinjau cepat agar nyaman dijalankan "
            "di laptop biasa maupun Streamlit Community Cloud."
        )
        st.markdown("[Buka Repositori GitHub](https://github.com/MahdiHbb1/cryptoml-sbox)")

    # Clean top brand bar + segmented 3-page switcher
    st.markdown(
        '<div class="brand-bar">'
        f'<div class="brand-mark">{SVG_BRAND}'
        '<div><div class="brand-name">CryptoML-SBox</div>'
        '<div class="brand-sub">Konstruksi S-Box Kriptografi Berbasis Machine Learning · UNNES</div></div></div>'
        '<span class="pill pill-done">AES-128 · XGBoost · RF · MLP</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="nav-radio-wrap">', unsafe_allow_html=True)
    active_page = st.radio(
        "Navigasi Halaman",
        PAGES,
        key="active_page",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if active_page == PAGES[0]:
        render_page_overview(catalog, df_metrics)
    elif active_page == PAGES[1]:
        render_page_architecture()
    else:
        render_page_simulation(catalog)


if __name__ == "__main__":
    main()
