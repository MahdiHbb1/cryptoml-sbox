"""Six-page Streamlit dashboard for CryptoML-SBox v2 (Polynomial-Free) and Legacy v1."""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src.web_helpers import (
    compute_sac_matrix,
    decrypt_text_aes,
    encrypt_text_aes,
    evaluate_sbox_full,
    format_sbox_grid,
    list_available_sboxes,
    load_mandrill_benchmark_table,
    normalize_16byte_secret,
    parse_uploaded_sbox,
    run_image_simulation,
)

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MANDRILL_PATH = REPO_ROOT / "data" / "mandril_color.jpg"
MANDRILL_LEGACY_PATH = RESULTS_DIR / "mandrill_metrics.json"
ARCHIFY_HTML_PATH = REPO_ROOT / "assets" / "cryptoml_sbox_archify.html"

V2_MODELS_DIR = REPO_ROOT / "models"
V2_RESULTS_DIR = RESULTS_DIR / "v2"
V2_MANDRILL_PATH = V2_RESULTS_DIR / "mandrill_v2_metrics.json"
V2_SUMMARY_PATH = V2_RESULTS_DIR / "comparative_summary.json"
V2_SPLIT_MANIFEST_PATH = REPO_ROOT / "datasets" / "split_manifest.json"

PAGES = [
    "Overview",
    "Dataset/Training",
    "Reconstruction",
    "Crypto Evaluation",
    "Image Encryption",
    "Comparative Analysis",
]

METHOD_LABELS = {
    "rf": "Random Forest (v2)",
    "xgboost": "XGBoost (v2)",
    "mlp": "MLP (v2)",
    "random_top_k": "Random Top-K",
    "random_swap": "Random Swap",
    "random_search": "Random Search",
    "random_permutation": "Initial Permutation",
}

METHOD_COLORS = {
    "rf": "#0F5C5C",
    "xgboost": "#1A7F7A",
    "mlp": "#8A4307",
    "random_top_k": "#5D7080",
    "random_swap": "#7E90A0",
    "random_search": "#9FB0BE",
    "random_permutation": "#C4CFD8",
}

st.set_page_config(
    page_title="CryptoML-SBox v2 — ML-Guided S-Box Reconstruction",
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
  text-wrap: balance;
}

p, .lead-copy, .section-sub, .step-body {
  text-wrap: pretty;
}

code, pre, .mono-val {
  font-family: 'JetBrains Mono', monospace !important;
  font-variant-numeric: tabular-nums;
}

table, .stDataFrame, [data-testid="stMetricValue"] {
  font-variant-numeric: tabular-nums;
}

[data-testid="stImageCaption"] {
  color: var(--ink-muted) !important;
}

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
  flex-wrap: wrap;
  gap: 0.4rem;
  background: var(--surface-subtle);
  padding: 0.3rem;
  border-radius: 10px;
  border: 1px solid var(--border);
  width: fit-content;
  max-width: 100%;
}

.nav-radio-wrap div[role="radiogroup"] > label {
  background: transparent;
  padding: 0.46rem 0.95rem;
  border-radius: 7px;
  margin: 0 !important;
  cursor: pointer;
  transition: background-color 140ms ease, color 140ms ease, box-shadow 140ms ease;
}

.nav-radio-wrap div[role="radiogroup"] > label > div:first-child {
  display: none;
}

.nav-radio-wrap div[role="radiogroup"] > label p {
  font-size: 0.94rem;
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

.lead-title {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-size: clamp(1.9rem, 3vw, 2.6rem);
  line-height: 1.12;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--ink);
  margin: 0 0 0.85rem 0;
}

.lead-copy {
  font-size: 1.08rem;
  line-height: 1.62;
  color: #2A3843;
  max-width: 64ch;
  margin: 0 0 1.2rem 0;
}

.section-heading {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-size: 1.48rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin: 2rem 0 0.45rem 0;
}

.section-sub {
  font-size: 0.99rem;
  color: var(--ink-muted);
  max-width: 74ch;
  margin: 0 0 1.1rem 0;
  line-height: 1.56;
}

.proof-panel {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.1rem 1.2rem;
  box-shadow: var(--shadow-sm);
}

.stat-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1.15rem;
  margin-top: 1.25rem;
  padding: 1rem 1.2rem;
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
  font-size: 1.32rem;
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

.step-row {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 1.75rem;
  margin-top: 0.7rem;
}

@media (max-width: 820px) {
  .step-row {
    grid-template-columns: 1fr;
    gap: 1.15rem;
  }
}

.step-col {
  border-top: 2px solid var(--ink);
  padding-top: 0.85rem;
}

.step-head {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  margin-bottom: 0.4rem;
}

.step-title {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--ink);
  margin: 0;
}

.step-body {
  font-size: 0.96rem;
  line-height: 1.58;
  color: #344450;
  margin: 0;
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.2rem 0.62rem;
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

.table-scroll {
  width: 100%;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: var(--surface);
}

.team-table {
  width: 100%;
  min-width: 640px;
  border-collapse: collapse;
  background: var(--surface);
  font-size: 0.94rem;
}

.team-table th {
  text-align: left;
  padding: 0.7rem 0.95rem;
  background: var(--surface-subtle);
  color: var(--ink-muted);
  font-weight: 600;
  font-size: 0.84rem;
  border-bottom: 1px solid var(--border);
}

.team-table td {
  padding: 0.8rem 0.95rem;
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

.insight-note {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.8rem 1rem;
  font-size: 0.93rem;
  color: #2D3D4A;
  margin-top: 0.75rem;
  line-height: 1.56;
}

.archify-frame {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 0.35rem;
  box-shadow: var(--shadow-sm);
}

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
  font-size: 0.95rem;
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
  font-size: 0.95rem;
  margin: 0.8rem 0 1rem 0;
}

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

@media (max-width: 640px) {
  .block-container {
    padding-top: 3.35rem;
    padding-left: 1rem;
    padding-right: 1rem;
  }

  .nav-radio-wrap div[role="radiogroup"] > label {
    padding: 0.4rem 0.74rem;
  }

  .nav-radio-wrap div[role="radiogroup"] > label p {
    font-size: 0.86rem;
  }

  .stButton > button {
    white-space: normal;
  }
}

@media (prefers-reduced-motion: reduce) {
  .nav-radio-wrap div[role="radiogroup"] > label {
    transition: none;
  }
}
</style>
"""

SVG_BRAND = (
    '<svg aria-hidden="true" width="26" height="26" viewBox="0 0 24 24" fill="none" '
    'stroke="#0F5C5C" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="3" y="3" width="7" height="7" rx="1.5"/>'
    '<rect x="14" y="3" width="7" height="7" rx="1.5"/>'
    '<rect x="3" y="14" width="7" height="7" rx="1.5"/>'
    '<path d="M14 17.5h7M17.5 14v7"/></svg>'
)

SVG_GRID = (
    '<svg aria-hidden="true" width="19" height="19" viewBox="0 0 24 24" fill="none" '
    'stroke="#0F5C5C" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M3 3h18v18H3zM9 3v18M15 3v18M3 9h18M3 15h18"/></svg>'
)

SVG_SLIDERS = (
    '<svg aria-hidden="true" width="19" height="19" viewBox="0 0 24 24" fill="none" '
    'stroke="#0F5C5C" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/></svg>'
)

SVG_SHIELD = (
    '<svg aria-hidden="true" width="19" height="19" viewBox="0 0 24 24" fill="none" '
    'stroke="#0F5C5C" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/></svg>'
)

SVG_CHECK = (
    '<svg aria-hidden="true" width="18" height="18" viewBox="0 0 24 24" fill="none" '
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


def _split_metrics_tables(
    df_all: pd.DataFrame,
    focus_seed: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pisahkan secara tegas tabel v2 (seed fokus & semua seed) vs Legacy v1."""
    if df_all.empty:
        return df_all, df_all, df_all
    baselines = df_all[df_all["Kategori"] == "Baseline"]
    v2_all = df_all[df_all["Kategori"] == "Model ML v2"]
    v2_focus_models = v2_all[v2_all["S-Box"].str.contains(f"seed {focus_seed}", regex=False)]
    if v2_focus_models.empty and not v2_all.empty:
        v2_focus_models = v2_all.head(3)
    df_v2_focus = pd.concat([baselines, v2_focus_models], ignore_index=True)
    legacy_models = df_all[df_all["Kategori"] == "Model ML"]
    df_legacy = pd.concat([baselines, legacy_models], ignore_index=True)
    return df_v2_focus, v2_all.reset_index(drop=True), df_legacy


@st.cache_data(show_spinner=False)
def load_legacy_ml_summaries() -> pd.DataFrame:
    entries = [
        ("Random Forest (v1)", "Yafi (@twizzcode)", RESULTS_DIR / "rf_metrics.json"),
        ("XGBoost (v1)", "Mahdi (@MahdiHbb1)", RESULTS_DIR / "xgb_metrics.json"),
        ("MLP (v1)", "Naila (@NAILAAR22)", RESULTS_DIR / "mlp_metrics.json"),
    ]
    rows = []
    for model_name, owner, path in entries:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        ml = data.get("ml_metrics", {})
        rows.append(
            {
                "Model ML": model_name,
                "Pengembang": owner,
                "Dataset": f"{ml.get('n_train', 0)} latih / {ml.get('n_test', 0)} uji",
                "R² (Test)": round(float(ml.get("r2", 0.0)), 4),
                "MAE": round(float(ml.get("mae", 0.0)), 4),
                "RMSE": round(float(ml.get("rmse", 0.0)), 4),
                "Waktu Latih (s)": round(float(ml.get("train_time_sec", 0.0)), 2),
                "Kandidat Disaring": f"{int(data.get('n_screen_candidates', 0)):,} (Top-{int(data.get('top_k_evaluated', 0))})",
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_mandrill_v2() -> tuple[dict, pd.DataFrame]:
    return load_mandrill_benchmark_table(V2_MANDRILL_PATH)


@st.cache_data(show_spinner=False)
def load_mandrill_legacy() -> tuple[dict, pd.DataFrame]:
    return load_mandrill_benchmark_table(MANDRILL_LEGACY_PATH)


@st.cache_data(show_spinner=False)
def load_comparative_summary() -> tuple[dict | None, str | None]:
    if not V2_SUMMARY_PATH.is_file():
        return None, None
    try:
        return json.loads(V2_SUMMARY_PATH.read_text(encoding="utf-8")), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def load_figure_bytes(path_str: str) -> bytes:
    path = Path(path_str)
    return path.read_bytes() if path.exists() else b""


@st.cache_data(show_spinner=False)
def cached_evaluate_sbox(sbox_bytes: bytes) -> dict:
    return evaluate_sbox_full(np.frombuffer(sbox_bytes, dtype=np.uint8))


@st.cache_data(show_spinner=False)
def cached_sac_matrix(sbox_bytes: bytes) -> np.ndarray:
    return compute_sac_matrix(np.frombuffer(sbox_bytes, dtype=np.uint8))


@st.cache_data(show_spinner=False)
def load_mandrill_array() -> np.ndarray:
    img = Image.open(MANDRILL_PATH).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


@st.cache_data(show_spinner=False)
def build_hero_mandrill_preview(sbox_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Enkripsi preview cepat Mandrill 224x224 untuk demonstrasi langsung di halaman depan."""
    sbox = np.frombuffer(sbox_bytes, dtype=np.uint8)
    plain = load_mandrill_array()
    key = normalize_16byte_secret("CryptoML-SBox26!", mode="utf8")
    iv = normalize_16byte_secret("MandrillCBC2-IV!", mode="utf8")
    sim = run_image_simulation(plain, sbox=sbox, key=key, iv=iv, mode="CBC2", max_dim=224)
    return sim["plain"], sim["cipher"]


@st.cache_data(show_spinner=False)
def load_archify_html() -> str:
    if ARCHIFY_HTML_PATH.exists():
        return ARCHIFY_HTML_PATH.read_text(encoding="utf-8")
    return ""


@st.cache_resource(show_spinner=False)
def load_v2_model(model_dir: str, manifest_mtime: int, model_mtime: int):
    from src.ml_reconstruction import load_trusted_model

    return load_trusted_model(model_dir)


@st.cache_data(show_spinner=False)
def cached_v2_reconstruction(model_dir: str, seed: int, manifest_mtime: int, model_mtime: int):
    from src.ml_reconstruction import reconstruct
    from src.train_rf_v2 import evaluate

    model, _ = load_v2_model(model_dir, manifest_mtime, model_mtime)
    return reconstruct(
        lambda box: evaluate(box)["composite_score"],
        model.predict,
        seed=seed,
        mode="fast",
        max_evaluations=46,
    )


def _fig_to_clean_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C5D1DC")
    ax.spines["bottom"].set_color("#C5D1DC")
    ax.tick_params(colors="#344450", labelsize=9)


def render_comparison_chart(df_metrics: pd.DataFrame, *, metric_mode: str = "v2"):
    labels = [
        name.replace(" (FIPS-197)", "").replace(" (2024/2025)", "")
        for name in df_metrics["S-Box"]
    ]
    colors = [
        "#90A2B2" if cat == "Baseline" else ("#8A4307" if "MLP" in lbl else "#0F5C5C")
        for lbl, cat in zip(labels, df_metrics["Kategori"])
    ]
    fig_h = max(3.1, 0.56 * len(labels) + 1.1)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, fig_h), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    y_pos = np.arange(len(labels))
    if metric_mode == "v2":
        panels = [
            (
                axes[0],
                df_metrics["NL"].to_numpy(dtype=float),
                "Nonlinearity / NL  (lebih besar lebih baik · AES = 112)",
                "{:.0f}",
                1.18,
            ),
            (
                axes[1],
                df_metrics["SAC Max Err"].to_numpy(dtype=float),
                "Deviasi Maksimum SAC  |P − 0.5|  (lebih kecil lebih baik)",
                "{:.5f}",
                1.30,
            ),
        ]
    else:
        panels = [
            (
                axes[0],
                df_metrics["SAC Max Err"].to_numpy(dtype=float),
                "Deviasi Maksimum SAC  |P − 0.5|  (lebih kecil lebih baik)",
                "{:.5f}",
                1.30,
            ),
            (
                axes[1],
                np.abs(df_metrics["BIC-SAC"].to_numpy(dtype=float) - 0.5),
                "Jarak BIC-SAC ke Ideal 0.5  (lebih kecil lebih baik)",
                "{:.5f}",
                1.30,
            ),
        ]

    for ax, vals, title, fmt, pad_factor in panels:
        ax.set_facecolor("#FFFFFF")
        bars = ax.barh(y_pos, vals, color=colors, height=0.54, edgecolor="none")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9.2, color="#141D24")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9.8, fontweight="bold", color="#141D24", loc="left", pad=10)
        ax.grid(axis="x", linestyle="--", linewidth=0.6, color="#E1E8EE")
        ax.set_axisbelow(True)
        _fig_to_clean_style(ax)
        max_v = max(float(np.max(vals)), 1e-6)
        ax.set_xlim(0, max_v * pad_factor)
        for bar, val in zip(bars, vals):
            ax.text(
                val + max_v * 0.02,
                bar.get_y() + bar.get_height() / 2,
                fmt.format(val),
                va="center",
                ha="left",
                fontsize=8.6,
                family="monospace",
                color="#141D24",
            )

    fig.tight_layout(w_pad=3.0)
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
    ax.set_title("Matriks SAC 8×8 (Nilai Ideal = 0.500)", fontsize=10.5, fontweight="bold", color="#141D24", pad=10)

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
        ax.set_xlabel("Intensitas Piksel (0–255)", fontsize=8.8, color="#344450")
        ax.set_ylabel("Frekuensi", fontsize=8.8, color="#344450")
        ax.set_title(title, fontsize=10, fontweight="bold", color="#141D24", loc="left", pad=8)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#E1E8EE")
        _fig_to_clean_style(ax)

    fig.tight_layout(w_pad=2.4)
    return fig


def render_convergence_chart(runs_for_seed: list[dict], seed: int):
    fig, ax = plt.subplots(figsize=(10.2, 3.8), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    order = ["rf", "xgboost", "mlp", "random_top_k", "random_swap", "random_search"]
    by_method = {row["method"]: row for row in runs_for_seed}
    for method in order:
        row = by_method.get(method)
        if not row or not row.get("history"):
            continue
        hist = row["history"]
        x = [0] + [int(h["full_evals_cumulative"]) for h in hist]
        y = [float(row["initial_score"])] + [float(h["global_best_score"]) for h in hist]
        is_ml = method in ("rf", "xgboost", "mlp")
        ax.plot(
            x,
            y,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "#465664"),
            linewidth=2.1 if is_ml else 1.4,
            linestyle="-" if is_ml else "--",
            alpha=0.95 if is_ml else 0.8,
        )

    ax.set_xlabel("Evaluasi Penuh Kumulatif (evaluate_sbox)", fontsize=9.2, color="#344450")
    ax.set_ylabel("Skor Komposit Terbaik (Global Best)", fontsize=9.2, color="#344450")
    ax.set_title(
        f"Kurva Konvergensi Rekonstruksi Berpasangan — Seed {seed} (Budget Setara)",
        fontsize=10.2,
        fontweight="bold",
        color="#141D24",
        loc="left",
        pad=10,
    )
    ax.grid(True, linestyle="--", linewidth=0.55, color="#E1E8EE")
    ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#D5DEE6", fontsize=8.4, ncol=3, loc="lower right")
    _fig_to_clean_style(ax)
    fig.tight_layout()
    return fig


def render_feature_importance_chart(manifests: dict[str, dict]):
    valid = [
        (name, m.get("metrics", {}).get("feature_importances"))
        for name, m in manifests.items()
        if isinstance(m.get("metrics", {}).get("feature_importances"), dict)
    ]
    if not valid:
        return None
    fig, axes = plt.subplots(1, len(valid), figsize=(5.2 * len(valid), 3.0), dpi=150)
    if len(valid) == 1:
        axes = [axes]
    fig.patch.set_facecolor("#FFFFFF")
    for ax, (name, imp) in zip(axes, valid):
        items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
        labels = [k for k, _ in items]
        vals = [float(v) for _, v in items]
        y_pos = np.arange(len(labels))
        ax.set_facecolor("#FFFFFF")
        bars = ax.barh(y_pos, vals, color="#0F5C5C" if "RF" in name or "Random" in name else "#1A7F7A", height=0.52)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9, family="monospace", color="#141D24")
        ax.invert_yaxis()
        ax.set_title(f"Feature Importance — {name}", fontsize=9.8, fontweight="bold", color="#141D24", loc="left", pad=8)
        ax.grid(axis="x", linestyle="--", linewidth=0.55, color="#E1E8EE")
        ax.set_axisbelow(True)
        _fig_to_clean_style(ax)
        max_v = max(max(vals), 1e-6)
        ax.set_xlim(0, max_v * 1.25)
        for bar, val in zip(bars, vals):
            ax.text(
                val + max_v * 0.025,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                ha="left",
                fontsize=8.4,
                family="monospace",
                color="#141D24",
            )
    fig.tight_layout(w_pad=2.8)
    return fig


def render_paired_comparison_charts(paired_summary: dict, differences: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")

    order = [m for m in ("rf", "xgboost", "mlp", "random_top_k", "random_swap", "random_search", "random_permutation") if m in paired_summary]
    labels = [METHOD_LABELS.get(m, m) for m in order]
    means = [float(paired_summary[m]["mean_score_delta"]) for m in order]
    sds = [float(paired_summary[m]["sd_score_delta"] or 0.0) for m in order]
    colors = [METHOD_COLORS.get(m, "#465664") for m in order]
    y_pos = np.arange(len(order))

    ax0 = axes[0]
    ax0.set_facecolor("#FFFFFF")
    ax0.barh(y_pos, means, xerr=sds, color=colors, height=0.54, capsize=3, ecolor="#344450")
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(labels, fontsize=9, color="#141D24")
    ax0.invert_yaxis()
    ax0.set_xlabel("Rata-rata Δ Composite Score (± SD)", fontsize=9, color="#344450")
    ax0.set_title("Kenaikan Skor Rata-Rata (5 Seed Berpasangan)", fontsize=9.8, fontweight="bold", color="#141D24", loc="left", pad=8)
    ax0.grid(axis="x", linestyle="--", linewidth=0.55, color="#E1E8EE")
    ax0.set_axisbelow(True)
    _fig_to_clean_style(ax0)

    diff_order = [m for m in ("rf", "xgboost", "mlp", "random_swap", "random_search") if m in differences]
    d_labels = [METHOD_LABELS.get(m, m) for m in diff_order]
    d_means = [float(differences[m]["mean_score_delta_difference"]) for m in diff_order]
    d_lo = [d_means[i] - float(differences[m]["ci95"][0]) if differences[m].get("ci95") else 0.0 for i, m in enumerate(diff_order)]
    d_hi = [float(differences[m]["ci95"][1]) - d_means[i] if differences[m].get("ci95") else 0.0 for i, m in enumerate(diff_order)]
    d_colors = [METHOD_COLORS.get(m, "#465664") for m in diff_order]
    dy_pos = np.arange(len(diff_order))

    ax1 = axes[1]
    ax1.set_facecolor("#FFFFFF")
    ax1.axvline(0.0, color="#8A4307", linestyle="--", linewidth=1.1, zorder=1)
    ax1.errorbar(
        d_means,
        dy_pos,
        xerr=[d_lo, d_hi],
        fmt="o",
        color="#0F5C5C",
        ecolor="#465664",
        elinewidth=1.6,
        capsize=4,
        markersize=6.5,
        zorder=3,
    )
    for yi, m_val, col in zip(dy_pos, d_means, d_colors):
        ax1.scatter([m_val], [yi], color=col, s=55, zorder=4)
    ax1.set_yticks(dy_pos)
    ax1.set_yticklabels(d_labels, fontsize=9, color="#141D24")
    ax1.invert_yaxis()
    ax1.set_xlabel("Selisih Δ Skor vs Random Top-K (CI 95%)", fontsize=9, color="#344450")
    ax1.set_title("Selisih Berpasangan vs Random Top-K (Garis 0 = Setara)", fontsize=9.8, fontweight="bold", color="#141D24", loc="left", pad=8)
    ax1.grid(axis="x", linestyle="--", linewidth=0.55, color="#E1E8EE")
    ax1.set_axisbelow(True)
    _fig_to_clean_style(ax1)

    fig.tight_layout(w_pad=2.8)
    return fig


def render_ablation_chart(ablation: dict):
    models = ["rf", "xgboost", "mlp"]
    sets = [s for s in ("perm4", "proxy", "cheap") if any(f"{m}_{s}" in ablation for m in models)]
    if not sets:
        return None
    set_labels = {
        "perm4": "Permutasi 4 Fitur (v2.1)",
        "proxy": "Proxy Diff tanpa SAC/FP",
        "cheap": "Fitur Murah + SAC 8×8 & FP",
    }
    set_colors = {"perm4": "#90A2B2", "proxy": "#8A4307", "cheap": "#0F5C5C"}

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.1), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    x = np.arange(len(models))
    width = 0.25 if len(sets) == 3 else 0.34

    for ax, metric_key, title in [
        (axes[0], "r2", "Test R² pada Held-Out Families (lebih tinggi lebih baik)"),
        (axes[1], "spearman", "Spearman Rank Correlation ρ (lebih tinggi lebih baik)"),
    ]:
        ax.set_facecolor("#FFFFFF")
        ax.axhline(0.0, color="#465664", linewidth=0.9, linestyle="--")
        all_vals = []
        for idx, feat_set in enumerate(sets):
            vals = [float((ablation.get(f"{m}_{feat_set}") or {}).get(metric_key) or 0.0) for m in models]
            all_vals.extend(vals)
            offset = (idx - (len(sets) - 1) / 2.0) * width
            bars = ax.bar(x + offset, vals, width=width * 0.92, label=set_labels.get(feat_set, feat_set), color=set_colors.get(feat_set, "#465664"))
            for bar, val in zip(bars, vals):
                va = "bottom" if val >= 0 else "top"
                pad = 0.025 if val >= 0 else -0.025
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + pad,
                    f"{val:+.2f}",
                    ha="center",
                    va=va,
                    fontsize=7.8,
                    family="monospace",
                    color="#141D24",
                )
        lo = min(min(all_vals, default=0.0), 0.0)
        hi = max(max(all_vals, default=0.0), 0.0)
        span = max(hi - lo, 0.4)
        ax.set_ylim(lo - 0.18 * span, hi + 0.22 * span)
        ax.set_xticks(x)
        ax.set_xticklabels(["Random Forest", "XGBoost", "MLP"], fontsize=9.2, color="#141D24")
        ax.set_title(title, fontsize=9.6, fontweight="bold", color="#141D24", loc="left", pad=10)
        ax.grid(axis="y", linestyle="--", linewidth=0.55, color="#E1E8EE")
        ax.set_axisbelow(True)
        _fig_to_clean_style(ax)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_leg,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=len(sets),
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#D5DEE6",
        fontsize=8.5,
    )
    fig.tight_layout(rect=[0.0, 0.11, 1.0, 1.0], w_pad=2.6)
    return fig


def render_pixel_correlation_chart(df_mandrill_v2: pd.DataFrame):
    if df_mandrill_v2.empty or "Corr Cipher" not in df_mandrill_v2.columns:
        return None
    rgb = df_mandrill_v2[df_mandrill_v2["Mode"] == "RGB"]
    gray = df_mandrill_v2[df_mandrill_v2["Mode"] == "Grayscale"]
    if rgb.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), dpi=150)
    fig.patch.set_facecolor("#FFFFFF")
    for ax, subset, title in [
        (axes[0], rgb, "Korelasi Piksel Cipher — RGB (Mean |r|, Ideal ≈ 0)"),
        (axes[1], gray, "Korelasi Piksel Cipher — Grayscale (Mean |r|, Ideal ≈ 0)"),
    ]:
        if subset.empty:
            ax.axis("off")
            continue
        labels = [str(s).replace(" (FIPS-197)", "").replace(" (seed 100)", "") for s in subset["S-Box"]]
        vals = subset["Corr Cipher"].to_numpy(dtype=float)
        colors = ["#90A2B2" if ("AES" in s or "K44" in s) else "#0F5C5C" for s in labels]
        y_pos = np.arange(len(labels))
        ax.set_facecolor("#FFFFFF")
        bars = ax.barh(y_pos, vals, color=colors, height=0.52)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.8, color="#141D24")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9.5, fontweight="bold", color="#141D24", loc="left", pad=8)
        ax.grid(axis="x", linestyle="--", linewidth=0.55, color="#E1E8EE")
        ax.set_axisbelow(True)
        _fig_to_clean_style(ax)
        max_v = max(float(np.max(vals)), 1e-5)
        ax.set_xlim(0, max_v * 1.32)
        for bar, val in zip(bars, vals):
            ax.text(
                val + max_v * 0.025,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.6f}",
                va="center",
                ha="left",
                fontsize=8.2,
                family="monospace",
                color="#141D24",
            )
    fig.tight_layout(w_pad=2.8)
    return fig


def _array_to_png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def _set_active_page(page_name: str) -> None:
    st.session_state["active_page"] = page_name


def _default_sbox_index(names: list[str]) -> int:
    for candidate in ("S-Box XGBoost v2 (seed 100)", "S-Box RF v2 (seed 100)", "AES Standar (FIPS-197)"):
        if candidate in names:
            return names.index(candidate)
    return 0


def render_page_overview(catalog: dict[str, dict], df_v2_focus: pd.DataFrame):
    _, df_mandrill_v2 = load_mandrill_v2()
    xgb_name = next(
        (
            name
            for name in ("S-Box XGBoost v2 (seed 100)", "S-Box RF v2 (seed 100)", "AES Standar (FIPS-197)")
            if name in catalog
        ),
        "AES Standar (FIPS-197)",
    )
    xgb_entry = catalog[xgb_name]
    xgb_m = cached_evaluate_sbox(xgb_entry["sbox"].tobytes())
    aes_m = cached_evaluate_sbox(catalog["AES Standar (FIPS-197)"]["sbox"].tobytes())
    preview_plain, preview_cipher = build_hero_mandrill_preview(xgb_entry["sbox"].tobytes())

    hero_entropy = "Belum tersedia"
    hero_entropy_note = "Jalankan `python -m src.run_v2_benchmark`"
    hero_npcr = "Belum tersedia"
    hero_npcr_note = "Belum ada artefak Mandrill v2"
    if not df_mandrill_v2.empty:
        rgb_row = df_mandrill_v2[
            (df_mandrill_v2["S-Box"] == xgb_name) & (df_mandrill_v2["Mode"] == "RGB")
        ]
        if not rgb_row.empty:
            r = rgb_row.iloc[0]
            hero_entropy = f"{r['Entropy Cipher']:.5f}"
            hero_entropy_note = f"Plain {r['Entropy Plain']:.5f} · χ² p={r['p-value']:.4f}"
            hero_npcr = f"{r['NPCR (%)']:.2f}%"
            lossless_cnt = int((df_mandrill_v2["Lossless"] == "Ya").sum())
            corr_txt = f" · |r|={r['Corr Cipher']:.5f}" if "Corr Cipher" in r else ""
            hero_npcr_note = f"UACI {r['UACI (%)']:.2f}%{corr_txt} · {lossless_cnt}/{len(df_mandrill_v2)} lossless"

    lead_left, lead_right = st.columns([1.15, 1.0], gap="large")
    with lead_left:
        st.markdown(
            '<h1 class="lead-title">Rekonstruksi S-Box Berpandu Machine Learning (Bebas Polinomial)</h1>'
            '<p class="lead-copy">'
            '<strong>CryptoML-SBox v2</strong> meneliti konstruksi <strong>Substitution-Box (S-Box)</strong> '
            '8-bit tanpa polinomial <em>irreducible</em>, tanpa invers multiplikatif GF(2⁸), tanpa matriks '
            '<em>affine</em>, dan tanpa <em>chaotic map</em>. Sistem memulai dari permutasi acak bijektif, melatih '
            '<strong>Random Forest</strong>, <strong>XGBoost</strong>, dan <strong>MLP</strong> pada split keluarga '
            '70/15/15 bebas kebocoran, mengarahkan rekonstruksi iteratif berbudget setara, lalu menguji enkripsi '
            'citra <strong>Mandrill</strong> pada <strong>AES-128 Modifikasi</strong>. Seluruh metrik empiris '
            'dilaporkan apa adanya dan bukan bukti keamanan kriptografi formal.'
            '</p>',
            unsafe_allow_html=True,
        )
        btn_col1, btn_col2, btn_col3 = st.columns(3, gap="small")
        with btn_col1:
            st.button(
                "Rekonstruksi ML",
                type="primary",
                width="stretch",
                on_click=_set_active_page,
                args=(PAGES[2],),
            )
        with btn_col2:
            st.button(
                "Analisis Komparatif",
                width="stretch",
                on_click=_set_active_page,
                args=(PAGES[5],),
            )
        with btn_col3:
            st.button(
                "Uji Citra Mandrill",
                width="stretch",
                on_click=_set_active_page,
                args=(PAGES[4],),
            )

    with lead_right:
        st.markdown('<div class="proof-panel">', unsafe_allow_html=True)
        img_c1, img_c2 = st.columns(2, gap="small")
        with img_c1:
            st.image(preview_plain, caption="Mandrill Asli (RGB)", width="stretch")
        with img_c2:
            st.image(
                preview_cipher,
                caption=f"AES-128 Modifikasi CBC2 · {xgb_name}",
                width="stretch",
            )
        st.markdown(
            '<div style="font-size:0.88rem; color:#465664; margin-top:0.35rem; line-height:1.55;">'
            'Pratinjau dienkripsi langsung oleh <code class="mono-val">aes_cipher_runner.py</code> '
            f'memakai <strong>{html.escape(xgb_name)}</strong> '
            f'(<code class="mono-val">polynomial-free · NL={xgb_m["nl"]} · DU={xgb_m["du"]} · FP={xgb_m["fixed_points"]}</code>). '
            'Operasi GF(2⁸) hanya dipertahankan pada <code>MixColumns</code>/<code>InvMixColumns</code> standar AES; '
            'dekripsi terverifikasi 100% lossless.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="stat-strip">'
        '<div class="stat-item"><div class="stat-item-label">Nonlinearity v2 (Seed 100)</div>'
        f'<div class="stat-item-value">{xgb_m["nl"]}</div>'
        f'<div class="stat-item-note">DU {xgb_m["du"]} · LAP {xgb_m["lap"]:.4f} · AES NL {aes_m["nl"]}</div></div>'
        '<div class="stat-item"><div class="stat-item-label">SAC Mean &amp; Max Error</div>'
        f'<div class="stat-item-value">{xgb_m["sac_mean"]:.5f}</div>'
        f'<div class="stat-item-note">Max Err {xgb_m["sac_max_err"]:.5f} · Skor {xgb_m["composite_score"]:.2f}</div></div>'
        '<div class="stat-item"><div class="stat-item-label">Entropi Mandrill v2 (RGB)</div>'
        f'<div class="stat-item-value">{hero_entropy}</div>'
        f'<div class="stat-item-note">{hero_entropy_note}</div></div>'
        '<div class="stat-item"><div class="stat-item-label">Difusi NPCR &amp; Korelasi</div>'
        f'<div class="stat-item-value">{hero_npcr}</div>'
        f'<div class="stat-item-note">{hero_npcr_note}</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h2 class="section-heading">Alur Arsitektur Pipeline v2 dalam 3 Tahap</h2>'
        '<p class="section-sub">Memisahkan secara tegas <strong>Phase A (Training Offline)</strong> dan '
        '<strong>Phase B (ML-Guided Iterative Reconstruction)</strong> sehingga dashboard tidak melatih ulang model saat dibuka.</p>'
        '<div class="step-row">'
        f'<div class="step-col"><div class="step-head">{SVG_GRID}'
        '<h3 class="step-title">1. Dataset Permutasi Bebas Polinomial</h3></div>'
        '<p class="step-body">1.500 permutasi bijektif unik dari 150 keluarga dibangun murni melalui inisialisasi '
        'permutasi acak 0–255 dan 4 operator mutasi bijektif (<code>single_swap</code>, <code>multi_swap</code>, '
        '<code>three_cycle</code>, <code>segment_reversal</code>), lalu dibagi 70/15/15 per keluarga tanpa kebocoran.</p></div>'
        f'<div class="step-col"><div class="step-head">{SVG_SLIDERS}'
        '<h3 class="step-title">2. Rekonstruksi Iteratif Terpandu ML</h3></div>'
        '<p class="step-body"><strong>Random Forest</strong>, <strong>XGBoost</strong>, dan <strong>MLP</strong> '
        'terlatih dipakai memeringkat pool kandidat mutasi untuk memilih Top-K terbaik, sementara penerimaan langkah '
        'selalu diverifikasi oleh skor aktual <code>evaluate_sbox</code> dan dibandingkan terhadap 4 baseline non-ML.</p></div>'
        f'<div class="step-col"><div class="step-head">{SVG_SHIELD}'
        '<h3 class="step-title">3. Uji Kriptografi &amp; Citra Mandrill</h3></div>'
        '<p class="step-body">S-Box hasil rekonstruksi diuji pada Metrik 1 lengkap, lalu dipasang sebagai '
        '<strong>AES-128 Modifikasi</strong> mode <code>CBC2</code> untuk menguji Entropy, Histogram χ², Korelasi Piksel '
        '(H/V/D), NPCR, UACI, NBCR, dan dekripsi 100% <em>lossless</em> pada Mandrill RGB &amp; Grayscale.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<h2 class="section-heading">Ringkasan S-Box v2 Polynomial-Free vs Baseline (Seed 100)</h2>'
        '<p class="section-sub">Menampilkan pembanding standar (AES &amp; K44) dan tiga S-Box hasil rekonstruksi v2 '
        '(RF, XGBoost, MLP pada seed tetap 100). Arsip <strong>Legacy v1 (GF(2⁸))</strong> dipisahkan pada halaman '
        '<em>Crypto Evaluation</em> agar tidak tercampur.</p>',
        unsafe_allow_html=True,
    )
    if not df_v2_focus.empty:
        st.dataframe(
            df_v2_focus,
            width="stretch",
            hide_index=True,
            height=38 * len(df_v2_focus) + 42,
        )
        chart_fig = render_comparison_chart(df_v2_focus, metric_mode="v2")
        st.pyplot(chart_fig, width="stretch")
        plt.close(chart_fig)
        st.markdown(
            '<div class="insight-note">'
            '<strong>Catatan ilmiah:</strong> Tanpa struktur aljabar inversi GF(2⁸), ruang permutasi bebas polinomial '
            'secara alami menghasilkan Nonlinearity di bawah AES/K44 (<code>NL = 112</code>). Fokus utama penelitian v2 '
            'adalah menguji kontribusi panduan ML dibanding baseline pencarian non-ML pada budget evaluasi yang sama '
            '(lihat halaman <em>Comparative Analysis</em>).'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<h2 class="section-heading">Kontribusi Tim &amp; Integrasi v2</h2>'
        '<p class="section-sub">Hasil eksperimen RF, MLP, evaluator Metrik 1, dan metrik citra Mandrill dipertahankan '
        'murni sesuai implementasi masing-masing anggota tim pada kontrak <code>docs/CONTRACT_V2_RECONSTRUCTION.md</code>.</p>'
        '<div class="table-scroll"><table class="team-table">'
        '<thead><tr><th>Anggota</th><th>Tanggung Jawab Modul (v1 &amp; v2)</th><th>Artefak Utama</th><th>Status</th></tr></thead>'
        '<tbody>'
        '<tr><td><div class="team-member">Naila</div><div class="team-handle">@NAILAAR22 · Issue #16</div></td>'
        '<td>Evaluator Metrik 1 (<code>sbox_metrics.py</code> v2.0.0), pemisahan <code>composite_score.py</code>, baseline AES/K44, &amp; model <strong>MLP</strong></td>'
        '<td><code>src/composite_score.py</code>, <code>data/baselines/</code>, <code>models/mlp/</code></td>'
        '<td><span class="pill pill-done">Selesai · Terintegrasi</span></td></tr>'
        '<tr><td><div class="team-member">Yafi</div><div class="team-handle">@twizzcode · Issue #17</div></td>'
        '<td>Generator permutasi bebas polinomial (<code>permutation_v2.py</code>), dataset keluarga 70/15/15 (<code>permutation-v2.1</code>), engine rekonstruksi bersama, &amp; model <strong>Random Forest</strong></td>'
        '<td><code>src/permutation_v2.py</code>, <code>src/ml_reconstruction.py</code>, <code>models/rf/</code></td>'
        '<td><span class="pill pill-done">Selesai · Terintegrasi</span></td></tr>'
        '<tr><td><div class="team-member">Mahdi</div><div class="team-handle">@MahdiHbb1 · Issue #18</div></td>'
        '<td>Audit Tahap 0 &amp; kontrak v2, model <strong>XGBoost v2</strong> + studi ablasi, eksperimen komparatif 5 seed, benchmark Mandrill v2 (korelasi piksel), &amp; dashboard 6 halaman</td>'
        '<td><code>src/train_xgboost_v2.py</code>, <code>results/v2/</code>, <code>app.py</code>, <code>REFACTOR_REPORT.md</code></td>'
        '<td><span class="pill pill-done">Selesai · Terintegrasi</span></td></tr>'
        '<tr><td><div class="team-member">Mas Rizqi</div><div class="team-handle">@rizqinp2</div></td>'
        '<td>Fondasi evaluator keamanan citra Mandrill (Metrik 2 &amp; 3: Entropy, Histogram χ², NPCR, UACI, NBCR) &amp; visualisasi laporan</td>'
        '<td><code>src/image_metrics.py</code>, <code>src/mandrill_eval.py</code>, <code>results/figures/</code></td>'
        '<td><span class="pill pill-done">Selesai · Terintegrasi</span></td></tr>'
        '</tbody></table></div>',
        unsafe_allow_html=True,
    )

    archify_html = load_archify_html()
    if archify_html:
        st.markdown('<h2 class="section-heading">Diagram Arsitektur Interaktif (Archify)</h2>', unsafe_allow_html=True)
        with st.expander("Buka Diagram Arsitektur & Alur Sistem Interaktif (Archify)", expanded=False):
            c_info, c_dl = st.columns([3.0, 1.0], vertical_alignment="center")
            with c_info:
                st.caption(
                    "Klik node untuk menelusuri komponen repositori; gunakan tombol Views, Light/Dark, atau Export PNG/SVG di dalam diagram."
                )
            with c_dl:
                st.download_button(
                    "Unduh Diagram (.html)",
                    data=archify_html.encode("utf-8"),
                    file_name="cryptoml_sbox_architecture_archify.html",
                    mime="text/html",
                    width="stretch",
                )
            st.markdown('<div class="archify-frame">', unsafe_allow_html=True)
            components.html(archify_html, height=820, scrolling=True)
            st.markdown("</div>", unsafe_allow_html=True)


def render_page_dataset():
    st.markdown('<h1 class="lead-title">Dataset &amp; Training (Phase A)</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Phase A berjalan secara offline untuk membentuk dataset permutasi bebas polinomial, '
        'memverifikasi split keluarga 70/15/15 tanpa kebocoran, dan menyimpan bundle model terlatih ke <code>models/</code>. '
        'Dashboard ini tidak pernah melatih ulang model saat dimuat.</p>',
        unsafe_allow_html=True,
    )

    if V2_SPLIT_MANIFEST_PATH.is_file():
        try:
            sm = json.loads(V2_SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            st.error(f"Split manifest tidak dapat dibaca: {exc}")
        else:
            ratios = sm.get("ratios_actual", {})
            n_tr = len(sm.get("train_ids", []))
            n_val = len(sm.get("val_ids", []))
            n_te = len(sm.get("test_ids", []))
            disjoint_ok = bool(sm.get("disjoint_family_verified")) and bool(sm.get("disjoint_hash_verified"))
            st.markdown(
                '<div class="stat-strip">'
                '<div class="stat-item"><div class="stat-item-label">Total Sampel Unik</div>'
                f'<div class="stat-item-value">{sm.get("n_samples_total", 0):,}</div>'
                f'<div class="stat-item-note">{sm.get("n_families_total", 0)} keluarga permutasi · seed {sm.get("split_seed", "-")}</div></div>'
                '<div class="stat-item"><div class="stat-item-label">Split Aktual (Train / Val / Test)</div>'
                f'<div class="stat-item-value">{n_tr} / {n_val} / {n_te}</div>'
                f'<div class="stat-item-note">{ratios.get("train", 0)*100:.1f}% / {ratios.get("val", 0)*100:.1f}% / {ratios.get("test", 0)*100:.1f}%</div></div>'
                '<div class="stat-item"><div class="stat-item-label">Versi Schema &amp; Fitur</div>'
                f'<div class="stat-item-value">{html.escape(str(sm.get("feature_version", "-")))}</div>'
                f'<div class="stat-item-note">Evaluator {html.escape(str(sm.get("evaluator_version", "-")))} · CS {html.escape(str(sm.get("composite_score_version", "-")))}</div></div>'
                '<div class="stat-item"><div class="stat-item-label">Verifikasi Anti-Leakage</div>'
                f'<div class="stat-item-value">{"Lolos" if disjoint_ok else "Gagal"}</div>'
                f'<div class="stat-item-note">Disjoint family: {"Ya" if sm.get("disjoint_family_verified") else "Tidak"} · Hash unik: {"Ya" if sm.get("disjoint_hash_verified") else "Tidak"}</div></div>'
                '</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Lihat Fingerprint SHA-256 & Metadata Split Manifest"):
                summary_meta = {
                    k: v for k, v in sm.items() if k not in ("train_ids", "val_ids", "test_ids")
                }
                summary_meta["counts"] = {"train": n_tr, "validation": n_val, "test": n_te}
                st.json(summary_meta)
    else:
        st.info("Artefak belum tersedia: `datasets/split_manifest.json` belum dibuat.")

    st.markdown('<h2 class="section-heading">Performa Model Tersimpan (Held-Out Families)</h2>', unsafe_allow_html=True)
    model_dirs = {
        "Random Forest": V2_MODELS_DIR / "rf",
        "XGBoost": V2_MODELS_DIR / "xgboost",
        "MLP": V2_MODELS_DIR / "mlp",
    }
    rows = []
    loaded_manifests: dict[str, dict] = {}
    for name, directory in model_dirs.items():
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            rows.append({
                "Model": name, "Status": "Artefak belum tersedia", "Val RMSE": None,
                "Test MAE": None, "Test RMSE": None, "Test R²": None, "Test Spearman": None,
                "Top-3 Hit (%)": None, "Top-3 Regret": None, "Train (s)": None, "Infer (s)": None,
            })
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rows.append({
                "Model": name, "Status": f"Manifest tidak valid: {exc}", "Val RMSE": None,
                "Test MAE": None, "Test RMSE": None, "Test R²": None, "Test Spearman": None,
                "Top-3 Hit (%)": None, "Top-3 Regret": None, "Train (s)": None, "Infer (s)": None,
            })
            continue
        loaded_manifests[name] = manifest
        metrics = manifest.get("metrics", {})
        val_m = metrics.get("validation_metrics") or {}
        hit_rate = metrics.get("topk_hit_rate")
        rows.append({
            "Model": name,
            "Status": f"Tersedia ({manifest.get('n_features', '-')} fitur)",
            "Val RMSE": round(float(val_m["rmse"]), 4) if "rmse" in val_m else None,
            "Test MAE": round(float(metrics["mae"]), 4) if "mae" in metrics else None,
            "Test RMSE": round(float(metrics["rmse"]), 4) if "rmse" in metrics else None,
            "Test R²": round(float(metrics["r2"]), 4) if "r2" in metrics else None,
            "Test Spearman": round(float(metrics["spearman"]), 4) if metrics.get("spearman") is not None else None,
            "Top-3 Hit (%)": round(float(hit_rate) * 100.0, 2) if hit_rate is not None else None,
            "Top-3 Regret": round(float(metrics["topk_regret"]), 4) if "topk_regret" in metrics else None,
            "Train (s)": round(float(metrics["train_time_sec"]), 3) if "train_time_sec" in metrics else None,
            "Infer (s)": round(float(metrics["inference_time_sec"]), 4) if "inference_time_sec" in metrics else None,
        })

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=38 * len(rows) + 42)
    st.markdown(
        '<div class="insight-note">'
        '<strong>Interpretasi jujur hasil training v2 (<code>permutation-v2.1</code>):</strong> Ketiga model dilatih '
        'pada 4 fitur permutasi murni (<code>hamming_mean</code>, <code>hamming_std</code>, <code>displacement_mean</code>, '
        '<code>displacement_std</code>) pada split keluarga identik tanpa kebocoran data. Nilai <code>Test R² &lt; 0</code> '
        'dilaporkan secara terbuka: 4 fitur jarak permutasi saja belum mengungguli prediktor rata-rata konstan pada '
        'keluarga permutasi baru. Pengaruh penambahan fitur kriptografis murah dianalisis secara khusus pada studi '
        '<strong>Ablation</strong> di halaman <em>Comparative Analysis</em>.'
        '</div>',
        unsafe_allow_html=True,
    )

    fi_fig = render_feature_importance_chart(loaded_manifests)
    if fi_fig is not None:
        st.markdown('<h2 class="section-heading">Feature Importance (Model Berbasis Pohon)</h2>', unsafe_allow_html=True)
        st.pyplot(fi_fig, width="stretch")
        plt.close(fi_fig)


def render_page_reconstruction():
    st.markdown('<h1 class="lead-title">ML-Guided Iterative Reconstruction (Phase B)</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-sub">Pada Phase B, model terlatih dimuat dari <code>models/</code> (dengan validasi ketat '
        'schema fitur dan fingerprint SHA-256) untuk memeringkat kandidat mutasi bijektif secara iteratif. '
        'Penerimaan kandidat tetap berdasarkan <strong>skor aktual</strong> dari <code>evaluate_sbox</code>.</p>',
        unsafe_allow_html=True,
    )

    summary, summary_err = load_comparative_summary()
    if summary and summary.get("paired"):
        st.markdown('<h2 class="section-heading">Kurva Konvergensi Pra-Hitung (Multi-Seed Berpasangan)</h2>', unsafe_allow_html=True)
        seeds = sorted({int(r["seed"]) for r in summary["paired"]})
        selected_seed = st.selectbox("Pilih Seed Eksperimen Berpasangan", seeds, index=0, key="recon_pre_seed")
        seed_runs = [r for r in summary["paired"] if int(r["seed"]) == selected_seed]
        conv_fig = render_convergence_chart(seed_runs, selected_seed)
        st.pyplot(conv_fig, width="stretch")
        plt.close(conv_fig)
    elif summary_err:
        st.error(f"Ringkasan komparatif tidak dapat dibaca: {summary_err}")

    st.markdown('<h2 class="section-heading">Simulasi Live Reconstruction Ringan (Opt-In)</h2>', unsafe_allow_html=True)
    model_options = {
        "XGBoost (models/xgboost)": ("xgboost", V2_MODELS_DIR / "xgboost"),
        "Random Forest — Yafi (models/rf)": ("rf", V2_MODELS_DIR / "rf"),
        "MLP — Naila (models/mlp)": ("mlp", V2_MODELS_DIR / "mlp"),
    }
    c1, c2, c3 = st.columns([1.6, 1.0, 1.2], vertical_alignment="bottom")
    with c1:
        chosen_label = st.selectbox("Pilih Model Terlatih", list(model_options.keys()), key="live_recon_model")
    kind, model_dir = model_options[chosen_label]
    with c2:
        seed = int(st.number_input("Seed Permutasi Awal", min_value=0, max_value=2**31 - 1, value=42, step=1))
    with c3:
        run_clicked = st.button("Jalankan Rekonstruksi Ringan", type="primary", width="stretch")

    manifest_path = model_dir / "manifest.json"
    model_path = model_dir / f"{kind}.joblib"
    if not manifest_path.is_file() or not model_path.is_file():
        st.info(f"Artefak model `{kind}` belum tersedia atau belum lengkap di `{model_dir.relative_to(REPO_ROOT)}`; live reconstruction dinonaktifkan.")
        return

    manifest_mtime = manifest_path.stat().st_mtime_ns
    model_mtime = model_path.stat().st_mtime_ns
    cache_key = (str(model_dir), seed, manifest_mtime, model_mtime)
    if run_clicked:
        with st.spinner(f"Menjalankan rekonstruksi ringan dengan model {chosen_label} (maks. 46 evaluasi penuh)..."):
            try:
                result = cached_v2_reconstruction(*cache_key)
            except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as exc:
                st.error(f"Model v2 tidak dapat digunakan: {exc}")
            else:
                st.session_state["v2_reconstruction_key"] = cache_key
                st.session_state["v2_reconstruction_result"] = result

    result = st.session_state.get("v2_reconstruction_result")
    if st.session_state.get("v2_reconstruction_key") != cache_key or result is None:
        st.caption(
            "Rekonstruksi hanya berjalan setelah tombol ditekan (mode `fast`, maks. 46 evaluasi penuh, ter-cache per model & seed)."
        )
        return

    delta = result["best_score"] - result["initial_score"]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Skor Awal", f"{result['initial_score']:.3f}")
    k2.metric("Skor Terbaik", f"{result['best_score']:.3f}", delta=f"{delta:+.3f}")
    k3.metric("Evaluasi Penuh", f"{result['full_evaluations']}")
    k4.metric("Cache Hits / Misses", f"{result['cache_hits']} / {result['cache_misses']}")
    k5.metric("Waktu Total", f"{result['total_time_sec']:.2f} s")
    st.caption(
        f"Rincian waktu — Fitur: {result['feature_time_sec']:.3f}s · Inferensi: {result['inference_time_sec']:.3f}s · "
        f"Evaluasi penuh: {result['evaluation_time_sec']:.3f}s. Composite Score adalah proxy eksperimen, bukan bukti keamanan formal."
    )

    history = pd.DataFrame(result["history"])
    if not history.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), dpi=150)
        fig.patch.set_facecolor("#FFFFFF")
        ax0, ax1 = axes
        ax0.set_facecolor("#FFFFFF")
        ax0.plot(history["iteration"], history["best_score"], color="#0F5C5C", linewidth=2.0, label="Global Best")
        ax0.plot(history["iteration"], history["actual_score"], color="#8A4307", linewidth=1.1, alpha=0.65, linestyle="--", label="Aktual Top-K")
        ax0.set_xlabel("Iterasi", fontsize=9, color="#344450")
        ax0.set_ylabel("Composite Score", fontsize=9, color="#344450")
        ax0.set_title("Perkembangan Skor per Iterasi", fontsize=9.8, fontweight="bold", color="#141D24", loc="left")
        ax0.legend(frameon=False, fontsize=8.3)
        ax0.grid(True, linestyle="--", linewidth=0.55, color="#E1E8EE")
        _fig_to_clean_style(ax0)

        ax1.set_facecolor("#FFFFFF")
        valid_pred = history.dropna(subset=["predicted_score", "actual_score"])
        if not valid_pred.empty:
            ax1.scatter(valid_pred["predicted_score"], valid_pred["actual_score"], color="#0F5C5C", alpha=0.8, s=36)
        ax1.set_xlabel("Skor Prediksi Model ML", fontsize=9, color="#344450")
        ax1.set_ylabel("Skor Aktual (evaluate_sbox)", fontsize=9, color="#344450")
        ax1.set_title("Prediksi ML vs Skor Aktual Kandidat Terpilih", fontsize=9.8, fontweight="bold", color="#141D24", loc="left")
        ax1.grid(True, linestyle="--", linewidth=0.55, color="#E1E8EE")
        _fig_to_clean_style(ax1)
        fig.tight_layout(w_pad=2.4)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    left, right = st.columns(2)
    with left:
        st.markdown("**S-Box Permutasi Awal (Hex 16×16)**")
        st.dataframe(format_sbox_grid(result["initial_sbox"]), width="stretch", height=360)
    with right:
        st.markdown("**S-Box Hasil Rekonstruksi Terbaik (Hex 16×16)**")
        st.dataframe(format_sbox_grid(result["sbox"]), width="stretch", height=360)

    if not history.empty:
        with st.expander("Lihat Tabel Log Iterasi Rekonstruksi"):
            show_cols = [
                c
                for c in (
                    "iteration", "operator_selected", "n_candidates_unique",
                    "predicted_score", "actual_score", "best_score", "accepted",
                    "full_evals_cumulative", "iter_total_time_sec",
                )
                if c in history.columns
            ]
            st.dataframe(history[show_cols], hide_index=True, width="stretch")


def render_sbox_inspector(catalog: dict[str, dict]):
    st.markdown(
        '<p class="section-sub" style="margin-top:0.2rem;">'
        'Inspeksi detail tabel 16×16, matriks SAC 8×8, dan seluruh Metrik 1 untuk S-Box hasil rekonstruksi v2, '
        'baseline AES/K44, arsip Legacy v1, atau unggah file S-Box Anda sendiri (<code>.npy</code> / <code>.csv</code> / <code>.txt</code>).</p>',
        unsafe_allow_html=True,
    )

    source_mode = st.radio(
        "Sumber S-Box",
        ["Katalog Proyek (v2, Baseline, Legacy v1)", "Unggah File S-Box (.npy / .csv / .txt)"],
        horizontal=True,
        label_visibility="collapsed",
        key="inspector_source_mode",
    )

    selected_name = ""
    selected_sbox: np.ndarray | None = None
    selected_meta = ""

    if source_mode.startswith("Katalog"):
        names = list(catalog.keys())
        default_idx = _default_sbox_index(names)
        selected_name = st.selectbox("Pilih S-Box untuk diinspeksi", names, index=default_idx, key="inspector_sbox")
        entry = catalog[selected_name]
        selected_sbox = entry["sbox"]
        selected_meta = (
            f"Kategori: {entry['status']} · Polinomial: {entry.get('poly') or 'Bebas polinomial'} · "
            f"Konstanta: {entry.get('constant') or '-'} · {entry.get('notes', '')}"
        )
    else:
        uploaded = st.file_uploader(
            "Unggah S-Box 256 elemen (.npy, .csv, atau .txt)",
            type=["npy", "csv", "txt"],
            key="inspector_upload",
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

    metrics = cached_evaluate_sbox(selected_sbox.tobytes())
    aes_ref = cached_evaluate_sbox(catalog["AES Standar (FIPS-197)"]["sbox"].tobytes())
    sac_mat = cached_sac_matrix(selected_sbox.tobytes())

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
                key="inspector_grid_fmt",
            )
        grid_df = format_sbox_grid(selected_sbox, fmt="hex" if fmt_choice == "Heksadesimal" else "dec")
        st.dataframe(grid_df, width="stretch", height=430)

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
                width="stretch",
            )
        with dl2:
            st.download_button(
                "Unduh Tabel 16×16 (.csv)",
                data=csv_text.encode("utf-8"),
                file_name="sbox_16x16.csv",
                mime="text/csv",
                width="stretch",
            )

    with heat_col:
        sac_fig = render_sac_heatmap(sac_mat)
        st.pyplot(sac_fig, width="stretch")
        plt.close(sac_fig)
        st.caption(
            f"Bijective: {'Ya' if metrics['bijective'] else 'Tidak'} · "
            f"Balanced: {'Ya' if metrics['balanced'] else 'Tidak'} · "
            f"BIC-NL: {metrics['bic_nl']} · Skor Gabungan: {metrics['composite_score']:.3f}"
        )


def render_page_crypto_evaluation(
    catalog: dict[str, dict],
    df_v2_focus: pd.DataFrame,
    df_v2_all: pd.DataFrame,
    df_legacy: pd.DataFrame,
):
    st.markdown('<h1 class="lead-title">Cryptographic Evaluation (Metrik 1)</h1>', unsafe_allow_html=True)
    st.warning(
        "Metrik empiris (NL, SAC, BIC, DU/DAP, LAP, FP) dan Composite Score adalah indikator perbandingan "
        "eksperimental, bukan bukti keamanan kriptografi formal."
    )

    tab_v2, tab_legacy, tab_inspect = st.tabs(
        [
            "Hasil v2 Polynomial-Free",
            "Arsip Legacy v1 (GF(2⁸))",
            "Inspeksi 16×16 & Matriks SAC",
        ]
    )

    with tab_v2:
        if df_v2_all.empty:
            st.info("Artefak evaluasi kriptografi v2 belum tersedia; hasil legacy tidak dicampur dengan v2.")
        else:
            st.subheader("Perbandingan Seed Fokus (Seed 100) vs Baseline AES & K44")
            st.dataframe(
                df_v2_focus,
                hide_index=True,
                width="stretch",
                height=38 * len(df_v2_focus) + 42,
            )
            fig_v2 = render_comparison_chart(df_v2_focus, metric_mode="v2")
            st.pyplot(fig_v2, width="stretch")
            plt.close(fig_v2)

            st.subheader("Seluruh Artefak v2 (RF, XGBoost, MLP × 5 Seed)")
            filter_model = st.radio(
                "Filter model v2",
                ["Semua (15)", "RF", "XGBoost", "MLP"],
                horizontal=True,
                label_visibility="collapsed",
                key="crypto_v2_filter",
            )
            if filter_model == "Semua (15)":
                df_show = df_v2_all
            else:
                df_show = df_v2_all[df_v2_all["S-Box"].str.contains(f" {filter_model} ", regex=False)]
            st.dataframe(df_show, hide_index=True, width="stretch")
            st.download_button(
                "Unduh Tabel Metrik v2 (.csv)",
                data=df_show.to_csv(index=False).encode("utf-8"),
                file_name="crypto_metrics_v2.csv",
                mime="text/csv",
            )

    with tab_legacy:
        st.caption(
            "Arsip hasil penelitian tahap awal (v1) yang menggunakan generator polinomial irreducible GF(2⁸) + "
            "matriks affine dan chaotic map. Dipisahkan tegas dari hasil v2."
        )
        if df_legacy.empty:
            st.info("Hasil legacy v1 tidak tersedia.")
        else:
            st.dataframe(
                df_legacy,
                hide_index=True,
                width="stretch",
                height=38 * len(df_legacy) + 42,
            )
            fig_leg = render_comparison_chart(df_legacy, metric_mode="legacy")
            st.pyplot(fig_leg, width="stretch")
            plt.close(fig_leg)
            df_ml_leg = load_legacy_ml_summaries()
            if not df_ml_leg.empty:
                st.subheader("Ringkasan Model ML Legacy v1 (Dataset Campuran GF(2⁸) + Chaotic)")
                st.dataframe(df_ml_leg, hide_index=True, width="stretch", height=38 * len(df_ml_leg) + 42)

    with tab_inspect:
        render_sbox_inspector(catalog)


def render_sim_image_tab(catalog: dict[str, dict]):
    st.markdown(
        '<p class="section-sub" style="margin-top:0.2rem;">'
        'Uji langsung enkripsi dan dekripsi citra <strong>Mandrill</strong> atau gambar Anda sendiri '
        'memakai <strong>AES-128 Modifikasi</strong> dengan custom S-Box. Sistem menghitung Entropy, Histogram χ², '
        'Korelasi Piksel (H/V/D), NPCR, UACI, NBCR (perubahan 1 bit piksel terakhir), serta verifikasi dekripsi <em>lossless</em>.</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns([1.35, 1.0, 1.05, 1.1])
    names = list(catalog.keys())
    default_idx = _default_sbox_index(names)
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
            ["256×256 (Cepat ~0.8 detik)", "128×128 (Instan)", "512×512 (Penuh ~3.2 detik)"],
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
        iv_text = st.text_input("Initialization Vector / IV (16 karakter)", value="MandrillCBC2-IV!", key="img_iv")

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
            pil_img = Image.open(uploaded_img) if uploaded_img is not None else Image.open(MANDRILL_PATH)
            pil_img = pil_img.convert("L" if color_mode.startswith("Grayscale") else "RGB")
            img_arr = np.asarray(pil_img, dtype=np.uint8)

            aes_mode = aes_mode_label.split()[0]
            max_dim = 128 if res_label.startswith("128") else (256 if res_label.startswith("256") else 512)

            with st.spinner("Menjalankan AES-128 Modifikasi, uji difusi 1-bit, korelasi piksel, dan dekripsi lossless..."):
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
        nbcr_txt = f" · NBCR {sim['nbcr']:.2f}%" if "nbcr" in sim else ""
        st.markdown(
            f'<div class="verify-ok">{SVG_CHECK}'
            f'<span>Dekripsi 100% Lossless ({html.escape(payload["sbox_name"])} · AES-128 Modifikasi {mode_used} · '
            f'Resolusi {sim["plain"].shape[1]}×{sim["plain"].shape[0]}{nbcr_txt} · '
            f'Enkripsi {sim["encrypt_sec"]:.2f}s · Dekripsi {sim["decrypt_sec"]:.2f}s)</span></div>',
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
        st.image(sim["plain"], caption="Citra Asli (Plaintext)", width="stretch")
    with p2:
        st.image(sim["cipher"], caption=f"Terenkripsi (AES-128 Modifikasi {mode_used})", width="stretch")
    with p3:
        st.image(sim["decrypted"], caption="Hasil Dekripsi (Identik 100%)", width="stretch")

    corr_info = sim.get("pixel_correlation")
    cols_k = st.columns(6 if corr_info else 5)
    cols_k[0].metric("Entropy Plain", f"{metrics['entropy_plain']:.5f}")
    cols_k[1].metric(
        "Entropy Cipher",
        f"{metrics['entropy_cipher']:.5f}",
        delta=f"{metrics['entropy_cipher'] - metrics['entropy_plain']:+.5f}",
    )
    cols_k[2].metric("NPCR (1-bit flip)", f"{metrics['npcr']:.4f}%", delta="Ideal ~99.61%", delta_color="off")
    cols_k[3].metric("UACI (1-bit flip)", f"{metrics['uaci']:.4f}%", delta="Ideal ~33.46%", delta_color="off")
    cols_k[4].metric("Histogram χ²", f"{metrics['hist_chi2']:.1f}", delta=f"p = {metrics['hist_p_value']:.4f}", delta_color="off")
    if corr_info:
        cc = corr_info["cipher"]
        pc = corr_info["plain"]
        cols_k[5].metric(
            "Korelasi Cipher |r|",
            f"{cc['mean_abs']:.5f}",
            delta=f"Plain {pc['mean_abs']:.3f}",
            delta_color="off",
        )

    hist_fig = render_histogram_comparison(sim["plain"], sim["cipher"])
    st.pyplot(hist_fig, width="stretch")
    plt.close(hist_fig)

    d1, d2, _ = st.columns([1.0, 1.0, 2.0])
    with d1:
        st.download_button(
            "Unduh Citra Terenkripsi (.png)",
            data=_array_to_png_bytes(sim["cipher"]),
            file_name=f"cipher_{mode_used.lower()}.png",
            mime="image/png",
            width="stretch",
        )
    with d2:
        st.download_button(
            "Unduh Hasil Dekripsi (.png)",
            data=_array_to_png_bytes(sim["decrypted"]),
            file_name="decrypted_lossless.png",
            mime="image/png",
            width="stretch",
        )


def render_sim_text_tab(catalog: dict[str, dict]):
    st.markdown(
        '<p class="section-sub" style="margin-top:0.2rem;">'
        'Uji <em>drop-in replacement</em> S-Box hasil rekonstruksi v2 maupun baseline pada enkripsi dan dekripsi '
        'pesan teks <strong>AES-128 Modifikasi</strong>. Coba ubah satu huruf kunci saat dekripsi untuk membuktikan '
        'sensitivitas kunci.</p>',
        unsafe_allow_html=True,
    )

    names = list(catalog.keys())
    default_idx = _default_sbox_index(names)

    t1, t2, t3, t4 = st.columns([1.4, 0.9, 1.1, 1.1])
    with t1:
        sbox_name = st.selectbox("S-Box AES-128", names, index=default_idx, key="txt_sbox")
    with t2:
        mode = st.selectbox("Mode AES", ["CBC2", "CBC", "ECB"], key="txt_mode")
    with t3:
        key_str = st.text_input("Kunci (16 karakter UTF-8)", value="KunciRahasia2026", key="txt_key")
    with t4:
        iv_str = st.text_input("IV (16 karakter UTF-8)", value="VectorAwal2026!!", key="txt_iv")

    enc_col, dec_col = st.columns(2, gap="large")
    with enc_col:
        st.markdown("**Enkripsi Pesan Teks**")
        plain_input = st.text_area(
            "Pesan Asli (Plaintext)",
            value="Proyek Akhir Kriptografi UNNES: Pengujian S-Box CryptoML-SBox v2 pada AES-128 Modifikasi (CBC2).",
            height=135,
        )
        if st.button("Enkripsi Teks", type="primary", width="stretch"):
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
        if st.button("Dekripsi Ciphertext", width="stretch"):
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


def render_page_image_encryption(catalog: dict[str, dict]):
    st.markdown('<h1 class="lead-title">Image Encryption (Mandrill)</h1>', unsafe_allow_html=True)
    st.caption(
        "Seluruh S-Box non-FIPS-197 diberi label AES-128 Modifikasi (custom S-Box pada SubBytes, InvSubBytes, dan KeyExpansion). "
        "Entropy, Histogram χ², Korelasi Piksel, NPCR/NBCR, dan UACI adalah metrik empiris, bukan bukti keamanan formal."
    )

    tab_bench, tab_image, tab_text = st.tabs(
        ["Benchmark Mandrill v2", "Simulasi Interaktif Citra", "Simulasi Teks AES-128"]
    )

    with tab_bench:
        raw_v2, df_v2 = load_mandrill_v2()
        if df_v2.empty:
            st.info("Artefak `results/v2/mandrill_v2_metrics.json` belum tersedia.")
        else:
            lossless_cnt = int((df_v2["Lossless"] == "Ya").sum())
            st.markdown(
                f'<div class="verify-ok">{SVG_CHECK}'
                f'<span>Benchmark Mandrill v2 (512×512 · Mode {html.escape(str(raw_v2.get("cipher_mode", "CBC2")))} · '
                f'SHA-256 citra <code>{html.escape(str(raw_v2.get("image_sha256", "-"))[:16])}…</code> · '
                f'{lossless_cnt}/{len(df_v2)} run terverifikasi dekripsi 100% lossless)</span></div>',
                unsafe_allow_html=True,
            )
            mode_filter = st.radio(
                "Filter mode citra v2",
                ["Semua (10)", "RGB (5)", "Grayscale (5)"],
                horizontal=True,
                label_visibility="collapsed",
                key="mandrill_v2_filter",
            )
            if mode_filter.startswith("RGB"):
                df_show = df_v2[df_v2["Mode"] == "RGB"]
            elif mode_filter.startswith("Grayscale"):
                df_show = df_v2[df_v2["Mode"] == "Grayscale"]
            else:
                df_show = df_v2
            st.dataframe(df_show, hide_index=True, width="stretch", height=38 * len(df_show) + 42)

            corr_fig = render_pixel_correlation_chart(df_v2)
            if corr_fig is not None:
                st.pyplot(corr_fig, width="stretch")
                plt.close(corr_fig)
                rgb_plain_r = df_v2.loc[df_v2["Mode"] == "RGB", "Corr Plain"].iloc[0] if not df_v2[df_v2["Mode"] == "RGB"].empty else 0.8485
                gray_plain_r = df_v2.loc[df_v2["Mode"] == "Grayscale", "Corr Plain"].iloc[0] if not df_v2[df_v2["Mode"] == "Grayscale"].empty else 0.7838
                st.markdown(
                    '<div class="insight-note">'
                    '<strong>Analisis korelasi piksel &amp; difusi:</strong> Citra Mandrill asli memiliki korelasi '
                    f'piksel bertetangga tinggi (<code>Mean |r| = {rgb_plain_r:.4f}</code> pada RGB dan '
                    f'<code>{gray_plain_r:.4f}</code> pada Grayscale). Setelah enkripsi AES-128 Modifikasi mode '
                    '<code>CBC2</code>, seluruh S-Box (AES, K44, RF v2, XGBoost v2, dan MLP v2) meruntuhkan korelasi '
                    'mendekati nol (<code>|r| &lt; 0.003</code>) dengan NPCR ~99.6%, UACI ~33.46%, NBCR ~50.0%, '
                    'serta dekripsi identik byte-per-byte.'
                    '</div>',
                    unsafe_allow_html=True,
                )

        _, df_leg = load_mandrill_legacy()
        if not df_leg.empty:
            with st.expander("Arsip Benchmark Mandrill Legacy v1 & Grafik Laporan"):
                st.dataframe(df_leg, hide_index=True, width="stretch", height=38 * len(df_leg) + 42)
                for fname, cap in [
                    ("mandrill_entropy_chi2.png", "Legacy v1 — Entropy Shannon plaintext vs cipher & Uji Chi-Square"),
                    ("mandrill_diffusion.png", "Legacy v1 — Selisih NPCR, UACI, dan NBCR terhadap nilai ideal"),
                    ("mandrill_encryption_xgb.png", "Legacy v1 — Visualisasi enkripsi, dekripsi, dan histogram RGB"),
                ]:
                    img_bytes = load_figure_bytes(str(FIGURES_DIR / fname))
                    if img_bytes:
                        st.image(img_bytes, caption=cap, width="stretch")

    with tab_image:
        render_sim_image_tab(catalog)
    with tab_text:
        render_sim_text_tab(catalog)


def render_page_comparative_analysis():
    st.markdown('<h1 class="lead-title">Comparative Analysis (v2 Multi-Seed &amp; Ablation)</h1>', unsafe_allow_html=True)
    st.caption(
        "Seluruh tabel dan grafik di bawah berasal langsung dari hasil eksperimen berpasangan tersimpan "
        "(`results/v2/comparative_summary.json`). Hasil negatif/netral dilaporkan secara jujur apa adanya."
    )

    summary, err = load_comparative_summary()
    if err:
        st.error(f"Ringkasan komparatif tidak dapat dibaca: {err}")
        return
    if not summary:
        st.info("Hasil eksperimen v2 belum tersedia; tidak ada angka contoh yang ditampilkan.")
        return

    cfg = summary.get("config", {})
    sources = cfg.get("model_sources", {})
    st.markdown(
        '<div class="stat-strip">'
        '<div class="stat-item"><div class="stat-item-label">Seed Berpasangan</div>'
        f'<div class="stat-item-value">{len(cfg.get("seeds", []))} Seed</div>'
        f'<div class="stat-item-note">{html.escape(", ".join(map(str, cfg.get("seeds", []))))}</div></div>'
        '<div class="stat-item"><div class="stat-item-label">Budget Evaluasi Penuh</div>'
        f'<div class="stat-item-value">{cfg.get("max_evaluations", "-")}</div>'
        '<div class="stat-item-note">Identik untuk seluruh metode pencarian</div></div>'
        '<div class="stat-item"><div class="stat-item-label">Sumber Model ML</div>'
        f'<div class="stat-item-value">{len(sources) if sources else 3} Model</div>'
        f'<div class="stat-item-note">{html.escape(", ".join(f"{k}:{v}" for k, v in sources.items()) if sources else "RF, XGBoost, MLP")}</div></div>'
        '<div class="stat-item"><div class="stat-item-label">Dataset Build Time</div>'
        f'<div class="stat-item-value">{float(summary.get("dataset_sec", 0.0)):.1f} s</div>'
        f'<div class="stat-item-note">{cfg.get("families", "-")} keluarga × {cfg.get("family_size", "-")} sampel</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    paired_summary = summary.get("paired_summary", {})
    differences = summary.get("paired_differences_vs_random_top_k", {})
    if paired_summary and differences:
        st.markdown('<h2 class="section-heading">Perbandingan Metode ML vs Baseline Non-ML</h2>', unsafe_allow_html=True)
        comp_fig = render_paired_comparison_charts(paired_summary, differences)
        st.pyplot(comp_fig, width="stretch")
        plt.close(comp_fig)

        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            st.markdown("**Ringkasan Rata-Rata per Metode (5 Seed)**")
            ps_rows = []
            for method, vals in paired_summary.items():
                ps_rows.append({
                    "Metode": METHOD_LABELS.get(method, method),
                    "Mean Δ Skor": round(float(vals["mean_score_delta"]), 4),
                    "SD": round(float(vals["sd_score_delta"]), 4) if vals.get("sd_score_delta") is not None else 0.0,
                    "Eval Penuh": round(float(vals["mean_full_evaluations"]), 1),
                    "Waktu (s)": round(float(vals["mean_total_time_sec"]), 3),
                })
            st.dataframe(pd.DataFrame(ps_rows), hide_index=True, width="stretch")
        with col_r:
            st.markdown("**Selisih Berpasangan vs Random Top-K (CI 95%)**")
            df_rows = []
            for method, vals in differences.items():
                if method == "random_top_k":
                    continue
                ci = vals.get("ci95")
                ci_str = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "-"
                df_rows.append({
                    "Metode": METHOD_LABELS.get(method, method),
                    "Mean Selisih": round(float(vals["mean_score_delta_difference"]), 4),
                    "SD Selisih": round(float(vals["sd_score_delta_difference"]), 4) if vals.get("sd_score_delta_difference") is not None else None,
                    "Interval 95%": ci_str,
                })
            st.dataframe(pd.DataFrame(df_rows), hide_index=True, width="stretch")

    ablation = summary.get("ablation", {})
    if ablation:
        st.markdown('<h2 class="section-heading">Studi Ablation Fitur (Audit Leakage &amp; Daya Prediksi)</h2>', unsafe_allow_html=True)
        abl_fig = render_ablation_chart(ablation)
        if abl_fig is not None:
            st.pyplot(abl_fig, width="stretch")
            plt.close(abl_fig)

        abl_rows = []
        for key, vals in ablation.items():
            abl_rows.append({
                "Model / Set Fitur": key,
                "Jumlah Fitur": vals.get("n_features"),
                "MAE": round(float(vals["mae"]), 4),
                "RMSE": round(float(vals["rmse"]), 4),
                "R²": round(float(vals["r2"]), 4),
                "Spearman ρ": round(float(vals["spearman"]), 4) if vals.get("spearman") is not None else None,
                "Top-5 Regret": round(float(vals["top_k_regret"]), 4),
                "Train (s)": round(float(vals["train_sec"]), 3),
            })
        st.dataframe(pd.DataFrame(abl_rows), hide_index=True, width="stretch")
        st.markdown(
            '<div class="insight-note">'
            '<strong>Temuan kunci studi ablation:</strong> Membandingkan 4 fitur permutasi murni (<code>perm4</code>), '
            'fitur proxy diferensial tanpa SAC/FP (<code>proxy</code>), dan fitur murah dengan matriks SAC 8×8 serta '
            'fixed points (<code>cheap</code>) pada held-out test set yang sama. Hasil ini menjelaskan secara transparan '
            'mengapa pipeline produksi v2.1 (4 fitur permutasi murni) belum mengungguli pencarian acak, sekaligus '
            'menunjukkan kontribusi spesifik komponen SAC/FP murah dalam memprediksi Composite Score.'
            '</div>',
            unsafe_allow_html=True,
        )

    runs = summary.get("paired", [])
    if runs:
        with st.expander("Lihat Detail Hasil per Seed & Pemisahan Waktu Komputasi (35 Run)"):
            run_rows = [
                {
                    "Seed": int(r["seed"]),
                    "Metode": METHOD_LABELS.get(r["method"], r["method"]),
                    "Skor Awal": round(float(r["initial_score"]), 3),
                    "Skor Akhir": round(float(r["best_score"]), 3),
                    "Δ Skor": round(float(r["score_delta"]), 3),
                    "Eval Penuh": int(r["full_evaluations"]),
                    "Fitur (s)": round(float(r["feature_time_sec"]), 4),
                    "Infer (s)": round(float(r["inference_time_sec"]), 4),
                    "Eval (s)": round(float(r["evaluation_time_sec"]), 3),
                    "Total (s)": round(float(r["total_time_sec"]), 3),
                }
                for r in runs
            ]
            st.dataframe(pd.DataFrame(run_rows), hide_index=True, width="stretch")


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = PAGES[0]

    try:
        catalog, df_all = load_catalog_and_metrics()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Katalog S-Box tidak dapat dimuat: {exc}")
        st.stop()

    df_v2_focus, df_v2_all, df_legacy = _split_metrics_tables(df_all, focus_seed=100)

    with st.sidebar:
        st.markdown("### Status Repositori")
        n_v2 = sum(1 for info in catalog.values() if info["status"] == "Model ML v2")
        n_leg = sum(1 for info in catalog.values() if info["status"] == "Model ML")
        st.caption(f"Katalog aktif: **2** baseline · **{n_v2}** artefak v2 · **{n_leg}** arsip legacy v1")
        st.markdown("**Baseline & Model v2 (Seed 100):**")
        for name, info in catalog.items():
            if info["status"] == "Baseline" or (info["status"] == "Model ML v2" and info.get("seed") == 100):
                st.markdown(f"- **{name}** (`{info['status']}`)")
        st.divider()
        st.caption(
            "Dashboard memuat artefak pra-hitung tanpa retraining saat startup. "
            "Rekonstruksi live bersifat opt-in dengan caching."
        )
        st.markdown("[Buka Repositori GitHub](https://github.com/MahdiHbb1/cryptoml-sbox)")

    st.markdown(
        '<div class="brand-bar">'
        f'<div class="brand-mark">{SVG_BRAND}'
        '<div><div class="brand-name">CryptoML-SBox v2</div>'
        '<div class="brand-sub">Polynomial-Free ML-Guided S-Box Reconstruction · Teknik Informatika UNNES</div></div></div>'
        '<span class="pill pill-done">RF · XGBoost · MLP · AES-128 Modifikasi</span>'
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
        render_page_overview(catalog, df_v2_focus)
    elif active_page == PAGES[1]:
        render_page_dataset()
    elif active_page == PAGES[2]:
        render_page_reconstruction()
    elif active_page == PAGES[3]:
        render_page_crypto_evaluation(catalog, df_v2_focus, df_v2_all, df_legacy)
    elif active_page == PAGES[4]:
        render_page_image_encryption(catalog)
    else:
        render_page_comparative_analysis()


if __name__ == "__main__":
    main()
