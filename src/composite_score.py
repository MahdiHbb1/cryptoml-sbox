"""Composite Score untuk optimasi kandidat S-box.

Composite Score digunakan sebagai objective optimasi untuk
meranking kandidat S-box selama proses pencarian.

Formula
-------
score =
    NL
    + 0.25 * BIC_NL
    - 2.0 * DU
    - 120.0 * SAC_max_err
    - 40.0 * abs(SAC_mean - 0.5)
    - 30.0 * abs(BIC_SAC - 0.5)
    - 1.5 * FixedPoints
    - 0.05 * WHS_cost

Komponen
---------
NL
    Nonlinearity (lebih besar lebih baik)

BIC_NL
    Bit Independence Criterion - Nonlinearity
    (lebih besar lebih baik)

DU
    Differential Uniformity
    (lebih kecil lebih baik)

SAC_max_err
    Deviasi maksimum terhadap nilai ideal 0.5
    (lebih kecil lebih baik)

SAC_mean
    Nilai ideal = 0.5

BIC_SAC
    Nilai ideal = 0.5

FixedPoints
    Banyaknya fixed point
    (lebih kecil lebih baik)

WHS_cost
    Walsh-Hadamard Spectrum cost
    (lebih kecil lebih baik)

Catatan
--------
Bobot dipertahankan sesuai kontrak proyek dan
tidak diubah pada evaluator v2.

Composite Score merupakan objective optimasi,
bukan bukti keamanan kriptografi.
"""

from __future__ import annotations

COMPOSITE_SCORE_VERSION = "2.0.0"


def compute_composite_score(metrics: dict) -> float:
    """Hitung Composite Score dari hasil evaluasi S-box."""

    if not metrics.get("bijective", False):
        raise ValueError(
            "Composite Score hanya didefinisikan untuk S-box bijektif."
        )

    return float(
        metrics["nl"]
        + 0.25 * metrics["bic_nl"]
        - 2.0 * metrics["du"]
        - 120.0 * metrics["sac_max_err"]
        - 40.0 * abs(metrics["sac_mean"] - 0.5)
        - 30.0 * abs(metrics["bic_sac"] - 0.5)
        - 1.5 * metrics["fixed_points"]
        - 0.05 * metrics["whs_cost"]
    )