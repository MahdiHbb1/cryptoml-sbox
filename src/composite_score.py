"""Composite Score: satu-satunya sumber rumus objective."""
from __future__ import annotations


def compute_composite_score(metrics: dict) -> float:
    if not metrics.get("bijective", False):
        raise ValueError("Composite Score hanya didefinisikan untuk S-box bijektif.")
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