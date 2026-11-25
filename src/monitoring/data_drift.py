"""Data drift detection for the ML Model Monitoring Dashboard.

Compares production feature distributions against the reference dataset using
Population Stability Index (PSI), Kolmogorov-Smirnov (KS) test for numeric
features, and chi-square test for categorical features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.utils.metrics import chi_square_test, ks_statistic, psi


@dataclass
class FeatureDriftResult:
    """Drift result for a single feature."""

    feature: str
    drift_type: str  # "numeric" or "categorical"
    statistic: float
    p_value: float
    psi_value: float
    drifted: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataDriftReport:
    """Aggregate data drift report across all features."""

    reference_shape: tuple[int, int]
    production_shape: tuple[int, int]
    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    drifted_features: list[str] = field(default_factory=list)
    overall_drifted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_shape": self.reference_shape,
            "production_shape": self.production_shape,
            "overall_drifted": self.overall_drifted,
            "n_drifted_features": len(self.drifted_features),
            "drifted_features": self.drifted_features,
            "features": [
                {
                    "feature": r.feature,
                    "drift_type": r.drift_type,
                    "statistic": r.statistic,
                    "p_value": r.p_value,
                    "psi_value": r.psi_value,
                    "drifted": r.drifted,
                    "details": r.details,
                }
                for r in self.feature_results
            ],
        }


def detect_data_drift(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    config: dict | None = None,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> DataDriftReport:
    """Detect data drift between reference and production datasets."""
    if config is None:
        from src.data.data_loader import load_config

        config = load_config()

    drift_cfg = config.get("drift", {})
    ks_threshold = drift_cfg.get("ks_p_value_threshold", 0.05)
    psi_threshold = drift_cfg.get("psi_threshold", 0.2)
    chi2_threshold = drift_cfg.get("chi2_p_value_threshold", 0.05)

    if numeric_columns is None:
        numeric_columns = reference.select_dtypes(include=["number"]).columns.tolist()
    if categorical_columns is None:
        categorical_columns = reference.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    target_column = config.get("data", {}).get("target_column", "target")
    numeric_columns = [c for c in numeric_columns if c != target_column]
    categorical_columns = [c for c in categorical_columns if c != target_column]

    feature_results: list[FeatureDriftResult] = []
    drifted_features: list[str] = []

    for col in numeric_columns:
        if col not in reference.columns or col not in production.columns:
            continue
        ref_vals = reference[col].dropna().values
        prod_vals = production[col].dropna().values

        ks_stat, ks_p = ks_statistic(ref_vals, prod_vals)
        psi_val = psi(ref_vals, prod_vals)

        drifted = (ks_p is not None and ks_p < ks_threshold) or (
            psi_val is not None and psi_val >= psi_threshold
        )

        feature_results.append(
            FeatureDriftResult(
                feature=col,
                drift_type="numeric",
                statistic=ks_stat,
                p_value=ks_p,
                psi_value=psi_val,
                drifted=drifted,
                details={
                    "ks_threshold": ks_threshold,
                    "psi_threshold": psi_threshold,
                    "reference_mean": float(ref_vals.mean()) if len(ref_vals) else None,
                    "production_mean": float(prod_vals.mean()) if len(prod_vals) else None,
                },
            )
        )
        if drifted:
            drifted_features.append(col)

    for col in categorical_columns:
        if col not in reference.columns or col not in production.columns:
            continue
        ref_vals = reference[col].dropna().values
        prod_vals = production[col].dropna().values

        chi2_stat, chi2_p = chi_square_test(ref_vals, prod_vals)

        # Categorical PSI using a shared vocabulary for consistent encoding
        all_categories = sorted(set(ref_vals.tolist()) | set(prod_vals.tolist()))
        cat_to_code = {cat: i for i, cat in enumerate(all_categories)}

        # Encode using shared vocabulary; missing categories get code -1 (will be
        # excluded from PSI calculation by the epsilon clipping in psi())
        ref_codes = np.array([cat_to_code.get(c, -1) for c in ref_vals], dtype=float)
        prod_codes = np.array([cat_to_code.get(c, -1) for c in prod_vals], dtype=float)

        # Filter out any unmapped codes (-1) before PSI
        ref_codes = ref_codes[ref_codes >= 0]
        prod_codes = prod_codes[prod_codes >= 0]

        if len(ref_codes) > 0 and len(prod_codes) > 0:
            psi_val = psi(ref_codes, prod_codes, buckets=len(all_categories))
        else:
            psi_val = float("nan")

        drifted = (chi2_p is not None and chi2_p < chi2_threshold) or (
            psi_val is not None and psi_val >= psi_threshold
        )

        feature_results.append(
            FeatureDriftResult(
                feature=col,
                drift_type="categorical",
                statistic=chi2_stat,
                p_value=chi2_p,
                psi_value=psi_val,
                drifted=drifted,
                details={
                    "chi2_threshold": chi2_threshold,
                    "psi_threshold": psi_threshold,
                    "reference_categories": sorted(ref_vals.tolist()),
                    "production_categories": sorted(prod_vals.tolist()),
                    "shared_vocabulary_size": len(all_categories),
                },
            )
        )
        if drifted:
            drifted_features.append(col)

    return DataDriftReport(
        reference_shape=reference.shape,
        production_shape=production.shape,
        feature_results=feature_results,
        drifted_features=drifted_features,
        overall_drifted=len(drifted_features) > 0,
    )