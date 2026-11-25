"""Data drift detection for the ML Model Monitoring Dashboard.

Compares production feature distributions against the reference dataset using
PSI, KS test for numeric features, and chi-square test for categorical features.
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
    drift_type: str
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


def _categorical_psi(
    expected: np.ndarray,
    actual: np.ndarray,
) -> float:
    """Calculate PSI directly from categorical frequency distributions."""

    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    categories = np.union1d(expected, actual)

    expected_counts = np.array(
        [(expected == category).sum() for category in categories],
        dtype=float,
    )
    actual_counts = np.array(
        [(actual == category).sum() for category in categories],
        dtype=float,
    )

    expected_props = expected_counts / len(expected)
    actual_props = actual_counts / len(actual)

    epsilon = 1e-6
    expected_props = np.clip(expected_props, epsilon, None)
    actual_props = np.clip(actual_props, epsilon, None)

    return float(
        np.sum(
            (actual_props - expected_props)
            * np.log(actual_props / expected_props)
        )
    )


def detect_data_drift(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    config: dict | None = None,
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
) -> DataDriftReport:
    """Detect data drift between reference and production datasets."""

    if not isinstance(reference, pd.DataFrame) or not isinstance(
        production, pd.DataFrame
    ):
        raise TypeError("reference and production must be pandas DataFrames")

    if reference.empty or production.empty:
        raise ValueError("Reference and production datasets cannot be empty")

    if config is None:
        from src.data.data_loader import load_config

        config = load_config()

    drift_cfg = config.get("drift", {})

    ks_threshold = drift_cfg.get("ks_p_value_threshold", 0.05)
    psi_threshold = drift_cfg.get("psi_threshold", 0.2)
    chi2_threshold = drift_cfg.get("chi2_p_value_threshold", 0.05)

    target_column = config.get("data", {}).get("target_column", "target")

    if numeric_columns is None:
        numeric_columns = reference.select_dtypes(
            include=["number"]
        ).columns.tolist()

    if categorical_columns is None:
        categorical_columns = reference.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    numeric_columns = [
        col for col in numeric_columns if col != target_column
    ]
    categorical_columns = [
        col for col in categorical_columns if col != target_column
    ]

    feature_results: list[FeatureDriftResult] = []
    drifted_features: list[str] = []

    # Numeric feature drift
    for col in numeric_columns:

        if col not in reference.columns or col not in production.columns:
            continue

        ref_vals = pd.to_numeric(
            reference[col], errors="coerce"
        ).dropna().to_numpy()

        prod_vals = pd.to_numeric(
            production[col], errors="coerce"
        ).dropna().to_numpy()

        ks_stat, ks_p = ks_statistic(ref_vals, prod_vals)
        psi_val = psi(ref_vals, prod_vals)

        drifted = (
            (np.isfinite(ks_p) and ks_p < ks_threshold)
            or (np.isfinite(psi_val) and psi_val >= psi_threshold)
        )

        feature_results.append(
            FeatureDriftResult(
                feature=col,
                drift_type="numeric",
                statistic=ks_stat,
                p_value=ks_p,
                psi_value=psi_val,
                drifted=bool(drifted),
                details={
                    "ks_threshold": ks_threshold,
                    "psi_threshold": psi_threshold,
                    "reference_mean": (
                        float(ref_vals.mean()) if len(ref_vals) else None
                    ),
                    "production_mean": (
                        float(prod_vals.mean()) if len(prod_vals) else None
                    ),
                },
            )
        )

        if drifted:
            drifted_features.append(col)

    # Categorical feature drift
    for col in categorical_columns:

        if col not in reference.columns or col not in production.columns:
            continue

        ref_vals = (
            reference[col].dropna().astype(str).to_numpy()
        )
        prod_vals = (
            production[col].dropna().astype(str).to_numpy()
        )

        chi2_stat, chi2_p = chi_square_test(ref_vals, prod_vals)

        psi_val = _categorical_psi(ref_vals, prod_vals)

        drifted = (
            (np.isfinite(chi2_p) and chi2_p < chi2_threshold)
            or (np.isfinite(psi_val) and psi_val >= psi_threshold)
        )

        feature_results.append(
            FeatureDriftResult(
                feature=col,
                drift_type="categorical",
                statistic=chi2_stat,
                p_value=chi2_p,
                psi_value=psi_val,
                drifted=bool(drifted),
                details={
                    "chi2_threshold": chi2_threshold,
                    "psi_threshold": psi_threshold,
                    "reference_categories": sorted(set(ref_vals)),
                    "production_categories": sorted(set(prod_vals)),
                    "shared_vocabulary_size": len(
                        set(ref_vals) | set(prod_vals)
                    ),
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