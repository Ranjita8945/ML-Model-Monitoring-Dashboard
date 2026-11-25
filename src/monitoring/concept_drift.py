"""Concept drift detection for the ML Model Monitoring Dashboard.

Concept drift occurs when the relationship between inputs and target changes
over time, i.e. P(Y|X) != P_ref(Y|X). Because ground truth labels may arrive
with delay, concept drift is inferred indirectly by comparing production
model performance against a reference (training-time) baseline.

Drift types distinguished:
    - Data drift:      P(X) changes (handled in data_drift.py)
    - Prediction drift: P(Ŷ) changes — model outputs shift
    - Label drift:     P(Y) changes — actual label distribution shifts
    - Concept drift:   P(Y|X) changes — inferred from performance
                       degradation vs reference baseline

Limitations:
    Performance degradation is an *indirect* signal of concept drift. Data
    drift (covariate shift) can also cause performance degradation without any
    change in P(Y|X). Concept drift should be confirmed with targeted
    analysis when ground truth labels are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.data_loader import load_config, resolve_path
from src.utils.metrics import classification_metrics, confusion_matrix_counts, psi
from src.utils.helpers import load_json, safe_read_csv

@dataclass
class DriftCheckResult:
    """Result of a single drift check."""

    check: str
    drift_type: str  # "data", "prediction", "label", "concept"
    drifted: bool
    statistic: float
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptDriftReport:
    """Aggregate concept drift report.

    Attributes:
        n_predictions: Total number of prediction records.
        n_with_labels: Number of records with actual_label available.
        checks: List of individual drift check results.
        reference_metrics: Reference (training-time) performance metrics.
        production_metrics: Latest production window performance metrics.
        overall_drifted: True if concept-level drift (performance degradation)
            is detected.
        status: "no_data", "passed", or "drifted".
    """

    n_predictions: int
    n_with_labels: int
    checks: list[DriftCheckResult] = field(default_factory=list)
    reference_metrics: dict[str, float] = field(default_factory=dict)
    production_metrics: dict[str, float] = field(default_factory=dict)
    overall_drifted: bool = False
    status: str = "no_data"  # "no_data" | "passed" | "drifted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_predictions": self.n_predictions,
            "n_with_labels": self.n_with_labels,
            "status": self.status,
            "overall_drifted": self.overall_drifted,
            "reference_metrics": self.reference_metrics,
            "production_metrics": self.production_metrics,
            "checks": [
                {
                    "check": c.check,
                    "drift_type": c.drift_type,
                    "drifted": c.drifted,
                    "statistic": c.statistic,
                    "threshold": c.threshold,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }

def _load_reference_metrics(config: dict) -> dict[str, float]:
    """Load reference (training-time) metrics from model metadata."""
    metadata_path = resolve_path(config.get("paths", {}).get("model_metadata", ""))
    if metadata_path and metadata_path.exists():
        try:
            metadata = load_json(metadata_path)
            return metadata.get("metrics", {})
        except (ValueError, OSError):
            return {}
    return {}


def detect_prediction_drift(
    reference_predictions: pd.Series,
    production_predictions: pd.Series,
    psi_threshold: float = 0.2,
) -> DriftCheckResult:
    """Detect prediction drift: P(Ŷ) changes between reference and production.

    This is NOT concept drift — it only indicates that the model's output
    distribution has shifted, which may or may not be accompanied by concept
    drift.
    """
    ref_vals = reference_predictions.dropna().values
    prod_vals = production_predictions.dropna().values

    if len(ref_vals) == 0 or len(prod_vals) == 0:
        return DriftCheckResult(
            check="prediction_distribution_psi",
            drift_type="prediction",
            drifted=False,
            statistic=float("nan"),
            threshold=psi_threshold,
            details={"reference_n": 0, "production_n": 0, "error": "insufficient data"},
        )

    psi_val = psi(ref_vals, prod_vals)
    drifted = not np.isnan(psi_val) and psi_val >= psi_threshold

    return DriftCheckResult(
        check="prediction_distribution_psi",
        drift_type="prediction",
        drifted=drifted,
        statistic=0.0 if np.isnan(psi_val) else float(psi_val),
        threshold=psi_threshold,
        details={
            "reference_n": int(len(ref_vals)),
            "production_n": int(len(prod_vals)),
            "note": (
                "Prediction distribution shift alone does not confirm concept drift."
            ),
        },
    )
def detect_label_drift(
    reference_labels: pd.Series,
    production_labels: pd.Series,
    threshold: float = 0.15,
) -> DriftCheckResult:
    """Detect label drift: P(Y) changes between reference and production.

    Compares the positive-class rate between reference and production label
    distributions. A shift beyond ``threshold`` indicates label drift.
    """
    ref_vals = reference_labels.dropna().values
    prod_vals = production_labels.dropna().values

    if len(ref_vals) == 0 or len(prod_vals) == 0:
        return DriftCheckResult(
            check="label_distribution_shift",
            drift_type="label",
            drifted=False,
            statistic=float("nan"),
            threshold=threshold,
            details={"reference_n": 0, "production_n": 0, "error": "insufficient data"},
        )

    ref_positive_rate = float(np.mean(ref_vals == 1))
    prod_positive_rate = float(np.mean(prod_vals == 1))
    shift = abs(prod_positive_rate - ref_positive_rate)
    drifted = shift > threshold

    return DriftCheckResult(
        check="label_distribution_shift",
        drift_type="label",
        drifted=drifted,
        statistic=0.0 if np.isnan(shift) else float(shift),
        threshold=threshold,
        details={
            "reference_positive_rate": ref_positive_rate,
            "production_positive_rate": prod_positive_rate,
            "reference_n": int(len(ref_vals)),
            "production_n": int(len(prod_vals)),
        },
    )


def detect_concept_drift_from_performance(
    reference_metrics: dict[str, float],
    production_metrics: dict[str, float],
    degradation_threshold: float = 0.10,
) -> DriftCheckResult:
    """Detect concept drift from performance degradation vs reference baseline.

    Concept drift is inferred when production performance metrics fall below
    the reference baseline by more than ``degradation_threshold`` (relative
    drop). For example, if reference accuracy is 0.90 and threshold is 0.10,
    concept drift is flagged when production accuracy drops below 0.81.

    This is an indirect signal: performance degradation can also be caused by
    data drift, so concept drift should be confirmed with additional
    investigation.
    """
    if not reference_metrics or not production_metrics:
        return DriftCheckResult(
            check="performance_degradation",
            drift_type="concept",
            drifted=False,
            statistic=0.0,
            threshold=degradation_threshold,
            details={"error": "missing reference or production metrics"},
        )

    max_relative_drop = 0.0
    metric_details: dict[str, Any] = {}

    for metric in ["accuracy", "precision", "recall", "f1"]:
        ref_val = reference_metrics.get(metric)
        prod_val = production_metrics.get(metric)

        if ref_val is None or prod_val is None or np.isnan(ref_val) or np.isnan(prod_val):
            metric_details[metric] = {
                "reference": ref_val,
                "production": prod_val,
                "relative_drop": None,
            }
            continue

        if ref_val > 0:
            relative_drop = (ref_val - prod_val) / ref_val
        else:
            relative_drop = 0.0 if prod_val <= ref_val else float("nan")

        metric_details[metric] = {
            "reference": float(ref_val),
            "production": float(prod_val),
            "relative_drop": None if np.isnan(relative_drop) else float(relative_drop),
        }
        if not np.isnan(relative_drop) and relative_drop > max_relative_drop:
            max_relative_drop = float(relative_drop)

    drifted = max_relative_drop > degradation_threshold

    return DriftCheckResult(
        check="performance_degradation",
        drift_type="concept",
        drifted=drifted,
        statistic=max_relative_drop,
        threshold=degradation_threshold,
        details={
            "degradation_threshold": degradation_threshold,
            "metrics": metric_details,
            "limitation": (
                "Performance degradation is an indirect signal of concept drift. "
                "Data drift (covariate shift) can also cause performance "
                "degradation without any change in P(Y|X). Confirm with "
                "targeted analysis."
            ),
        },
    )


def evaluate_with_ground_truth(
    y_true: list[int] | np.ndarray | pd.Series | None = None,
    y_pred: list[int] | np.ndarray | pd.Series | None = None,
    y_proba: list[float] | np.ndarray | pd.Series | None = None,
    predictions_log: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Evaluate model performance using ground truth labels.

    Supports direct metric computation from y_true/y_pred/y_proba, or extraction
    from a predictions_log DataFrame containing 'actual_label' and 'prediction'
    columns.

    Returns a dictionary with 'n_samples', 'metrics', and 'confusion_matrix'.
    If no ground truth is available, returns an 'error' key.
    """
    from src.utils.metrics import classification_metrics, confusion_matrix_counts

    if predictions_log is not None:
        if not isinstance(predictions_log, pd.DataFrame) or predictions_log.empty:
            return {"error": "predictions_log is empty or invalid", "n_samples": 0}
        if "actual_label" not in predictions_log.columns or "prediction" not in predictions_log.columns:
            return {"error": "predictions_log missing required columns", "n_samples": 0}
        labeled = predictions_log.dropna(subset=["actual_label", "prediction"])
        if labeled.empty:
            return {"error": "no labeled records in predictions_log", "n_samples": 0}
        y_true = labeled["actual_label"].values
        y_pred = labeled["prediction"].values
        y_proba = (
            labeled["prediction_proba"].values
            if "prediction_proba" in labeled.columns
            else None
        )

    if y_true is None or y_pred is None:
        return {"error": "no ground truth provided", "n_samples": 0}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return {"error": "empty ground truth", "n_samples": 0}

    if y_proba is not None:
        y_proba = np.asarray(y_proba)

    metrics = classification_metrics(y_true, y_pred, y_proba)
    cm = confusion_matrix_counts(y_true, y_pred)

    return {
        "n_samples": int(len(y_true)),
        "metrics": metrics,
        "confusion_matrix": cm,
    }


def detect_concept_drift(
    predictions_log: pd.DataFrame | None = None,
    config: dict | None = None,
    reference_predictions: pd.DataFrame | None = None,
    reference_labels: pd.DataFrame | None = None,
    window_size: int = 100,
    degradation_threshold: float = 0.10,
    psi_threshold: float = 0.2,
    label_drift_threshold: float = 0.15,
) -> ConceptDriftReport:
    """Detect all drift types from prediction logs."""
    if config is None:
        config = load_config()

    drift_cfg = config.get("drift", {})
    psi_threshold = drift_cfg.get("psi_threshold", psi_threshold)
    label_drift_threshold = drift_cfg.get(
        "label_drift_threshold", label_drift_threshold
    )
    degradation_threshold = drift_cfg.get(
        "degradation_threshold", degradation_threshold
    )

    if predictions_log is None:
        predictions_log = safe_read_csv(config["paths"]["predictions_log"])

    if predictions_log.empty:
        return ConceptDriftReport(
            n_predictions=0,
            n_with_labels=0,
            status="no_data",
        )

    n_predictions = len(predictions_log)
    # n_with_labels counts rows where BOTH actual_label AND prediction are present
    if "actual_label" in predictions_log.columns and "prediction" in predictions_log.columns:
        labeled = predictions_log.dropna(subset=["actual_label", "prediction"])
    else:
        labeled = predictions_log.iloc[0:0]  # empty copy with same columns
    n_with_labels = len(labeled)

    checks: list[DriftCheckResult] = []

    if reference_predictions is not None and "prediction" in reference_predictions.columns and "prediction" in predictions_log.columns:
        checks.append(
            detect_prediction_drift(
                reference_predictions["prediction"],
                predictions_log["prediction"],
                psi_threshold=psi_threshold,
            )
        )
    else:
        checks.append(
            DriftCheckResult(
                check="prediction_distribution_psi",
                drift_type="prediction",
                drifted=False,
                statistic=0.0,
                threshold=psi_threshold,
                details={"error": "no reference predictions provided"},
            )
        )

    # --- Label drift: P(Y) changes ---
    if reference_labels is not None and "actual_label" in reference_labels.columns:
        ref_labels = reference_labels["actual_label"]
    else:
        ref_labels = pd.Series(dtype=float)

    if "actual_label" in labeled.columns and not labeled.empty:
        prod_labels = labeled["actual_label"]
    else:
        prod_labels = pd.Series(dtype=float)
    checks.append(
        detect_label_drift(
            ref_labels,
            prod_labels,
            threshold=label_drift_threshold,
        )
    )

    # --- Concept drift: P(Y|X) changes (inferred from performance) ---
    reference_metrics = _load_reference_metrics(config)

    production_metrics: dict[str, float] = {}
    if not labeled.empty and window_size > 0 and "actual_label" in labeled.columns and "prediction" in labeled.columns:
        
        
        # Select the most recent production records.
        latest_window = labeled.tail(window_size)
        y_true = latest_window["actual_label"].values
        y_pred = latest_window["prediction"].values
        y_proba = (
            latest_window["prediction_proba"].values
            if "prediction_proba" in latest_window.columns
            else None
        )

        production_metrics = classification_metrics(y_true, y_pred, y_proba)
    elif not labeled.empty and "actual_label" in labeled.columns and "prediction" in labeled.columns:
        # Fall back to all labeled data if windowing not applicable
        y_true = labeled["actual_label"].values
        y_pred = labeled["prediction"].values
        y_proba = (
            labeled["prediction_proba"].values
            if "prediction_proba" in labeled.columns
            else None
        )
        production_metrics = classification_metrics(y_true, y_pred, y_proba)

    checks.append(
        detect_concept_drift_from_performance(
            reference_metrics,
            production_metrics,
            degradation_threshold=degradation_threshold,
        )
    )

    # --- Determine overall status ---
    concept_drifted = any(
        c.drift_type == "concept" and c.drifted for c in checks
    )
    any_drifted = any(c.drifted for c in checks)

    if n_with_labels == 0:
        status = "no_data"
    elif concept_drifted:
        status = "drifted"
    elif any_drifted:
        status = "drifted"
    else:
        status = "passed"

    overall_drifted = concept_drifted

    return ConceptDriftReport(
        n_predictions=n_predictions,
        n_with_labels=n_with_labels,
        checks=checks,
        reference_metrics=reference_metrics,
        production_metrics=production_metrics,
        overall_drifted=overall_drifted,
        status=status,
    )