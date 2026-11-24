"""Concept drift detection for the ML Model Monitoring Dashboard.

Concept drift occurs when the relationship between inputs and predictions
changes over time. This module monitors prediction distribution shifts and
supports delayed ground-truth updates for label-drift analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.utils.metrics import classification_metrics, confusion_matrix_counts, psi
from src.utils.helpers import safe_read_csv


@dataclass
class ConceptDriftResult:
    """Result of a concept drift check."""

    check: str
    drifted: bool
    statistic: float
    threshold: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptDriftReport:
    """Aggregate concept drift report."""

    n_predictions: int
    n_with_labels: int
    checks: list[ConceptDriftResult] = field(default_factory=list)
    overall_drifted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_predictions": self.n_predictions,
            "n_with_labels": self.n_with_labels,
            "overall_drifted": self.overall_drifted,
            "checks": [
                {
                    "check": c.check,
                    "drifted": c.drifted,
                    "statistic": c.statistic,
                    "threshold": c.threshold,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


def detect_concept_drift(
    predictions_log: pd.DataFrame | None = None,
    config: dict | None = None,
    reference_predictions: pd.DataFrame | None = None,
) -> ConceptDriftReport:
    """Detect concept drift from prediction logs."""
    if config is None:
        from src.data.data_loader import load_config

        config = load_config()

    drift_cfg = config.get("drift", {})
    psi_threshold = drift_cfg.get("psi_threshold", 0.2)

    if predictions_log is None:
        predictions_log = safe_read_csv(config["paths"]["predictions_log"])

    if predictions_log.empty:
        return ConceptDriftReport(
            n_predictions=0,
            n_with_labels=0,
            overall_drifted=False,
        )

    n_predictions = len(predictions_log)

    checks: list[ConceptDriftResult] = []
    if reference_predictions is not None and "prediction" in reference_predictions.columns:
        ref_preds = reference_predictions["prediction"].dropna().values
        prod_preds = predictions_log["prediction"].dropna().values
        psi_val = psi(ref_preds, prod_preds)
        drifted = psi_val >= psi_threshold
        checks.append(
            ConceptDriftResult(
                check="prediction_distribution_psi",
                drifted=drifted,
                statistic=psi_val,
                threshold=psi_threshold,
                details={"reference_n": len(ref_preds), "production_n": len(prod_preds)},
            )
        )

    if "actual_label" in predictions_log.columns:
        labeled = predictions_log.dropna(subset=["actual_label"])
        n_with_labels = len(labeled)
        if n_with_labels > 0:
            label_dist = labeled["actual_label"].value_counts(normalize=True).sort_index()
            expected_positive_rate = 0.5
            actual_positive_rate = float(label_dist.get(1, 0.0))
            label_shift = abs(actual_positive_rate - expected_positive_rate)
            label_drifted = label_shift > 0.15
            checks.append(
                ConceptDriftResult(
                    check="label_distribution_shift",
                    drifted=label_drifted,
                    statistic=label_shift,
                    threshold=0.15,
                    details={
                        "positive_rate": actual_positive_rate,
                        "expected_positive_rate": expected_positive_rate,
                        "n_labeled": n_with_labels,
                    },
                )
            )
        else:
            n_with_labels = 0
    else:
        n_with_labels = 0

    overall_drifted = any(c.drifted for c in checks)

    return ConceptDriftReport(
        n_predictions=n_predictions,
        n_with_labels=n_with_labels,
        checks=checks,
        overall_drifted=overall_drifted,
    )


def evaluate_with_ground_truth(
    y_true: list[int] | None = None,
    y_pred: list[int] | None = None,
    y_proba: list[float] | None = None,
    predictions_log: pd.DataFrame | None = None,
    config: dict | None = None,
) -> dict[str, Any]:
    """Evaluate model performance using ground truth labels from predictions log."""
    if config is None:
        from src.data.data_loader import load_config

        config = load_config()

    if y_true is None and predictions_log is not None:
        labeled = predictions_log.dropna(subset=["actual_label"])
        if len(labeled) > 0:
            y_true = labeled["actual_label"].tolist()
            y_pred = labeled["prediction"].tolist()
            if "prediction_proba" in labeled.columns:
                y_proba = labeled["prediction_proba"].tolist()

    if y_true is None or y_pred is None:
        return {
            "error": "No ground truth available for evaluation",
            "n_samples": 0,
        }

    metrics = classification_metrics(y_true, y_pred, y_proba)
    cm = confusion_matrix_counts(y_true, y_pred)

    return {
        "n_samples": len(y_true),
        "metrics": metrics,
        "confusion_matrix": cm,
    }
