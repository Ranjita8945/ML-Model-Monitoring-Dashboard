"""Model performance monitoring for the ML Model Monitoring Dashboard.

Computes rolling classification metrics over production windows and alerts
when metrics fall below configured thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.utils.metrics import classification_metrics, confusion_matrix_counts


@dataclass
class PerformanceAlert:
    """A single performance alert."""

    metric: str
    current_value: float
    threshold: float
    window: str
    severity: str  # "warning" or "critical"


@dataclass
class PerformanceReport:
    """Aggregate performance monitoring report."""

    window_size: int
    n_windows: int
    window_results: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[PerformanceAlert] = field(default_factory=list)
    overall_passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "n_windows": self.n_windows,
            "overall_passed": self.overall_passed,
            "n_alerts": len(self.alerts),
            "alerts": [
                {
                    "metric": a.metric,
                    "current_value": a.current_value,
                    "threshold": a.threshold,
                    "window": a.window,
                    "severity": a.severity,
                }
                for a in self.alerts
            ],
            "windows": self.window_results,
        }


def rolling_performance(
    predictions_log: pd.DataFrame,
    window_size: int = 100,
    config: dict | None = None,
) -> PerformanceReport:
    """Compute rolling classification metrics over sliding windows."""
    if config is None:
        from src.data.data_loader import load_config

        config = load_config()

    perf_cfg = config.get("performance", {})
    min_accuracy = perf_cfg.get("min_accuracy", 0.70)
    min_precision = perf_cfg.get("min_precision", 0.65)
    min_recall = perf_cfg.get("min_recall", 0.65)
    min_f1 = perf_cfg.get("min_f1", 0.65)
    min_roc_auc = perf_cfg.get("min_roc_auc", 0.70)

    available = [c for c in ["actual_label", "prediction"] if c in predictions_log.columns]
    if not available:
        return PerformanceReport(
            window_size=window_size,
            n_windows=0,
            overall_passed=True,
        )

    labeled = predictions_log.dropna(subset=available).copy()

    if labeled.empty or "actual_label" not in labeled.columns or "prediction" not in labeled.columns:
        return PerformanceReport(
            window_size=window_size,
            n_windows=0,
            overall_passed=True,
        )

    if window_size <= 0:
         raise ValueError("window_size must be greater than 0")

    n_windows = (len(labeled) + window_size - 1) // window_size
    window_results: list[dict[str, Any]] = []
    alerts: list[PerformanceAlert] = []
    overall_passed = True

    for i in range(n_windows):
        start = i * window_size
        end = min(start + window_size, len(labeled))
        window_df = labeled.iloc[start:end]

        y_true = window_df["actual_label"].values
        y_pred = window_df["prediction"].values
        y_proba = (
            window_df["prediction_proba"].values
            if "prediction_proba" in window_df.columns
            else None
        )

        metrics = classification_metrics(y_true, y_pred, y_proba)
        cm = confusion_matrix_counts(y_true, y_pred)

        window_label = f"window_{i+1}_rows_{start}-{end}"
        window_results.append(
            {
                "window": window_label,
                "n_samples": len(window_df),
                "metrics": metrics,
                "confusion_matrix": cm,
            }
        )

        window_passed = True
        if metrics["accuracy"] < min_accuracy:
            alerts.append(
                PerformanceAlert(
                    metric="accuracy",
                    current_value=metrics["accuracy"],
                    threshold=min_accuracy,
                    window=window_label,
                    severity="critical" if metrics["accuracy"] < min_accuracy * 0.8 else "warning",
                )
            )
            window_passed = False
        if metrics["precision"] < min_precision:
            alerts.append(
                PerformanceAlert(
                    metric="precision",
                    current_value=metrics["precision"],
                    threshold=min_precision,
                    window=window_label,
                    severity="warning",
                )
            )
            window_passed = False
        if metrics["recall"] < min_recall:
            alerts.append(
                PerformanceAlert(
                    metric="recall",
                    current_value=metrics["recall"],
                    threshold=min_recall,
                    window=window_label,
                    severity="warning",
                )
            )
            window_passed = False
        if metrics["f1"] < min_f1:
            alerts.append(
                PerformanceAlert(
                    metric="f1",
                    current_value=metrics["f1"],
                    threshold=min_f1,
                    window=window_label,
                    severity="warning",
                )
            )
            window_passed = False
        if y_proba is not None and metrics.get("roc_auc") is not None and metrics["roc_auc"] < min_roc_auc:
            alerts.append(
                PerformanceAlert(
                    metric="roc_auc",
                    current_value=metrics["roc_auc"],
                    threshold=min_roc_auc,
                    window=window_label,
                    severity="warning",
                )
            )
            window_passed = False

        if not window_passed:
            overall_passed = False

    return PerformanceReport(
        window_size=window_size,
        n_windows=n_windows,
        window_results=window_results,
        alerts=alerts,
        overall_passed=overall_passed,
    )
