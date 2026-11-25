
"""Monitoring report generation for the ML Model Monitoring Dashboard.

Orchestrates data drift, concept drift, and performance monitoring, then
writes JSON reports to drift_reports/ and performance_reports/.
"""

from __future__ import annotations

from typing import Any

from src.data.data_loader import (
    load_config,
    load_production_data,
    load_reference_data,
    load_predictions_log,
    resolve_path,
)
from src.monitoring.concept_drift import detect_concept_drift
from src.monitoring.data_drift import detect_data_drift
from src.monitoring.performance_monitor import rolling_performance
from src.utils.helpers import ensure_dir, save_json, utc_now


def run_monitoring_cycle(
    config: dict | None = None,
) -> dict[str, Any]:
    """Run a complete monitoring cycle and save monitoring reports."""

    if config is None:
        config = load_config()

    try:
        reference = load_reference_data(config)
        production = load_production_data(config)
        predictions = load_predictions_log(config)

        # Data drift
        drift_report = detect_data_drift(
            reference,
            production,
            config=config,
        )

        # Concept drift
        concept_report = detect_concept_drift(
            predictions,
            config=config,
        )

        # Performance monitoring
        perf_cfg = config.get("performance", {})
        window_size = perf_cfg.get("window_size", 100)

        perf_report = rolling_performance(
            predictions,
            window_size=window_size,
            config=config,
        )

        # Prepare report directories
        drift_reports_dir = resolve_path(
            config["paths"]["drift_reports_dir"]
        )
        perf_reports_dir = resolve_path(
            config["paths"]["performance_reports_dir"]
        )

        ensure_dir(drift_reports_dir)
        ensure_dir(perf_reports_dir)

        timestamp = (
            utc_now()
            .replace(":", "-")
            .replace(".", "-")
        )

        drift_path = (
            drift_reports_dir / f"drift_report_{timestamp}.json"
        )
        perf_path = (
            perf_reports_dir / f"performance_report_{timestamp}.json"
        )

        # Save reports
        save_json(drift_report.to_dict(), drift_path)
        save_json(perf_report.to_dict(), perf_path)

        return {
            "timestamp": timestamp,
            "data_drift": drift_report.to_dict(),
            "concept_drift": concept_report.to_dict(),
            "performance": perf_report.to_dict(),
            "drift_report_path": str(drift_path),
            "performance_report_path": str(perf_path),
        }

    except Exception as exc:
        raise RuntimeError(
            f"Monitoring cycle failed: {exc}"
        ) from exc
