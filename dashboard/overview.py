
"""Overview page: model metadata, latest validation, active alerts."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from src.data.data_loader import load_config, load_predictions_log, resolve_path
from src.utils.helpers import load_json


def _load_model_metadata() -> dict:
    config = load_config()
    metadata_path = resolve_path(config["paths"]["model_metadata"])

    if metadata_path.exists():
        return load_json(metadata_path)

    return {}


def _load_latest_report(directory: Path) -> dict:
    """Load the latest JSON report from a directory."""
    if not directory.exists():
        return {}

    reports = sorted(directory.glob("*.json"))

    if not reports:
        return {}

    try:
        return load_json(reports[-1])
    except Exception as exc:
        st.warning(f"Could not load the latest report: {exc}")
        return {}


def render_overview() -> None:
    st.title("Overview")

    config = load_config()
    metadata = _load_model_metadata()

    # Model information
    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Model Type", metadata.get("model_type", "N/A"))

    with col2:
        st.metric("Features", metadata.get("n_features", "N/A"))

    with col3:
        trained_at = metadata.get("trained_at", "N/A")
        st.metric(
            "Trained At",
            str(trained_at)[:10] if trained_at != "N/A" else "N/A",
        )

    # Training metrics
    metrics = metadata.get("metrics", {})

    if metrics:
        st.subheader("Training Metrics")

        metrics_cols = st.columns(len(metrics))

        for i, (key, value) in enumerate(metrics.items()):
            with metrics_cols[i]:
                try:
                    st.metric(
                        key.replace("_", " ").title(),
                        f"{float(value):.4f}",
                    )
                except (ValueError, TypeError):
                    st.metric(
                        key.replace("_", " ").title(),
                        str(value),
                    )

    # Recent predictions
    st.subheader("Recent Predictions")

    predictions = load_predictions_log(config)

    if not predictions.empty:
        recent = predictions.tail(10)

        display_columns = [
            column
            for column in [
                "prediction_id",
                "timestamp",
                "prediction",
                "prediction_proba",
                "model_version",
            ]
            if column in recent.columns
        ]

        st.dataframe(
            recent[display_columns],
            use_container_width=True,
        )
    else:
        st.info(
            "No predictions logged yet. "
            "Run the pipeline to generate predictions."
        )

    # Active alerts
    st.subheader("Active Alerts")

    drift_reports_dir = resolve_path(
        config["paths"]["drift_reports_dir"]
    )

    perf_reports_dir = resolve_path(
        config["paths"]["performance_reports_dir"]
    )

    latest_drift = _load_latest_report(drift_reports_dir)
    latest_perf = _load_latest_report(perf_reports_dir)

    alert_count = 0

    if latest_drift:
        if latest_drift.get("overall_drifted", False):
            drifted_count = len(
                latest_drift.get("drifted_features", [])
            )

            st.error(
                f"🚨 Data drift detected in {drifted_count} feature(s)"
            )
            alert_count += 1
        else:
            st.success("✅ No data drift detected")
    else:
        st.info("No data drift report available yet.")

    if latest_perf:
        if not latest_perf.get("overall_passed", True):
            n_alerts = latest_perf.get("n_alerts", 0)

            st.error(
                f"🚨 {n_alerts} performance alert(s)"
            )
            alert_count += 1
        else:
            st.success("✅ Performance within thresholds")
    else:
        st.info("No performance report available yet.")

    if alert_count == 0:
        st.info("No active alerts.")
