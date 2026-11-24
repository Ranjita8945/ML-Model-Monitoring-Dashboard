"""Overview page: model metadata, latest validation, active alerts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.data_loader import load_config, load_predictions_log, resolve_path
from src.utils.helpers import load_json, safe_read_csv


def _load_model_metadata() -> dict:
    config = load_config()
    metadata_path = resolve_path(config["paths"]["model_metadata"])
    if metadata_path.exists():
        return load_json(metadata_path)
    return {}


def render_overview() -> None:
    st.title("Overview")

    config = load_config()
    metadata = _load_model_metadata()

    # --- Model Info ---
    st.subheader("Model Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model Type", metadata.get("model_type", "N/A"))
    with col2:
        st.metric("Features", metadata.get("n_features", "N/A"))
    with col3:
        trained_at = metadata.get("trained_at", "N/A")
        st.metric("Trained At", str(trained_at)[:10] if trained_at != "N/A" else "N/A")

    if metadata.get("metrics"):
        st.subheader("Training Metrics")
        metrics_cols = st.columns(len(metadata["metrics"]))
        for i, (k, v) in enumerate(metadata["metrics"].items()):
            with metrics_cols[i]:
                st.metric(k.replace("_", " ").title(), f"{v:.4f}")

    # --- Latest Predictions ---
    st.subheader("Recent Predictions")
    predictions = load_predictions_log(config)
    if not predictions.empty:
        recent = predictions.tail(10)
        st.dataframe(
            recent[["prediction_id", "timestamp", "prediction", "prediction_proba", "model_version"]],
            use_container_width=True,
        )
    else:
        st.info("No predictions logged yet. Run the pipeline to generate predictions.")

    # --- Active Alerts ---
    st.subheader("Active Alerts")
    drift_reports_dir = resolve_path(config["paths"]["drift_reports_dir"])
    perf_reports_dir = resolve_path(config["paths"]["performance_reports_dir"])

    alert_count = 0
    if drift_reports_dir.exists():
        drift_reports = sorted(drift_reports_dir.glob("*.json"))
        if drift_reports:
            latest_drift = load_json(drift_reports[-1])
            if latest_drift.get("overall_drifted"):
                st.error(f"🚨 Data drift detected in {len(latest_drift.get('drifted_features', []))} feature(s)")
                alert_count += 1
            else:
                st.success("✅ No data drift detected")

    if perf_reports_dir.exists():
        perf_reports = sorted(perf_reports_dir.glob("*.json"))
        if perf_reports:
            latest_perf = load_json(perf_reports[-1])
            if not latest_perf.get("overall_passed", True):
                st.error(f"🚨 {latest_perf.get('n_alerts', 0)} performance alert(s)")
                alert_count += 1
            else:
                st.success("✅ Performance within thresholds")

    if alert_count == 0:
        st.info("No active alerts.")
