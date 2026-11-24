"""Data Drift page: feature distribution comparisons, PSI/KS statistics."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from src.data.data_loader import load_config, load_production_data, load_reference_data, resolve_path
from src.monitoring.data_drift import detect_data_drift
from src.utils.helpers import load_json


def render_drift_dashboard() -> None:
    st.title("Data Drift")

    config = load_config()

    # --- Load data ---
    reference = load_reference_data(config)
    production = load_production_data(config)

    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Reference Rows", len(reference))
    with col2:
        st.metric("Production Rows", len(production))
    with col3:
        st.metric("Features", len(reference.columns) - 1)

    # --- Detect drift ---
    drift_report = detect_data_drift(reference, production, config=config)

    # --- Drift summary ---
    st.subheader("Drift Summary")
    if drift_report.overall_drifted:
        st.error(
            f"🚨 Data drift detected in {len(drift_report.drifted_features)} feature(s): "
            + ", ".join(drift_report.drifted_features)
        )
    else:
        st.success("✅ No significant data drift detected")

    # --- Per-feature drift table ---
    st.subheader("Feature-Level Drift Details")
    if drift_report.feature_results:
        rows = []
        for f in drift_report.feature_results:
            rows.append(
                {
                    "Feature": f.feature,
                    "Type": f.drift_type,
                    "Statistic": round(f.statistic, 4) if f.statistic == f.statistic else "N/A",
                    "P-Value": round(f.p_value, 4) if f.p_value == f.p_value else "N/A",
                    "PSI": round(f.psi_value, 4) if f.psi_value == f.psi_value else "N/A",
                    "Drifted": "🚨 Yes" if f.drifted else "✅ No",
                }
            )
        st.dataframe(rows, use_container_width=True)

    # --- Distribution plots for drifted numeric features ---
    st.subheader("Distribution Comparisons")
    numeric_drifted = [
        f for f in drift_report.feature_results
        if f.drift_type == "numeric" and f.drifted
    ]

    if numeric_drifted:
        feature_to_plot = st.selectbox(
            "Select feature to visualize",
            [f.feature for f in numeric_drifted],
        )
        if feature_to_plot:
            ref_vals = reference[feature_to_plot].dropna()
            prod_vals = production[feature_to_plot].dropna()

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=ref_vals,
                    name="Reference",
                    opacity=0.7,
                    nbinsx=30,
                    marker_color="#2E86AB",
                )
            )
            fig.add_trace(
                go.Histogram(
                    x=prod_vals,
                    name="Production",
                    opacity=0.7,
                    nbinsx=30,
                    marker_color="#F18F01",
                )
            )
            fig.update_layout(
                title=f"Distribution: {feature_to_plot}",
                barmode="overlay",
                xaxis_title=feature_to_plot,
                yaxis_title="Count",
                legend=dict(x=0.7, y=0.95),
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No drifted numeric features to visualize.")

    # --- Latest drift report files ---
    st.subheader("Drift Reports")
    drift_reports_dir = resolve_path(config["paths"]["drift_reports_dir"])
    if drift_reports_dir.exists():
        reports = sorted(drift_reports_dir.glob("*.json"), reverse=True)[:5]
        if reports:
            for r in reports:
                st.text(r.name)
        else:
            st.info("No drift reports generated yet.")
    else:
        st.info("Drift reports directory not found.")
