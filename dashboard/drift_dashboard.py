"""Data Drift page: feature distribution comparisons, PSI/KS statistics."""

from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st

from src.data.data_loader import (
    load_config,
    load_production_data,
    load_reference_data,
    resolve_path,
)
from src.monitoring.data_drift import detect_data_drift


def _safe_round(value, digits: int = 4):
    """Round finite numeric values; return N/A otherwise."""
    try:
        number = float(value)
        return round(number, digits) if math.isfinite(number) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def render_drift_dashboard() -> None:
    st.title("Data Drift")

    config = load_config()

    # Load datasets
    try:
        reference = load_reference_data(config)
        production = load_production_data(config)
    except Exception as exc:
        st.error("Unable to load reference or production data.")
        with st.expander("Error details"):
            st.exception(exc)
        return

    if reference.empty or production.empty:
        st.warning("Reference or production dataset is empty.")
        return

    # Dataset overview
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Reference Rows", len(reference))

    with col2:
        st.metric("Production Rows", len(production))

    with col3:
        st.metric("Features", len(reference.columns))

    # Detect drift
    try:
        drift_report = detect_data_drift(
            reference,
            production,
            config=config,
        )
    except Exception as exc:
        st.error("Data drift detection failed.")
        with st.expander("Error details"):
            st.exception(exc)
        return

    # Drift summary
    st.subheader("Drift Summary")

    if drift_report.overall_drifted:
        st.error(
            f"🚨 Data drift detected in "
            f"{len(drift_report.drifted_features)} feature(s): "
            + ", ".join(drift_report.drifted_features)
        )
    else:
        st.success("✅ No significant data drift detected")

    # Feature-level drift details
    st.subheader("Feature-Level Drift Details")

    if drift_report.feature_results:
        rows = []

        for feature in drift_report.feature_results:
            rows.append(
                {
                    "Feature": feature.feature,
                    "Type": feature.drift_type,
                    "Statistic": _safe_round(feature.statistic),
                    "P-Value": _safe_round(feature.p_value),
                    "PSI": _safe_round(feature.psi_value),
                    "Drifted": (
                        "🚨 Yes" if feature.drifted else "✅ No"
                    ),
                }
            )

        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No feature-level drift results available.")

    # Distribution comparisons for drifted numeric features
    st.subheader("Distribution Comparisons")

    numeric_drifted = [
        feature
        for feature in drift_report.feature_results
        if feature.drift_type == "numeric" and feature.drifted
    ]

    if numeric_drifted:
        feature_to_plot = st.selectbox(
            "Select feature to visualize",
            [feature.feature for feature in numeric_drifted],
        )

        if (
            feature_to_plot in reference.columns
            and feature_to_plot in production.columns
        ):
            ref_vals = reference[feature_to_plot].dropna()
            prod_vals = production[feature_to_plot].dropna()

            fig = go.Figure()

            fig.add_trace(
                go.Histogram(
                    x=ref_vals,
                    name="Reference",
                    opacity=0.7,
                    nbinsx=30,
                )
            )

            fig.add_trace(
                go.Histogram(
                    x=prod_vals,
                    name="Production",
                    opacity=0.7,
                    nbinsx=30,
                )
            )

            fig.update_layout(
                title=f"Distribution: {feature_to_plot}",
                barmode="overlay",
                xaxis_title=feature_to_plot,
                yaxis_title="Count",
                template="plotly_white",
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Selected feature is missing from one of the datasets.")
    else:
        st.info("No drifted numeric features to visualize.")

    # Latest drift report files
    st.subheader("Drift Reports")

    drift_reports_dir = resolve_path(
        config["paths"]["drift_reports_dir"]
    )

    if drift_reports_dir.exists():
        reports = sorted(
            drift_reports_dir.glob("*.json"),
            reverse=True,
        )[:5]

        if reports:
            for report in reports:
                st.text(report.name)
        else:
            st.info("No drift reports generated yet.")
    else:
        st.info("Drift reports directory not found.")