"""Predictions page: recent prediction log table with filters."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.data_loader import load_config, load_predictions_log, resolve_path


def render_prediction_dashboard() -> None:
    st.title("Predictions")

    config = load_config()
    predictions = load_predictions_log(config)

    if predictions.empty:
        st.info("No predictions logged yet. Run the pipeline to generate predictions.")
        return

    # --- Filters ---
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    with col1:
        model_versions = ["All"] + sorted(predictions["model_version"].dropna().unique().tolist())
        selected_model = st.selectbox("Model Version", model_versions)
    with col2:
        pred_values = ["All"] + sorted(predictions["prediction"].dropna().unique().tolist())
        selected_pred = st.selectbox("Prediction", pred_values)
    with col3:
        n_rows = st.number_input("Rows to show", min_value=10, max_value=len(predictions), value=min(100, len(predictions)), step=10)

    # --- Apply filters ---
    filtered = predictions.copy()
    if selected_model != "All":
        filtered = filtered[filtered["model_version"] == selected_model]
    if selected_pred != "All":
        filtered = filtered[filtered["prediction"] == selected_pred]

    filtered = filtered.tail(int(n_rows))

    # --- Summary stats ---
    st.subheader("Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total (Filtered)", len(filtered))
    with col2:
        if not filtered.empty:
            pos_rate = (filtered["prediction"] == 1).mean() * 100
            st.metric("Positive Prediction Rate", f"{pos_rate:.1f}%")
        else:
            st.metric("Positive Prediction Rate", "0%")
    with col3:
        labeled = filtered.dropna(subset=["actual_label"])
        st.metric("Labeled", len(labeled))
    with col4:
        if not labeled.empty:
            accuracy = (labeled["prediction"] == labeled["actual_label"]).mean() * 100
            st.metric("Accuracy (Labeled)", f"{accuracy:.1f}%")
        else:
            st.metric("Accuracy (Labeled)", "N/A")

    # --- Prediction distribution ---
    st.subheader("Prediction Distribution")
    if not filtered.empty:
        pred_counts = filtered["prediction"].value_counts().sort_index()
        fig = px.bar(
            x=[f"Class {int(k)}" for k in pred_counts.index],
            y=pred_counts.values,
            color=[f"Class {int(k)}" for k in pred_counts.index],
            color_discrete_map={"Class 0": "#2E86AB", "Class 1": "#F18F01"},
            labels={"x": "Prediction", "y": "Count"},
        )
        fig.update_layout(showlegend=False, height=350, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # --- Prediction log table ---
    st.subheader("Prediction Log")
    display_cols = [
        c for c in ["prediction_id", "timestamp", "prediction", "prediction_proba", "actual_label", "model_version"]
        if c in filtered.columns
    ]
    st.dataframe(filtered[display_cols], use_container_width=True, height=400)

    # --- Features column (expanded) ---
    with st.expander("View Feature Details"):
        if not filtered.empty and "features" in filtered.columns:
            row_idx = st.number_input(
                "Row index to inspect",
                min_value=0,
                max_value=len(filtered) - 1,
                value=0,
            )
            features_json = filtered.iloc[row_idx]["features"]
            try:
                features_dict = json.loads(features_json)
                st.json(features_dict)
            except (json.JSONDecodeError, TypeError):
                st.text(features_json)
