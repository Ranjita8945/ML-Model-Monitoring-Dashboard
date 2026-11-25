"""Predictions page: recent prediction log table with filters."""

from __future__ import annotations

import json

import plotly.express as px
import streamlit as st

from src.data.data_loader import load_config, load_predictions_log


def render_prediction_dashboard() -> None:
    st.title("Predictions")

    config = load_config()
    predictions = load_predictions_log(config)

    if predictions.empty:
        st.info(
            "No predictions logged yet. "
            "Run the pipeline to generate predictions."
        )
        return

    # Filters
    st.subheader("Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "model_version" in predictions.columns:
            versions = predictions["model_version"].dropna().unique().tolist()
            versions = sorted(versions, key=str)
            model_versions = ["All"] + versions
            selected_model = st.selectbox(
                "Model Version",
                model_versions,
            )
        else:
            selected_model = "All"

    with col2:
        if "prediction" in predictions.columns:
            values = predictions["prediction"].dropna().unique().tolist()
            values = sorted(values, key=str)
            prediction_values = ["All"] + values
            selected_prediction = st.selectbox(
                "Prediction",
                prediction_values,
            )
        else:
            selected_prediction = "All"

    with col3:
        n_rows = st.number_input(
            "Rows to show",
            min_value=1,
            max_value=len(predictions),
            value=min(100, len(predictions)),
            step=1,
        )

    # Apply filters
    filtered = predictions.copy()

    if selected_model != "All" and "model_version" in filtered.columns:
        filtered = filtered[
            filtered["model_version"] == selected_model
        ]

    if selected_prediction != "All" and "prediction" in filtered.columns:
        filtered = filtered[
            filtered["prediction"] == selected_prediction
        ]

    filtered = filtered.tail(int(n_rows))

    if filtered.empty:
        st.warning("No predictions match the selected filters.")
        return

    # Summary statistics
    st.subheader("Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total (Filtered)", len(filtered))

    with col2:
        if "prediction" in filtered.columns:
            positive_rate = (
                filtered["prediction"] == 1
            ).mean() * 100

            st.metric(
                "Positive Prediction Rate",
                f"{positive_rate:.1f}%",
            )
        else:
            st.metric("Positive Prediction Rate", "N/A")

    with col3:
        if "actual_label" in filtered.columns:
            labeled = filtered.dropna(subset=["actual_label"])
        else:
            labeled = filtered.iloc[0:0]

        st.metric("Labeled", len(labeled))

    with col4:
        if (
            not labeled.empty
            and "prediction" in labeled.columns
        ):
            accuracy = (
                labeled["prediction"] == labeled["actual_label"]
            ).mean() * 100

            st.metric(
                "Accuracy (Labeled)",
                f"{accuracy:.1f}%",
            )
        else:
            st.metric("Accuracy (Labeled)", "N/A")

    # Prediction distribution
    st.subheader("Prediction Distribution")

    if "prediction" in filtered.columns:
        prediction_counts = filtered["prediction"].value_counts().sort_index(
            key=lambda index: index.map(str)
        )

        class_labels = [
            f"Class {value}"
            for value in prediction_counts.index
        ]

        fig = px.bar(
            x=class_labels,
            y=prediction_counts.values,
            color=class_labels,
            labels={
                "x": "Prediction",
                "y": "Count",
                "color": "Prediction",
            },
        )

        fig.update_layout(
            showlegend=False,
            height=350,
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Prediction column is unavailable.")

    # Prediction log table
    st.subheader("Prediction Log")

    display_columns = [
        column
        for column in [
            "prediction_id",
            "timestamp",
            "prediction",
            "prediction_proba",
            "actual_label",
            "model_version",
        ]
        if column in filtered.columns
    ]

    st.dataframe(
        filtered[display_columns],
        use_container_width=True,
        height=400,
    )

    # Feature details
    with st.expander("View Feature Details"):
        if "features" not in filtered.columns:
            st.info("Feature details are unavailable.")
            return

        row_index = st.number_input(
            "Row index to inspect",
            min_value=0,
            max_value=len(filtered) - 1,
            value=0,
            step=1,
        )

        features = filtered.iloc[int(row_index)]["features"]

        try:
            if isinstance(features, str):
                features = json.loads(features)

            st.json(features)

        except (json.JSONDecodeError, TypeError, ValueError):
            st.text(str(features))