"""Performance page: rolling metrics and confusion matrix."""

from __future__ import annotations

import math

import plotly.graph_objects as go
import streamlit as st

from src.data.data_loader import (
    load_config,
    load_predictions_log,
    resolve_path,
)
from src.monitoring.performance_monitor import rolling_performance


def _safe_metric(value) -> str:
    """Format a metric safely, including missing or non-finite values."""
    try:
        number = float(value)
        return f"{number:.4f}" if math.isfinite(number) else "N/A"
    except (TypeError, ValueError):
        return "N/A"


def render_performance_dashboard() -> None:
    st.title("Performance")

    config = load_config()
    predictions = load_predictions_log(config)

    # Dataset overview
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Predictions", len(predictions))

    if "actual_label" in predictions.columns:
        labeled = predictions.dropna(subset=["actual_label"])
    else:
        labeled = predictions.iloc[0:0]

    with col2:
        st.metric("Labeled Predictions", len(labeled))

    with col3:
        coverage = (
            len(labeled) / len(predictions) * 100
            if not predictions.empty
            else 0
        )
        st.metric("Label Coverage", f"{coverage:.1f}%")

    # Rolling performance
    st.subheader("Rolling Performance Metrics")

    perf_cfg = config.get("performance", {})
    window_size = perf_cfg.get("window_size", 100)

    required_columns = {"actual_label", "prediction"}

    if predictions.empty:
        st.info("No predictions logged yet. Run the pipeline to generate predictions.")

    elif not required_columns.issubset(predictions.columns):
        st.warning(
            "Performance monitoring requires both "
            "'actual_label' and 'prediction' columns."
        )

    else:
        labeled = predictions.dropna(
            subset=["actual_label", "prediction"]
        )

        if labeled.empty:
            st.info(
                "No labeled predictions available for performance monitoring."
            )

        else:
            try:
                if int(window_size) <= 0:
                    raise ValueError("Window size must be greater than zero.")

                perf_report = rolling_performance(
                    labeled,
                    window_size=int(window_size),
                    config=config,
                )

            except Exception as exc:
                st.error("Unable to calculate performance metrics.")
                with st.expander("Error details"):
                    st.exception(exc)
                perf_report = None

            if perf_report is not None:

                # Rolling metrics chart
                st.subheader("Metrics Over Time")

                if perf_report.window_results:
                    windows = [
                        result["window"]
                        for result in perf_report.window_results
                    ]

                    accuracies = [
                        result["metrics"].get("accuracy")
                        for result in perf_report.window_results
                    ]

                    f1_scores = [
                        result["metrics"].get("f1")
                        for result in perf_report.window_results
                    ]

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=windows,
                            y=accuracies,
                            mode="lines+markers",
                            name="Accuracy",
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=windows,
                            y=f1_scores,
                            mode="lines+markers",
                            name="F1 Score",
                        )
                    )

                    fig.update_layout(
                        title="Rolling Metrics Over Windows",
                        xaxis_title="Window",
                        yaxis_title="Score",
                        yaxis=dict(range=[0, 1]),
                        template="plotly_white",
                        height=400,
                    )

                    st.plotly_chart(fig, width="stretch")

                    # Performance alerts
                    st.subheader("Performance Alerts")

                    if perf_report.alerts:
                        for alert in perf_report.alerts:
                            severity_icon = (
                                "🔴"
                                if alert.severity == "critical"
                                else "🟡"
                            )

                            st.warning(
                                f"{severity_icon} {alert.metric}: "
                                f"{_safe_metric(alert.current_value)} "
                                f"(threshold: {_safe_metric(alert.threshold)}) "
                                f"in {alert.window}"
                            )
                    else:
                        st.success("✅ No performance alerts")

                    # Overall status
                    if perf_report.overall_passed:
                        st.success(
                            "✅ Overall performance within thresholds"
                        )
                    else:
                        st.error(
                            "🚨 Performance below thresholds "
                            "in one or more windows"
                        )

                    # Latest window confusion matrix
                    st.subheader("Latest Window Confusion Matrix")

                    latest_result = perf_report.window_results[-1]
                    latest_cm = latest_result.get("confusion_matrix", {})

                    cm_keys = {
                        "true_negative",
                        "false_positive",
                        "false_negative",
                        "true_positive",
                    }

                    if cm_keys.issubset(latest_cm):
                        cm_fig = go.Figure(
                            data=go.Heatmap(
                                z=[
                                    [
                                        latest_cm["true_negative"],
                                        latest_cm["false_positive"],
                                    ],
                                    [
                                        latest_cm["false_negative"],
                                        latest_cm["true_positive"],
                                    ],
                                ],
                                x=["Pred: 0", "Pred: 1"],
                                y=["Actual: 0", "Actual: 1"],
                                colorscale="Blues",
                                showscale=True,
                            )
                        )

                        cm_fig.update_layout(
                            title="Confusion Matrix (Latest Window)",
                            height=400,
                        )

                        st.plotly_chart(cm_fig, width="stretch")
                    else:
                        st.info(
                            "Confusion matrix is unavailable "
                            "for the latest window."
                        )

                    # Latest window metrics
                    st.subheader("Latest Window Metrics")

                    latest_metrics = latest_result.get("metrics", {})

                    if latest_metrics:
                        metric_columns = st.columns(
                            min(len(latest_metrics), 4)
                        )

                        for index, (key, value) in enumerate(
                            latest_metrics.items()
                        ):
                            with metric_columns[
                                index % len(metric_columns)
                            ]:
                                st.metric(
                                    key.replace("_", " ").title(),
                                    _safe_metric(value),
                                )
                    else:
                        st.info("No latest-window metrics available.")

                else:
                    st.info(
                        "Not enough labeled predictions to generate "
                        "rolling performance windows."
                    )

    # Latest performance report files
    st.subheader("Performance Reports")

    perf_reports_dir = resolve_path(
        config["paths"]["performance_reports_dir"]
    )

    if perf_reports_dir.exists():
        reports = sorted(
            perf_reports_dir.glob("*.json"),
            reverse=True,
        )[:5]

        if reports:
            for report in reports:
                st.text(report.name)
        else:
            st.info("No performance reports generated yet.")
    else:
        st.info("Performance reports directory not found.")