"""Performance page: rolling metrics, confusion matrix, ROC curve."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from src.data.data_loader import load_config, load_predictions_log, resolve_path
from src.monitoring.performance_monitor import rolling_performance


def render_performance_dashboard() -> None:
    st.title("Performance")

    config = load_config()
    predictions = load_predictions_log(config)

    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", len(predictions))
    with col2:
        labeled = predictions.dropna(subset=["actual_label"])
        st.metric("Labeled Predictions", len(labeled))
    with col3:
        if not labeled.empty and not predictions.empty:
            st.metric("Label Coverage", f"{len(labeled) / len(predictions) * 100:.1f}%")
        else:
            st.metric("Label Coverage", "0%")

    # --- Rolling performance ---
    st.subheader("Rolling Performance Metrics")
    perf_cfg = config.get("performance", {})
    window_size = perf_cfg.get("window_size", 100)

    if not predictions.empty and "actual_label" in predictions.columns:
        labeled = predictions.dropna(subset=["actual_label", "prediction"])
        if not labeled.empty:
            perf_report = rolling_performance(labeled, window_size=window_size, config=config)

            # Metrics over time chart
            if perf_report.window_results:
                windows = [w["window"] for w in perf_report.window_results]
                accuracies = [w["metrics"]["accuracy"] for w in perf_report.window_results]
                f1s = [w["metrics"]["f1"] for w in perf_report.window_results]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=windows, y=accuracies, mode="lines+markers", name="Accuracy", line=dict(color="#2E86AB")))
                fig.add_trace(go.Scatter(x=windows, y=f1s, mode="lines+markers", name="F1", line=dict(color="#F18F01")))
                fig.update_layout(
                    title="Rolling Metrics Over Windows",
                    xaxis_title="Window",
                    yaxis_title="Score",
                    yaxis=dict(range=[0, 1]),
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

            # Alerts
            if perf_report.alerts:
                st.subheader("Performance Alerts")
                for alert in perf_report.alerts:
                    severity_color = "🔴" if alert.severity == "critical" else "🟡"
                    st.warning(
                        f"{severity_color} {alert.metric}: {alert.current_value:.4f} "
                        f"(below threshold {alert.threshold}) in {alert.window}"
                    )
            else:
                st.success("✅ No performance alerts")

            # Overall status
            if perf_report.overall_passed:
                st.success("✅ Overall performance within thresholds")
            else:
                st.error("🚨 Performance below thresholds in one or more windows")

            # Latest window confusion matrix
            st.subheader("Latest Window Confusion Matrix")
            if perf_report.window_results:
                latest_cm = perf_report.window_results[-1]["confusion_matrix"]
                cm_fig = go.Figure(
                    data=go.Heatmap(
                        z=[[latest_cm["true_negative"], latest_cm["false_positive"]],
                           [latest_cm["false_negative"], latest_cm["true_positive"]]],
                        x=["Pred: 0", "Pred: 1"],
                        y=["Actual: 0", "Actual: 1"],
                        colorscale="Blues",
                        showscale=True,
                    )
                )
                cm_fig.update_layout(title="Confusion Matrix (Latest Window)", height=400)
                st.plotly_chart(cm_fig, use_container_width=True)

                # Latest metrics
                latest_metrics = perf_report.window_results[-1]["metrics"]
                mcols = st.columns(len(latest_metrics))
                for i, (k, v) in enumerate(latest_metrics.items()):
                    with mcols[i]:
                        st.metric(k.replace("_", " ").title(), f"{v:.4f}")
        else:
            st.info("No labeled predictions available for performance monitoring.")
    else:
        st.info("No predictions logged yet. Run the pipeline to generate predictions.")

    # --- Latest performance report ---
    st.subheader("Performance Reports")
    perf_reports_dir = resolve_path(config["paths"]["performance_reports_dir"])
    if perf_reports_dir.exists():
        reports = sorted(perf_reports_dir.glob("*.json"), reverse=True)[:5]
        if reports:
            for r in reports:
                st.text(r.name)
        else:
            st.info("No performance reports generated yet.")
