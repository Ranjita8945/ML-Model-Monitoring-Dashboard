"""Streamlit multi-page dashboard router.

Pages:
    - Overview: model metadata, latest validation, active alerts
    - Data Drift: feature distribution comparisons, PSI/KS statistics
    - Performance: rolling metrics, confusion matrix, ROC curve
    - Predictions: recent prediction log table with filters
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when Streamlit runs this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from dashboard.overview import render_overview
from dashboard.drift_dashboard import render_drift_dashboard
from dashboard.performance_dashboard import render_performance_dashboard
from dashboard.prediction_dashboard import render_prediction_dashboard

PAGES = {
    "Overview": render_overview,
    "Data Drift": render_drift_dashboard,
    "Performance": render_performance_dashboard,
    "Predictions": render_prediction_dashboard,
}

st.set_page_config(
    page_title="ML Model Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("📊 ML Monitoring")
st.sidebar.markdown("---")
selection = st.sidebar.radio("Navigate", list(PAGES.keys()))
st.sidebar.markdown("---")
st.sidebar.caption("ML Model Monitoring Dashboard v0.1.0")

page = PAGES[selection]
page()
