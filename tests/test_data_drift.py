"""Tests for src/monitoring/data_drift.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.data_drift import detect_data_drift


class TestDetectDataDrift:
    def test_no_drift_on_similar_data(self):
        rng = np.random.RandomState(42)
        reference = pd.DataFrame(
            {
                "feature_0": rng.randn(200),
                "feature_1": rng.randn(200),
                "target": rng.randint(0, 2, 200),
            }
        )
        production = pd.DataFrame(
            {
                "feature_0": rng.randn(200),
                "feature_1": rng.randn(200),
                "target": rng.randint(0, 2, 200),
            }
        )
        config = {"data": {"target_column": "target"}, "drift": {"ks_p_value_threshold": 0.05, "psi_threshold": 0.2, "chi2_p_value_threshold": 0.05}}
        report = detect_data_drift(reference, production, config=config)
        assert report.reference_shape == (200, 3)
        assert report.production_shape == (200, 3)

    def test_detects_drift_on_shifted_data(self):
        reference = pd.DataFrame(
            {
                "feature_0": np.random.RandomState(42).randn(500),
                "feature_1": np.random.RandomState(42).randn(500),
                "target": np.random.randint(0, 2, 500),
            }
        )
        production = pd.DataFrame(
            {
                "feature_0": np.random.RandomState(42).randn(500) + 3.0,  # shifted
                "feature_1": np.random.RandomState(42).randn(500),
                "target": np.random.randint(0, 2, 500),
            }
        )
        config = {"data": {"target_column": "target"}, "drift": {"ks_p_value_threshold": 0.05, "psi_threshold": 0.2, "chi2_p_value_threshold": 0.05}}
        report = detect_data_drift(reference, production, config=config)
        # Should detect drift on feature_0
        drifted_features = report.drifted_features
        assert "feature_0" in drifted_features, f"Expected feature_0 to be drifted, got {drifted_features}"
        assert report.overall_drifted is True

    def test_report_to_dict(self):
        reference = pd.DataFrame({"feature_0": [1.0, 2.0], "target": [0, 1]})
        production = pd.DataFrame({"feature_0": [1.0, 2.0], "target": [0, 1]})
        config = {"data": {"target_column": "target"}, "drift": {"ks_p_value_threshold": 0.05, "psi_threshold": 0.2, "chi2_p_value_threshold": 0.05}}
        report = detect_data_drift(reference, production, config=config)
        d = report.to_dict()
        assert "reference_shape" in d
        assert "production_shape" in d
        assert "overall_drifted" in d
        assert "drifted_features" in d
        assert "features" in d
        assert isinstance(d["features"], list)
