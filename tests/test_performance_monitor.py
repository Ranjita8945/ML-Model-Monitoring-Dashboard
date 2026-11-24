"""Tests for src/monitoring/performance_monitor.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.performance_monitor import rolling_performance


class TestRollingPerformance:
    def test_empty_predictions(self):
        report = rolling_performance(pd.DataFrame(), window_size=100, config={})
        assert report.n_windows == 0
        assert report.overall_passed is True

    def test_no_labels(self):
        predictions = pd.DataFrame({"prediction": [0, 1, 0, 1]})
        report = rolling_performance(predictions, window_size=2, config={})
        assert report.n_windows == 0
        assert report.overall_passed is True

    def test_all_windows_pass(self):
        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.randint(0, 2, n)
        y_pred = y_true.copy()  # perfect predictions
        # Probabilities correlated with labels for meaningful ROC-AUC
        y_proba = np.where(y_true == 1, rng.uniform(0.7, 1.0, n), rng.uniform(0.0, 0.3, n))
        predictions = pd.DataFrame(
            {"actual_label": y_true, "prediction": y_pred, "prediction_proba": y_proba}
        )
        config = {
            "performance": {
                "min_accuracy": 0.70,
                "min_precision": 0.65,
                "min_recall": 0.65,
                "min_f1": 0.65,
                "min_roc_auc": 0.70,
            }
        }
        report = rolling_performance(predictions, window_size=50, config=config)
        assert report.n_windows == 4
        assert report.overall_passed is True
        assert len(report.alerts) == 0

    def test_accuracy_alert(self):
        n = 100
        y_true = np.ones(n)
        y_pred = np.zeros(n)  # always wrong
        predictions = pd.DataFrame(
            {"actual_label": y_true, "prediction": y_pred}
        )
        config = {
            "performance": {
                "min_accuracy": 0.70,
                "min_precision": 0.65,
                "min_recall": 0.65,
                "min_f1": 0.65,
                "min_roc_auc": 0.70,
            }
        }
        report = rolling_performance(predictions, window_size=50, config=config)
        assert report.overall_passed is False
        assert len(report.alerts) > 0
        accuracy_alerts = [a for a in report.alerts if a.metric == "accuracy"]
        assert len(accuracy_alerts) > 0

    def test_report_to_dict(self):
        n = 50
        predictions = pd.DataFrame(
            {
                "actual_label": np.random.randint(0, 2, n),
                "prediction": np.random.randint(0, 2, n),
            }
        )
        config = {
            "performance": {
                "min_accuracy": 0.70,
                "min_precision": 0.65,
                "min_recall": 0.65,
                "min_f1": 0.65,
                "min_roc_auc": 0.70,
            }
        }
        report = rolling_performance(predictions, window_size=25, config=config)
        d = report.to_dict()
        assert "window_size" in d
        assert "n_windows" in d
        assert "overall_passed" in d
        assert "alerts" in d
        assert "windows" in d
