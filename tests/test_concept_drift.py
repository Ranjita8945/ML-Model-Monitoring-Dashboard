"""Tests for src/monitoring/concept_drift.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.monitoring.concept_drift import detect_concept_drift, evaluate_with_ground_truth


class TestDetectConceptDrift:
    def test_empty_predictions(self):
        report = detect_concept_drift(pd.DataFrame(), config={})
        assert report.n_predictions == 0
        assert report.overall_drifted is False

    def test_no_labels_no_drift(self):
        predictions = pd.DataFrame(
            {"prediction": [0, 1, 0, 1], "prediction_proba": [0.1, 0.9, 0.2, 0.8]}
        )
        report = detect_concept_drift(predictions, config={})
        assert report.n_predictions == 4
        assert report.n_with_labels == 0
        assert report.overall_drifted is False

    def test_with_labels(self):
        predictions = pd.DataFrame(
            {
                "prediction": [0, 1, 0, 1],
                "actual_label": [0, 1, 0, 1],
                "prediction_proba": [0.1, 0.9, 0.2, 0.8],
            }
        )
        report = detect_concept_drift(predictions, config={})
        assert report.n_with_labels == 4
        assert isinstance(report.overall_drifted, bool)

    def test_report_to_dict(self):
        predictions = pd.DataFrame(
            {"prediction": [0, 1], "actual_label": [0, 1], "prediction_proba": [0.1, 0.9]}
        )
        report = detect_concept_drift(predictions, config={})
        d = report.to_dict()
        assert "n_predictions" in d
        assert "n_with_labels" in d
        assert "overall_drifted" in d
        assert "checks" in d


class TestEvaluateWithGroundTruth:
    def test_perfect_predictions(self):
        result = evaluate_with_ground_truth(
            y_true=[0, 1, 0, 1],
            y_pred=[0, 1, 0, 1],
            y_proba=[0.1, 0.9, 0.2, 0.8],
        )
        assert result["n_samples"] == 4
        assert result["metrics"]["accuracy"] == 1.0
        assert result["confusion_matrix"]["true_positive"] == 2

    def test_no_ground_truth(self):
        result = evaluate_with_ground_truth()
        assert "error" in result
        assert result["n_samples"] == 0

    def test_from_predictions_log(self):
        predictions = pd.DataFrame(
            {
                "prediction": [0, 1, 0, 1],
                "actual_label": [0, 1, 0, 1],
                "prediction_proba": [0.1, 0.9, 0.2, 0.8],
            }
        )
        result = evaluate_with_ground_truth(predictions_log=predictions)
        assert result["n_samples"] == 4
        assert result["metrics"]["accuracy"] == 1.0
