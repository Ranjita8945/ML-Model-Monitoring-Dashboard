"""Tests for src/utils/metrics.py"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.metrics import chi_square_test, classification_metrics, confusion_matrix_counts, ks_statistic, psi


class TestPSI:
    def test_identical_distributions_low_psi(self):
        rng = np.random.RandomState(42)
        expected = rng.randn(1000)
        actual = rng.randn(1000)
        value = psi(expected, actual)
        assert value < 0.1  # similar distributions should have low PSI

    def test_shifted_distributions_high_psi(self):
        expected = np.random.RandomState(42).randn(1000)
        actual = np.random.RandomState(42).randn(1000) + 2.0  # shifted by 2 std devs
        value = psi(expected, actual)
        assert value >= 0.1  # shifted distributions should have higher PSI

    def test_returns_float(self):
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        value = psi(expected, actual)
        assert isinstance(value, float)


class TestKSStatistic:
    def test_identical_distributions_high_pvalue(self):
        rng = np.random.RandomState(42)
        expected = rng.randn(500)
        actual = rng.randn(500)
        stat, p_value = ks_statistic(expected, actual)
        assert p_value > 0.05  # similar distributions -> high p-value

    def test_different_distributions_low_pvalue(self):
        expected = np.random.RandomState(42).randn(500)
        actual = np.random.RandomState(42).randn(500) + 3.0
        stat, p_value = ks_statistic(expected, actual)
        assert p_value < 0.05  # different distributions -> low p-value

    def test_returns_tuple(self):
        expected = np.array([1.0, 2.0, 3.0])
        actual = np.array([1.0, 2.0, 3.0])
        result = ks_statistic(expected, actual)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestChiSquareTest:
    def test_same_distribution(self):
        expected = np.array(["A", "B", "C"] * 100)
        actual = np.array(["A", "B", "C"] * 100)
        stat, p_value = chi_square_test(expected, actual)
        assert p_value > 0.05  # same distribution -> high p-value

    def test_different_distribution(self):
        expected = np.array(["A"] * 300)
        actual = np.array(["B"] * 300)
        stat, p_value = chi_square_test(expected, actual)
        assert p_value < 0.05  # very different -> low p-value


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_with_proba(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])
        metrics = classification_metrics(y_true, y_pred, y_proba)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] == 1.0

    def test_pandas_series_input(self):
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = pd.Series([0, 1, 0, 1])
        metrics = classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0


class TestConfusionMatrixCounts:
    def test_known_values(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = confusion_matrix_counts(y_true, y_pred)
        assert cm["true_negative"] == 1
        assert cm["false_positive"] == 1
        assert cm["false_negative"] == 1
        assert cm["true_positive"] == 1
