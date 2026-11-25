"""Shared statistical metrics for drift detection and model monitoring.

Includes Population Stability Index (PSI), Kolmogorov-Smirnov (KS) test,
chi-square test for categorical drift, and classification metric wrappers.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    buckets: int = 10,
) -> float:
    """Compute Population Stability Index (PSI) between two distributions."""

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    if buckets < 1:
        raise ValueError("buckets must be at least 1")

    # Use unique quantile boundaries to handle duplicate values.
    breakpoints = np.unique(
        np.percentile(expected, np.linspace(0, 100, buckets + 1))
    )

    # A constant reference distribution has no meaningful quantile bins.
    if len(breakpoints) < 2:
        return 0.0 if np.all(actual == expected[0]) else float("inf")

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_props = expected_counts / len(expected)
    actual_props = actual_counts / len(actual)

    epsilon = 1e-6
    expected_props = np.clip(expected_props, epsilon, None)
    actual_props = np.clip(actual_props, epsilon, None)

    psi_values = (
        (actual_props - expected_props)
        * np.log(actual_props / expected_props)
    )

    return float(np.sum(psi_values))


def ks_statistic(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test."""

    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return float("nan"), float("nan")

    stat, p_value = stats.ks_2samp(expected, actual)

    return float(stat), float(p_value)


def chi_square_test(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Compare two categorical distributions using a chi-square test."""

    expected = np.asarray(expected)
    actual = np.asarray(actual)

    # Remove missing categorical values.
    expected = expected[pd.notna(expected)]
    actual = actual[pd.notna(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return float("nan"), float("nan")

    categories = np.union1d(expected, actual)

    if len(categories) < 2:
        return float("nan"), float("nan")

    ref_counts = np.array(
        [(expected == category).sum() for category in categories]
    )
    actual_counts = np.array(
        [(actual == category).sum() for category in categories]
    )

    contingency = np.vstack([ref_counts, actual_counts])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            stat, p_value, _, _ = stats.chi2_contingency(contingency)
    except (ValueError, Warning):
        return float("nan"), float("nan")

    return float(stat), float(p_value)


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """Compute standard binary classification metrics."""

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths")

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)

        if len(y_proba) != len(y_true):
            raise ValueError("y_proba and y_true must have equal lengths")

        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics


def confusion_matrix_counts(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, int]:
    """Return binary confusion matrix components."""

    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths")

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }