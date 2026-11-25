"""Shared statistical metrics for drift detection and model monitoring.

Includes Population Stability Index (PSI), Kolmogorov-Smirnov (KS) test,
chi-square test for categorical drift, and classification metric wrappers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    buckets: int = 10,
) -> float:
    """Compute Population Stability Index (PSI) between two distributions.

    PSI measures how much a distribution has shifted compared to a reference.
    Common interpretation:
        PSI < 0.10: insignificant change
        0.10 <= PSI < 0.25: minor change
        PSI >= 0.25: significant change

    Args:
        expected: Reference distribution values.
        actual: Production/distribution values to compare.
        buckets: Number of quantile buckets to use.

    Returns:
        PSI value (float).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Remove NaN/inf
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return float("nan")

    # Breakpoints from expected quantiles
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_percentiles = np.percentile(expected, breakpoints)
    expected_percentiles[0] = -np.inf
    expected_percentiles[-1] = np.inf

    # Histograms
    expected_counts, _ = np.histogram(expected, bins=expected_percentiles)
    actual_counts, _ = np.histogram(actual, bins=expected_percentiles)

    # Convert to proportions with small epsilon to avoid division by zero
    expected_props = expected_counts / len(expected)
    actual_props = actual_counts / len(actual)
    epsilon = 1e-6
    expected_props = np.clip(expected_props, epsilon, 1.0)
    actual_props = np.clip(actual_props, epsilon, 1.0)

    psi_values = (actual_props - expected_props) * np.log(actual_props / expected_props)
    return float(np.sum(psi_values))


def ks_statistic(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Args:
        expected: Reference sample.
        actual: Sample to compare against reference.

    Returns:
        (KS statistic, p-value).
    """
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
    """Chi-square test of independence for categorical distributions.

    Builds a 2×k contingency table (reference × production counts per category)
    and uses scipy.stats.chi2_contingency to test whether the two categorical
    distributions differ. This correctly handles different dataset sizes and
    categories missing from either dataset.

    Args:
        expected: Reference categorical values.
        actual: Production categorical values.

    Returns:
        (chi-square statistic, p-value).
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    if len(expected) == 0 or len(actual) == 0:
        return float("nan"), float("nan")

    categories = np.unique(np.concatenate([expected, actual]))

    # Build contingency table: row 0 = reference, row 1 = production
    ref_counts = np.array([np.sum(expected == c) for c in categories], dtype=float)
    prod_counts = np.array([np.sum(actual == c) for c in categories], dtype=float)
    contingency = np.vstack([ref_counts, prod_counts])

    # Chi-square test of independence on the contingency table
    # This handles different sample sizes naturally via expected frequencies
    try:
        stat, p_value, dof, expected_freq = stats.chi2_contingency(contingency)
    except ValueError:
        # e.g., if a row/column sums to zero
        return float("nan"), float("nan")

    return float(stat), float(p_value)


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for the positive class (optional).

    Returns:
        Dictionary with accuracy, precision, recall, f1, and optionally roc_auc.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")

    return metrics


def confusion_matrix_counts(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, int]:
    """Return confusion matrix components as a dictionary."""
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
