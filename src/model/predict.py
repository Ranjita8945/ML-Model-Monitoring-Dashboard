"""Prediction utilities for the ML Model Monitoring Dashboard.

Loads a trained model artifact and generates predictions on new data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

from src.data.data_loader import load_config, resolve_path


def load_model(model_path: str | Path | None = None) -> Any:
    """Load a serialized model from disk.

    Args:
        model_path: Path to the .pkl file. If None, reads from config.yaml.

    Returns:
        The deserialized model object.
    """
    if model_path is None:
        config = load_config()
        model_path = resolve_path(config["paths"]["model_artifact"])
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    model = joblib.load(model_path)
    return model


def predict(
    X: pd.DataFrame | np.ndarray,
    model_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Generate predictions and probabilities for input data.

    Args:
        X: Feature matrix (DataFrame or ndarray).
        model_path: Path to model artifact. If None, reads from config.yaml.

    Returns:
        (predictions, probabilities) where probabilities may be None if the
        model does not expose predict_proba.
    """
    model = load_model(model_path)
    y_pred = model.predict(X)
    y_proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] > 1:
            y_proba = proba[:, 1]
        else:
            y_proba = proba[:, 0]
    return y_pred, y_proba


def predict_dataframe(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """Predict on a DataFrame and return a copy with prediction columns added.

    Args:
        df: Input DataFrame containing feature columns.
        feature_columns: List of feature column names. If None, uses all
            columns except 'target' and 'prediction_id'.
        model_path: Path to model artifact.

    Returns:
        DataFrame with 'prediction' and 'prediction_proba' columns appended.
    """
    if feature_columns is None:
        exclude = {"target", "prediction_id"}
        feature_columns = [c for c in df.columns if c not in exclude]

    X = df[feature_columns]
    y_pred, y_proba = predict(X, model_path=model_path)

    result = df.copy()
    result["prediction"] = y_pred
    if y_proba is not None:
        result["prediction_proba"] = y_proba
    return result
