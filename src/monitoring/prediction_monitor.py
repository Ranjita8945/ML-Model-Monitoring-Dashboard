"""Prediction logging and retrieval for the ML Model Monitoring Dashboard.

Appends structured prediction records (timestamp, features, prediction,
probability, actual_label, model_version) to prediction_logs.csv and provides
helpers to read recent windows for monitoring.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.data_loader import load_config, load_predictions_log, resolve_path


def _default_columns() -> list[str]:
    return [
        "prediction_id",
        "timestamp",
        "features",
        "prediction",
        "prediction_proba",
        "actual_label",
        "model_version",
    ]


def log_predictions(
    records: list[dict[str, Any]],
    config: dict | None = None,
) -> Path:
    """Append prediction records to the predictions log CSV.

    Args:
        records: List of dicts, each containing at least:
            - features (dict or str): input feature values
            - prediction (int): predicted class
            - prediction_proba (float, optional): probability of positive class
            - actual_label (int, optional): ground truth (may arrive later)
            - model_version (str, optional): model version identifier
        config: Configuration dict. If None, loads from config.yaml.

    Returns:
        Path to the predictions log file.
    """
    if config is None:
        config = load_config()

    log_path = resolve_path(config["paths"]["predictions_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    model_version = "unknown"
    model_metadata_path = config.get("paths", {}).get("model_metadata")
    if model_metadata_path:
        model_metadata_path = resolve_path(model_metadata_path)
        if model_metadata_path.exists():
            with open(model_metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            model_version = metadata.get("trained_at", "unknown")

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for rec in records:
        row = {
            "prediction_id": rec.get("prediction_id", f"pred_{uuid.uuid4().hex[:12]}"),
            "timestamp": rec.get("timestamp", now),
            "features": json.dumps(rec.get("features", {}), sort_keys=True),
            "prediction": rec.get("prediction"),
            "prediction_proba": rec.get("prediction_proba"),
            "actual_label": rec.get("actual_label"),
            "model_version": rec.get("model_version", model_version),
        }
        rows.append(row)

    df_new = pd.DataFrame(rows, columns=_default_columns())

    if log_path.exists() and log_path.stat().st_size > 0:
        try:
            df_existing = pd.read_csv(log_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        except pd.errors.EmptyDataError:
            df_combined = df_new
    else:
        df_combined = df_new

    df_combined.to_csv(log_path, index=False)
    print(f"[INFO] Logged {len(df_new)} prediction(s) to {log_path}")
    return log_path


def log_single_prediction(
    features: dict[str, Any],
    prediction: int,
    prediction_proba: float | None = None,
    actual_label: int | None = None,
    model_version: str | None = None,
    config: dict | None = None,
) -> Path:
    """Log a single prediction record."""
    return log_predictions(
        [
            {
                "features": features,
                "prediction": prediction,
                "prediction_proba": prediction_proba,
                "actual_label": actual_label,
                "model_version": model_version,
            }
        ],
        config=config,
    )


def get_recent_predictions(
    n: int = 100,
    config: dict | None = None,
) -> pd.DataFrame:
    """Retrieve the most recent N prediction records."""
    if config is None:
        config = load_config()
    df = load_predictions_log(config)
    if df.empty:
        return df
    df = df.sort_values("timestamp", ascending=False).head(n)
    return df


def update_actual_labels(
    updates: list[dict[str, Any]],
    config: dict | None = None,
) -> Path:
    """Update actual_label for existing prediction records by prediction_id.

    Used when ground truth arrives with delay (concept drift monitoring).
    """
    if config is None:
        config = load_config()
    log_path = resolve_path(config["paths"]["predictions_log"])
    if not log_path.exists():
        raise FileNotFoundError(f"Predictions log not found: {log_path}")

    df = pd.read_csv(log_path)
    for upd in updates:
        pid = upd["prediction_id"]
        mask = df["prediction_id"] == pid
        if mask.any():
            df.loc[mask, "actual_label"] = upd.get("actual_label")

    df.to_csv(log_path, index=False)
    print(f"[INFO] Updated actual labels for {len(updates)} record(s)")
    return log_path
