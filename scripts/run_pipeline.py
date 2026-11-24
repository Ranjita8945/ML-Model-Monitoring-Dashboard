"""End-to-end pipeline: train -> predict -> log predictions.

Usage:
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.data_loader import load_config, load_production_data, load_reference_data, split_features_target
from src.model.predict import predict_dataframe
from src.model.train import main as train_main
from src.monitoring.prediction_monitor import log_predictions


def run_pipeline() -> None:
    """Train model, predict on production data, and log predictions."""
    config = load_config()

    # Step 1: Train
    print("[PIPELINE] Step 1/3: Training model...")
    train_main()

    # Step 2: Predict on production data
    print("[PIPELINE] Step 2/3: Generating predictions on production data...")
    df = load_production_data(config)
    target_column = config["data"]["target_column"]
    id_column = config["data"]["id_column"]

    # Exclude target, id, and non-numeric columns from features
    exclude = {target_column, id_column}
    feature_columns = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    result = predict_dataframe(df, feature_columns=feature_columns)

    # Step 3: Log predictions
    print("[PIPELINE] Step 3/3: Logging predictions...")
    records = []
    for _, row in result.iterrows():
        features = {c: row[c] for c in feature_columns}
        records.append({
            "features": features,
            "prediction": int(row["prediction"]),
            "prediction_proba": float(row["prediction_proba"]) if "prediction_proba" in row and pd.notna(row["prediction_proba"]) else None,
            "actual_label": int(row[target_column]) if target_column in row else None,
            "model_version": "v1.0",
        })

    log_path = log_predictions(records, config=config)
    print(f"[PIPELINE] Complete. Predictions logged to {log_path}")


if __name__ == "__main__":
    run_pipeline()
