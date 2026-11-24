"""Model training for the ML Model Monitoring Dashboard.

Trains a scikit-learn classifier on the reference dataset and persists the
artifact (trained_model.pkl) plus metadata (model_metadata.json) via joblib.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.data_loader import load_config, load_reference_data, resolve_path, split_features_target


def load_training_config() -> dict[str, Any]:
    """Load and return the model training section of config.yaml."""
    config = load_config()
    return config.get("model", {})


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    config: dict[str, Any] | None = None,
) -> RandomForestClassifier:
    """Train a RandomForestClassifier.

    Args:
        X: Feature matrix.
        y: Target vector.
        config: Training configuration dict. If None, loads from config.yaml.

    Returns:
        Fitted RandomForestClassifier.
    """
    if config is None:
        config = load_training_config()

    random_state = config.get("random_state", 42)
    n_estimators = config.get("n_estimators", 100)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def evaluate_on_split(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Compute classification metrics on a held-out split."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }
    return metrics


def save_artifacts(
    model: RandomForestClassifier,
    feature_names: list[str],
    metrics: dict[str, float],
    config: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Persist model and metadata to disk.

    Returns:
        (model_path, metadata_path)
    """
    if config is None:
        config = load_config()

    model_dir = resolve_path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_path(config["paths"]["model_artifact"])
    metadata_path = resolve_path(config["paths"]["model_metadata"])

    # Serialize model
    joblib.dump(model, model_path)

    # Build metadata
    metadata = {
        "model_type": "RandomForestClassifier",
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Model saved to {model_path}")
    print(f"[INFO] Metadata saved to {metadata_path}")
    return model_path, metadata_path


def main() -> None:
    """CLI entrypoint: load reference data, train, evaluate, save artifacts."""
    config = load_config()
    model_config = config.get("model", {})

    # Load reference data
    df = load_reference_data(config)
    target_column = config["data"]["target_column"]
    X, y = split_features_target(df, target_column=target_column)

    # Keep only numeric feature columns for model training
    X = X.select_dtypes(include=["number"])
    feature_names = list(X.columns)

    # Train/test split
    test_size = model_config.get("test_size", 0.2)
    random_state = model_config.get("random_state", 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Train
    model = train_model(X_train.values, y_train.values, config=model_config)

    # Evaluate
    metrics = evaluate_on_split(model, X_test.values, y_test.values)
    print("[INFO] Training metrics:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")

    # Save
    save_artifacts(model, feature_names, metrics, config=config)
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
