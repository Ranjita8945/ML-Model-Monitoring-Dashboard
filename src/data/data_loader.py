"""Data loading utilities for the ML Model Monitoring Dashboard.

Loads reference and production CSV datasets, splits them into features/target,
and provides access to configuration-driven paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load YAML configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a path relative to the project root."""
    p = Path(relative)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def load_reference_data(config: dict | None = None) -> pd.DataFrame:
    """Load the reference dataset."""
    if config is None:
        config = load_config()
    path = resolve_path(config["paths"]["reference_data"])
    return pd.read_csv(path)


def load_production_data(config: dict | None = None) -> pd.DataFrame:
    """Load the production dataset."""
    if config is None:
        config = load_config()
    path = resolve_path(config["paths"]["production_data"])
    return pd.read_csv(path)


def load_predictions_log(config: dict | None = None) -> pd.DataFrame:
    """Load the prediction log CSV, or return an empty DataFrame if missing."""
    if config is None:
        config = load_config()
    path = resolve_path(config["paths"]["predictions_log"])
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def split_features_target(
    df: pd.DataFrame, target_column: str = "target"
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features (X) and target (y)."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
