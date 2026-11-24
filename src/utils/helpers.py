"""Shared helper utilities for the ML Model Monitoring Dashboard.

Includes path resolution, timestamp helpers, JSON/CSV IO, and lightweight
plotting helpers used across the monitoring and dashboard modules.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(data: dict[str, Any], path: str | Path, indent: int = 2) -> None:
    """Save a dictionary as a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON file and return a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def utc_now() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (including parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV file, returning an empty DataFrame if missing or empty."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def append_to_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Append a DataFrame to a CSV file, writing header only if file is new."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_header = not p.exists() or p.stat().st_size == 0
    df.to_csv(p, mode="a", header=write_header, index=False)
