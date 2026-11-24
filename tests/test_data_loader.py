"""Tests for src/data/data_loader.py"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.data_loader import (
    load_config,
    load_production_data,
    load_reference_data,
    resolve_path,
    split_features_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestLoadConfig:
    def test_load_config_returns_dict(self):
        config = load_config()
        assert isinstance(config, dict)
        assert "paths" in config
        assert "data" in config
        assert "drift" in config


class TestResolvePath:
    def test_resolve_relative_path(self):
        p = resolve_path("data/reference/reference_data.csv")
        assert p == PROJECT_ROOT / "data" / "reference" / "reference_data.csv"

    def test_resolve_absolute_path_unchanged(self):
        abs_path = PROJECT_ROOT / "config.yaml"
        p = resolve_path(str(abs_path))
        assert p == abs_path


class TestSplitFeaturesTarget:
    def test_split_returns_X_and_y(self):
        df = pd.DataFrame(
            {"feature_0": [1.0, 2.0, 3.0], "feature_1": [4.0, 5.0, 6.0], "target": [0, 1, 0]}
        )
        X, y = split_features_target(df, target_column="target")
        assert list(X.columns) == ["feature_0", "feature_1"]
        assert list(y) == [0, 1, 0]
        assert "target" not in X.columns

    def test_split_missing_target_raises(self):
        df = pd.DataFrame({"feature_0": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Target column"):
            split_features_target(df, target_column="target")


class TestLoadReferenceData:
    def test_loads_reference_csv(self, tmp_path):
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()
        ref_csv = ref_dir / "reference_data.csv"
        pd.DataFrame({"feature_0": [1.0, 2.0], "target": [0, 1]}).to_csv(ref_csv, index=False)

        config = {"paths": {"reference_data": str(ref_csv)}}
        df = load_reference_data(config)
        assert len(df) == 2
        assert list(df.columns) == ["feature_0", "target"]


class TestLoadProductionData:
    def test_loads_production_csv(self, tmp_path):
        prod_dir = tmp_path / "production"
        prod_dir.mkdir()
        prod_csv = prod_dir / "production_data.csv"
        pd.DataFrame({"feature_0": [1.0, 2.0], "target": [0, 1]}).to_csv(prod_csv, index=False)

        config = {"paths": {"production_data": str(prod_csv)}}
        df = load_production_data(config)
        assert len(df) == 2
        assert list(df.columns) == ["feature_0", "target"]
