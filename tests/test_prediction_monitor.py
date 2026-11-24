"""Tests for src/monitoring/prediction_monitor.py"""

from __future__ import annotations

from pathlib import Path

import json
import pandas as pd
import pytest

from src.monitoring.prediction_monitor import (
    get_recent_predictions,
    log_predictions,
    log_single_prediction,
    update_actual_labels,
)


class TestLogSinglePrediction:
    def test_logs_and_retrieves(self, tmp_path, monkeypatch):
        log_path = tmp_path / "predictions.csv"

        config = {"paths": {"predictions_log": str(log_path)}}

        log_single_prediction(
            features={"feature_0": 1.0, "feature_1": 2.0},
            prediction=1,
            prediction_proba=0.85,
            model_version="v1.0",
            config=config,
        )

        assert log_path.exists()
        df = pd.read_csv(log_path)
        assert len(df) == 1
        assert df.iloc[0]["prediction"] == 1
        assert df.iloc[0]["prediction_proba"] == 0.85
        assert df.iloc[0]["model_version"] == "v1.0"

        features = json.loads(df.iloc[0]["features"])
        assert features["feature_0"] == 1.0


class TestLogPredictionsBatch:
    def test_logs_multiple_records(self, tmp_path):
        log_path = tmp_path / "batch_predictions.csv"
        config = {"paths": {"predictions_log": str(log_path)}}

        records = [
            {"features": {"a": 1.0}, "prediction": 0, "prediction_proba": 0.3, "model_version": "v1"},
            {"features": {"a": 2.0}, "prediction": 1, "prediction_proba": 0.7, "model_version": "v1"},
        ]

        log_predictions(records, config=config)

        df = pd.read_csv(log_path)
        assert len(df) == 2
        assert list(df["prediction"]) == [0, 1]

    def test_appends_to_existing(self, tmp_path):
        log_path = tmp_path / "append_predictions.csv"
        config = {"paths": {"predictions_log": str(log_path)}}

        log_predictions(
            [{"features": {"a": 1.0}, "prediction": 0, "model_version": "v1"}],
            config=config,
        )
        log_predictions(
            [{"features": {"a": 2.0}, "prediction": 1, "model_version": "v1"}],
            config=config,
        )

        df = pd.read_csv(log_path)
        assert len(df) == 2


class TestGetRecentPredictions:
    def test_returns_recent_n(self, tmp_path):
        log_path = tmp_path / "recent_predictions.csv"
        config = {"paths": {"predictions_log": str(log_path)}}

        log_predictions(
            [
                {"features": {"a": float(i)}, "prediction": i % 2, "model_version": "v1"}
                for i in range(5)
            ],
            config=config,
        )

        recent = get_recent_predictions(n=3, config=config)
        assert len(recent) == 3

    def test_empty_when_no_log(self, tmp_path):
        log_path = tmp_path / "nonexistent.csv"
        config = {"paths": {"predictions_log": str(log_path)}}

        df = get_recent_predictions(n=10, config=config)
        assert df.empty


class TestUpdateActualLabels:
    def test_updates_actual_label(self, tmp_path):
        log_path = tmp_path / "label_update.csv"
        config = {"paths": {"predictions_log": str(log_path)}}

        log_predictions(
            [
                {"prediction_id": "pred_001", "features": {"a": 1.0}, "prediction": 0, "model_version": "v1"},
                {"prediction_id": "pred_002", "features": {"a": 2.0}, "prediction": 1, "model_version": "v1"},
            ],
            config=config,
        )

        update_actual_labels(
            [
                {"prediction_id": "pred_001", "actual_label": 1},
            ],
            config=config,
        )

        df = pd.read_csv(log_path)
        row = df[df["prediction_id"] == "pred_001"].iloc[0]
        assert row["actual_label"] == 1

        row2 = df[df["prediction_id"] == "pred_002"].iloc[0]
        assert pd.isna(row2["actual_label"]) or row2["actual_label"] != 1

    def test_update_missing_id_no_crash(self, tmp_path):
        log_path = tmp_path / "missing_id.csv"
        config = {"paths": {"predictions_log": str(log_path)}}

        log_predictions(
            [{"prediction_id": "pred_001", "features": {"a": 1.0}, "prediction": 0, "model_version": "v1"}],
            config=config,
        )

        # Should not raise even if ID doesn't exist
        update_actual_labels(
            [{"prediction_id": "pred_999", "actual_label": 1}],
            config=config,
        )
