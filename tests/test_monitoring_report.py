"""Tests for src/monitoring/monitoring_report.py"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest

from src.monitoring.monitoring_report import run_monitoring_cycle


def _np_random_normal(n):
    return np.random.RandomState(42).randn(n)


class TestRunMonitoringCycle:
    def test_full_cycle(self, tmp_path, monkeypatch):
        # Create reference and production data
        reference = pd.DataFrame(
            {
                "feature_0": _np_random_normal(200),
                "feature_1": _np_random_normal(200),
                "target": np.random.randint(0, 2, 200),
            }
        )
        production = pd.DataFrame(
            {
                "feature_0": _np_random_normal(200) + 1.0,
                "feature_1": _np_random_normal(200),
                "target": np.random.randint(0, 2, 200),
            }
        )

        # Write temp CSVs
        ref_dir = tmp_path / "reference"
        ref_dir.mkdir()
        ref_csv = ref_dir / "reference_data.csv"
        reference.to_csv(ref_csv, index=False)

        prod_dir = tmp_path / "production"
        prod_dir.mkdir()
        prod_csv = prod_dir / "production_data.csv"
        production.to_csv(prod_csv, index=False)

        pred_dir = tmp_path / "predictions"
        pred_dir.mkdir()
        pred_csv = pred_dir / "prediction_logs.csv"
        pd.DataFrame(
            {
                "prediction_id": ["p1", "p2"],
                "timestamp": ["2024-01-01", "2024-01-02"],
                "features": ["{}", "{}"],
                "prediction": [0, 1],
                "prediction_proba": [0.3, 0.7],
                "actual_label": [0, 1],
                "model_version": ["v1", "v1"],
            }
        ).to_csv(pred_csv, index=False)

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "model_metadata.json").write_text("{}")

        config = {
            "paths": {
                "reference_data": str(ref_csv),
                "production_data": str(prod_csv),
                "predictions_log": str(pred_csv),
                "model_dir": str(model_dir),
                "model_metadata": str(model_dir / "model_metadata.json"),
                "reports_dir": str(tmp_path / "reports"),
                "drift_reports_dir": str(tmp_path / "drift_reports"),
                "performance_reports_dir": str(tmp_path / "performance_reports"),
            },
            "data": {"target_column": "target"},
            "drift": {
                "ks_p_value_threshold": 0.05,
                "psi_threshold": 0.2,
                "chi2_p_value_threshold": 0.05,
            },
            "performance": {
                "min_accuracy": 0.70,
                "min_precision": 0.65,
                "min_recall": 0.65,
                "min_f1": 0.65,
                "min_roc_auc": 0.70,
                "window_size": 50,
            },
        }

        result = run_monitoring_cycle(config=config)

        assert "timestamp" in result
        assert "data_drift" in result
        assert "concept_drift" in result
        assert "performance" in result
        assert "drift_report_path" in result
        assert "performance_report_path" in result

        # Reports should have been written to disk
        import os
        assert os.path.exists(result["drift_report_path"])
        assert os.path.exists(result["performance_report_path"])



