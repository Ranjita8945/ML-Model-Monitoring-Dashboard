"""Tests for src/model/train.py, evaluate.py, predict.py"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.model.evaluate import compute_classification_metrics, compute_confusion_matrix, evaluate_model
from src.model.predict import load_model, predict, predict_dataframe
from src.model.train import evaluate_on_split, save_artifacts, train_model


class TestTrainModel:
    def test_train_returns_fitted_model(self):
        X = np.random.RandomState(42).randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = train_model(X, y, config={"random_state": 42, "n_estimators": 10})
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
        preds = model.predict(X)
        assert len(preds) == 100


class TestEvaluateOnSplit:
    def test_returns_metrics_dict(self):
        from sklearn.ensemble import RandomForestClassifier

        X = np.random.RandomState(42).randn(100, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
        metrics = evaluate_on_split(model, X, y)
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

class TestSaveArtifacts:
    def test_saves_model_and_metadata(self, tmp_path):
        from sklearn.ensemble import RandomForestClassifier

        X = np.random.RandomState(42).randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)

        config = {
            "paths": {
                "model_dir": str(tmp_path / "models"),
                "model_artifact": str(tmp_path / "models" / "trained_model.pkl"),
                "model_metadata": str(tmp_path / "models" / "model_metadata.json"),
            }
        }
        metrics = {"accuracy": 0.9, "f1": 0.85}
        model_path, metadata_path = save_artifacts(model, ["f0", "f1", "f2"], metrics, config=config)

        assert model_path.exists()
        assert metadata_path.exists()

        loaded = joblib.load(model_path)
        assert loaded.predict(X).shape == (50,)

        import json

        with open(metadata_path) as f:
            meta = json.load(f)
        assert meta["metrics"]["accuracy"] == 0.9
        assert meta["feature_names"] == ["f0", "f1", "f2"]


class TestComputeClassificationMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_with_proba(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])
        metrics = compute_classification_metrics(y_true, y_pred, y_proba)
        assert "roc_auc" in metrics
        assert metrics["roc_auc"] == 1.0


class TestComputeConfusionMatrix:
    def test_known_values(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm["true_negative"] == 1
        assert cm["false_positive"] == 1
        assert cm["false_negative"] == 1
        assert cm["true_positive"] == 1


class TestEvaluateModel:
    def test_returns_metrics_and_cm(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = evaluate_model(y_true, y_pred)
        assert "metrics" in result
        assert "confusion_matrix" in result


class TestLoadModel:
    def test_loads_saved_model(self, tmp_path):
        from sklearn.ensemble import RandomForestClassifier

        X = np.random.RandomState(42).randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)

        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path)

        loaded = load_model(str(model_path))
        assert loaded is not None
        assert loaded.predict(X).shape == (50,)

    def test_load_model_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model("nonexistent_model.pkl")


class TestPredict:
    def test_predict_returns_arrays(self):
        from sklearn.ensemble import RandomForestClassifier

        X = np.random.RandomState(42).randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)

        model_path = Path(__file__).parents[2] / "models" / "test_temp.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        try:
            y_pred, y_proba = predict(X, model_path=str(model_path))
            assert len(y_pred) == 50
            assert y_proba is not None
            assert len(y_proba) == 50
        finally:
            model_path.unlink(missing_ok=True)


class TestPredictDataFrame:
    def test_adds_prediction_columns(self):
        from sklearn.ensemble import RandomForestClassifier

        X = pd.DataFrame(
            {"f0": np.random.RandomState(42).randn(50), "f1": np.random.RandomState(43).randn(50)}
        )
        y = (X["f0"] > 0).astype(int)
        model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X.values, y.values)

        model_path = Path(__file__).parents[2] / "models" / "test_temp2.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        try:
            result = predict_dataframe(X, feature_columns=["f0", "f1"], model_path=str(model_path))
            assert "prediction" in result.columns
            assert "prediction_proba" in result.columns
            assert len(result) == 50
        finally:
            model_path.unlink(missing_ok=True)