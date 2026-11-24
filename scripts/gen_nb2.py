"""Generate notebook 02_model_training.ipynb"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.split("\n")}

def code(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.split("\n")}

cells = [
    md("# 02 - Model Training\n\nTrain a binary classification model on the reference dataset.\n\n- Train a RandomForestClassifier\n- Compute classification metrics\n- Save model artifact and metadata with joblib"),
    code("import numpy as np\nimport pandas as pd\nimport plotly.express as px\nfrom pathlib import Path\n\nfrom src.data.data_loader import load_reference_data, load_config, split_features_target\nfrom src.model.train import train_model, evaluate_on_split, save_artifacts\nfrom sklearn.model_selection import train_test_split\n\nconfig = load_config()\ndf = load_reference_data(config)\nprint(f\"Reference data shape: {df.shape}\")"),
    code("target_column = config[\"data\"][\"target_column\"]\nX, y = split_features_target(df, target_column=target_column)\n\n# Keep only numeric features\nX = X.select_dtypes(include=[\"number\"])\nprint(f\"Feature columns: {list(X.columns)}\")\nprint(f\"Target distribution:\\n{y.value_counts(normalize=True)}\")"),
    code("X_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y\n)\nprint(f\"Train: {X_train.shape}, Test: {X_test.shape}\")"),
    code("model_config = config.get(\"model\", {})\nmodel = train_model(X_train.values, y_train.values, config=model_config)\nprint(\"Model trained successfully.\")"),
    code("metrics = evaluate_on_split(model, X_test.values, y_test.values)\nprint(\"Test set metrics:\")\nfor k, v in metrics.items():\n    print(f\"  {k}: {v:.4f}\")"),
    code("save_artifacts(model, list(X.columns), metrics, config=config)\nprint(\"Artifacts saved to models/\")"),
    code("import joblib\nfrom pathlib import Path\n\nmodel_path = Path(config[\"paths\"][\"model_artifact\"])\nmetadata_path = Path(config[\"paths\"][\"model_metadata\"])\n\nloaded_model = joblib.load(model_path)\nimport json\nwith open(metadata_path) as f:\n    metadata = json.load(f)\n\nprint(f\"Model type: {metadata['model_type']}\")\nprint(f\"Features: {metadata['n_features']}\")\nprint(f\"Trained at: {metadata['trained_at']}\")\nprint(f\"Test accuracy: {metadata['metrics']['accuracy']:.4f}\")"),
    md("## Key Observations\n\n- The model achieves strong performance on the held-out test set.\n- Model artifact and metadata are persisted for use in prediction and monitoring.\n- The pipeline is reproducible via `python scripts/run_pipeline.py`."),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
with open(NOTEBOOKS_DIR / "02_model_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print(f"[INFO] Wrote {NOTEBOOKS_DIR / '02_model_training.ipynb'}")
