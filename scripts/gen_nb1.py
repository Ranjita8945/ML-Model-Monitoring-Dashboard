"""Generate notebook 01_data_exploration.ipynb"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.split("\n")}

def code(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.split("\n")}

cells = [
    md("# 01 - Data Exploration\n\nExplore reference and production datasets.\n\n- Load reference and production data\n- Inspect distributions, data types, missing values\n- Compare feature statistics\n- Identify potential data drift candidates"),
    code("import pandas as pd\nimport numpy as np\nimport plotly.express as px\nfrom pathlib import Path\n\nPROJECT_ROOT = Path.cwd()\nprint(f\"Project root: {PROJECT_ROOT}\")"),
    code("from src.data.data_loader import load_reference_data, load_production_data, load_config\n\nconfig = load_config()\nreference = load_reference_data(config)\nproduction = load_production_data(config)\n\nprint(f\"Reference shape: {reference.shape}\")\nprint(f\"Production shape: {production.shape}\")\nprint()\nprint(\"Reference columns:\", list(reference.columns))\nprint(\"Production columns:\", list(production.columns))"),
    code("display(reference.head())\ndisplay(production.head())"),
    code("print(\"=== REFERENCE ===\")\nprint(reference.dtypes)\nprint()\nprint(\"Missing values:\")\nprint(reference.isnull().sum())\nprint()\nprint(\"=== PRODUCTION ===\")\nprint(production.dtypes)\nprint()\nprint(\"Missing values:\")\nprint(production.isnull().sum())"),
    code("fig_ref = px.histogram(reference, x=\"target\", title=\"Reference Target Distribution\", labels={\"target\": \"Target Class\"}, color_discrete_sequence=[\"#2E86AB\"])\nfig_prod = px.histogram(production, x=\"target\", title=\"Production Target Distribution\", labels={\"target\": \"Target Class\"}, color_discrete_sequence=[\"#F18F01\"])\n\nfig_ref.show()\nfig_prod.show()"),
    code("if \"category\" in reference.columns:\n    fig_cat_ref = px.histogram(reference, x=\"category\", title=\"Reference Category Distribution\", color_discrete_sequence=[\"#2E86AB\"])\n    fig_cat_prod = px.histogram(production, x=\"category\", title=\"Production Category Distribution\", color_discrete_sequence=[\"#F18F01\"])\n    fig_cat_ref.show()\n    fig_cat_prod.show()\n\n    print(\"Reference category proportions:\")\n    print(reference[\"category\"].value_counts(normalize=True).sort_index())\n    print()\n    print(\"Production category proportions:\")\n    print(production[\"category\"].value_counts(normalize=True).sort_index())"),
    code("numeric_cols = reference.select_dtypes(include=[\"number\"]).columns.tolist()\nnumeric_cols = [c for c in numeric_cols if c != \"target\"]\n\ncomparison = pd.DataFrame({\n    \"feature\": numeric_cols,\n    \"ref_mean\": [reference[c].mean() for c in numeric_cols],\n    \"prod_mean\": [production[c].mean() for c in numeric_cols],\n    \"ref_std\": [reference[c].std() for c in numeric_cols],\n    \"prod_std\": [production[c].std() for c in numeric_cols],\n})\ncomparison[\"mean_diff_pct\"] = ((comparison[\"prod_mean\"] - comparison[\"ref_mean\"]) / comparison[\"ref_mean\"].abs() * 100).round(2)\n\ndisplay(comparison)"),
    code("import plotly.graph_objects as go\n\nfor col in numeric_cols[:4]:\n    fig = go.Figure()\n    fig.add_trace(go.Histogram(x=reference[col].dropna(), name=\"Reference\", opacity=0.7, marker_color=\"#2E86AB\", nbinsx=30))\n    fig.add_trace(go.Histogram(x=production[col].dropna(), name=\"Production\", opacity=0.7, marker_color=\"#F18F01\", nbinsx=30))\n    fig.update_layout(title=f\"Distribution: {col}\", barmode=\"overlay\", xaxis_title=col, yaxis_title=\"Count\", template=\"plotly_white\")\n    fig.show()"),
    md("## Key Observations\n\n- The production dataset includes a `prediction_id` column for tracking individual predictions.\n- Categorical feature `category` shows a distribution shift between reference and production.\n- Several numeric features show mean shifts, indicating potential covariate drift.\n- These observations justify running data drift detection in the monitoring pipeline."),
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
with open(NOTEBOOKS_DIR / "01_data_exploration.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print(f"[INFO] Wrote {NOTEBOOKS_DIR / '01_data_exploration.ipynb'}")
