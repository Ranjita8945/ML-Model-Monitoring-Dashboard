"""Generate notebook 03_monitoring_experiments.ipynb"""
import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parents[1] / "notebooks"

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.split("\n")}

def code(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.split("\n")}

cells = [
    md("# 03 - Monitoring Experiments\n\nRun data drift, concept drift, and performance monitoring on production data.\n\n- Run full monitoring cycle\n- Inspect drift reports\n- Visualize drift statistics and performance alerts"),
    code("import json\nimport pandas as pd\nimport plotly.express as px\nfrom pathlib import Path\n\nfrom src.monitoring.monitoring_report import run_monitoring_cycle\nfrom src.data.data_loader import load_config, load_reference_data, load_production_data, load_predictions_log\nfrom src.monitoring.data_drift import detect_data_drift\nfrom src.monitoring.performance_monitor import rolling_performance\nfrom src.monitoring.concept_drift import detect_concept_drift\n\nconfig = load_config()\nprint(\"Configuration loaded.\")"),
    code("results = run_monitoring_cycle(config=config)\nprint(f\"Monitoring cycle completed at: {results['timestamp']}\")\nprint()\nprint(\"=== DATA DRIFT ===\")\nprint(f\"Overall drifted: {results['data_drift']['overall_drifted']}\")\nprint(f\"Drifted features: {results['data_drift']['drifted_features']}\")\nprint()\nprint(\"=== CONCEPT DRIFT ===\")\nprint(f\"N predictions: {results['concept_drift']['n_predictions']}\")\nprint(f\"Overall drifted: {results['concept_drift']['overall_drifted']}\")\nprint()\nprint(\"=== PERFORMANCE ===\")\nprint(f\"Overall passed: {results['performance']['overall_passed']}\")\nprint(f\"N alerts: {results['performance']['n_alerts']}\")"),
    code("reference = load_reference_data(config)\nproduction = load_production_data(config)\ndrift_report = detect_data_drift(reference, production, config=config)\n\ndrift_rows = []\nfor f in drift_report.feature_results:\n    drift_rows.append({\n        \"feature\": f.feature,\n        \"type\": f.drift_type,\n        \"statistic\": round(f.statistic, 4),\n        \"p_value\": round(f.p_value, 4),\n        \"psi\": round(f.psi_value, 4),\n        \"drifted\": f.drifted,\n    })\ndrift_df = pd.DataFrame(drift_rows)\ndisplay(drift_df)"),
    code("drifted = drift_df[drift_df[\"drifted\"] == True]\nif not drifted.empty:\n    fig = px.bar(\n        drift_df,\n        x=\"feature\",\n        y=\"psi\",\n        color=\"drifted\",\n        color_discrete_map={True: \"#E4572E\", False: \"#2E86AB\"},\n        title=\"Population Stability Index (PSI) by Feature\",\n        labels={\"psi\": \"PSI Value\", \"feature\": \"Feature\"},\n    )\n    fig.add_hline(y=0.2, line_dash=\"dash\", line_color=\"red\", annotation_text=\"PSI Threshold (0.2)\")\n    fig.show()\nelse:\n    print(\"No features exceeded the PSI drift threshold.\")"),
    code("predictions = load_predictions_log(config)\nprint(f\"Prediction log shape: {predictions.shape}\")\nprint(predictions.head())\n\nperf_cfg = config.get(\"performance\", {})\nwindow_size = perf_cfg.get(\"window_size\", 100)\nlabeled = predictions.dropna(subset=[\"actual_label\", \"prediction\"])\nif not labeled.empty:\n    perf_report = rolling_performance(labeled, window_size=window_size, config=config)\n    print(f\"Windows evaluated: {perf_report.n_windows}\")\n    print(f\"Overall passed: {perf_report.overall_passed}\")\n    print(f\"Alerts: {len(perf_report.alerts)}\")\n    for alert in perf_report.alerts[:5]:\n        print(f\"  [{alert.severity}] {alert.metric}: {alert.current_value:.4f} < {alert.threshold}\")\nelse:\n    print(\"No labeled predictions available for performance monitoring.\")"),
    md("## Key Observations\n\n- Data drift is detected on features with shifted distributions.\n- Performance metrics are computed over rolling windows and compared against thresholds.\n- Reports are written as JSON files to `drift_reports/` and `performance_reports/`.\n- The full monitoring cycle can be re-run via `python app.py --mode monitor`."),
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
with open(NOTEBOOKS_DIR / "03_monitoring_experiments.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)
print(f"[INFO] Wrote {NOTEBOOKS_DIR / '03_monitoring_experiments.ipynb'}")
