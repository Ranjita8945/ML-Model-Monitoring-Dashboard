# ML Model Monitoring Dashboard

A production-ready ML model monitoring dashboard for detecting data drift, concept drift,
model performance degradation, prediction behaviour shifts, and production data quality issues.

## Project Overview

This project provides a complete ML model monitoring solution built with Python, Streamlit,
and scikit-learn. It monitors machine learning models in production by tracking:

1. **Data Drift** — changes in input feature distributions (P(X))
2. **Concept Drift** — changes in the relationship between inputs and target (P(Y|X))
3. **Model Performance Degradation** — decline in classification metrics over time
4. **Prediction Behaviour** — shifts in model output distribution (P(Ŷ))
5. **Production Data Quality** — schema, null rates, and categorical validity checks

## Problem Statement

Machine learning models degrade over time due to changes in data distributions and
underlying relationships. Without monitoring, these issues go undetected until
business impact occurs. This dashboard provides early warning signals for:

- **Data drift**: Input features shifting away from training distribution
- **Concept drift**: The relationship between features and target changing
- **Performance decay**: Accuracy, precision, recall, or F1 dropping below thresholds
- **Prediction shifts**: Model outputting different class proportions than expected
- **Data quality issues**: Missing values, schema violations, unexpected categories

## Objectives

- Detect data drift using PSI, KS test, and chi-square statistics
- Detect concept drift via performance degradation against a reference baseline
- Monitor classification performance over rolling windows
- Track prediction behaviour and label distribution shifts
- Provide a Streamlit dashboard for visual exploration
- Generate JSON reports for audit and alerting


## Key Features

| Feature | Description |
|---------|-------------|
| **Data Drift Detection** | KS test for numeric features, chi-square + PSI for categorical features |
| **Concept Drift Detection** | Performance-based inference with rolling windows |
| **Performance Monitoring** | Rolling accuracy, precision, recall, F1, ROC-AUC with alerting |
| **Prediction Logging** | Structured prediction logs with delayed label support |
| **Data Validation** | Schema, dtype, null rate, categorical, and range validation |
| **Streamlit Dashboard** | Multi-page web UI for exploring drift and performance |
| **Report Generation** | JSON reports for data drift, concept drift, and performance |

## System Architecture

```
ML Model Monitoring Dashboard
├── app.py                      # CLI entrypoint (train, monitor, pipeline, dashboard)
├── config.yaml                 # Configuration (paths, thresholds, model params)
├── src/
│   ├── data/
│   │   ├── data_loader.py      # CSV loading, config loading, path resolution
│   │   └── data_validator.py   # Schema, dtype, null, categorical, range checks
│   ├── model/
│   │   ├── train.py            # RandomForest training, evaluation, artifact persistence
│   │   ├── predict.py          # Model loading and prediction
│   │   └── evaluate.py         # Classification metrics and confusion matrix
│   ├── monitoring/
│   │   ├── data_drift.py       # Data drift detection (PSI, KS, chi-square)
│   │   ├── concept_drift.py    # Concept drift (performance-based), prediction/label drift
│   │   ├── performance_monitor.py  # Rolling window performance monitoring
│   │   ├── prediction_monitor.py   # Prediction logging and label updates
│   │   └── monitoring_report.py    # Orchestrates full monitoring cycle
│   └── utils/
│       ├── metrics.py          # PSI, KS, chi-square, classification metrics
│       └── helpers.py          # YAML/JSON IO, path resolution, CSV helpers
├── dashboard/
│   ├── app.py                  # Streamlit multi-page router
│   ├── overview.py             # Overview page (model info, alerts)
│   ├── drift_dashboard.py      # Data drift page
│   ├── performance_dashboard.py # Performance page
│   └── prediction_dashboard.py  # Predictions page
├── tests/                      # Pytest test suite
├── data/                       # Reference, production, and prediction data
├── models/                     # Trained model artifacts and metadata
├── drift_reports/              # Generated drift reports
├── performance_reports/        # Generated performance reports
└── logs/                       # Application logs
```

## Project Folder Structure

```
ml-model-monitoring-dashboard/
├── app.py
├── config.yaml
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data/
│   ├── model/
│   ├── monitoring/
│   └── utils/
├── dashboard/
├── tests/
├── data/
├── models/
├── drift_reports/
├── performance_reports/
├── reports/
└── logs/
```

## Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.10 |
| Data Processing | Pandas, NumPy | 2.0.3, 1.24.4 |
| Statistics | SciPy | 1.11.1 |
| Machine Learning | scikit-learn | 1.3.0 |
| Model Serialization | Joblib | 1.3.1 |
| Dashboard | Streamlit | 1.28.1 |
| Visualization | Plotly, Matplotlib | 5.15.0, 3.7.2 |
| Configuration | PyYAML | 6.0.1 |
| Testing | Pytest | 7.4.3 |

## Installation Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Ranjita8945/ML-Model-Monitoring-Dashboard.git
cd ML-Model-Monitoring-Dashboard
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

## Statistical Techniques Used

### Population Stability Index (PSI)
Measures how much a distribution has shifted compared to a reference.
- PSI < 0.10: Insignificant change
- 0.10 ≤ PSI < 0.25: Minor change
- PSI ≥ 0.25: Significant change

### Kolmogorov-Smirnov (KS) Test
Two-sample KS test for comparing numeric feature distributions between
reference and production. Low p-value (< 0.05) indicates statistically
significant distribution differences.

### Chi-Square Test of Independence
Uses `scipy.stats.chi2_contingency` on a 2×k contingency table (reference
× production counts per category) to test whether categorical distributions
differ. Handles different dataset sizes and missing categories correctly.

### Classification Metrics
Accuracy, precision, recall, F1-score, and ROC-AUC computed using
scikit-learn with `zero_division=0` for edge cases (single class present).

### Concept Drift Detection
Concept drift is inferred indirectly from performance degradation:
- Production performance metrics are compared against a reference baseline
- A relative drop > threshold (default 10%) in any metric flags concept drift
- **Limitation**: Performance degradation is an *indirect* signal — data
  drift (covariate shift) can also cause performance degradation without
  any change in P(Y|X). Concept drift should be confirmed with targeted
  analysis when ground truth labels are available.

## Example Monitoring Results

After running `python app.py --mode monitor`, reports are generated as JSON:

**Drift report** (`drift_reports/drift_report_*.json`):
```json
{
  "overall_drifted": false,
  "n_drifted_features": 0,
  "drifted_features": [],
  "features": [
    {
      "feature": "feature_0",
      "drift_type": "numeric",
      "statistic": 0.023,
      "p_value": 0.845,
      "psi_value": 0.012,
      "drifted": false
    }
  ]
}
```

**Concept drift report** (embedded in monitoring cycle output):
```json
{
  "status": "passed",
  "overall_drifted": false,
  "n_predictions": 2000,
  "n_with_labels": 2000,
  "reference_metrics": {"accuracy": 0.91, "precision": 0.90, "recall": 0.92, "f1": 0.91},
  "production_metrics": {"accuracy": 0.88, "precision": 0.87, "recall": 0.89, "f1": 0.88}
}
```

**Performance report** (`performance_reports/performance_report_*.json`):
```json
{
  "window_size": 100,
  "n_windows": 20,
  "overall_passed": true,
  "n_alerts": 0,
  "alerts": [],
  "windows": [...]
}
```

## Testing Instructions

Run the full test suite with Pytest:

```bash
python -m pytest tests/ -v
```

### Test Coverage

| Test File | What It Tests |
|-----------|--------------|
| `test_data_loader.py` | Config loading, path resolution, data loading, feature/target split |
| `test_data_validator.py` | Schema, dtype, null, categorical, range validation |
| `test_metrics.py` | PSI, KS test, chi-square, classification metrics, confusion matrix |
| `test_data_drift.py` | Data drift detection (numeric and categorical) |
| `test_concept_drift.py` | Concept drift, prediction drift, label drift, ground truth evaluation |
| `test_performance_monitor.py` | Rolling window performance, alerting |
| `test_prediction_monitor.py` | Prediction logging, label updates, retrieval |
| `test_monitoring_report.py` | Full monitoring cycle orchestration |
| `test_model.py` | Model training, evaluation, serialization, prediction |

All 70 tests pass with the current implementation.

## Limitations

1. **Concept drift is inferred indirectly**: Performance degradation can be caused
   by data drift (covariate shift) without any actual change in P(Y|X). Confirm
   with targeted analysis when ground truth labels are available.

2. **Binary classification only**: The current implementation focuses on binary
   classification. Multi-class support would require macro/micro averaging.

3. **Delayed labels**: Concept drift detection requires ground truth labels,
   which may arrive with significant delay in production. Use
   `update_actual_labels()` to record delayed labels.

4. **No automated retraining**: The dashboard detects issues but does not
   automatically retrain models. Human intervention is required.

5. **Reference baseline required**: Concept drift detection needs reference
   (training-time) metrics from model metadata. Without metadata, performance
   comparison is not possible.

## Future Improvements

- Multi-class classification support
- Drift detection for numerical features using Wasserstein distance
- Automated alerting via email/Slack/webhooks
- Model retraining pipeline integration
- Drift magnitude ranking and prioritization
- Historical drift trend analysis
- Data quality rules engine (customizable thresholds)
- Docker containerization for deployment
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Environment Setup

No environment variables or API keys are required. All configuration is in `config.yaml`.

The configuration file controls:
- **Paths**: Reference data, production data, predictions log, model artifacts, reports
- **Data schema**: Target column, feature columns, categorical/numeric column lists
- **Drift thresholds**: KS p-value, PSI, chi-square p-value thresholds
- **Performance thresholds**: Minimum accuracy, precision, recall, F1, ROC-AUC
- **Model**: RandomForest hyperparameters (n_estimators, random_state, test_size)

## How to Train the Model

```bash
python app.py --mode train
```

This will:
1. Load the reference dataset from `data/reference/reference_data.csv`
2. Train a RandomForestClassifier on numeric features
3. Evaluate on a held-out test split
4. Save the model to `models/trained_model.pkl`
5. Save metadata (including metrics) to `models/model_metadata.json`

## How to Run Monitoring

```bash
python app.py --mode monitor
```

This will:
1. Load reference and production data
2. Run data drift detection (PSI, KS, chi-square)
3. Run concept drift detection (performance vs reference baseline)
4. Run rolling window performance monitoring
5. Save JSON reports to `drift_reports/` and `performance_reports/`

## How to Run the Full Pipeline

```bash
python app.py --mode pipeline
```

Runs training followed by monitoring in a single command.

## How to Launch the Dashboard

```bash
python app.py
# or
python app.py --mode dashboard
```

This launches the Streamlit dashboard at `http://localhost:8501`.

The dashboard has four pages:
- **Overview**: Model metadata, training metrics, recent predictions, active alerts
- **Data Drift**: Feature-level drift statistics, distribution visualizations
- **Performance**: Rolling metrics over time, confusion matrix, alerts
- **Predictions**: Prediction log table with filters and feature inspection