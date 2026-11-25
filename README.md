# ML Model Monitoring Dashboard

A Python-based machine learning monitoring dashboard for detecting data drift, potential concept drift, and model performance degradation.

Built using Python, Scikit-learn, Pandas, SciPy, Plotly, and Streamlit.

## Project Overview

Machine learning models can become less reliable over time as input data distributions and relationships between features and targets change.

This project demonstrates how to monitor a trained classification model using statistical drift detection, rolling performance metrics, prediction logging, and an interactive dashboard.

### What This Project Monitors

1. **Data Drift:** Changes in input feature distributions (P(X)).
2. **Concept Drift:** Potential changes in the relationship between input features and target (P(Y|X)), inferred through performance degradation.
3. **Model Performance:** Changes in accuracy, precision, recall, F1-score, and other classification metrics.
4. **Prediction Behaviour:** Prediction logs and changes in prediction distributions.
5. **Delayed Ground Truth:** Updating prediction records when actual labels become available.

## Problem Statement

Machine learning models are generally trained using historical data. However, real-world data can change over time, potentially reducing model performance.

Without monitoring, these changes may go unnoticed.

This project implements monitoring techniques that help identify:

- Input feature distribution changes.
- Potential concept drift.
- Declining model performance.
- Changes in prediction behaviour.
- Performance alerts across monitoring windows.

## Key Features

| Feature | Description |
|---|---|
| Data Drift Detection | Numeric and categorical drift detection |
| PSI | Population Stability Index |
| KS Test | Statistical comparison of numeric distributions |
| Chi-Square Test | Statistical comparison of categorical distributions |
| Concept Drift | Performance-based concept drift indication |
| Performance Monitoring | Rolling classification metrics |
| Prediction Logging | Structured prediction records |
| Delayed Label Updates | Updating actual labels using prediction IDs |
| Interactive Dashboard | Streamlit interface with four pages |
| Monitoring Reports | JSON reports for monitoring results |
| Automated Tests | Pytest-based test suite |

## System Architecture

```text
ML-Model-Monitoring-Dashboard/
│
├── app.py
├── config.yaml
├── requirements.txt
├── README.md
│
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── data_validator.py
│   │
│   ├── model/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   │
│   ├── monitoring/
│   │   ├── data_drift.py
│   │   ├── concept_drift.py
│   │   ├── performance_monitor.py
│   │   ├── prediction_monitor.py
│   │   └── monitoring_report.py
│   │
│   └── utils/
│       ├── metrics.py
│       └── helpers.py
│
├── dashboard/
│   ├── app.py
│   ├── overview.py
│   ├── drift_dashboard.py
│   ├── performance_dashboard.py
│   └── prediction_dashboard.py
│
├── tests/
├── data/
├── models/
├── drift_reports/
├── performance_reports/
└── logs/
```

## Dashboard Pages

### 1. Overview

- Model information and metadata.
- Training metrics.
- Recent predictions.
- Active monitoring alerts.

### 2. Data Drift

- Reference and production dataset comparison.
- Feature-level drift statistics.
- PSI and KS statistics.
- Distribution comparison visualizations.

### 3. Performance

- Rolling accuracy and F1-score.
- Performance alerts.
- Latest window confusion matrix.
- Classification performance metrics.

### 4. Predictions

- Recent prediction records.
- Model version filtering.
- Prediction class filtering.
- Prediction distribution visualization.
- Actual label and accuracy information.
- Feature-level inspection.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.10 |
| Data Processing | Pandas, NumPy |
| Statistical Analysis | SciPy |
| Machine Learning | Scikit-learn |
| Model Serialization | Joblib |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib |
| Configuration | PyYAML |
| Testing | Pytest |

## Statistical Techniques

### Population Stability Index (PSI)

PSI measures changes between reference and production distributions.

Common interpretation guidelines:

- PSI < 0.10: Little or no significant change.
- 0.10 ≤ PSI < 0.25: Moderate change.
- PSI ≥ 0.25: Significant change.

These thresholds are guidelines and should be selected according to the application.

### Kolmogorov-Smirnov (KS) Test

The two-sample KS test compares numeric feature distributions.

A p-value below 0.05 is commonly used as an indication of a statistically significant difference.

### Chi-Square Test

The chi-square test compares categorical distributions between reference and production datasets.

It helps identify changes in category frequencies.

### Classification Metrics

The project uses classification metrics including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC, where applicable
- Confusion matrix

### Concept Drift Detection

Concept drift occurs when the relationship between input features and the target variable changes over time.

This project uses changes in model performance across monitoring windows as an indication of potential concept drift.

**Note:** Performance degradation can have multiple causes. It does not independently prove concept drift.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Ranjita8945/ML-Model-Monitoring-Dashboard.git
cd ML-Model-Monitoring-Dashboard
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

The root `app.py` provides four execution modes.

### Launch the Dashboard

```bash
python app.py
```

Alternatively:

```bash
streamlit run dashboard/app.py
```

### Train the Model

```bash
python app.py --mode train
```

### Run a Monitoring Cycle

```bash
python app.py --mode monitor
```

### Run the Complete Pipeline

```bash
python app.py --mode pipeline
```

The pipeline executes model training followed by a monitoring cycle.

## Running Tests

Run the complete test suite:

```bash
pytest -v
```

### Test Results

Latest recorded test result:

**70 passed, 2 warnings**

The warnings relate to Scikit-learn feature-name consistency and do not indicate test failures.

## Project Limitations

- Monitoring is demonstrated using generated or simulated production data.
- The project is not connected to a live production model-serving system.
- Concept drift is inferred through performance degradation rather than directly proven.
- Statistical thresholds may require adjustment for different datasets and applications.
- Monitoring results depend on the availability of actual labels for performance evaluation.

## Future Improvements

- Integrate monitoring with a live model-serving API.
- Add automated monitoring schedules.
- Implement more advanced drift detection techniques.
- Add configurable alert notifications.
- Integrate experiment tracking and model version management.
- Deploy the dashboard to a cloud platform.

## Author

**Ranjita Dandgulkar**

B.Tech – Artificial Intelligence and Machine Learning

GitHub: https://github.com/Ranjita8945
