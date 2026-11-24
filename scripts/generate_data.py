"""Synthetic data generator for the ML Model Monitoring Dashboard.

Creates:
    - data/reference/reference_data.csv  : baseline (no drift) classification dataset
    - data/production/production_data.csv: production dataset with injected drift

The production dataset includes both covariate drift (feature distribution shift)
and concept drift (shift in the relationship between features and target) so that
the monitoring pipeline has something to detect.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = PROJECT_ROOT / "data" / "reference" / "reference_data.csv"
PRODUCTION_PATH = PROJECT_ROOT / "data" / "production" / "production_data.csv"


def generate_reference_data(
    n_samples: int = 2000,
    n_features: int = 8,
    n_informative: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a balanced binary classification reference dataset.

    Returns a DataFrame with feature columns (feature_0..feature_n) and a
    ``target`` column (0/1).
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[0.5, 0.5],
        class_sep=1.0,
        random_state=random_state,
    )
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    # Add a categorical feature for chi-square drift testing
    np.random.seed(random_state)
    df["category"] = np.random.choice(["A", "B", "C"], size=n_samples, p=[0.5, 0.3, 0.2])
    return df


def generate_production_data(
    reference: pd.DataFrame,
    covariate_drift: float = 0.5,
    concept_drift: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate production data with injected drift relative to reference.

    Args:
        reference: Reference DataFrame.
        covariate_drift: Magnitude of feature mean shift (in std-dev units).
        concept_drift: Probability of flipping the target label (concept drift).
        random_state: Seed for reproducibility.

    Returns:
        Production DataFrame with the same schema as reference plus a
        ``prediction_id`` column.
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(reference)
    n_features = sum(1 for c in reference.columns if c.startswith("feature_"))

    df = reference.copy()

    # --- Covariate drift: shift means of a subset of numeric features ---
    feature_cols = [c for c in reference.columns if c.startswith("feature_")]
    drift_features = feature_cols[: max(1, n_features // 2)]
    for col in drift_features:
        std = df[col].std()
        shift = covariate_drift * std
        df[col] = df[col] + rng.normal(loc=shift, scale=std * 0.1, size=n_samples)

    # --- Concept drift: flip a fraction of target labels ---
    flip_mask = rng.random(n_samples) < concept_drift
    df.loc[flip_mask, "target"] = 1 - df.loc[flip_mask, "target"].astype(int)

    # --- Categorical distribution shift ---
    if "category" in df.columns:
        new_probs = [0.2, 0.5, 0.3]  # shifted from [0.5, 0.3, 0.2]
        df["category"] = rng.choice(["A", "B", "C"], size=n_samples, p=new_probs)

    # Add prediction_id for tracking
    df.insert(0, "prediction_id", [f"prod_{i:06d}" for i in range(n_samples)])

    return df


def save_dataframes(reference: pd.DataFrame, production: pd.DataFrame) -> None:
    """Persist reference and production DataFrames to CSV."""
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRODUCTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(REFERENCE_PATH, index=False)
    production.to_csv(PRODUCTION_PATH, index=False)
    print(f"[INFO] Reference data saved to {REFERENCE_PATH} ({len(reference)} rows)")
    print(f"[INFO] Production data saved to {PRODUCTION_PATH} ({len(production)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic reference and production datasets"
    )
    parser.add_argument("--samples", type=int, default=2000, help="Rows per dataset")
    parser.add_argument("--features", type=int, default=8, help="Number of features")
    parser.add_argument(
        "--covariate-drift", type=float, default=0.5, help="Covariate drift magnitude"
    )
    parser.add_argument(
        "--concept-drift", type=float, default=0.3, help="Concept drift magnitude"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("[INFO] Generating reference data...")
    reference = generate_reference_data(
        n_samples=args.samples,
        n_features=args.features,
        random_state=args.seed,
    )

    print("[INFO] Generating production data with drift...")
    production = generate_production_data(
        reference,
        covariate_drift=args.covariate_drift,
        concept_drift=args.concept_drift,
        random_state=args.seed,
    )

    save_dataframes(reference, production)
    print("[INFO] Data generation complete.")


if __name__ == "__main__":
    main()
