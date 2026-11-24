"""Data validation for the ML Model Monitoring Dashboard.

Validates schema, dtypes, null rates, and categorical vocabularies of datasets
against a reference schema. Returns structured validation results suitable for
dashboard display and alerting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    check: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Aggregate validation report across multiple checks."""

    passed: bool = True
    results: list[ValidationResult] = field(default_factory=list)

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)
        if not result.passed:
            self.passed = False

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total_checks": len(self.results),
            "failed_checks": [r.check for r in self.results if not r.passed],
            "details": {
                r.check: {"passed": r.passed, "message": r.message, "details": r.details}
                for r in self.results
            },
        }


def validate_schema(df: pd.DataFrame, expected_columns: list[str]) -> ValidationResult:
    """Check that all expected columns are present."""
    missing = [c for c in expected_columns if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_columns]
    passed = len(missing) == 0
    message = "Schema OK" if passed else f"Missing columns: {missing}"
    return ValidationResult(
        check="schema",
        passed=passed,
        message=message,
        details={"missing": missing, "extra": extra},
    )


def validate_dtypes(
    df: pd.DataFrame, expected_dtypes: dict[str, str]
) -> ValidationResult:
    """Check that columns have expected dtypes."""
    mismatches = {}
    for col, expected in expected_dtypes.items():
        if col in df.columns:
            actual = str(df[col].dtype)
            if actual != expected:
                mismatches[col] = {"expected": expected, "actual": actual}
    passed = len(mismatches) == 0
    message = "Dtypes OK" if passed else f"Dtype mismatches: {mismatches}"
    return ValidationResult(
        check="dtypes",
        passed=passed,
        message=message,
        details={"mismatches": mismatches},
    )


def validate_nulls(
    df: pd.DataFrame, max_null_rate: float = 0.05
) -> ValidationResult:
    """Check that no column exceeds the maximum null rate."""
    null_rates = df.isnull().mean()
    failing = {col: float(rate) for col, rate in null_rates.items() if rate > max_null_rate}
    passed = len(failing) == 0
    message = "Null rates OK" if passed else f"Columns exceeding null threshold: {failing}"
    return ValidationResult(
        check="nulls",
        passed=passed,
        message=message,
        details={"failing_columns": failing, "max_null_rate": max_null_rate},
    )

def validate_categorical(
    df: pd.DataFrame, column: str, allowed_values: list[str]
) -> ValidationResult:
    """Check that a categorical column only contains allowed values."""
    if column not in df.columns:
        return ValidationResult(
            check=f"categorical_{column}",
            passed=False,
            message=f"Column '{column}' not found",
            details={"column": column},
        )
    actual_values = sorted(df[column].dropna().unique().tolist())
    unexpected = [v for v in actual_values if v not in allowed_values]
    passed = len(unexpected) == 0
    message = (
        f"Categorical '{column}' OK"
        if passed
        else f"Unexpected values in '{column}': {unexpected}"
    )
    return ValidationResult(
        check=f"categorical_{column}",
        passed=passed,
        message=message,
        details={
            "column": column,
            "allowed": allowed_values,
            "actual": actual_values,
            "unexpected": unexpected,
        },
    )


def validate_column_ranges(
    df: pd.DataFrame, column: str, min_val: float, max_val: float
) -> ValidationResult:
    """Check that a numeric column stays within [min_val, max_val]."""
    if column not in df.columns:
        return ValidationResult(
            check=f"range_{column}",
            passed=False,
            message=f"Column '{column}' not found",
            details={"column": column},
        )
    col_min = float(df[column].min())
    col_max = float(df[column].max())
    passed = (col_min >= min_val) and (col_max <= max_val)
    message = (
        f"Range '{column}' OK"
        if passed
        else f"Column '{column}' out of range [{min_val}, {max_val}]: [{col_min}, {col_max}]"
    )
    return ValidationResult(
        check=f"range_{column}",
        passed=passed,
        message=message,
        details={
            "column": column,
            "min_expected": min_val,
            "max_expected": max_val,
            "min_actual": col_min,
            "max_actual": col_max,
        },
    )


def validate_dataset(
    df: pd.DataFrame,
    expected_columns: list[str] | None = None,
    expected_dtypes: dict[str, str] | None = None,
    max_null_rate: float = 0.05,
    categorical_checks: dict[str, list[str]] | None = None,
    range_checks: dict[str, tuple[float, float]] | None = None,
) -> ValidationReport:
    """Run a battery of validation checks on a DataFrame."""
    report = ValidationReport()

    if expected_columns is not None:
        report.add(validate_schema(df, expected_columns))

    if expected_dtypes is not None:
        report.add(validate_dtypes(df, expected_dtypes))

    report.add(validate_nulls(df, max_null_rate=max_null_rate))

    if categorical_checks:
        for col, allowed in categorical_checks.items():
            report.add(validate_categorical(df, col, allowed))

    if range_checks:
        for col, (min_val, max_val) in range_checks.items():
            report.add(validate_column_ranges(df, col, min_val, max_val))

    return report


def compare_reference_production(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    expected_columns: list[str] | None = None,
    categorical_columns: dict[str, list[str]] | None = None,
    max_null_rate: float = 0.05,
) -> ValidationReport:
    """Validate production data against reference schema."""
    if expected_columns is None:
        expected_columns = list(reference.columns)

    report = ValidationReport()
    report.add(validate_schema(production, expected_columns))
    report.add(validate_nulls(production, max_null_rate=max_null_rate))

    if categorical_columns:
        for col, allowed in categorical_columns.items():
            report.add(validate_categorical(production, col, allowed))

    return report