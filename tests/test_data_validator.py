"""Tests for src/data/data_validator.py"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.data_validator import (
    ValidationReport,
    compare_reference_production,
    validate_column_ranges,
    validate_categorical,
    validate_dataset,
    validate_dtypes,
    validate_nulls,
    validate_schema,
)


class TestValidateSchema:
    def test_all_columns_present(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = validate_schema(df, ["a", "b", "c"])
        assert result.passed is True
        assert result.message == "Schema OK"

    def test_missing_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = validate_schema(df, ["a", "b", "c"])
        assert result.passed is False
        assert "c" in result.details["missing"]

    def test_extra_columns_allowed(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result = validate_schema(df, ["a", "b"])
        assert result.passed is True  # extra columns do not fail schema check


class TestValidateDtypes:
    def test_dtypes_match(self):
        df = pd.DataFrame({"a": [1, 2], "b": [1.0, 2.0]})
        result = validate_dtypes(df, {"a": "int64", "b": "float64"})
        assert result.passed is True

    def test_dtype_mismatch(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        result = validate_dtypes(df, {"a": "int64"})
        assert result.passed is False
        assert "a" in result.details["mismatches"]


class TestValidateNulls:
    def test_no_nulls_passes(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = validate_nulls(df, max_null_rate=0.05)
        assert result.passed is True

    def test_nulls_above_threshold_fail(self):
        df = pd.DataFrame({"a": [1.0, None, None, None]})
        result = validate_nulls(df, max_null_rate=0.05)
        assert result.passed is False
        assert "a" in result.details["failing_columns"]


class TestValidateCategorical:
    def test_allowed_values_pass(self):
        df = pd.DataFrame({"category": ["A", "B", "A", "C"]})
        result = validate_categorical(df, "category", ["A", "B", "C"])
        assert result.passed is True

    def test_unexpected_values_fail(self):
        df = pd.DataFrame({"category": ["A", "B", "D"]})
        result = validate_categorical(df, "category", ["A", "B", "C"])
        assert result.passed is False
        assert "D" in result.details["unexpected"]

    def test_missing_column_fail(self):
        df = pd.DataFrame({"a": [1, 2]})
        result = validate_categorical(df, "category", ["A", "B"])
        assert result.passed is False

class TestValidateColumnRanges:
    def test_within_range_passes(self):
        df = pd.DataFrame({"a": [1.0, 5.0, 10.0]})
        result = validate_column_ranges(df, "a", 0.0, 20.0)
        assert result.passed is True

    def test_outside_range_fails(self):
        df = pd.DataFrame({"a": [1.0, 25.0, 10.0]})
        result = validate_column_ranges(df, "a", 0.0, 20.0)
        assert result.passed is False


class TestValidateDataset:
    def test_all_checks_pass(self):
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": ["x", "y", "x"], "target": [0, 1, 0]}
        )
        report = validate_dataset(
            df,
            expected_columns=["a", "b", "target"],
            expected_dtypes={"a": "float64", "target": "int64"},
            max_null_rate=0.05,
            categorical_checks={"b": ["x", "y"]},
            range_checks={"a": (0.0, 10.0)},
        )
        assert report.passed is True

    def test_some_checks_fail(self):
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": ["x", "z", "x"], "target": [0, 1, 0]}
        )
        report = validate_dataset(
            df,
            expected_columns=["a", "b", "target"],
            categorical_checks={"b": ["x", "y"]},
        )
        assert report.passed is False
        assert "categorical_b" in report.summary()["failed_checks"]


class TestCompareReferenceProduction:
    def test_production_matches_reference_schema(self):
        reference = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "target": [0, 1]})
        production = pd.DataFrame({"a": [3.0, 4.0], "b": ["x", "y"], "target": [1, 0]})
        report = compare_reference_production(reference, production)
        assert report.passed is True

    def test_production_missing_column(self):
        reference = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "target": [0, 1]})
        production = pd.DataFrame({"a": [3.0, 4.0], "target": [1, 0]})
        report = compare_reference_production(reference, production)
        assert report.passed is False
        assert "schema" in report.summary()["failed_checks"]

    def test_validation_report_summary_structure(self):
        report = ValidationReport()
        from src.data.data_validator import validate_schema

        df = pd.DataFrame({"a": [1, 2]})
        report.add(validate_schema(df, ["a", "b"]))
        summary = report.summary()
        assert summary["passed"] is False
        assert "schema" in summary["failed_checks"]
        assert summary["total_checks"] == 1