"""
data_validation.py — Data validation rules for the German Credit (credit-g) dataset.

Provides a DataValidator class and a validate_dataframe() helper that enforce:
  - Schema: all expected columns present with correct dtype families
  - Completeness: no missing values (configurable threshold)
  - Range: numeric features within documented domain limits
  - Cardinality: categorical features only contain known categories
  - Consistency: cross-field logical constraints
  - Duplicates: no duplicate rows

Validation results are returned as a ValidationReport (dataclass) containing
a list of ValidationError objects.  The caller decides whether to raise or log.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# ─── SCHEMA DEFINITION ────────────────────────────────────────────────────────

#: Expected numeric columns and their valid [min, max] inclusive ranges.
#: Derived from the UCI / OpenML credit-g documentation.
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "duration":                 (1.0,   72.0),   # months
    "credit_amount":            (250.0, 18_424.0),
    "installment_commitment":   (1.0,   4.0),    # % of income bucket
    "residence_since":          (1.0,   4.0),    # years bucket
    "age":                      (18.0,  75.0),   # years
    "existing_credits":         (1.0,   4.0),    # count
    "num_dependents":           (1.0,   2.0),    # count
}

#: Expected categorical columns and their allowed category sets.
#: Values are the raw strings used in the OpenML version 1 frame.
CATEGORICAL_VALUES: dict[str, frozenset[str]] = {
    "checking_status": frozenset({
        "no checking", "<0", "0<=X<200", ">=200",
    }),
    "credit_history": frozenset({
        "no credits/all paid", "all paid", "existing paid",
        "delayed previously", "critical/other existing credit",
    }),
    "purpose": frozenset({
        "new car", "used car", "furniture/equipment", "radio/tv",
        "domestic appliance", "repairs", "education", "vacation",
        "retraining", "business", "other",
    }),
    "savings_status": frozenset({
        "no known savings", "<100", "100<=X<500", "500<=X<1000", ">=1000",
    }),
    "employment": frozenset({
        "unemployed", "<1", "1<=X<4", "4<=X<7", ">=7",
    }),
    "personal_status": frozenset({
        "male div/sep", "female div/dep/mar", "male single",
        "male mar/wid", "female single",
    }),
    "other_parties": frozenset({
        "none", "co applicant", "guarantor",
    }),
    "property_magnitude": frozenset({
        "real estate", "life insurance", "car", "no known property",
    }),
    "other_payment_plans": frozenset({
        "bank", "stores", "none",
    }),
    "housing": frozenset({
        "rent", "own", "for free",
    }),
    "job": frozenset({
        "unemp/unskilled non res", "unskilled resident",
        "skilled", "high qualif/self emp/mgmt",
    }),
    "own_telephone": frozenset({"none", "yes"}),
    "foreign_worker": frozenset({"yes", "no"}),
}

#: All expected feature columns (excluding the target 'class').
EXPECTED_COLUMNS: frozenset[str] = (
    frozenset(NUMERIC_RANGES.keys()) | frozenset(CATEGORICAL_VALUES.keys())
)

#: Dtype families (broad) mapped to column name patterns / explicit lists.
DTYPE_FAMILIES: dict[str, list[str]] = {
    "numeric":      list(NUMERIC_RANGES.keys()),
    "categorical":  list(CATEGORICAL_VALUES.keys()),
}


# ─── RESULT TYPES ─────────────────────────────────────────────────────────────

@dataclass
class ValidationError:
    """A single data quality violation."""
    rule:     str
    column:   str
    message:  str
    severity: str = "error"   # "error" | "warning"
    details:  Any = None      # extra context (bad values, counts, …)

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.rule} / {self.column}: {self.message}"


@dataclass
class ValidationReport:
    """Aggregated results from DataValidator.validate()."""
    errors:   list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)

    # ── convenience ─────────────────────────────────────────────────────────
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    @property
    def total_issues(self) -> int:
        return len(self.errors) + len(self.warnings)

    def add(self, item: ValidationError) -> None:
        if item.severity == "warning":
            self.warnings.append(item)
        else:
            self.errors.append(item)

    def raise_if_invalid(self) -> None:
        """Raise ValueError summarising all errors (useful in pipeline scripts)."""
        if not self.is_valid:
            msgs = "\n  ".join(str(e) for e in self.errors)
            raise ValueError(
                f"Data validation failed with {len(self.errors)} error(s):\n  {msgs}"
            )

    def print_summary(self) -> None:
        status = "PASSED" if self.is_valid else "FAILED"
        print(f"\n── Data Validation Report [{status}] ──")
        if not self.errors and not self.warnings:
            print("  All checks passed.")
            return
        for item in self.errors + self.warnings:
            print(f"  {item}")
        print()


# ─── VALIDATOR ────────────────────────────────────────────────────────────────

class DataValidator:
    """
    Applies a suite of validation rules to a pandas DataFrame that represents
    the German Credit (credit-g) feature matrix (no target column).

    Parameters
    ----------
    missing_threshold : float
        Maximum allowed fraction of missing values per column (default 0.0
        means *no* missing values are permitted).
    range_tolerance : float
        Fractional tolerance applied to numeric range bounds.  For example
        0.05 widens each [min, max] by 5 % on each side (useful in production
        where the scoring distribution may shift slightly).
    extra_columns : str
        How to treat unexpected extra columns: ``"error"`` (default) or
        ``"warning"`` or ``"ignore"``.
    """

    def __init__(
        self,
        *,
        missing_threshold: float = 0.0,
        range_tolerance: float = 0.0,
        extra_columns: str = "warning",
    ) -> None:
        if not 0.0 <= missing_threshold <= 1.0:
            raise ValueError("missing_threshold must be in [0, 1]")
        if not 0.0 <= range_tolerance < 1.0:
            raise ValueError("range_tolerance must be in [0, 1)")
        if extra_columns not in ("error", "warning", "ignore"):
            raise ValueError("extra_columns must be 'error', 'warning', or 'ignore'")

        self.missing_threshold = missing_threshold
        self.range_tolerance = range_tolerance
        self.extra_columns = extra_columns

    # ── public entry point ───────────────────────────────────────────────────

    def validate(self, df: pd.DataFrame) -> ValidationReport:
        """
        Run all validation rules and return a :class:`ValidationReport`.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame — may or may not include the ``'class'`` column.

        Returns
        -------
        ValidationReport
        """
        report = ValidationReport()

        # Drop target column if present so it does not interfere.
        if "class" in df.columns:
            df = df.drop(columns=["class"])

        self._check_schema(df, report)
        self._check_completeness(df, report)
        self._check_numeric_ranges(df, report)
        self._check_categorical_values(df, report)
        self._check_consistency(df, report)
        self._check_duplicates(df, report)

        return report

    # ── rule implementations ─────────────────────────────────────────────────

    def _check_schema(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Rule: all expected columns present; no unexpected extras."""
        present = set(df.columns)

        # Missing required columns → hard error.
        missing_cols = EXPECTED_COLUMNS - present
        for col in sorted(missing_cols):
            report.add(ValidationError(
                rule="schema.missing_column",
                column=col,
                message=f"Required column '{col}' is absent from the DataFrame.",
                severity="error",
            ))

        # Extra / unknown columns.
        extra_cols = present - EXPECTED_COLUMNS
        if extra_cols and self.extra_columns != "ignore":
            for col in sorted(extra_cols):
                report.add(ValidationError(
                    rule="schema.extra_column",
                    column=col,
                    message=f"Unexpected column '{col}' found in the DataFrame.",
                    severity=self.extra_columns,
                ))

        # Dtype family checks for present columns.
        for col in DTYPE_FAMILIES["numeric"]:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                report.add(ValidationError(
                    rule="schema.wrong_dtype",
                    column=col,
                    message=(
                        f"Column '{col}' is expected to be numeric but has "
                        f"dtype '{df[col].dtype}'."
                    ),
                    severity="error",
                ))

        for col in DTYPE_FAMILIES["categorical"]:
            if col in df.columns:
                if not (
                    pd.api.types.is_object_dtype(df[col])
                    or pd.api.types.is_categorical_dtype(df[col])
                    or pd.api.types.is_string_dtype(df[col])
                ):
                    report.add(ValidationError(
                        rule="schema.wrong_dtype",
                        column=col,
                        message=(
                            f"Column '{col}' is expected to be categorical/string "
                            f"but has dtype '{df[col].dtype}'."
                        ),
                        severity="warning",
                    ))

    def _check_completeness(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Rule: missing values per column must be ≤ missing_threshold."""
        n = len(df)
        if n == 0:
            report.add(ValidationError(
                rule="completeness.empty_dataframe",
                column="*",
                message="DataFrame is empty (zero rows).",
                severity="error",
            ))
            return

        for col in df.columns:
            if col not in EXPECTED_COLUMNS:
                continue
            null_count = int(df[col].isna().sum())
            null_frac = null_count / n
            if null_frac > self.missing_threshold:
                sev = "error" if self.missing_threshold == 0.0 else "warning"
                report.add(ValidationError(
                    rule="completeness.missing_values",
                    column=col,
                    message=(
                        f"Column '{col}' has {null_count} missing values "
                        f"({null_frac:.1%}), exceeding threshold "
                        f"({self.missing_threshold:.1%})."
                    ),
                    severity=sev,
                    details={"null_count": null_count, "null_fraction": null_frac},
                ))

    def _check_numeric_ranges(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Rule: numeric values must lie within [min, max] (± tolerance)."""
        for col, (lo, hi) in NUMERIC_RANGES.items():
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if series.empty:
                continue

            tol_lo = lo - abs(lo) * self.range_tolerance
            tol_hi = hi + abs(hi) * self.range_tolerance

            out_low = series[series < tol_lo]
            out_high = series[series > tol_hi]

            if not out_low.empty:
                report.add(ValidationError(
                    rule="range.below_minimum",
                    column=col,
                    message=(
                        f"Column '{col}' has {len(out_low)} value(s) below "
                        f"minimum {lo} (min observed: {series.min():.4g})."
                    ),
                    severity="error",
                    details={
                        "count": len(out_low),
                        "min_observed": float(series.min()),
                        "expected_min": lo,
                    },
                ))

            if not out_high.empty:
                report.add(ValidationError(
                    rule="range.above_maximum",
                    column=col,
                    message=(
                        f"Column '{col}' has {len(out_high)} value(s) above "
                        f"maximum {hi} (max observed: {series.max():.4g})."
                    ),
                    severity="error",
                    details={
                        "count": len(out_high),
                        "max_observed": float(series.max()),
                        "expected_max": hi,
                    },
                ))

    def _check_categorical_values(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Rule: categorical columns must only contain known category values."""
        for col, allowed in CATEGORICAL_VALUES.items():
            if col not in df.columns:
                continue
            actual = set(df[col].dropna().astype(str).unique())
            unknown = actual - allowed
            if unknown:
                report.add(ValidationError(
                    rule="cardinality.unknown_category",
                    column=col,
                    message=(
                        f"Column '{col}' contains {len(unknown)} unknown "
                        f"category value(s): {sorted(unknown)!r}."
                    ),
                    severity="error",
                    details={
                        "unknown_values": sorted(unknown),
                        "allowed_values": sorted(allowed),
                    },
                ))

    def _check_consistency(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Rule: cross-column logical constraints."""

        # Rule C1: age must be ≥ 18 (legal minimum for a credit contract).
        if "age" in df.columns:
            underage = df[pd.to_numeric(df["age"], errors="coerce") < 18]
            if not underage.empty:
                report.add(ValidationError(
                    rule="consistency.age_below_legal_minimum",
                    column="age",
                    message=(
                        f"{len(underage)} row(s) have age < 18, "
                        "which is below the legal minimum for a credit contract."
                    ),
                    severity="error",
                    details={"count": len(underage)},
                ))

        # Rule C2: installment_commitment (% of disposable income) must be
        #          positive and a whole-number bucket (1–4).
        if "installment_commitment" in df.columns:
            ic = pd.to_numeric(df["installment_commitment"], errors="coerce").dropna()
            non_integer = ic[ic % 1 != 0]
            if not non_integer.empty:
                report.add(ValidationError(
                    rule="consistency.installment_commitment_not_integer",
                    column="installment_commitment",
                    message=(
                        f"{len(non_integer)} row(s) have non-integer "
                        "installment_commitment values (expected whole-number buckets 1–4)."
                    ),
                    severity="warning",
                    details={"count": len(non_integer)},
                ))

        # Rule C3: existing_credits must be a positive integer.
        if "existing_credits" in df.columns:
            ec = pd.to_numeric(df["existing_credits"], errors="coerce").dropna()
            non_pos = ec[ec < 1]
            if not non_pos.empty:
                report.add(ValidationError(
                    rule="consistency.existing_credits_non_positive",
                    column="existing_credits",
                    message=(
                        f"{len(non_pos)} row(s) have existing_credits < 1 "
                        "(must have at least the current application)."
                    ),
                    severity="error",
                    details={"count": len(non_pos)},
                ))

        # Rule C4: credit_amount must be positive.
        if "credit_amount" in df.columns:
            ca = pd.to_numeric(df["credit_amount"], errors="coerce").dropna()
            non_pos = ca[ca <= 0]
            if not non_pos.empty:
                report.add(ValidationError(
                    rule="consistency.credit_amount_non_positive",
                    column="credit_amount",
                    message=(
                        f"{len(non_pos)} row(s) have credit_amount ≤ 0 "
                        "(loan amount must be strictly positive)."
                    ),
                    severity="error",
                    details={"count": len(non_pos)},
                ))

        # Rule C5: duration must be a positive whole number of months.
        if "duration" in df.columns:
            dur = pd.to_numeric(df["duration"], errors="coerce").dropna()
            non_pos = dur[dur <= 0]
            if not non_pos.empty:
                report.add(ValidationError(
                    rule="consistency.duration_non_positive",
                    column="duration",
                    message=(
                        f"{len(non_pos)} row(s) have duration ≤ 0 months."
                    ),
                    severity="error",
                    details={"count": len(non_pos)},
                ))

        # Rule C6: foreign_worker and own_telephone must be binary yes/no.
        for col in ("foreign_worker", "own_telephone"):
            if col in df.columns:
                allowed = CATEGORICAL_VALUES[col]
                actual_vals = set(df[col].dropna().astype(str).unique())
                bad = actual_vals - allowed
                if bad:
                    report.add(ValidationError(
                        rule=f"consistency.{col}_invalid_binary",
                        column=col,
                        message=(
                            f"Column '{col}' must be one of {sorted(allowed)!r} "
                            f"but found: {sorted(bad)!r}."
                        ),
                        severity="error",
                        details={"bad_values": sorted(bad)},
                    ))

    def _check_duplicates(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Rule: no exact duplicate rows (usually indicates a data ingestion issue)."""
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            dup_frac = dup_count / max(len(df), 1)
            report.add(ValidationError(
                rule="quality.duplicate_rows",
                column="*",
                message=(
                    f"Found {dup_count} duplicate row(s) "
                    f"({dup_frac:.1%} of the dataset)."
                ),
                severity="warning",
                details={"duplicate_count": dup_count, "duplicate_fraction": dup_frac},
            ))


# ─── CONVENIENCE FUNCTION ─────────────────────────────────────────────────────

def validate_dataframe(
    df: pd.DataFrame,
    *,
    missing_threshold: float = 0.0,
    range_tolerance: float = 0.0,
    extra_columns: str = "warning",
    raise_on_error: bool = True,
    verbose: bool = True,
) -> ValidationReport:
    """
    Validate *df* against the German Credit dataset rules.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (with or without the ``'class'`` target column).
    missing_threshold : float
        Allowed missing-value fraction per column (0.0 = none allowed).
    range_tolerance : float
        Fractional tolerance on numeric range bounds.
    extra_columns : str
        ``"error"`` | ``"warning"`` | ``"ignore"`` for unexpected columns.
    raise_on_error : bool
        If ``True`` (default), raise a :class:`ValueError` when errors are found.
    verbose : bool
        If ``True`` (default), print the validation summary.

    Returns
    -------
    ValidationReport
        Contains all discovered errors and warnings.

    Raises
    ------
    ValueError
        When *raise_on_error* is ``True`` and validation errors are found.
    """
    validator = DataValidator(
        missing_threshold=missing_threshold,
        range_tolerance=range_tolerance,
        extra_columns=extra_columns,
    )
    report = validator.validate(df)

    if verbose:
        report.print_summary()

    if raise_on_error:
        report.raise_if_invalid()

    return report
