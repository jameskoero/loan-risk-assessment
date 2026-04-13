"""
Unit tests for src/loan_risk_assessment.py

Run with:
    pytest tests/ -v
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

# Ensure the repo root is on sys.path so `src` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.loan_risk_assessment import (
    engineer_features,
    build_preprocessor,
    optimise_threshold,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df():
    """Minimal synthetic dataframe that mimics German Credit columns."""
    np.random.seed(0)
    n = 50
    return pd.DataFrame(
        {
            "duration": np.random.randint(6, 72, n),
            "credit_amount": np.random.randint(500, 15000, n),
            "installment_commitment": np.random.randint(1, 4, n),
            "age": np.random.randint(20, 75, n),
            "residence_since": np.random.randint(1, 4, n),
            "existing_credits": np.random.randint(1, 4, n),
            "num_dependents": np.random.randint(1, 2, n),
            "checking_status": np.random.choice(["<0", "0<=X<200", ">=200", "no checking"], n),
            "credit_history": np.random.choice(["critical/other", "existing paid", "delayed", "no credits"], n),
            "purpose": np.random.choice(["car", "furniture", "radio/tv", "domestic", "education"], n),
            "savings_status": np.random.choice(["<100", "100<=X<500", ">=1000", "no known savings"], n),
            "employment": np.random.choice(["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"], n),
            "personal_status": np.random.choice(["male single", "female div/dep", "male div/sep"], n),
            "other_parties": np.random.choice(["none", "co applicant", "guarantor"], n),
            "property_magnitude": np.random.choice(["real estate", "life insurance", "car", "no known property"], n),
            "other_payment_plans": np.random.choice(["bank", "stores", "none"], n),
            "housing": np.random.choice(["rent", "own", "for free"], n),
            "job": np.random.choice(["unskilled resident", "unskilled non res", "skilled", "high qualif/self emp/mgmt"], n),
            "own_telephone": np.random.choice(["none", "yes"], n),
            "foreign_worker": np.random.choice(["yes", "no"], n),
            "default": np.random.randint(0, 2, n),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# engineer_features
# ─────────────────────────────────────────────────────────────────────────────

def test_engineer_features_adds_columns(sample_df):
    result = engineer_features(sample_df)
    expected_new = {
        "debt_to_credit_ratio",
        "loan_income_proxy",
        "long_term_loan",
        "high_credit",
        "senior_applicant",
        "age_group",
    }
    assert expected_new.issubset(set(result.columns))


def test_engineer_features_does_not_drop_original_columns(sample_df):
    original_cols = set(sample_df.columns)
    result = engineer_features(sample_df)
    assert original_cols.issubset(set(result.columns))


def test_engineer_features_binary_flags(sample_df):
    result = engineer_features(sample_df)
    for col in ["long_term_loan", "high_credit", "senior_applicant"]:
        assert set(result[col].unique()).issubset({0, 1})


def test_engineer_features_does_not_mutate_input(sample_df):
    original_columns = list(sample_df.columns)
    engineer_features(sample_df)
    assert list(sample_df.columns) == original_columns


# ─────────────────────────────────────────────────────────────────────────────
# build_preprocessor
# ─────────────────────────────────────────────────────────────────────────────

def test_build_preprocessor_returns_three_items(sample_df):
    df = engineer_features(sample_df)
    X = df.drop(columns=["default"])
    result = build_preprocessor(X)
    assert len(result) == 3


def test_build_preprocessor_transform_shape(sample_df):
    df = engineer_features(sample_df)
    X = df.drop(columns=["default"])
    prep, _, _ = build_preprocessor(X)
    prep.fit(X)
    X_transformed = prep.transform(X)
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] > X.shape[1]  # OHE expands categorical columns


def test_build_preprocessor_no_nans_after_transform(sample_df):
    df = engineer_features(sample_df)
    X = df.drop(columns=["default"])
    prep, _, _ = build_preprocessor(X)
    prep.fit(X)
    X_transformed = prep.transform(X)
    assert not np.isnan(X_transformed).any()


# ─────────────────────────────────────────────────────────────────────────────
# optimise_threshold
# ─────────────────────────────────────────────────────────────────────────────

def test_optimise_threshold_returns_float():
    np.random.seed(42)
    y_te = np.random.randint(0, 2, 100)
    y_proba = np.random.rand(100)
    result = optimise_threshold(y_te, y_proba)
    assert isinstance(result, float)


def test_optimise_threshold_in_valid_range():
    np.random.seed(42)
    y_te = np.array([0] * 70 + [1] * 30)
    y_proba = np.random.rand(100)
    result = optimise_threshold(y_te, y_proba)
    assert 0.1 <= result <= 0.9
