"""
tests/test_loan_risk_assessment.py — Basic unit tests for the loan risk assessment pipeline.

Run with:
    pytest tests/test_loan_risk_assessment.py -v
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Add parent directory so we can import the package module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.loan_risk_assessment import (
    build_preprocessor,
    engineer_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal synthetic DataFrame that mirrors the German Credit dataset schema."""
    np.random.seed(0)
    n = 50
    return pd.DataFrame(
        {
            "duration":                np.random.randint(6, 60, n),
            "credit_amount":           np.random.randint(500, 15000, n),
            "installment_commitment":  np.random.randint(1, 4, n),
            "residence_since":         np.random.randint(1, 4, n),
            "age":                     np.random.randint(20, 75, n),
            "existing_credits":        np.random.randint(1, 4, n),
            "num_dependents":          np.random.randint(1, 3, n),
            "checking_status":         np.random.choice(["no checking", "<0", "0<=X<200", ">=200"], n),
            "credit_history":          np.random.choice(["existing paid", "all paid", "delayed previously"], n),
            "purpose":                 np.random.choice(["furniture/equipment", "new car", "education"], n),
            "savings_status":          np.random.choice(["<100", "100<=X<500", "no known savings"], n),
            "employment":              np.random.choice(["unemployed", "1<=X<4", ">=7"], n),
            "personal_status":         np.random.choice(["male single", "female div/dep/mar"], n),
            "other_parties":           np.random.choice(["none", "guarantor"], n),
            "property_magnitude":      np.random.choice(["real estate", "car", "no known property"], n),
            "other_payment_plans":     np.random.choice(["none", "bank"], n),
            "housing":                 np.random.choice(["own", "free", "rent"], n),
            "job":                     np.random.choice(["skilled", "unskilled resident"], n),
            "own_telephone":           np.random.choice(["yes", "none"], n),
            "foreign_worker":          np.random.choice(["yes", "no"], n),
            "default":                 np.random.randint(0, 2, n),
        }
    )


# ── engineer_features ─────────────────────────────────────────────────────────

class TestEngineerFeatures:
    EXPECTED_NEW_COLS = [
        "debt_to_credit_ratio",
        "loan_income_proxy",
        "long_term_loan",
        "high_credit",
        "senior_applicant",
        "age_group",
    ]

    def test_new_columns_added(self, sample_df):
        result = engineer_features(sample_df)
        for col in self.EXPECTED_NEW_COLS:
            assert col in result.columns, f"Missing engineered column: {col}"

    def test_original_columns_preserved(self, sample_df):
        result = engineer_features(sample_df)
        for col in sample_df.columns:
            assert col in result.columns, f"Original column '{col}' was dropped"

    def test_no_rows_dropped(self, sample_df):
        result = engineer_features(sample_df)
        assert len(result) == len(sample_df)

    def test_long_term_loan_is_binary(self, sample_df):
        result = engineer_features(sample_df)
        assert set(result["long_term_loan"].unique()).issubset({0, 1})

    def test_high_credit_is_binary(self, sample_df):
        result = engineer_features(sample_df)
        assert set(result["high_credit"].unique()).issubset({0, 1})

    def test_senior_applicant_is_binary(self, sample_df):
        result = engineer_features(sample_df)
        assert set(result["senior_applicant"].unique()).issubset({0, 1})

    def test_age_group_values(self, sample_df):
        result = engineer_features(sample_df)
        valid_groups = {"u25", "25_35", "35_50", "o50", "nan"}
        assert set(result["age_group"].unique()).issubset(valid_groups)

    def test_debt_ratio_non_negative(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["debt_to_credit_ratio"] >= 0).all()

    def test_does_not_mutate_input(self, sample_df):
        original_cols = list(sample_df.columns)
        engineer_features(sample_df)
        assert list(sample_df.columns) == original_cols


# ── build_preprocessor ────────────────────────────────────────────────────────

class TestBuildPreprocessor:
    def test_returns_three_values(self, sample_df):
        X = sample_df.drop(columns=["default"])
        result = build_preprocessor(X)
        assert len(result) == 3

    def test_preprocessor_fits_and_transforms(self, sample_df):
        X = engineer_features(sample_df).drop(columns=["default"])
        prep, num_f, cat_f = build_preprocessor(X)
        prep.fit(X)
        X_transformed = prep.transform(X)
        # Result should be a 2D numeric array
        assert X_transformed.ndim == 2
        assert X_transformed.shape[0] == len(X)

    def test_numeric_and_categorical_split(self, sample_df):
        X = engineer_features(sample_df).drop(columns=["default"])
        _, num_f, cat_f = build_preprocessor(X)
        # Every feature ends up in exactly one group
        assert set(num_f).isdisjoint(set(cat_f))
        assert set(num_f) | set(cat_f) == set(X.columns)

    def test_feature_names_out(self, sample_df):
        X = engineer_features(sample_df).drop(columns=["default"])
        prep, _, _ = build_preprocessor(X)
        prep.fit(X)
        names = prep.get_feature_names_out()
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)

    def test_handles_unseen_categories(self, sample_df):
        """Transformer should not raise when test data has an unseen category."""
        X = engineer_features(sample_df).drop(columns=["default"])
        prep, _, _ = build_preprocessor(X)
        prep.fit(X)
        # Introduce an unseen category
        X_new = X.copy()
        X_new.loc[0, "purpose"] = "UNSEEN_CATEGORY"
        # Should not raise (OneHotEncoder has handle_unknown='ignore')
        prep.transform(X_new)


# ── predict.py helpers ────────────────────────────────────────────────────────

class TestPredictHelpers:
    """Import and test the helper functions from predict.py."""

    @pytest.fixture(autouse=True)
    def import_predict(self):
        import importlib
        import predict as pred_module
        self.pred = pred_module

    def test_label_risk_low(self):
        assert self.pred.label_risk(0.10) == "Low"

    def test_label_risk_medium(self):
        assert self.pred.label_risk(0.50) == "Medium"

    def test_label_risk_high(self):
        assert self.pred.label_risk(0.75) == "High"

    def test_label_risk_boundary_low_medium(self):
        # 0.35 is the boundary — belongs to Medium
        assert self.pred.label_risk(0.35) == "Medium"

    def test_label_risk_boundary_medium_high(self):
        # 0.60 is the boundary — belongs to High
        assert self.pred.label_risk(0.60) == "High"


# ── Saved model round-trip (skipped if model not yet generated) ───────────────

@pytest.mark.skipif(
    not os.path.exists("models/loan_risk_model.joblib"),
    reason="Trained model not found — run src/loan_risk_assessment.py first"
)
class TestSavedModel:
    def test_model_loads(self):
        import joblib
        pipeline = joblib.load("models/loan_risk_model.joblib")
        assert pipeline is not None

    def test_model_predicts(self, sample_df):
        import joblib
        pipeline = joblib.load("models/loan_risk_model.joblib")
        X = engineer_features(sample_df).drop(columns=["default"])
        proba = pipeline.predict_proba(X)[:, 1]
        assert proba.shape == (len(X),)
        assert ((proba >= 0) & (proba <= 1)).all()

    def test_metadata_has_required_keys(self):
        with open("models/model_metadata.json") as f:
            meta = json.load(f)
        required = {"model_type", "cv_roc_auc", "test_roc_auc", "optimal_threshold"}
        assert required.issubset(meta.keys())
