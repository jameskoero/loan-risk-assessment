"""
Data preprocessing for loan risk assessment: encoding, scaling, imputation.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


NUMERIC_COLS = [
    'loan_amount', 'loan_term', 'interest_rate', 'annual_income',
    'debt_to_income_ratio', 'credit_score', 'employment_length',
    'num_credit_lines', 'num_derogatory_marks',
]

CATEGORICAL_COLS = ['home_ownership', 'loan_purpose', 'verification_status']

ENGINEERED_NUMERIC_COLS = [
    'monthly_payment', 'debt_burden', 'credit_utilization_risk',
    'loan_to_income', 'payment_to_income', 'risk_score_raw',
]

TARGET_COL = 'default'


class LoanDataPreprocessor:
    """Fit/transform pipeline for loan data: imputation, encoding, scaling."""

    def __init__(self):
        self.numeric_cols: list[str] = []
        self.categorical_cols: list[str] = CATEGORICAL_COLS[:]
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.feature_names_: list[str] = []
        self._fitted = False

    def _detect_numeric_cols(self, df: pd.DataFrame) -> list[str]:
        base = [c for c in NUMERIC_COLS if c in df.columns]
        engineered = [c for c in ENGINEERED_NUMERIC_COLS if c in df.columns]
        return base + engineered

    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Fit all transformers on df and return (X_processed, y).
        Separates the target column before processing.
        """
        df = df.copy()

        y = df[TARGET_COL].copy() if TARGET_COL in df.columns else pd.Series(dtype=int)
        df = df.drop(columns=[TARGET_COL], errors='ignore')

        self.numeric_cols = self._detect_numeric_cols(df)
        present_cat = [c for c in self.categorical_cols if c in df.columns]
        self.categorical_cols = present_cat

        # Impute
        df[self.numeric_cols] = self.numeric_imputer.fit_transform(df[self.numeric_cols])
        if self.categorical_cols:
            df[self.categorical_cols] = self.categorical_imputer.fit_transform(df[self.categorical_cols])

        # Scale numerics
        num_scaled = self.scaler.fit_transform(df[self.numeric_cols])
        num_df = pd.DataFrame(num_scaled, columns=self.numeric_cols, index=df.index)

        # Encode categoricals
        if self.categorical_cols:
            cat_encoded = self.encoder.fit_transform(df[self.categorical_cols])
            cat_names = self.encoder.get_feature_names_out(self.categorical_cols).tolist()
            cat_df = pd.DataFrame(cat_encoded, columns=cat_names, index=df.index)
            X = pd.concat([num_df, cat_df], axis=1)
        else:
            X = num_df

        self.feature_names_ = X.columns.tolist()
        self._fitted = True
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted transformations to new data."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        df = df.copy()
        df = df.drop(columns=[TARGET_COL], errors='ignore')

        # Ensure all expected columns are present; fill missing with NaN
        for col in self.numeric_cols:
            if col not in df.columns:
                df[col] = np.nan
        for col in self.categorical_cols:
            if col not in df.columns:
                df[col] = np.nan

        df[self.numeric_cols] = self.numeric_imputer.transform(df[self.numeric_cols])
        if self.categorical_cols:
            df[self.categorical_cols] = self.categorical_imputer.transform(df[self.categorical_cols])

        num_scaled = self.scaler.transform(df[self.numeric_cols])
        num_df = pd.DataFrame(num_scaled, columns=self.numeric_cols, index=df.index)

        if self.categorical_cols:
            cat_encoded = self.encoder.transform(df[self.categorical_cols])
            cat_names = self.encoder.get_feature_names_out(self.categorical_cols).tolist()
            cat_df = pd.DataFrame(cat_encoded, columns=cat_names, index=df.index)
            X = pd.concat([num_df, cat_df], axis=1)
        else:
            X = num_df

        # Reindex to match training feature set
        X = X.reindex(columns=self.feature_names_, fill_value=0.0)
        return X

    def save(self, path: str) -> None:
        """Persist the fitted preprocessor to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'LoanDataPreprocessor':
        """Load a previously saved preprocessor from disk."""
        return joblib.load(path)
