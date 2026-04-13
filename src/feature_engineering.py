"""
Feature engineering for loan risk assessment.
Adds derived financial risk indicators to the raw dataset.
"""
import numpy as np
import pandas as pd


def _monthly_payment(loan_amount: pd.Series, interest_rate: pd.Series, loan_term: pd.Series) -> pd.Series:
    """
    Compute monthly payment using the standard amortisation formula.
    Falls back to a simple flat payment when interest_rate == 0.
    """
    monthly_rate = interest_rate / 1200.0  # annual % → monthly decimal
    # Avoid division by zero: use simple formula where rate is effectively 0
    zero_rate = monthly_rate == 0.0
    safe_rate = monthly_rate.where(~zero_rate, other=1e-10)

    payment = loan_amount * safe_rate / (1 - (1 + safe_rate) ** (-loan_term))
    # For zero-rate rows, fall back to loan_amount / loan_term
    payment = payment.where(~zero_rate, other=loan_amount / loan_term)
    return payment


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived risk features to a loan DataFrame.

    New columns
    -----------
    monthly_payment         : Amortised monthly instalment (£/$/currency).
    debt_burden             : Estimated monthly debt obligation.
    credit_utilization_risk : 0 = low risk (high score), 1 = high risk (low score).
    loan_to_income          : Loan amount relative to annual income.
    payment_to_income       : Monthly payment relative to monthly income.
    risk_score_raw          : Composite weighted risk indicator (higher → riskier).
    """
    df = df.copy()

    # --- Basic derived features ---
    df['monthly_payment'] = _monthly_payment(
        df['loan_amount'], df['interest_rate'], df['loan_term']
    )

    df['debt_burden'] = df['debt_to_income_ratio'] * df['annual_income'] / 12.0

    credit_score_clamped = df['credit_score'].clip(300, 850)
    df['credit_utilization_risk'] = 1.0 - (credit_score_clamped - 300.0) / 550.0

    # Guard against zero income — flag these rows and fill derived ratios with
    # a high-risk sentinel (1.0 for ratios) rather than silently using the median,
    # because zero-income applications are genuinely high-risk and should not be
    # obscured by an average value.
    zero_income_mask = df['annual_income'] == 0
    if zero_income_mask.any():
        import warnings
        n_zero = int(zero_income_mask.sum())
        warnings.warn(
            f"{n_zero} loan application(s) have annual_income=0. "
            "loan_to_income and payment_to_income will be set to 1.0 "
            "(maximum risk sentinel) for these rows.",
            UserWarning,
            stacklevel=2,
        )
    safe_income = df['annual_income'].replace(0, np.nan)
    df['loan_to_income'] = df['loan_amount'] / safe_income
    df['payment_to_income'] = df['monthly_payment'] / (safe_income / 12.0)

    # --- Composite risk score (higher = riskier) ---
    dti_norm = (df['debt_to_income_ratio'] / 60.0).clip(0, 1)
    derog_norm = (df['num_derogatory_marks'] / 10.0).clip(0, 1)
    rate_norm = ((df['interest_rate'] - 5.0) / 25.0).clip(0, 1)

    df['risk_score_raw'] = (
        0.35 * df['credit_utilization_risk']
        + 0.25 * dti_norm
        + 0.20 * derog_norm
        + 0.10 * rate_norm
        + 0.10 * df['payment_to_income'].clip(0, 1)
    )

    # Replace infinities (e.g. from remaining division edge-cases)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # For zero-income rows the ratio columns are NaN — set them to the
    # maximum-risk sentinel value (1.0) so these applications are treated as
    # high-risk rather than average.
    if zero_income_mask.any():
        for col in ('loan_to_income', 'payment_to_income'):
            df.loc[zero_income_mask, col] = 1.0

    # Fill any remaining NaN derived columns with column median (safe fallback)
    derived_cols = [
        'monthly_payment', 'debt_burden', 'credit_utilization_risk',
        'loan_to_income', 'payment_to_income', 'risk_score_raw',
    ]
    for col in derived_cols:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    return df
