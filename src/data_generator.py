"""
Synthetic loan dataset generator for default risk modeling.
"""
import numpy as np
import pandas as pd
import os


def generate_loan_dataset(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic loan dataset with ~20% default rate.

    Parameters
    ----------
    n_samples : int
        Number of loan records to generate.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame containing loan features and binary default target.
    """
    rng = np.random.RandomState(random_state)

    loan_amount = rng.uniform(5000, 100000, n_samples)
    loan_term = rng.choice([12, 24, 36, 48, 60], n_samples)
    interest_rate = rng.uniform(5, 30, n_samples)
    annual_income = rng.uniform(20000, 200000, n_samples)
    debt_to_income_ratio = rng.uniform(0, 60, n_samples)
    credit_score = rng.uniform(300, 850, n_samples)
    employment_length = rng.uniform(0, 30, n_samples)
    num_credit_lines = rng.randint(1, 31, n_samples)
    num_derogatory_marks = rng.choice(
        range(11), n_samples, p=[0.45, 0.25, 0.13, 0.07, 0.04, 0.02, 0.02, 0.01, 0.005, 0.003, 0.002]
    )
    home_ownership = rng.choice(['OWN', 'RENT', 'MORTGAGE'], n_samples, p=[0.15, 0.40, 0.45])
    loan_purpose = rng.choice(
        ['debt_consolidation', 'home_improvement', 'car', 'other'],
        n_samples,
        p=[0.45, 0.25, 0.15, 0.15],
    )
    verification_status = rng.choice(
        ['Verified', 'Source Verified', 'Not Verified'],
        n_samples,
        p=[0.35, 0.35, 0.30],
    )

    # Logistic-based default probability
    credit_score_norm = (credit_score - 300) / 550          # 0-1, higher is better
    dti_norm = debt_to_income_ratio / 60                     # 0-1, higher is worse
    income_norm = (annual_income - 20000) / 180000           # 0-1, higher is better
    loan_to_income = loan_amount / annual_income
    derog_norm = num_derogatory_marks / 10                   # 0-1, higher is worse
    rate_norm = (interest_rate - 5) / 25                     # 0-1, higher is worse

    # Positive contribution → increases default risk
    logit = (
        -3.5
        - 2.5 * credit_score_norm
        + 2.0 * dti_norm
        + 1.5 * derog_norm
        + 1.0 * rate_norm
        + 0.8 * loan_to_income
        - 0.6 * income_norm
        - 0.4 * (employment_length / 30)
        + 0.3 * (home_ownership == 'RENT').astype(float)
        + rng.normal(0, 0.5, n_samples)   # noise
    )

    default_prob = 1 / (1 + np.exp(-logit))

    # Calibrate to ~20% default rate
    threshold = np.percentile(default_prob, 80)
    default = (default_prob >= threshold).astype(int)

    df = pd.DataFrame({
        'loan_amount': np.round(loan_amount, 2),
        'loan_term': loan_term,
        'interest_rate': np.round(interest_rate, 2),
        'annual_income': np.round(annual_income, 2),
        'debt_to_income_ratio': np.round(debt_to_income_ratio, 2),
        'credit_score': np.round(credit_score).astype(int),
        'employment_length': np.round(employment_length, 1),
        'num_credit_lines': num_credit_lines,
        'num_derogatory_marks': num_derogatory_marks,
        'home_ownership': home_ownership,
        'loan_purpose': loan_purpose,
        'verification_status': verification_status,
        'default': default,
    })
    return df


def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to CSV at the given path, creating directories as needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
