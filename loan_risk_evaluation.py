"""
Loan Risk Evaluation Module
============================
Provides the ``evaluate_loan_risk`` function, which scores a single loan
application and returns a structured risk assessment without requiring a
pre-trained model (rule-based scoring) *or* with an optional scikit-learn
compatible model.

Rule-based scoring is intentionally transparent so lenders can audit every
decision without needing ML tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

RISK_TIERS = {
    "LOW": (80, 100),       # score >= 80 → low risk, approve
    "MEDIUM": (60, 80),     # 60 ≤ score < 80 → moderate risk, review
    "HIGH": (40, 60),       # 40 ≤ score < 60 → high risk, additional checks
    "VERY_HIGH": (0, 40),   # score < 40 → very high risk, likely decline
}


@dataclass
class LoanApplication:
    """Structured representation of a loan application.

    Attributes
    ----------
    credit_score:
        Applicant's credit score (300–850).
    annual_income:
        Gross annual income in the applicant's currency.
    loan_amount:
        Requested loan principal.
    loan_term_months:
        Repayment period in months (e.g. 12, 24, 60).
    existing_debt:
        Total outstanding debt obligations.
    employment_years:
        Number of years in current or most recent employment.
    num_late_payments:
        Count of late payments in credit history.
    has_collateral:
        Whether the applicant offers collateral.
    purpose:
        Loan purpose string (e.g. ``"car"``, ``"education"``, ``"business"``).
    """

    credit_score: float
    annual_income: float
    loan_amount: float
    loan_term_months: int
    existing_debt: float = 0.0
    employment_years: float = 0.0
    num_late_payments: int = 0
    has_collateral: bool = False
    purpose: str = "personal"


@dataclass
class RiskAssessment:
    """Result returned by :func:`evaluate_loan_risk`.

    Attributes
    ----------
    score:
        Composite risk score on a 0–100 scale (higher = lower risk).
    tier:
        One of ``"LOW"``, ``"MEDIUM"``, ``"HIGH"``, ``"VERY_HIGH"``.
    recommendation:
        Plain-text recommendation (``"APPROVE"``, ``"REVIEW"``, or
        ``"DECLINE"``).
    factors:
        Dictionary mapping factor names to their individual sub-scores so
        callers can surface explanations to end-users.
    messages:
        Human-readable list of risk drivers found during evaluation.
    """

    score: float
    tier: str
    recommendation: str
    factors: Dict[str, float] = field(default_factory=dict)
    messages: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------

def _score_credit(credit_score: float) -> float:
    """Map a 300-850 credit score to a 0-100 sub-score."""
    normalised = (credit_score - 300) / (850 - 300)
    return round(max(0.0, min(100.0, normalised * 100)), 2)


def _score_debt_to_income(annual_income: float, existing_debt: float,
                           loan_amount: float) -> float:
    """Score based on debt-to-income ratio (lower DTI → higher score)."""
    if annual_income <= 0:
        return 0.0
    total_debt = existing_debt + loan_amount
    dti = total_debt / annual_income
    # DTI thresholds: <=0.36 excellent, 0.36-0.43 good, 0.43-0.50 fair, >0.50 poor
    if dti <= 0.36:
        score = 100.0
    elif dti <= 0.43:
        score = 75.0
    elif dti <= 0.50:
        score = 50.0
    else:
        # Linear decay above 0.50; score reaches 0 at DTI = 1.0
        score = max(0.0, (1.0 - dti) * 100)
    return round(score, 2)


def _score_loan_to_income(annual_income: float, loan_amount: float) -> float:
    """Score based on loan-to-income ratio."""
    if annual_income <= 0:
        return 0.0
    lti = loan_amount / annual_income
    if lti <= 2.0:
        score = 100.0
    elif lti <= 4.0:
        score = 70.0
    elif lti <= 6.0:
        score = 40.0
    else:
        score = max(0.0, 100.0 - lti * 10)
    return round(score, 2)


def _score_employment(employment_years: float) -> float:
    """Score based on employment stability."""
    if employment_years >= 5:
        return 100.0
    elif employment_years >= 2:
        return 70.0
    elif employment_years >= 1:
        return 45.0
    elif employment_years >= 0.5:
        return 20.0
    return 0.0


def _score_payment_history(num_late_payments: int) -> float:
    """Score based on payment history (penalise late payments)."""
    if num_late_payments == 0:
        return 100.0
    elif num_late_payments == 1:
        return 70.0
    elif num_late_payments <= 3:
        return 40.0
    elif num_late_payments <= 6:
        return 15.0
    return 0.0


def _score_collateral(has_collateral: bool) -> float:
    """Bonus for secured loans."""
    return 100.0 if has_collateral else 50.0


def _determine_tier(score: float) -> str:
    if score >= 80:
        return "LOW"
    if score >= 60:
        return "MEDIUM"
    if score >= 40:
        return "HIGH"
    return "VERY_HIGH"


def _determine_recommendation(tier: str) -> str:
    return {
        "LOW": "APPROVE",
        "MEDIUM": "REVIEW",
        "HIGH": "REVIEW",
        "VERY_HIGH": "DECLINE",
    }[tier]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_loan_risk(
    application: LoanApplication,
    model: Optional[Any] = None,
) -> RiskAssessment:
    """Evaluate the default risk of a loan application.

    When *model* is ``None`` the function uses a transparent, rule-based
    scoring engine.  If a scikit-learn–compatible *model* is supplied (one
    that exposes a ``predict_proba`` method), its output probability is
    blended with the rule-based score (50/50) to produce the final result.

    Parameters
    ----------
    application:
        A :class:`LoanApplication` instance populated with the applicant's
        details.
    model:
        Optional pre-trained classifier with a ``predict_proba`` method.
        The model should be trained to predict loan default (class 1 = default)
        and must accept the same feature vector produced internally.

    Returns
    -------
    RiskAssessment
        A fully populated :class:`RiskAssessment` object.

    Raises
    ------
    ValueError
        If any required field contains an out-of-range or invalid value.
    """
    _validate(application)

    # --- Rule-based sub-scores --------------------------------------------
    factors: Dict[str, float] = {
        "credit_score":      _score_credit(application.credit_score),
        "debt_to_income":    _score_debt_to_income(
                                 application.annual_income,
                                 application.existing_debt,
                                 application.loan_amount),
        "loan_to_income":    _score_loan_to_income(
                                 application.annual_income,
                                 application.loan_amount),
        "employment":        _score_employment(application.employment_years),
        "payment_history":   _score_payment_history(application.num_late_payments),
        "collateral":        _score_collateral(application.has_collateral),
    }

    # Weighted combination (weights sum to 1.0)
    weights = {
        "credit_score":    0.35,
        "debt_to_income":  0.25,
        "loan_to_income":  0.15,
        "employment":      0.10,
        "payment_history": 0.10,
        "collateral":      0.05,
    }
    rule_score = sum(factors[k] * weights[k] for k in weights)

    # --- Optional model blending ------------------------------------------
    if model is not None:
        features = _build_feature_vector(application)
        proba_default = model.predict_proba([features])[0][1]
        model_score = (1.0 - proba_default) * 100.0
        final_score = 0.5 * rule_score + 0.5 * model_score
    else:
        final_score = rule_score

    final_score = round(max(0.0, min(100.0, final_score)), 2)
    tier = _determine_tier(final_score)
    recommendation = _determine_recommendation(tier)
    messages = _build_messages(application, factors)

    return RiskAssessment(
        score=final_score,
        tier=tier,
        recommendation=recommendation,
        factors=factors,
        messages=messages,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(app: LoanApplication) -> None:
    if not (300 <= app.credit_score <= 850):
        raise ValueError(
            f"credit_score must be between 300 and 850, got {app.credit_score}")
    if app.annual_income < 0:
        raise ValueError("annual_income must be non-negative")
    if app.loan_amount <= 0:
        raise ValueError("loan_amount must be positive")
    if app.loan_term_months <= 0:
        raise ValueError("loan_term_months must be positive")
    if app.existing_debt < 0:
        raise ValueError("existing_debt must be non-negative")
    if app.employment_years < 0:
        raise ValueError("employment_years must be non-negative")
    if app.num_late_payments < 0:
        raise ValueError("num_late_payments must be non-negative")


# ---------------------------------------------------------------------------
# Feature vector for ML model blending
# ---------------------------------------------------------------------------

def _build_feature_vector(app: LoanApplication) -> list:
    """Produce a numeric feature vector compatible with a trained model."""
    dti = (app.existing_debt + app.loan_amount) / max(app.annual_income, 1)
    lti = app.loan_amount / max(app.annual_income, 1)
    monthly_payment = app.loan_amount / app.loan_term_months
    payment_to_income = monthly_payment / max(app.annual_income / 12, 1)
    return [
        app.credit_score,
        app.annual_income,
        app.loan_amount,
        app.loan_term_months,
        app.existing_debt,
        app.employment_years,
        app.num_late_payments,
        int(app.has_collateral),
        dti,
        lti,
        payment_to_income,
    ]


# ---------------------------------------------------------------------------
# Human-readable messages
# ---------------------------------------------------------------------------

def _build_messages(app: LoanApplication,
                    factors: Dict[str, float]) -> list:
    msgs = []
    if factors["credit_score"] < 50:
        msgs.append(
            f"Low credit score ({app.credit_score:.0f}); significant default risk.")
    if factors["debt_to_income"] < 50:
        total_debt = app.existing_debt + app.loan_amount
        dti = total_debt / max(app.annual_income, 1)
        msgs.append(
            f"High debt-to-income ratio ({dti:.2f}); may indicate repayment strain.")
    if factors["loan_to_income"] < 50:
        lti = app.loan_amount / max(app.annual_income, 1)
        msgs.append(
            f"Loan-to-income ratio ({lti:.2f}) is elevated.")
    if factors["employment"] < 50:
        msgs.append(
            f"Short employment history ({app.employment_years:.1f} yrs); "
            "income stability uncertain.")
    if factors["payment_history"] < 70:
        msgs.append(
            f"{app.num_late_payments} late payment(s) detected in credit history.")
    if not app.has_collateral:
        msgs.append("Unsecured loan; no collateral provided.")
    if not msgs:
        msgs.append("No major risk factors identified.")
    return msgs
