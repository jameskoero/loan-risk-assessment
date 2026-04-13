from .models import (
    LoanApplication,
    RiskAssessmentResult,
    RiskLevel,
    EmploymentStatus,
)


# Weight constants for risk scoring
_CREDIT_SCORE_WEIGHT = 35
_DTI_WEIGHT = 30
_EMPLOYMENT_WEIGHT = 15
_LATE_PAYMENT_WEIGHT = 20


def _score_credit(credit_score: int) -> float:
    """Returns a 0-100 risk contribution (higher = riskier) based on credit score."""
    if credit_score >= 750:
        return 0.0
    if credit_score >= 700:
        return 20.0
    if credit_score >= 650:
        return 45.0
    if credit_score >= 600:
        return 70.0
    return 100.0


def _score_dti(debt_to_income: float) -> float:
    """Returns a 0-100 risk contribution based on debt-to-income ratio."""
    if debt_to_income <= 0.20:
        return 0.0
    if debt_to_income <= 0.35:
        return 25.0
    if debt_to_income <= 0.43:
        return 55.0
    if debt_to_income <= 0.50:
        return 80.0
    return 100.0


def _score_employment(status: EmploymentStatus) -> float:
    """Returns a 0-100 risk contribution based on employment status."""
    scores = {
        EmploymentStatus.employed: 0.0,
        EmploymentStatus.retired: 10.0,
        EmploymentStatus.self_employed: 40.0,
        EmploymentStatus.unemployed: 100.0,
    }
    return scores[status]


def _score_late_payments(num_late: int) -> float:
    """Returns a 0-100 risk contribution based on number of late payments."""
    if num_late == 0:
        return 0.0
    if num_late <= 1:
        return 30.0
    if num_late <= 3:
        return 65.0
    return 100.0


def assess_risk(application: LoanApplication) -> RiskAssessmentResult:
    """
    Assess the default risk of a loan application.

    Returns a RiskAssessmentResult with risk level, score, and contributing factors.
    """
    monthly_income = application.annual_income / 12
    monthly_loan_payment = application.loan_amount / application.loan_term_months
    total_monthly_debt = (application.existing_debt / 12) + monthly_loan_payment
    debt_to_income = total_monthly_debt / monthly_income
    loan_to_income = application.loan_amount / application.annual_income

    credit_component = _score_credit(application.credit_score) * _CREDIT_SCORE_WEIGHT / 100
    dti_component = _score_dti(debt_to_income) * _DTI_WEIGHT / 100
    employment_component = _score_employment(application.employment_status) * _EMPLOYMENT_WEIGHT / 100
    late_payment_component = _score_late_payments(application.num_late_payments) * _LATE_PAYMENT_WEIGHT / 100

    raw_score = credit_component + dti_component + employment_component + late_payment_component

    if raw_score <= 30:
        risk_level = RiskLevel.low
        recommendation = "Approve"
    elif raw_score <= 60:
        risk_level = RiskLevel.medium
        recommendation = "Approve with conditions"
    else:
        risk_level = RiskLevel.high
        recommendation = "Decline"

    factors = _build_factors(application, debt_to_income)

    return RiskAssessmentResult(
        applicant_id=application.applicant_id,
        risk_level=risk_level,
        risk_score=round(raw_score, 2),
        debt_to_income_ratio=round(debt_to_income, 4),
        loan_to_income_ratio=round(loan_to_income, 4),
        recommendation=recommendation,
        factors=factors,
    )


def _build_factors(application: LoanApplication, debt_to_income: float) -> list[str]:
    factors: list[str] = []

    if application.credit_score < 650:
        factors.append(f"Low credit score ({application.credit_score})")
    elif application.credit_score >= 750:
        factors.append(f"Excellent credit score ({application.credit_score})")

    if debt_to_income > 0.43:
        factors.append(f"High debt-to-income ratio ({debt_to_income:.1%})")
    elif debt_to_income <= 0.20:
        factors.append(f"Healthy debt-to-income ratio ({debt_to_income:.1%})")

    if application.employment_status == EmploymentStatus.unemployed:
        factors.append("Applicant is currently unemployed")
    elif application.employment_status == EmploymentStatus.self_employed:
        factors.append("Self-employed income may be variable")

    if application.num_late_payments > 0:
        factors.append(f"{application.num_late_payments} late payment(s) on record")

    loan_to_income = application.loan_amount / application.annual_income
    if loan_to_income > 3:
        factors.append(f"Loan amount is {loan_to_income:.1f}x annual income")

    return factors
