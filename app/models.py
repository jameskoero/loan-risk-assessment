from pydantic import BaseModel, Field
from enum import Enum


class EmploymentStatus(str, Enum):
    employed = "employed"
    self_employed = "self_employed"
    unemployed = "unemployed"
    retired = "retired"


class LoanPurpose(str, Enum):
    home = "home"
    auto = "auto"
    education = "education"
    business = "business"
    personal = "personal"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class LoanApplication(BaseModel):
    applicant_id: str = Field(..., description="Unique identifier for the applicant")
    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    annual_income: float = Field(..., gt=0, description="Annual income in USD")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in USD")
    loan_term_months: int = Field(..., gt=0, description="Loan term in months")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score (300-850)")
    employment_status: EmploymentStatus
    loan_purpose: LoanPurpose
    existing_debt: float = Field(0.0, ge=0, description="Total annual debt obligations in USD (e.g. existing loan payments per year)")
    num_late_payments: int = Field(0, ge=0, description="Number of late payments in history")


class RiskAssessmentResult(BaseModel):
    applicant_id: str
    risk_level: RiskLevel
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Risk score 0-100 (higher = riskier)")
    debt_to_income_ratio: float
    loan_to_income_ratio: float
    recommendation: str
    factors: list[str]
