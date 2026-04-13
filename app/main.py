from fastapi import FastAPI, HTTPException
from .models import LoanApplication, RiskAssessmentResult
from .assessor import assess_risk

app = FastAPI(
    title="Loan Risk Assessment API",
    description="Advanced Loan Default Risk Assessment",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/assess", response_model=RiskAssessmentResult, summary="Assess loan default risk")
def assess_loan(application: LoanApplication) -> RiskAssessmentResult:
    """
    Submit a loan application and receive a risk assessment.

    - **risk_level**: low / medium / high
    - **risk_score**: 0–100 (higher means riskier)
    - **recommendation**: Approve / Approve with conditions / Decline
    - **factors**: human-readable list of contributing risk factors
    """
    try:
        return assess_risk(application)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
