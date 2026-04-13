import pytest
from app.models import LoanApplication, EmploymentStatus, LoanPurpose, RiskLevel
from app.assessor import assess_risk


def _base_application(**overrides) -> LoanApplication:
    defaults = dict(
        applicant_id="test-001",
        age=35,
        annual_income=80000,
        loan_amount=20000,
        loan_term_months=60,
        credit_score=720,
        employment_status=EmploymentStatus.employed,
        loan_purpose=LoanPurpose.auto,
        existing_debt=5000,
        num_late_payments=0,
    )
    defaults.update(overrides)
    return LoanApplication(**defaults)


class TestRiskScoring:
    def test_low_risk_applicant(self):
        app = _base_application(credit_score=780, existing_debt=0)
        result = assess_risk(app)
        assert result.risk_level == RiskLevel.low
        assert result.recommendation == "Approve"
        assert result.risk_score < 30

    def test_high_risk_applicant(self):
        app = _base_application(
            credit_score=550,
            employment_status=EmploymentStatus.unemployed,
            existing_debt=40000,
            num_late_payments=5,
        )
        result = assess_risk(app)
        assert result.risk_level == RiskLevel.high
        assert result.recommendation == "Decline"
        assert result.risk_score > 60

    def test_medium_risk_applicant(self):
        app = _base_application(
            credit_score=640,
            existing_debt=25000,
            num_late_payments=2,
        )
        result = assess_risk(app)
        assert result.risk_level == RiskLevel.medium

    def test_dti_calculation(self):
        app = _base_application(annual_income=60000, loan_amount=12000, loan_term_months=12, existing_debt=0)
        result = assess_risk(app)
        # monthly payment = 1000, monthly income = 5000 => DTI = 0.20
        assert abs(result.debt_to_income_ratio - 0.20) < 0.01

    def test_loan_to_income_calculation(self):
        app = _base_application(annual_income=80000, loan_amount=40000)
        result = assess_risk(app)
        assert abs(result.loan_to_income_ratio - 0.5) < 0.01

    def test_factors_include_credit_issue(self):
        app = _base_application(credit_score=580)
        result = assess_risk(app)
        assert any("credit score" in f.lower() for f in result.factors)

    def test_factors_include_late_payments(self):
        app = _base_application(num_late_payments=2)
        result = assess_risk(app)
        assert any("late payment" in f.lower() for f in result.factors)

    def test_factors_include_unemployment(self):
        app = _base_application(employment_status=EmploymentStatus.unemployed)
        result = assess_risk(app)
        assert any("unemployed" in f.lower() for f in result.factors)

    def test_risk_score_bounds(self):
        for credit in [300, 500, 650, 750, 850]:
            app = _base_application(credit_score=credit)
            result = assess_risk(app)
            assert 0 <= result.risk_score <= 100

    def test_applicant_id_preserved(self):
        app = _base_application(applicant_id="abc-123")
        result = assess_risk(app)
        assert result.applicant_id == "abc-123"


class TestAPI:
    def test_health_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_assess_endpoint_low_risk(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        payload = {
            "applicant_id": "api-test-001",
            "age": 40,
            "annual_income": 100000,
            "loan_amount": 15000,
            "loan_term_months": 60,
            "credit_score": 800,
            "employment_status": "employed",
            "loan_purpose": "auto",
            "existing_debt": 0,
            "num_late_payments": 0,
        }
        response = client.post("/assess", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["risk_level"] == "low"
        assert data["recommendation"] == "Approve"

    def test_assess_endpoint_invalid_credit_score(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        payload = {
            "applicant_id": "bad-001",
            "age": 30,
            "annual_income": 50000,
            "loan_amount": 10000,
            "loan_term_months": 36,
            "credit_score": 200,  # invalid
            "employment_status": "employed",
            "loan_purpose": "personal",
        }
        response = client.post("/assess", json=payload)
        assert response.status_code == 422
