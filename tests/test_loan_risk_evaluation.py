"""
Tests for loan_risk_evaluation.py
"""
import pytest
from loan_risk_evaluation import (
    LoanApplication,
    RiskAssessment,
    evaluate_loan_risk,
    _score_credit,
    _score_debt_to_income,
    _score_loan_to_income,
    _score_employment,
    _score_payment_history,
    _score_collateral,
    _determine_tier,
    _determine_recommendation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def low_risk_application():
    return LoanApplication(
        credit_score=780,
        annual_income=100_000,
        loan_amount=20_000,
        loan_term_months=60,
        existing_debt=5_000,
        employment_years=8,
        num_late_payments=0,
        has_collateral=True,
        purpose="car",
    )


@pytest.fixture
def high_risk_application():
    return LoanApplication(
        credit_score=350,
        annual_income=30_000,
        loan_amount=40_000,
        loan_term_months=12,
        existing_debt=25_000,
        employment_years=0.3,
        num_late_payments=7,
        has_collateral=False,
        purpose="personal",
    )


@pytest.fixture
def medium_risk_application():
    return LoanApplication(
        credit_score=620,
        annual_income=60_000,
        loan_amount=15_000,
        loan_term_months=36,
        existing_debt=10_000,
        employment_years=3,
        num_late_payments=1,
        has_collateral=False,
        purpose="education",
    )


# ---------------------------------------------------------------------------
# Sub-score helpers
# ---------------------------------------------------------------------------

class TestScoreCredit:
    def test_minimum_score(self):
        assert _score_credit(300) == 0.0

    def test_maximum_score(self):
        assert _score_credit(850) == 100.0

    def test_midpoint(self):
        score = _score_credit(575)
        assert 45 < score < 55

    def test_score_is_monotonically_increasing(self):
        scores = [_score_credit(cs) for cs in range(300, 851, 50)]
        assert scores == sorted(scores)


class TestScoreDebtToIncome:
    def test_excellent_dti(self):
        # DTI = (0 + 10_000) / 100_000 = 0.10 → excellent
        assert _score_debt_to_income(100_000, 0, 10_000) == 100.0

    def test_good_dti(self):
        # DTI ≈ 0.40
        score = _score_debt_to_income(100_000, 0, 40_000)
        assert score == 75.0

    def test_fair_dti(self):
        # DTI ≈ 0.46
        score = _score_debt_to_income(100_000, 0, 46_000)
        assert score == 50.0

    def test_poor_dti(self):
        # DTI > 0.5
        score = _score_debt_to_income(50_000, 0, 40_000)
        assert score < 50.0

    def test_zero_income(self):
        assert _score_debt_to_income(0, 0, 10_000) == 0.0


class TestScoreLoanToIncome:
    def test_low_lti(self):
        assert _score_loan_to_income(100_000, 100_000) == 100.0

    def test_moderate_lti(self):
        score = _score_loan_to_income(100_000, 300_000)
        assert score == 70.0

    def test_high_lti(self):
        score = _score_loan_to_income(100_000, 500_000)
        assert score == 40.0

    def test_very_high_lti(self):
        score = _score_loan_to_income(100_000, 2_000_000)
        assert score == 0.0

    def test_zero_income(self):
        assert _score_loan_to_income(0, 10_000) == 0.0


class TestScoreEmployment:
    def test_stable_employment(self):
        assert _score_employment(10) == 100.0

    def test_moderate_employment(self):
        assert _score_employment(3) == 70.0

    def test_short_employment(self):
        assert _score_employment(1.5) == 45.0

    def test_very_short_employment(self):
        assert _score_employment(0.6) == 20.0

    def test_no_employment(self):
        assert _score_employment(0) == 0.0


class TestScorePaymentHistory:
    def test_perfect_history(self):
        assert _score_payment_history(0) == 100.0

    def test_one_late_payment(self):
        assert _score_payment_history(1) == 70.0

    def test_few_late_payments(self):
        assert _score_payment_history(2) == 40.0

    def test_many_late_payments(self):
        assert _score_payment_history(5) == 15.0

    def test_severe_history(self):
        assert _score_payment_history(10) == 0.0


class TestScoreCollateral:
    def test_with_collateral(self):
        assert _score_collateral(True) == 100.0

    def test_without_collateral(self):
        assert _score_collateral(False) == 50.0


class TestDetermineTier:
    def test_low_risk_tier(self):
        assert _determine_tier(85) == "LOW"

    def test_medium_risk_tier(self):
        assert _determine_tier(70) == "MEDIUM"

    def test_high_risk_tier(self):
        assert _determine_tier(50) == "HIGH"

    def test_very_high_risk_tier(self):
        assert _determine_tier(20) == "VERY_HIGH"

    def test_boundary_medium(self):
        assert _determine_tier(80) == "LOW"

    def test_boundary_high(self):
        assert _determine_tier(60) == "MEDIUM"

    def test_boundary_very_high(self):
        assert _determine_tier(40) == "HIGH"

    def test_perfect_score(self):
        assert _determine_tier(100) == "LOW"


class TestDetermineRecommendation:
    def test_low_tier_approve(self):
        assert _determine_recommendation("LOW") == "APPROVE"

    def test_medium_tier_review(self):
        assert _determine_recommendation("MEDIUM") == "REVIEW"

    def test_high_tier_review(self):
        assert _determine_recommendation("HIGH") == "REVIEW"

    def test_very_high_tier_decline(self):
        assert _determine_recommendation("VERY_HIGH") == "DECLINE"


# ---------------------------------------------------------------------------
# evaluate_loan_risk integration tests
# ---------------------------------------------------------------------------

class TestEvaluateLoanRisk:
    def test_returns_risk_assessment(self, low_risk_application):
        result = evaluate_loan_risk(low_risk_application)
        assert isinstance(result, RiskAssessment)

    def test_score_in_range(self, low_risk_application):
        result = evaluate_loan_risk(low_risk_application)
        assert 0.0 <= result.score <= 100.0

    def test_low_risk_approved(self, low_risk_application):
        result = evaluate_loan_risk(low_risk_application)
        assert result.tier == "LOW"
        assert result.recommendation == "APPROVE"

    def test_high_risk_declined(self, high_risk_application):
        result = evaluate_loan_risk(high_risk_application)
        assert result.tier == "VERY_HIGH"
        assert result.recommendation == "DECLINE"

    def test_medium_risk_review(self, medium_risk_application):
        result = evaluate_loan_risk(medium_risk_application)
        assert result.tier in ("MEDIUM", "HIGH")
        assert result.recommendation == "REVIEW"

    def test_factors_present(self, low_risk_application):
        result = evaluate_loan_risk(low_risk_application)
        assert "credit_score" in result.factors
        assert "debt_to_income" in result.factors
        assert "loan_to_income" in result.factors
        assert "employment" in result.factors
        assert "payment_history" in result.factors
        assert "collateral" in result.factors

    def test_messages_list(self, low_risk_application):
        result = evaluate_loan_risk(low_risk_application)
        assert isinstance(result.messages, list)
        assert len(result.messages) > 0

    def test_high_risk_messages_contain_issues(self, high_risk_application):
        result = evaluate_loan_risk(high_risk_application)
        combined = " ".join(result.messages).lower()
        assert any(
            kw in combined
            for kw in ("credit", "debt", "late", "collateral", "employment", "loan")
        )

    def test_low_risk_no_major_issues(self, low_risk_application):
        result = evaluate_loan_risk(low_risk_application)
        assert "No major risk factors identified." in result.messages

    def test_tier_consistent_with_score(self, low_risk_application):
        result = evaluate_loan_risk(low_risk_application)
        low, high = (0, 100)
        for tier, (lo, hi) in {
            "LOW": (80, 101), "MEDIUM": (60, 80),
            "HIGH": (40, 60), "VERY_HIGH": (0, 40),
        }.items():
            if result.tier == tier:
                assert lo <= result.score < hi or (tier == "LOW" and result.score >= 80)

    def test_low_risk_score_higher_than_high_risk(
        self, low_risk_application, high_risk_application
    ):
        low_result = evaluate_loan_risk(low_risk_application)
        high_result = evaluate_loan_risk(high_risk_application)
        assert low_result.score > high_result.score


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_credit_score_low(self):
        app = LoanApplication(
            credit_score=100, annual_income=50_000,
            loan_amount=10_000, loan_term_months=24)
        with pytest.raises(ValueError, match="credit_score"):
            evaluate_loan_risk(app)

    def test_invalid_credit_score_high(self):
        app = LoanApplication(
            credit_score=900, annual_income=50_000,
            loan_amount=10_000, loan_term_months=24)
        with pytest.raises(ValueError, match="credit_score"):
            evaluate_loan_risk(app)

    def test_negative_income(self):
        app = LoanApplication(
            credit_score=700, annual_income=-1,
            loan_amount=10_000, loan_term_months=24)
        with pytest.raises(ValueError, match="annual_income"):
            evaluate_loan_risk(app)

    def test_zero_loan_amount(self):
        app = LoanApplication(
            credit_score=700, annual_income=50_000,
            loan_amount=0, loan_term_months=24)
        with pytest.raises(ValueError, match="loan_amount"):
            evaluate_loan_risk(app)

    def test_negative_loan_amount(self):
        app = LoanApplication(
            credit_score=700, annual_income=50_000,
            loan_amount=-5_000, loan_term_months=24)
        with pytest.raises(ValueError, match="loan_amount"):
            evaluate_loan_risk(app)

    def test_zero_loan_term(self):
        app = LoanApplication(
            credit_score=700, annual_income=50_000,
            loan_amount=10_000, loan_term_months=0)
        with pytest.raises(ValueError, match="loan_term_months"):
            evaluate_loan_risk(app)

    def test_negative_existing_debt(self):
        app = LoanApplication(
            credit_score=700, annual_income=50_000,
            loan_amount=10_000, loan_term_months=24, existing_debt=-100)
        with pytest.raises(ValueError, match="existing_debt"):
            evaluate_loan_risk(app)

    def test_negative_employment_years(self):
        app = LoanApplication(
            credit_score=700, annual_income=50_000,
            loan_amount=10_000, loan_term_months=24, employment_years=-1)
        with pytest.raises(ValueError, match="employment_years"):
            evaluate_loan_risk(app)

    def test_negative_late_payments(self):
        app = LoanApplication(
            credit_score=700, annual_income=50_000,
            loan_amount=10_000, loan_term_months=24, num_late_payments=-1)
        with pytest.raises(ValueError, match="num_late_payments"):
            evaluate_loan_risk(app)


# ---------------------------------------------------------------------------
# Model blending tests
# ---------------------------------------------------------------------------

class FakeModel:
    """Minimal scikit-learn–compatible model stub."""

    def __init__(self, default_proba: float):
        self._default_proba = default_proba

    def predict_proba(self, X):
        return [[1 - self._default_proba, self._default_proba]]


class TestModelBlending:
    def test_model_low_default_proba_boosts_score(self, medium_risk_application):
        no_model = evaluate_loan_risk(medium_risk_application)
        with_good_model = evaluate_loan_risk(
            medium_risk_application, model=FakeModel(0.05))
        assert with_good_model.score >= no_model.score

    def test_model_high_default_proba_lowers_score(self, medium_risk_application):
        no_model = evaluate_loan_risk(medium_risk_application)
        with_bad_model = evaluate_loan_risk(
            medium_risk_application, model=FakeModel(0.95))
        assert with_bad_model.score <= no_model.score

    def test_blended_score_in_valid_range(self, low_risk_application):
        result = evaluate_loan_risk(low_risk_application, model=FakeModel(0.5))
        assert 0.0 <= result.score <= 100.0
