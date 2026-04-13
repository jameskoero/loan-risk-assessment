# loan-risk-assessment

Advanced Loan Default Risk Assessment

## Overview

`loan_risk_evaluation.py` provides the `evaluate_loan_risk` function — a
transparent, rule-based scoring engine (with optional ML model blending) that
assesses the default risk of a single loan application.

## Quick start

```python
from loan_risk_evaluation import LoanApplication, evaluate_loan_risk

app = LoanApplication(
    credit_score=720,
    annual_income=80_000,
    loan_amount=15_000,
    loan_term_months=48,
    existing_debt=8_000,
    employment_years=5,
    num_late_payments=0,
    has_collateral=False,
    purpose="car",
)

result = evaluate_loan_risk(app)
print(result.score)          # e.g. 77.35
print(result.tier)           # "MEDIUM"
print(result.recommendation) # "REVIEW"
print(result.messages)       # ["Unsecured loan; no collateral provided."]
```

## Scoring factors

| Factor | Weight | Description |
|--------|--------|-------------|
| Credit score | 35 % | Maps 300–850 range to 0–100 |
| Debt-to-income (DTI) | 25 % | Penalises DTI > 0.36 |
| Loan-to-income (LTI) | 15 % | Penalises LTI > 2× income |
| Employment stability | 10 % | Rewards ≥ 5 years employment |
| Payment history | 10 % | Penalises late payments |
| Collateral | 5 % | Bonus for secured loans |

## Risk tiers

| Tier | Score range | Recommendation |
|------|-------------|----------------|
| LOW | 80 – 100 | APPROVE |
| MEDIUM | 60 – 79 | REVIEW |
| HIGH | 40 – 59 | REVIEW |
| VERY_HIGH | 0 – 39 | DECLINE |

## ML model blending

Pass any scikit-learn–compatible classifier (with a `predict_proba` method)
as the `model` keyword argument.  The final score is the 50/50 average of the
rule-based score and the model's survival probability.

## Project structure

```
loan-risk-assessment/
├── loan_risk_evaluation.py   # Core evaluation function
├── tests/
│   └── test_loan_risk_evaluation.py
├── requirements.txt
└── README.md
```

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

## License

MIT © James Koero

