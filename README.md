# loan-risk-assessment

Advanced Loan Default Risk Assessment

A FastAPI service that evaluates the default risk of a loan application based on credit score, debt-to-income ratio, employment status, and payment history.

## Features

- Risk scoring (0–100) with **low / medium / high** classification
- Human-readable risk factors
- Approve / Approve with conditions / Decline recommendation
- REST API built with FastAPI

## Quick Start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API docs available at `http://localhost:8000/docs`.

## Run Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## API

### `POST /assess`

Submit a loan application and receive a risk assessment.

**Request body**

| Field | Type | Description |
|---|---|---|
| `applicant_id` | string | Unique identifier |
| `age` | int | Age (18–100) |
| `annual_income` | float | Annual income (USD) |
| `loan_amount` | float | Requested loan amount (USD) |
| `loan_term_months` | int | Loan term in months |
| `credit_score` | int | Credit score (300–850) |
| `employment_status` | enum | `employed` / `self_employed` / `unemployed` / `retired` |
| `loan_purpose` | enum | `home` / `auto` / `education` / `business` / `personal` |
| `existing_debt` | float | Existing annual debt obligations (USD) |
| `num_late_payments` | int | Number of late payments on record |

**Response**

```json
{
  "applicant_id": "app-001",
  "risk_level": "low",
  "risk_score": 12.5,
  "debt_to_income_ratio": 0.18,
  "loan_to_income_ratio": 0.25,
  "recommendation": "Approve",
  "factors": ["Excellent credit score (780)", "Healthy debt-to-income ratio (18.0%)"]
}
```
