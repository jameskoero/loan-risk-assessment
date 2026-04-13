# Loan Default Risk Assessment System

A production-quality machine learning pipeline for predicting loan default risk. The system generates synthetic loan data, engineers financial risk features, trains and compares multiple classifiers, evaluates model performance, and produces actionable risk scores with lending recommendations.

---

## Features

- **Synthetic Data Generation** — Realistic loan dataset with ~20% default rate, controlled by a logistic model seeded from real-world credit features.
- **Feature Engineering** — Derived indicators: monthly payment, debt burden, credit-utilisation risk, loan-to-income ratio, payment-to-income ratio, and a composite raw risk score.
- **Multiple Classifiers** — Logistic Regression, Random Forest, Gradient Boosting, and XGBoost trained in parallel and compared by cross-validated AUC-ROC.
- **Class Imbalance Handling** — SMOTE oversampling via `imbalanced-learn`.
- **Comprehensive Evaluation** — Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR, Brier Score, KS Statistic, plus ROC, PR, calibration, and confusion-matrix plots.
- **Risk Scoring** — Batch and single-loan scoring with four risk tiers and lending recommendations.
- **Persistence** — All models and the preprocessor are saved with `joblib` for later reuse.

---

## Installation

```bash
git clone <repo-url>
cd loan-risk-assessment

pip install -r requirements.txt
```

Python 3.10+ is recommended.

---

## Quick Start

```bash
# Run the full pipeline (default: 10 000 samples)
python main.py

# Customise sample size and output directory
python main.py --samples 5000 --output-dir results --random-state 99
```

Output artefacts are written to `output/` by default:

| Path | Contents |
|------|----------|
| `output/loan_data_raw.csv` | Raw generated dataset |
| `output/loan_data_engineered.csv` | Dataset after feature engineering |
| `output/preprocessor.pkl` | Fitted preprocessing pipeline |
| `output/models/` | Trained model files (`*.pkl`) |
| `output/model_comparison.csv` | Test-set metrics for all models |
| `output/reports/` | Plots and JSON metrics for the best model |

---

## Project Structure

```
loan-risk-assessment/
├── main.py                     # End-to-end pipeline entry point
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_generator.py       # Synthetic dataset creation
│   ├── data_preprocessing.py   # Imputation, encoding, scaling
│   ├── feature_engineering.py  # Derived financial features
│   ├── model_training.py       # Multi-model training + SMOTE
│   ├── model_evaluation.py     # Metrics, plots, comparison table
│   └── risk_scoring.py         # Probability → risk tier + recommendation
└── tests/
    ├── __init__.py
    └── test_pipeline.py        # pytest unit & smoke tests
```

---

## Risk Categories

| Category  | P(default) range | Recommendation       |
|-----------|-----------------|----------------------|
| Low       | < 0.20          | Approve              |
| Medium    | 0.20 – 0.40     | Review               |
| High      | 0.40 – 0.60     | Decline              |
| Very High | ≥ 0.60          | Decline – High Risk  |

---

## Model Performance (typical, 10 000 samples)

| Model               | AUC-ROC | F1   | Precision | Recall |
|---------------------|---------|------|-----------|--------|
| XGBoost             | ~0.92   | ~0.70| ~0.72     | ~0.68  |
| Gradient Boosting   | ~0.91   | ~0.68| ~0.71     | ~0.66  |
| Random Forest       | ~0.90   | ~0.67| ~0.70     | ~0.64  |
| Logistic Regression | ~0.85   | ~0.60| ~0.64     | ~0.56  |

Exact numbers vary with random seed and sample size.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Contributing

1. Fork the repository and create a feature branch.
2. Ensure `pytest tests/ -v` passes before opening a pull request.
3. Follow PEP 8 style guidelines.

## License

MIT
