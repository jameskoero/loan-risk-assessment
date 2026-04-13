# Loan Default Risk Assessment

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/loan-risk-assessment/blob/main/loan_risk_assessment.ipynb)

A production-grade ML pipeline for credit risk scoring on the **German Credit dataset** (OpenML `credit-g`).

---

## ✨ Features

| Capability | Details |
|---|---|
| **Data validation** | Schema, range, cardinality, consistency & duplicate checks via `data_validation.py` |
| **Class balancing** | SMOTE oversampling on the training split |
| **5 classifiers** | Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM |
| **SHAP explainability** | Bar chart + beeswarm for the best model |
| **Business cost matrix** | FN=5×FP; threshold optimised to minimise total cost |
| **4-tier risk scorecard** | Low / Medium / High / Very High |
| **16 plots** | Auto-saved to `plots/` |
| **Model export** | `loan_risk_model.joblib` + `model_metadata.json` |

---

## 📊 Model Results (German Credit — AUC)

| Model | ROC-AUC |
|---|---|
| Logistic Regression | ~0.76 |
| Random Forest | ~0.80 |
| Gradient Boosting | ~0.81 |
| XGBoost | ~0.80 |
| LightGBM | **~0.82** |

---

## 📁 Project Structure

```
loan-risk-assessment/
├── data_validation.py        # Data validation rules (schema, range, consistency)
├── loan_risk_assessment.py   # Full ML pipeline script
├── loan_risk_assessment.ipynb# Colab-ready notebook
├── generate_notebook.py      # Regenerates the .ipynb from the script
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quickstart

```bash
pip install -r requirements.txt
python loan_risk_assessment.py
```

Or open the notebook in Google Colab using the badge above.

---

## 🛡️ Data Validation

`data_validation.py` enforces the following rule categories before the pipeline runs:

- **Schema** — all 20 expected columns present with correct dtype families
- **Completeness** — no missing values (configurable threshold)
- **Range** — numeric features within documented domain bounds (e.g. `age` 18–75, `credit_amount` 250–18 424)
- **Cardinality** — categorical features only contain known category strings
- **Consistency** — cross-field constraints (e.g. `age ≥ 18`, `credit_amount > 0`, `duration > 0`)
- **Duplicates** — no exact duplicate rows

```python
from data_validation import validate_dataframe

report = validate_dataframe(df, raise_on_error=True)
```

---

## 📄 License

MIT © 2025 James Koero
