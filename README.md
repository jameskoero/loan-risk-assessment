# 🏦 Loan Default Risk Assessment

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/loan-risk-assessment/blob/main/notebooks/loan_risk_assessment.ipynb)

> **Advanced Credit Risk Modelling with Machine Learning**  
> Author: James Koero | Junior ML Engineer | Kisumu, Kenya

---

## 📋 Overview

This project is a **production-grade, end-to-end machine learning pipeline** that predicts whether a loan applicant will default, using the [German Credit Dataset (OpenML ID 31)](https://www.openml.org/d/31). It covers the complete data-science lifecycle:

- Exploratory Data Analysis (EDA)
- Feature engineering
- Class-imbalance handling with SMOTE
- Multi-model training and comparison
- Hyperparameter tuning via GridSearchCV
- Business-cost threshold optimisation
- Model explainability (SHAP values)
- Model serialisation with joblib

Financial institutions lose billions annually to loan defaults. Early, accurate risk prediction helps lenders:

- Reject high-risk applicants before disbursal
- Offer risk-adjusted interest rates
- Reduce non-performing loan (NPL) ratios
- Comply with responsible lending regulations

---

## 📂 Project Structure

```
loan-risk-assessment/
├── README.md                         # This file
├── LICENSE                           # MIT licence
├── requirements.txt                  # Pinned Python dependencies
├── .gitignore                        # Python + ML artifact exclusions
├── setup.py                          # Package installation
├── src/
│   ├── __init__.py
│   └── loan_risk_assessment.py       # Core ML pipeline
├── notebooks/
│   └── loan_risk_assessment.ipynb    # Interactive Jupyter notebook
├── tests/
│   ├── __init__.py
│   └── test_loan_risk_assessment.py  # Pytest test suite
├── docs/
│   ├── API.md                        # API reference
│   ├── ARCHITECTURE.md               # System design
│   ├── MODEL_PERFORMANCE.md          # Benchmark results
│   └── CONTRIBUTING.md              # Contribution guidelines
├── plots/                            # Generated plots (git-ignored)
└── models/                           # Saved model artifacts (git-ignored)
```

---

## 🗂️ Dataset

| Property | Value |
|----------|-------|
| **Name** | German Credit Data |
| **Source** | [OpenML ID 31](https://www.openml.org/d/31) |
| **Rows** | 1 000 loan applications |
| **Features** | 20 (numeric + categorical) |
| **Target** | `default` — 1 = bad credit risk, 0 = good credit risk |
| **Class balance** | 70 % good / 30 % bad |

The dataset is **automatically downloaded** at runtime via `sklearn.datasets.fetch_openml` — no manual download needed.

---

## ✨ Features & Capabilities

| Module | What it does |
|--------|-------------|
| `load_data` | Downloads German Credit data from OpenML |
| `run_eda` | Target distribution, numeric histograms, correlation heatmap |
| `engineer_features` | 6 domain-inspired features (debt ratio, age groups, …) |
| `build_preprocessor` | `ColumnTransformer` pipeline: impute → scale / one-hot encode |
| `train_models` | Trains 5 classifiers; prints accuracy / F1 / AUC table |
| `tune_best_model` | `GridSearchCV` over Gradient Boosting hyperparameters |
| `evaluation_plots` | ROC, PR, confusion matrices, metrics bar chart |
| `optimise_threshold` | Business cost-matrix threshold search |
| `plot_feature_importance` | Top-20 feature importances (bar chart) |
| `save_model` | Exports `Pipeline` + JSON metadata via joblib |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/jameskoero/loan-risk-assessment.git
cd loan-risk-assessment
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install as a package

```bash
pip install -e .
```

---

## ⚡ Quick Start

### Run the full pipeline

```bash
python -m src.loan_risk_assessment
```

or, from the repo root:

```bash
python src/loan_risk_assessment.py
```

### Use as a Python module

```python
from src.loan_risk_assessment import load_data, engineer_features, build_preprocessor

df = load_data()
df = engineer_features(df)
X = df.drop(columns=["default"])
y = df["default"]
preprocessor, num_features, cat_features = build_preprocessor(X)
```

### Open the Jupyter notebook

```bash
jupyter notebook notebooks/loan_risk_assessment.ipynb
```

Or open directly in [Google Colab](https://colab.research.google.com/github/jameskoero/loan-risk-assessment/blob/main/notebooks/loan_risk_assessment.ipynb).

---

## 📊 Model Performance

Typical results on the German Credit holdout set (80/20 split, `random_state=42`):

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| Logistic Regression | ~0.75 | ~0.57 | ~0.79 |
| Decision Tree | ~0.72 | ~0.57 | ~0.69 |
| Random Forest | ~0.78 | ~0.62 | ~0.82 |
| Gradient Boosting | ~0.79 | ~0.63 | ~0.84 |
| XGBoost/HGB | ~0.79 | ~0.63 | ~0.84 |
| **GB (Tuned)** | **~0.80** | **~0.65** | **~0.85** |

> See [`docs/MODEL_PERFORMANCE.md`](docs/MODEL_PERFORMANCE.md) for full benchmark details.

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 🤝 Contributing

Contributions are welcome! Please read [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) before opening a PR.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
