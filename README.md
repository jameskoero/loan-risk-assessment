# Loan Default Risk Assessment

### Advanced Credit Risk Modelling with Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jmskoero/loan-risk-assessment/blob/main/loan_risk_assessment.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-enabled-189ABB)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-explainability-9B59B6)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

A production-grade, end-to-end machine learning system for predicting loan default probability, built on the [German Credit Dataset (UCI / OpenML)](https://www.openml.org/d/31).

This project demonstrates advanced techniques across the full data science lifecycle: automated preprocessing pipelines, class imbalance correction, multi-model benchmarking, hyperparameter tuning, SHAP-based model explainability, and business-oriented financial threshold optimisation. The result is a system that does not just predict — it explains its decisions in terms that credit analysts and loan officers can act on.

> Built by **James Koero** — Junior ML Engineer, Kisumu, Kenya.
> "Making credit risk transparent, explainable, and financially optimal."

---

## Business Problem

Financial institutions lose billions annually to loan defaults. Simple rule-based screening misses complex, non-linear patterns in applicant data. An accurate, interpretable early-warning prediction system enables lenders to:

- Reject high-risk applicants before loan disbursal
- Offer risk-adjusted interest rates to borderline applicants
- Reduce Non-Performing Loan (NPL) ratios
- Minimise the financial cost of False Negatives — missed defaults that become bad debt
- Comply with responsible lending regulations through model transparency

---

## Key Features

| Feature | Details |
|---|---|
| End-to-End Automated Pipeline | Built using Scikit-Learn ColumnTransformer and Pipeline for clean, reproducible, production-ready preprocessing |
| SMOTE Imbalance Correction | Synthetic Minority Over-sampling to address the heavy class skew in credit datasets, ensuring high-risk applicants are reliably identified |
| 5-Model Benchmarking | Logistic Regression, Random Forest, Gradient Boosting, SVM, and KNN compared on a consistent evaluation framework |
| Hyperparameter Tuning | GridSearchCV on Gradient Boosting with 5-fold stratified cross-validation, optimising for ROC-AUC |
| Explainable AI (SHAP) | SHAP summary and beeswarm plots reveal why each applicant was flagged — turning a black-box model into a transparent decision tool |
| Financial Threshold Optimisation | Custom cost-matrix analysis identifies the optimal decision threshold that minimises total financial loss, not just classification error |
| Rich EDA | 6 visualisation blocks covering distributions, heatmaps, categorical default rates, and demographic breakdowns |
| Feature Engineering | 6 domain-driven engineered features including leverage ratio, age groups, and loan term flags |
| Risk Scorecard | 4-tier applicant segmentation: LOW / MEDIUM-LOW / MEDIUM-HIGH / HIGH |
| 16 Publication-Quality Plots | Saved automatically to the `plots/` folder on run |
| Joblib Model Export | Full serialised pipeline ready for API integration or production deployment |

---

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.72 | ~0.60 | ~0.56 | ~0.58 | ~0.76 |
| Decision Tree | ~0.70 | ~0.57 | ~0.60 | ~0.58 | ~0.69 |
| Random Forest | ~0.76 | ~0.64 | ~0.60 | ~0.62 | ~0.80 |
| Gradient Boosting | ~0.75 | ~0.65 | ~0.61 | ~0.63 | ~0.78 |
| **GB (Tuned)** | **~0.76** | **~0.67** | **~0.62** | **~0.64** | **~0.79** |

Exact scores vary with random seed and grid search results. The primary evaluation metric is **ROC-AUC**, which is most appropriate for imbalanced binary classification tasks. Final threshold selection is driven by the financial cost matrix rather than accuracy alone.

---

## Project Structure

```
loan-risk-assessment/
│
├── loan_risk_assessment.ipynb   <- Main Colab notebook (43 cells)
├── loan_risk_assessment.py      <- Standalone Python script
├── generate_notebook.py         <- Script to regenerate the .ipynb
├── requirements.txt             <- All dependencies
├── README.md                    <- This file
├── LICENSE                      <- MIT License
│
├── model/
│   ├── loan_risk_model.joblib   <- Trained pipeline (generated on run)
│   └── model_metadata.json      <- Model config and performance metrics
│
└── plots/                       <- Auto-generated visualisations (16 plots)
    ├── 01_target_distribution.png
    ├── 02_numeric_distributions.png
    ├── 03_correlation_heatmap.png
    ├── 04_categorical_default_rates.png
    ├── 05_boxplots_vs_default.png
    ├── 06_smote_comparison.png
    ├── 07_roc_pr_curves.png
    ├── 08_confusion_matrices.png
    ├── 09_metrics_comparison.png
    ├── 10_calibration_curves.png
    ├── 11_learning_curves.png
    ├── 12_shap_importance.png
    ├── 13_shap_beeswarm.png
    ├── 14_threshold_optimisation.png
    ├── 15_risk_scorecard.png
    └── 16_feature_importance.png
```

---

## Notebook Walkthrough

The main notebook (`loan_risk_assessment.ipynb`) is structured into six logical phases:

| Phase | Section | Description |
|---|---|---|
| 1 | Data Acquisition and Labelling | Auto-download from OpenML; map raw codes to human-readable categories (e.g. "A11" becomes "no checking account") |
| 2 | Exploratory Data Analysis | Visualise default rates across demographics, credit amounts, and loan durations using Seaborn and Matplotlib |
| 3 | Feature Engineering and Preprocessing | 6 engineered features; ColumnTransformer pipeline with scaling, encoding, and imputation in a non-leaking validation framework |
| 4 | SMOTE and Model Training | Class imbalance correction followed by training and benchmarking 5 classifier architectures |
| 5 | Hyperparameter Tuning and Evaluation | GridSearchCV on Gradient Boosting; full evaluation dashboard including ROC, PR curves, calibration, and learning curves |
| 6 | Explainability, Threshold Optimisation, and Deployment | SHAP global and local interpretability; cost-matrix threshold selection; risk scorecard; joblib pipeline export and `predict_loan_risk()` inference function |

---

## Quick Start

### Option 1 — Google Colab (Recommended)

Click the badge at the top of this README, or open the notebook directly:

```
https://colab.research.google.com/github/Jmskoero/loan-risk-assessment/blob/main/loan_risk_assessment.ipynb
```

All data is auto-downloaded from OpenML. No manual file upload is required.

### Option 2 — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Jmskoero/loan-risk-assessment.git
cd loan-risk-assessment

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Run the standalone script
python loan_risk_assessment.py

# 3b. Or regenerate and open the notebook
python generate_notebook.py
jupyter notebook loan_risk_assessment.ipynb
```

---

## Dependencies

```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
imbalanced-learn>=0.11.0
shap>=0.44.0
xgboost>=1.7.0
lightgbm>=4.0.0
joblib>=1.3.0
jupyter>=1.0.0
```

---

## Author

**James Koero**
Junior ML Engineer | Kisumu, Kenya
B.Sc. Physics (Major) / Mathematics (Minor) — Moi University, 2012
Self-taught ML practitioner since 2023

- Email: [jmskoero@gmail.com](mailto:jmskoero@gmail.com)
- LinkedIn: [linkedin.com/in/your-profile] *(to be updated)*
- GitHub: [James Koero](https://github.com/Jmskoero)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **Hans Hofmann** — creator of the German Credit Dataset (1994)
- **OpenML** — for hosting the dataset publicly
- **Lundberg & Lee (2017)** — SHAP: A Unified Approach to Interpreting Model Predictions
- **Chawla et al. (2002)** — SMOTE: Synthetic Minority Over-sampling Technique
