# 🏦 Loan Default Risk Assessment
### Advanced Credit Risk Modelling with Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/loan-risk-assessment/blob/main/loan_risk_assessment.ipynb)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-enabled-189ABB)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-explainability-9B59B6)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

A **production-grade, end-to-end machine learning system** for predicting loan default risk,
using the [German Credit Dataset (UCI / OpenML)](https://www.openml.org/d/31).

This project covers the **complete data science lifecycle** — raw data ingestion,
exploratory analysis, feature engineering, imbalance handling, multi-model training,
hyperparameter tuning, SHAP explainability, business-oriented threshold optimisation,
risk segmentation, and production model export.

> Built by **James Koero** — Junior ML Engineer, Kisumu, Kenya.  
> *"Making credit risk transparent, explainable, and financially optimal."*

---

## 🎯 Business Problem

Financial institutions lose billions annually to loan defaults. An accurate, early-warning
prediction system enables lenders to:

- ✅ Reject high-risk applicants **before** loan disbursal
- ✅ Offer **risk-adjusted interest rates** to borderline applicants
- ✅ Reduce **Non-Performing Loan (NPL) ratios**
- ✅ Comply with **responsible lending regulations**

---

## 🔬 Key Features

| Feature | Details |
|---------|---------|
| 📊 **Rich EDA** | 6 visualisation blocks: distributions, heatmaps, categorical analysis, box plots |
| 🛠️ **Feature Engineering** | 6 domain-driven new features (leverage ratio, age groups, loan term flags) |
| ⚗️ **ColumnTransformer Pipeline** | Clean, reproducible, production-ready preprocessing |
| ⚖️ **SMOTE** | Synthetic Minority Over-sampling to handle 70/30 class imbalance |
| 🤖 **5-Model Comparison** | LR → Decision Tree → Random Forest → Gradient Boosting → XGBoost |
| 🎯 **GridSearchCV Tuning** | Optimal hyperparameters with 5-fold stratified cross-validation |
| 🔍 **SHAP Explainability** | Feature importance + beeswarm plots (black-box → glass-box) |
| 💼 **Business Cost Matrix** | Threshold tuning to minimise total financial cost, not just accuracy |
| 📋 **Risk Scorecard** | 4-tier risk segmentation: LOW / MEDIUM-LOW / MEDIUM-HIGH / HIGH |
| 📈 **16 Publication-Quality Plots** | Saved automatically to `plots/` folder |
| 💾 **Joblib Model Export** | Full pipeline saved for production deployment |

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| Logistic Regression | ~0.72 | ~0.60 | ~0.56 | ~0.58 | ~0.76 |
| Decision Tree | ~0.70 | ~0.57 | ~0.60 | ~0.58 | ~0.69 |
| Random Forest | ~0.76 | ~0.64 | ~0.60 | ~0.62 | ~0.80 |
| Gradient Boosting | ~0.75 | ~0.65 | ~0.61 | ~0.63 | ~0.78 |
| **GB (Tuned) ✅** | **~0.76** | **~0.67** | **~0.62** | **~0.64** | **~0.79** |

> *Exact scores vary with random seed and grid search results.*  
> **Metric of choice: ROC-AUC** — most appropriate for imbalanced binary classification.

---

## 🗂️ Project Structure

```
loan-risk-assessment/
│
├── loan_risk_assessment.ipynb   ← Main Colab notebook (43 cells)
├── loan_risk_assessment.py      ← Standalone Python script
├── generate_notebook.py         ← Script to regenerate the .ipynb
├── requirements.txt             ← All dependencies
├── README.md                    ← This file
├── LICENSE                      ← MIT License
│
├── model/
│   ├── loan_risk_model.joblib   ← Trained pipeline (generated on run)
│   └── model_metadata.json      ← Model config and performance metrics
│
└── plots/                       ← Auto-generated visualisations (16 plots)
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

## 🚀 Quick Start

### ▶️ Option 1 — Google Colab (Recommended)

Click the badge at the top of this README, or:

```
https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/loan-risk-assessment/blob/main/loan_risk_assessment.ipynb
```

All data is auto-downloaded from OpenML — **no manual file upload required**.

### 💻 Option 2 — Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_GITHUB_USERNAME/loan-risk-assessment.git
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

## 📦 Dependencies

```txt
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

## 🔍 Notebook Walkthrough

| # | Section | Description |
|---|---------|-------------|
| 1 | Setup | Install and import all packages |
| 2 | Data Loading | Auto-download German Credit via OpenML |
| 3 | EDA | 5 visualisation blocks, distributions, correlations |
| 4 | Feature Engineering | 6 new domain-driven features |
| 5 | Preprocessing Pipeline | ColumnTransformer with imputation + encoding + scaling |
| 6 | SMOTE | Class imbalance correction |
| 7 | Model Training | 5 classifiers with full evaluation |
| 8 | Hyperparameter Tuning | GridSearchCV on Gradient Boosting |
| 9 | Evaluation Dashboard | ROC, PR, Confusion Matrices, Calibration, Learning Curves |
| 10 | SHAP Explainability | Global feature importance + beeswarm plots |
| 11 | Threshold Optimisation | Business cost matrix, optimal decision threshold |
| 12 | Risk Scorecard | 4-tier applicant segmentation |
| 13 | Feature Importance | Top-20 Gradient Boosting features |
| 14 | Model Export | Joblib pipeline + JSON metadata |
| 15 | Inference API | `predict_loan_risk()` function for new applicants |
| 16 | Summary | Results table, conclusions, future roadmap |

---

## 👤 Author

**James Koero**  
Junior ML Engineer | Kisumu, Kenya 🇰🇪  
B.Sc. Physics (Major) / Mathematics (Minor) — Moi University, 2012  
Self-taught ML practitioner since 2023

- 📧 *[your.email@example.com]*
- 💼 *[linkedin.com/in/your-profile]*
- 🐙 *[github.com/YOUR_GITHUB_USERNAME]*

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **Hans Hofmann** — creator of the German Credit Dataset (1994)
- **OpenML** — for hosting the dataset publicly
- **Lundberg & Lee (2017)** — SHAP: A Unified Approach to Interpreting Model Predictions
- **Chawla et al. (2002)** — SMOTE: Synthetic Minority Over-sampling Technique

---

*⭐ If you find this project useful, please give it a star on GitHub!*
