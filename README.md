# 🏦 Advanced Loan Default Risk Assessment

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/loan-risk-assessment/blob/main/loan_risk_assessment.ipynb)

A production-grade machine learning pipeline for credit risk scoring. Built on the German Credit dataset from OpenML, the project covers the full ML lifecycle — from raw data to an exported, deployment-ready model — with rigorous evaluation, explainability, and business-cost optimisation.

---

## ✨ Features

- 🔄 **SMOTE oversampling** to correct class imbalance before training
- 🤖 **5 classifiers** compared head-to-head: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM
- 🔍 **SHAP explainability** — bar plot and beeswarm for the best model
- 💰 **Business cost matrix** with FN penalty 5× FP, and per-model optimal threshold search
- 🏷️ **4-tier risk scorecard** (Low / Medium / High / Very High) from predicted probabilities
- 📈 **Calibration curves** for all 5 models and **learning curves** for the best model
- 🖼️ **16 professional plots** auto-saved to `plots/` at 150 dpi
- 📦 **Production export**: `loan_risk_model.joblib` + `model_metadata.json`

---

## 📊 Dataset

**German Credit Data** from [OpenML](https://www.openml.org/d/31)

| Property | Value |
|---|---|
| Samples | 1 000 |
| Features | 20 (7 numeric, 13 categorical) |
| Target | `good` / `bad` credit risk |
| Class ratio | 70 % good · 30 % bad |

Auto-downloaded via `sklearn.datasets.fetch_openml(name='credit-g', version=1)`.

---

## 🤖 Models

| Model | Approx. AUC |
|---|---|
| Logistic Regression | ~0.72 |
| Random Forest | ~0.78 |
| Gradient Boosting | ~0.79 |
| XGBoost | ~0.80 |
| LightGBM | ~0.79 |

---

## 🚀 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python loan_risk_assessment.py
```

Plots are written to `plots/`, the trained model to `loan_risk_model.joblib`, and evaluation metadata to `model_metadata.json`.

To regenerate the Jupyter notebook from the script:

```bash
python generate_notebook.py
```

---

## 📁 Project Structure

```
loan-risk-assessment/
├── loan_risk_assessment.py   # Main ML pipeline script
├── generate_notebook.py      # Rebuilds the .ipynb from the script
├── loan_risk_assessment.ipynb  # Colab-ready notebook
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── README.md                 # This file
├── plots/                    # Auto-generated PNG figures (16 files)
├── loan_risk_model.joblib    # Exported best model (after running script)
└── model_metadata.json       # Model metrics & metadata (after running script)
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) © 2025 James Koero.
