# Loan Default Risk Assessment

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org) [![scikit-learn](https://img.shields.io/badge/scikit--learn-GradientBoosting-F7931E?style=for-the-badge)](https://scikit-learn.org) [![Flask](https://img.shields.io/badge/Flask-Gunicorn-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com) [![License](https://img.shields.io/badge/License-MIT-C9A84C?style=for-the-badge)](LICENSE)

[![Live API](https://img.shields.io/badge/Live%20API-Render-46E3B7?style=flat-square&logo=render)](https://loan-risk-assessment-nsdw.onrender.com/health) [![Dashboard](https://img.shields.io/badge/Dashboard-Live%20on%20Vercel-000000?style=flat-square&logo=vercel)](https://jameskoero-loan-risk-assessment-h84.vercel.app/)

**A live, deployed ML system for predicting loan default risk**, trained on the German Credit dataset with SMOTE class balancing and a cost-optimized decision threshold.

---

## Live Deployments

| Service | URL | Status |
|---|---|---|
| **API** | [loan-risk-assessment-nsdw.onrender.com](https://loan-risk-assessment-nsdw.onrender.com) | Live on Render |
| **Health Check** | [/health](https://loan-risk-assessment-nsdw.onrender.com/health) | Returns status ok |
| **Dashboard** | [jameskoero-loan-risk-assessment-h84.vercel.app](https://jameskoero-loan-risk-assessment-h84.vercel.app/) | Live on Vercel |

Note: the API runs on Render's free tier. The first request after idle may take 30-60s to cold-start.

---

## Overview

This repository trains and serves a GradientBoostingClassifier that predicts loan default probability from the German Credit dataset (OpenML id=31, 1,000 applicants, 20 features). SMOTE corrects the 70/30 class imbalance, and the decision threshold is chosen via a business cost matrix rather than the naive 0.5 cutoff.

---

## Model Performance

| Metric | Score | Notes |
|---|---|---|
| CV ROC-AUC (5-fold) | **0.916** | On SMOTE-balanced training folds |
| Test ROC-AUC (held-out 20%) | **0.791** | Real-world generalization estimate |
| Gini Coefficient | **0.582** | 2xAUC-1; Basel III minimum is 0.35 |
| Optimal threshold | **0.22** | Cost matrix: FN penalty 5x, FP penalty 1x |
| F1 at 0.5 threshold | 0.632 | Without cost-based tuning |
| Precision at 0.22 | 0.45 | 45% of flagged applicants actually default |
| Recall at 0.22 | 0.833 | Catches 83.3% of real defaulters |
| F1 at 0.22 | 0.585 | Deployed operating point |

Best hyperparameters: learning_rate=0.05, max_depth=4, n_estimators=100, subsample=0.8

The cost matrix (COST_FN=5, COST_FP=1) is deliberately conservative: missing a real defaulter is weighted 5x worse than wrongly flagging a good applicant. This trades precision for recall.

---

## Visualizations

Regenerated directly from the current SMOTE-corrected model, matching the metrics above.

### Confusion Matrix

![Confusion Matrix](https://raw.githubusercontent.com/jameskoero/loan-risk-assessment/main/images/confusion_matrix.png)

### ROC Curve

![ROC Curve](https://raw.githubusercontent.com/jameskoero/loan-risk-assessment/main/images/roc_curve.png)

### Feature Importance

![Feature Importance](https://raw.githubusercontent.com/jameskoero/loan-risk-assessment/main/images/feature_importance.png)

### SHAP Summary

![SHAP Summary](https://raw.githubusercontent.com/jameskoero/loan-risk-assessment/main/images/shap_summary.png)

### SHAP Waterfall

![SHAP Waterfall](https://raw.githubusercontent.com/jameskoero/loan-risk-assessment/main/images/shap_waterfall.png)

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| Machine Learning | scikit-learn 1.6.1, GradientBoostingClassifier |
| Imbalance handling | imbalanced-learn 0.14.2 (SMOTE) |
| API | Flask 3.1.3, gunicorn 23.0.0 |
| Frontend | Static HTML/JS, zero build step |
| Backend hosting | Render |
| Frontend hosting | Vercel |
| Training environment | Google Colab |

Version pinning matters here: an earlier deploy failed because the serving environment's scikit-learn and numpy versions did not match the versions used to train and pickle the model. requirements.txt is now pinned exactly to the training environment.

---

## Project Structure

    loan-risk-assessment/
      app.py                      Flask REST API, serves /health and /predict
      render.yaml                 Render Blueprint deployment config
      requirements.txt            Pinned to exact training-environment versions
      frontend/index.html         Zero-build static dashboard
      models/loan_risk_model.joblib   Trained pipeline (preprocess + SMOTE + GB)
      models/model_metadata.json      Confirmed metrics from the training run
      src/loan_risk_assessment.py     Training pipeline
      images/                     Generated evaluation charts
      docs/  notebooks/  tests/
      predict.py                  CLI batch scoring tool
      setup.py  LICENSE  README.md

---

## Local Setup

    git clone https://github.com/jameskoero/loan-risk-assessment.git
    cd loan-risk-assessment
    pip install -r requirements.txt
    python src/loan_risk_assessment.py

Run in Google Colab rather than Termux: the full pipeline needs more RAM than a phone terminal reliably provides.

---

## API Reference

Base URL: https://loan-risk-assessment-nsdw.onrender.com

**GET /health** returns a JSON object with status ok.

**POST /predict** accepts the German Credit Data schema (20 fields): checking_status, duration, credit_history, purpose, credit_amount, savings_status, employment, installment_commitment, personal_status, other_parties, residence_since, property_magnitude, age, other_payment_plans, housing, existing_credits, job, num_dependents, own_telephone, foreign_worker.

The response contains risk_score, a probability between 0.0 and 1.0. The dashboard flags anything at or above 0.22 as HIGH RISK.

Field codes follow the original UCI German Credit encoding (for example A11 means checking account below 0 DM). The dashboard dropdowns show the full human-readable mapping.

---

## Business Framing

Expected Loss is modelled as EL = PD x LGD x EAD, consistent with IFRS 9 staging. The Gini coefficient of 0.582 exceeds the Basel III regulatory minimum of 0.35 for an acceptable discriminatory model.

---

## Author

**James Koero**
ML Engineer, Kisumu, Kenya

[GitHub](https://github.com/jameskoero) | [LinkedIn](https://linkedin.com/in/jameskoero)

---

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.
