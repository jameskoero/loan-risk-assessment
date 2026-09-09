# 🏦 Loan Default Risk Assessment



![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)




![ML](https://img.shields.io/badge/ML-GradientBoosting-orange)




![API](https://img.shields.io/badge/API-Flask-lightblue)




![License](https://img.shields.io/badge/License-MIT-lightgrey)




![Status](https://img.shields.io/badge/Status-Live-success)




![Test ROC--AUC](https://img.shields.io/badge/Test_ROC--AUC-0.791-success)




![CV ROC--AUC](https://img.shields.io/badge/CV_ROC--AUC-0.916-success)



> **Live, deployed ML system for predicting loan default risk**, trained on the German Credit dataset with SMOTE class balancing and a business-cost-optimized decision threshold.

---

## 🔴 Live Deployments

| Service | URL | Status |
|---|---|---|
| **API** | [loan-risk-assessment-nsdw.onrender.com](https://loan-risk-assessment-nsdw.onrender.com) | ✅ Live — Flask + gunicorn on Render |
| **Health Check** | [/health](https://loan-risk-assessment-nsdw.onrender.com/health) | ✅ `{"status": "ok"}` confirmed |
| **Frontend** | [jameskoero-loan-risk-assessment-h84.vercel.app](https://jameskoero-loan-risk-assessment-h84.vercel.app/) | ✅ Live — static HTML/JS on Vercel |

> ⚠️ The API runs on Render's free tier — first request after idle may take 30–60s to cold-start.

---

## 📖 Overview

This repository trains and serves a GradientBoostingClassifier that predicts loan default probability from the German Credit dataset (OpenML id=31), with SMOTE applied to the 70/30 class imbalance and a decision threshold selected via a business cost matrix rather than the default 0.5 cutoff.

---

## 📊 Model Performance

| Metric | Score | Notes |
|---|---|---|
| **CV ROC-AUC** (5-fold, GridSearchCV) | **0.916** | On SMOTE-balanced training folds |
| **Test ROC-AUC** (held-out 20%) | **0.791** | Real-world generalization estimate |
| **Gini Coefficient** | **0.582** | = 2×AUC−1; Basel III minimum acceptable is 0.35 |
| **Optimal threshold** | **0.22** | Selected via cost matrix (FN penalty 5×, FP penalty 1×) |
| **F1 at default 0.5 threshold** | 0.632 | If deployed without cost-based tuning |
| **Precision at 0.22 threshold** | 0.45 | Of applicants flagged high-risk, 45% actually default |
| **Recall at 0.22 threshold** | 0.833 | Catches 83.3% of real defaulters |
| **F1 at 0.22 threshold** | 0.585 | Deployed operating point |

**Why precision is lower at the deployed threshold:** the cost matrix (COST_FN=5, COST_FP=1) is deliberately conservative — missing a real defaulter is weighted 5× worse than wrongly flagging a good applicant.

Best hyperparameters: `learning_rate=0.05, max_depth=4, n_estimators=100, subsample=0.8`

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn 1.6.1, GradientBoostingClassifier |
| Imbalance handling | imbalanced-learn 0.14.2 (SMOTE) |
| API | Flask 3.1.3 + gunicorn 23.0.0 |
| Frontend | Static HTML/JS, zero build step |
| Backend hosting | Render |
| Frontend hosting | Vercel |
| Training environment | Google Colab |

> **Version pinning matters here**: an earlier deploy attempt failed because the serving environment's scikit-learn/numpy versions didn't match the training environment. `requirements.txt` is now pinned exactly.

---

## 🌐 API Reference

**Base URL:** `https://loan-risk-assessment-nsdw.onrender.com`

### GET /health
Returns `{ "status": "ok" }`

### POST /predict

Request uses the German Credit Data schema (20 fields): checking_status, duration, credit_history, purpose, credit_amount, savings_status, employment, installment_commitment, personal_status, other_parties, residence_since, property_magnitude, age, other_payment_plans, housing, existing_credits, job, num_dependents, own_telephone, foreign_worker.

Response: `{ "risk_score": 0.0 }` — a probability between 0.0 and 1.0. The frontend flags anything at or above 0.22 as HIGH RISK, matching the cost-optimized threshold.

---

## 💼 Business Framing

Expected Loss is modelled as EL = PD x LGD x EAD, consistent with IFRS 9 staging. The Gini coefficient (0.582) exceeds the Basel III regulatory minimum of 0.35.

---

## 📊 Model Performance

| Metric | Score | Notes |
|---|---|---|
| **CV ROC-AUC** (5-fold, GridSearchCV) | **0.916** | On SMOTE-balanced training folds |
| **Test ROC-AUC** (held-out 20%) | **0.791** | Real-world generalization estimate |
| **Gini Coefficient** | **0.582** | = 2×AUC−1; Basel III minimum acceptable is 0.35 |
| **Optimal threshold** | **0.22** | Selected via cost matrix (FN penalty 5×, FP penalty 1×) |
| **F1 at default 0.5 threshold** | 0.632 | If deployed without cost-based tuning |
| **Precision at 0.22 threshold** | 0.45 | Of applicants flagged high-risk, 45% actually default |
| **Recall at 0.22 threshold** | 0.833 | Catches 83.3% of real defaulters |
| **F1 at 0.22 threshold** | 0.585 | Deployed operating point |

**Why precision is lower at the deployed threshold:** the cost matrix (COST_FN=5, COST_FP=1) is deliberately conservative — missing a real defaulter is weighted 5× worse than wrongly flagging a good applicant.

Best hyperparameters: `learning_rate=0.05, max_depth=4, n_estimators=100, subsample=0.8`

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn 1.6.1, GradientBoostingClassifier |
| Imbalance handling | imbalanced-learn 0.14.2 (SMOTE) |
| API | Flask 3.1.3 + gunicorn 23.0.0 |
| Frontend | Static HTML/JS, zero build step |
| Backend hosting | Render |
| Frontend hosting | Vercel |
| Training environment | Google Colab |

> **Version pinning matters here**: an earlier deploy attempt failed because the serving environment's scikit-learn/numpy versions didn't match the training environment. `requirements.txt` is now pinned exactly.

---

## 🌐 API Reference

**Base URL:** `https://loan-risk-assessment-nsdw.onrender.com`

### GET /health
Returns `{ "status": "ok" }`

### POST /predict

Request uses the German Credit Data schema (20 fields): checking_status, duration, credit_history, purpose, credit_amount, savings_status, employment, installment_commitment, personal_status, other_parties, residence_since, property_magnitude, age, other_payment_plans, housing, existing_credits, job, num_dependents, own_telephone, foreign_worker.

Response: `{ "risk_score": 0.0 }` — a probability between 0.0 and 1.0. The frontend flags anything at or above 0.22 as HIGH RISK, matching the cost-optimized threshold.

---

## 💼 Business Framing

Expected Loss is modelled as EL = PD x LGD x EAD, consistent with IFRS 9 staging. The Gini coefficient (0.582) exceeds the Basel III regulatory minimum of 0.35.

---

## 👤 Author

**James Koero**
BSc Physics & Mathematics — Moi University, Kenya (2012)
Self-taught ML Engineer | Kisumu, Kenya
Email: jmskoero@gmail.com
GitHub: github.com/jameskoero

Academic Mentor: Prof. Johan Loeckx — Vrije Universiteit Brussel (VUB), Belgium

---

## 📄 License

Licensed under the MIT License — see LICENSE for details.
