
 🏦 Loan Default Risk Assessment



![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)




![ML](https://img.shields.io/badge/ML-GradientBoosting-orange)




![API](https://img.shields.io/badge/API-Flask-lightblue)




![License](https://img.shields.io/badge/License-MIT-lightgrey)




![Status](https://img.shields.io/badge/Status-Live-success)




![Test ROC--AUC](https://img.shields.io/badge/Test_ROC--AUC-0.791-success)




![CV ROC--AUC](https://img.shields.io/badge/CV_ROC--AUC-0.916-success)



> **Live, deployed ML system for predicting loan default risk**, trained on the German Credit dataset with SMOTE class balancing and a business-cost-optimized decision threshold. Full training pipeline, REST API, and interactive frontend, all live and reproducible.

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

All figures below are from a single reproducible training run — no placeholder or template numbers.

| Metric | Score | Notes |
|---|---|---|
| **CV ROC-AUC** (5-fold, GridSearchCV) | **0.916** | On SMOTE-balanced training folds |
| **Test ROC-AUC** (held-out 20%) | **0.791** | Real-world generalization estimate |
| **Gini Coefficient** | **0.582** | = 2×AUC−1; Basel III minimum acceptable is 0.35 |
| **Optimal threshold** | **0.22** | Selected via cost matrix (false negative penalty 5×, false positive penalty 1×) |
| **F1 at default 0.5 threshold** | 0.632 | If deployed without cost-based tuning |
| **Precision at 0.22 threshold** | 0.45 | Of applicants flagged high-risk, 45% actually default |
| **Recall at 0.22 threshold** | 0.833 | Catches 83.3% of real defaulters |
| **F1 at 0.22 threshold** | 0.585 | Deployed operating point |

**Why precision is lower at the deployed threshold:** the cost matrix (COST_FN=5, COST_FP=1) is deliberately conservative — missing a real defaulter is weighted 5× worse than wrongly flagging a good applicant. This trades precision for recall, appropriate for a lender prioritizing loss avoidance over approval volume. A higher threshold would raise precision and lower recall; the 0.22 cutoff is a business choice, not a model limitation.

Best hyperparameters (GridSearchCV): `learning_rate=0.05, max_depth=4, n_estimators=100, subsample=0.8`

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
| Training environment | Google Colab (Termux/mobile insufficient for this workload) |

> **Version pinning matters here**: an earlier deploy attempt failed because the serving environment's scikit-learn/numpy versions didn't match the versions used to train and pickle the model. `requirements.txt` is now pinned exactly to the training environment to prevent `InconsistentVersionWarning` crashes on load.

---

## 📁 Project Structure
