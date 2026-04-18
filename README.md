# 🏦 Loan Default Risk Assessment

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![ML](https://img.shields.io/badge/ML-GradientBoosting-orange)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)
![Tests](https://img.shields.io/badge/Tests-pytest-brightgreen)
![API](https://img.shields.io/badge/API-Flask-lightblue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-87%25-blue)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.92-success)

> **Production-ready ML system for predicting loan default risk** with explainable outputs using SHAP.
> Includes a full training pipeline, REST API, CLI inference tool, unit tests, and Jupyter walkthrough.
> Built to real-world financial industry standards by a self-taught ML engineer.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [ML Pipeline](#ml-pipeline)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Model Performance](#model-performance)
- [SHAP Explainability](#shap-explainability)
- [Visualizations](#visualizations)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Dataset](#dataset)
- [Business Impact](#business-impact)
- [Documentation](#documentation)
- [Author](#author)

---

## 📖 Overview

This repository provides a **complete, deployable loan default risk system** built to the standards expected in real-world fintech and banking environments. The system:

- **Predicts** the probability of loan default using a calibrated GradientBoosting model
- **Explains** every individual prediction using SHAP (SHapley Additive exPlanations)
- **Serves** predictions via a REST API (Flask) and a CLI batch scoring tool
- **Validates** itself via a unit test suite (pytest)
- **Documents** its architecture, API contract, and performance benchmarks

---

## 🧩 Problem Statement

Financial institutions lose billions of dollars annually to loan defaults. Traditional rule-based credit scoring models are rigid, opaque, and fail to explain *why* a borrower is high-risk — a requirement under financial regulations like GDPR Article 22 and Fair Lending laws.

```
❓ Can we accurately predict which borrowers will default?
❓ Can we explain WHY the model flags a borrower as high-risk?
❓ Can we serve these predictions via a production API?
✅ This project answers ALL THREE questions.
```

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph INPUT["📥 Input Layer"]
        A1[CSV File\nbatch scoring]
        A2[REST API\nJSON request]
        A3[Jupyter Notebook\nexploration]
    end

    subgraph PIPELINE["⚙️ ML Pipeline — src/"]
        B1[🧹 Preprocessing\nImputation · Encoding · Scaling]
        B2[🔧 Feature Engineering\nRatios · Flags · Interactions]
        B3[🤖 GradientBoosting\nClassifier Training]
        B4[💾 Model Artifact\nmodels/model.pkl]
    end

    subgraph EXPLAIN["🔍 Explainability Layer"]
        C1[SHAP TreeExplainer]
        C2[Global Summary Plot]
        C3[Individual Waterfall Plot]
    end

    subgraph OUTPUT["📤 Output Layer"]
        D1[Risk Score 0.0–1.0]
        D2[Decision: Approve / Review / Decline]
        D3[Explanation: Top risk factors]
        D4[API Response JSON]
    end

    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2 --> B3 --> B4
    B4 --> C1
    C1 --> C2
    C1 --> C3
    B4 --> D1
    D1 --> D2
    C3 --> D3
    D1 & D3 --> D4

    style INPUT fill:#1f6feb,color:#fff
    style PIPELINE fill:#d29922,color:#fff
    style EXPLAIN fill:#8b5cf6,color:#fff
    style OUTPUT fill:#238636,color:#fff
```

---

## ⚙️ ML Pipeline

```mermaid
flowchart LR
    A[📂 Raw Loan Data] --> B[🧹 Clean & Impute]
    B --> C[🔧 Feature Engineering]
    C --> D[✂️ Train/Test Split 80/20]
    D --> E[🤖 GradientBoosting]
    E --> F[📊 Evaluate AUC·F1·PR]
    F --> G[🔍 SHAP Explainability]
    G --> H[💾 Save Artifact]
    H --> I[🚀 Serve API / CLI]

    style A fill:#1f6feb,color:#fff
    style E fill:#d29922,color:#fff
    style G fill:#8b5cf6,color:#fff
    style I fill:#238636,color:#fff
```

---

## ✨ Key Features

- ✅ **End-to-end training pipeline** — `src/loan_risk_assessment.py`
- ✅ **Gradient Boosting model** with calibrated probability outputs
- ✅ **SHAP explainability** — global summary + per-borrower waterfall plots
- ✅ **REST API** (Flask) — `POST /predict` scores one or many applicants
- ✅ **CLI batch inference** — `predict.py` scores a full CSV file
- ✅ **Unit test suite** — `tests/` with pytest
- ✅ **Jupyter Notebook** — full interactive walkthrough
- ✅ **Architecture docs** — `docs/ARCHITECTURE.md`
- ✅ **API contract docs** — `docs/API.md`
- ✅ **Performance benchmark** — `docs/MODEL_PERFORMANCE.md`
- ✅ **Makefile** — one-command train, test, and serve

---

## 🛠️ Tech Stack

```mermaid
graph LR
    PY[Python 3.10+] --> SK[scikit-learn]
    PY --> SH[SHAP]
    PY --> FL[Flask]
    PY --> PD[Pandas / NumPy]
    PY --> VZ[Matplotlib / Seaborn]
    PY --> PT[pytest]
    SK --> GB[GradientBoostingClassifier]
    SH --> SP[Summary Plot]
    SH --> WF[Waterfall Plot]
    FL --> API[REST API POST /predict]

    style PY fill:#3776ab,color:#fff
    style GB fill:#f97316,color:#fff
    style SH fill:#8b5cf6,color:#fff
    style FL fill:#238636,color:#fff
```

| Component | Tool | Purpose |
|---|---|---|
| Language | Python 3.10+ | Core development |
| ML Framework | scikit-learn | Model training & evaluation |
| Algorithm | GradientBoostingClassifier | Default prediction |
| Explainability | SHAP | Model interpretation |
| API | Flask | REST endpoint serving |
| Data | Pandas, NumPy | Feature engineering |
| Visualization | Matplotlib, Seaborn | Plots & charts |
| Testing | pytest | Unit & integration tests |
| Notebook | Jupyter | Interactive walkthrough |
| Build | Makefile | Automation |

---

## 📊 Model Performance

```mermaid
xychart-beta
    title "Model Performance Metrics (%)"
    x-axis ["Accuracy", "ROC-AUC x100", "Precision", "Recall", "F1 Score"]
    y-axis "Score (%)" 0 --> 100
    bar [87, 92, 84, 81, 82]
```

| Metric | Score | Interpretation |
|---|---|---|
| ✅ Accuracy | **87%** | 87 in 100 applicants correctly classified |
| ✅ ROC-AUC | **0.92** | Excellent discrimination between classes |
| ✅ Precision | **84%** | 84% of flagged borrowers are true defaults |
| ✅ Recall | **81%** | Catches 81% of all actual defaults |
| ✅ F1 Score | **82%** | Strong precision-recall balance |

> Full benchmark details: [docs/MODEL_PERFORMANCE.md](docs/MODEL_PERFORMANCE.md)

---

## 🔍 SHAP Explainability

```mermaid
flowchart LR
    M[🤖 Trained GradientBoosting] --> E[🔍 SHAP TreeExplainer]
    E --> G[🌍 Global Summary\nTop drivers across portfolio]
    E --> I[👤 Individual Waterfall\nPer-borrower breakdown]
    E --> F[📊 Force Plot\nReal-time single prediction]

    style M fill:#d29922,color:#fff
    style E fill:#8b5cf6,color:#fff
    style G fill:#238636,color:#fff
    style I fill:#1f6feb,color:#fff
```

**Top identified risk drivers:**

| Rank | Feature | Effect on Default Risk |
|---|---|---|
| 🥇 1 | Credit history / derogatory marks | ↑ Strongly increases risk |
| 🥈 2 | Debt-to-income ratio | ↑ Increases risk |
| 🥉 3 | Loan amount relative to income | ↑ Increases risk |
| 4 | Employment length | ↓ Reduces risk |
| 5 | Open credit accounts | Varies |

---

## 🖼️ Visualizations

> 📌 Run `python src/loan_risk_assessment.py` to generate all plots into the `images/` folder.

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

### ROC Curve
![ROC Curve](images/roc_curve.png)

### SHAP Summary Plot — Global Feature Importance
![SHAP Summary](images/shap_summary.png)

### SHAP Waterfall Plot — Individual Prediction
![SHAP Waterfall](images/shap_waterfall.png)

### Feature Importance Bar Chart
![Feature Importance](images/feature_importance.png)

---

## 📁 Project Structure

```
loan-risk-assessment/
│
├── 📁 docs/
│   ├── ARCHITECTURE.md               ← System design & data flow
│   ├── API.md                        ← REST API contract & examples
│   ├── MODEL_PERFORMANCE.md          ← Benchmark results
│   └── CONTRIBUTING.md               ← Contribution guidelines
│
├── 📁 notebooks/
│   └── loan_risk_assessment.ipynb    ← Full interactive walkthrough
│
├── 📁 src/
│   ├── __init__.py
│   └── loan_risk_assessment.py       ← Core training pipeline
│
├── 📁 tests/
│   ├── __init__.py
│   └── test_loan_risk_assessment.py  ← Unit & integration tests
│
├── 📁 models/                        ← Saved artifacts (gitignored)
│   └── model.pkl
│
├── 📁 images/                        ← Generated plot outputs
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── shap_summary.png
│   ├── shap_waterfall.png
│   └── feature_importance.png
│
├── 📄 app.py                         ← Flask REST API server
├── 📄 predict.py                     ← CLI batch scoring tool
├── 📄 setup.py                       ← Package config
├── 📄 requirements.txt               ← Dependencies
├── 📄 Makefile                       ← Automation
├── 📄 CHANGELOG.md                   ← Version history
├── 📄 LICENSE                        ← MIT License
└── 📄 README.md                      ← This file
```

---

## 🚀 Installation

```bash
git clone https://github.com/jameskoero/loan-risk-assessment.git
cd loan-risk-assessment
pip install -r requirements.txt
pip install -e .
```

---

## ⚡ Quick Start

```bash
# Train model and save artifacts
python src/loan_risk_assessment.py

# Run unit tests
pytest tests/ -v

# Score a CSV of new applicants
python predict.py --input input.csv --output scores.csv

# Start the REST API
python app.py

# Using Makefile
make train
make test
make serve
```

---

## 🌐 API Reference

### Health Check
```http
GET /health
→ { "status": "ok", "model": "loaded" }
```

### Score a Single Applicant
```http
POST /predict
Content-Type: application/json

{
  "loan_amnt": 15000,
  "annual_inc": 55000,
  "dti": 18.5,
  "delinq_2yrs": 0,
  "open_acc": 8,
  "emp_length": 5
}
```

**Response:**
```json
{
  "risk_score": 0.23,
  "decision": "APPROVE",
  "top_factors": [
    { "feature": "dti", "impact": +0.08 },
    { "feature": "delinq_2yrs", "impact": -0.04 }
  ]
}
```

> Full API contract: [docs/API.md](docs/API.md)

---

## 📂 Dataset

| Dataset | Link | Size |
|---|---|---|
| Lending Club | [Kaggle](https://www.kaggle.com/datasets/wordsforthewise/lending-club) | 2.2M rows |
| Home Credit | [Kaggle](https://www.kaggle.com/c/home-credit-default-risk) | 300K rows |
| Give Me Some Credit | [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit) | 150K rows |

---

## 💼 Business Impact

```mermaid
graph TD
    M[🤖 Loan Risk System] --> A[👔 Loan Officers\nExplainable decisions]
    M --> B[📈 Risk Teams\nPortfolio SHAP insights]
    M --> C[⚖️ Compliance\nAudit-ready predictions]
    M --> D[💻 Engineering\nProduction API integration]

    style M fill:#d29922,color:#fff
    style A fill:#1f6feb,color:#fff
    style B fill:#238636,color:#fff
    style C fill:#8b5cf6,color:#fff
    style D fill:#f97316,color:#fff
```

---

## 📚 Documentation

| Document | Description |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design, data flow, components |
| [docs/API.md](docs/API.md) | REST API contract with examples |
| [docs/MODEL_PERFORMANCE.md](docs/MODEL_PERFORMANCE.md) | Benchmarks and methodology |
| [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) | How to contribute |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

---

## 👤 Author & Acknowledgements

**James Koero (Jayalo)**
BSc Physics & Mathematics — Moi University, Kenya (2012)
Self-taught ML Engineer | Kisumu, Kenya 🇰🇪
📧 [jmskoero@gmail.com](mailto:jmskoero@gmail.com)
🐙 [github.com/jameskoero](https://github.com/jameskoero)

**Academic Mentor:**
**Prof. Johan Loeckx** — Vrije Universiteit Brussel (VUB), Belgium
*Guidance on ML methodology, model evaluation, and production-grade standards*

---

## 📄 License

Licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

> *"Good models predict. Great models explain." — This project does both.*
