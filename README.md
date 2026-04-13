<div align="center">

# 🏦 Loan Risk Assessment
### *Advanced Loan Default Prediction with Explainable AI*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/loan-risk-assessment/blob/main/loan_risk_assessment.ipynb)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-blueviolet)](https://shap.readthedocs.io/)

> **"Empowering fair, data-driven credit decisions through transparent machine learning."**

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Tech Stack & Requirements](#-tech-stack--requirements)
- [Project Structure](#-project-structure)
- [Notebook Walkthrough](#-notebook-walkthrough)
- [Sample Results](#-sample-results)
- [Quick Start](#-quick-start)
- [Future Improvements (Roadmap)](#-future-improvements-roadmap)
- [How to Contribute](#-how-to-contribute)
- [Related Projects](#-related-projects)
- [Acknowledgements](#-acknowledgements)
- [Author](#-author)
- [License](#-license)

---

## 🔍 Overview

This project builds a **production-ready loan default risk classifier** using the classic **German Credit Dataset**. It demonstrates an end-to-end machine learning workflow — from raw data ingestion and exploratory analysis all the way to model explainability and business insights — making it an ideal reference for **fintech ML engineers**, **data scientists in banking**, and **self-learners** entering the credit-risk domain.

The goal is not just to maximise a metric, but to build a model that is **interpretable**, **fair**, and **actionable** for real-world lending decisions. Class imbalance is handled via **SMOTE**, multiple classifiers are benchmarked, and every prediction is explained using **SHAP values** — providing the "why" behind each credit decision.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📊 **Comprehensive EDA** | Univariate, bivariate, and multivariate analysis with rich visualisations |
| ⚖️ **Class Imbalance Handling** | SMOTE oversampling to address skewed default/non-default ratio |
| 🤖 **Multi-Model Benchmarking** | Logistic Regression, Random Forest, XGBoost, and LightGBM compared side-by-side |
| 🔍 **Explainable AI (XAI)** | Global and local SHAP explanations for every model |
| 📈 **Business Metrics** | Precision, Recall, F1, AUC-ROC, and cost-sensitive evaluation |
| 🔧 **Reproducible Pipelines** | scikit-learn `Pipeline` objects for clean, leak-free preprocessing |
| 💾 **Model Persistence** | Trained models saved with `joblib` for deployment |

---

## 🛠️ Tech Stack & Requirements

### Installation

```bash
git clone https://github.com/jameskoero/loan-risk-assessment.git
cd loan-risk-assessment
pip install -r requirements.txt
```

### `requirements.txt` (with annotations)

```text
# Core Data Manipulation & Numerical Computing
numpy==1.26.4          # Fundamental n-dimensional array library; underpins all ML libs
pandas==2.2.1          # DataFrame toolkit for data loading, cleaning, and feature engineering

# Machine Learning
scikit-learn==1.4.2    # Classifiers, preprocessing, pipelines, and evaluation metrics
xgboost==2.0.3         # Gradient-boosted trees; top performer on tabular credit data
lightgbm==4.6.0        # Microsoft's fast gradient boosting — handles large datasets efficiently
imbalanced-learn==0.12.2  # SMOTE and resampling strategies for class-imbalanced credit data

# Explainability
shap==0.45.0           # SHapley Additive exPlanations — model-agnostic feature importance

# Visualisation
matplotlib==3.8.4      # Base plotting library for charts, ROC curves, confusion matrices
seaborn==0.13.2        # Statistical visualisation — heatmaps, distribution plots
plotly==5.20.0         # Interactive charts for dynamic EDA dashboards

# Jupyter Environment
jupyter==1.0.0         # Classic Jupyter Notebook server
jupyterlab==4.2.5      # Next-generation Jupyter interface
ipywidgets==8.1.2      # Interactive widgets (sliders, dropdowns) in notebooks

# Statistical Analysis
scipy==1.13.0          # Statistical tests (chi-squared, KS) used in EDA
statsmodels==0.14.2    # Regression diagnostics, VIF, and statistical summaries

# Model Persistence & Utilities
joblib==1.4.0          # Efficient serialisation for scikit-learn models
tqdm==4.66.2           # Progress bars for loops and data pipelines
python-dotenv==1.0.1   # Environment variable management (.env file support)
```

---

## 📁 Project Structure

```
loan-risk-assessment/
├── loan_risk_assessment.ipynb   # Main Jupyter notebook (full pipeline)
├── requirements.txt             # Pinned Python dependencies
├── plots/                       # Auto-generated visualisation outputs
│   ├── eda/                     # EDA charts
│   ├── model/                   # Model performance plots
│   └── shap/                    # SHAP explanation plots
├── models/                      # Saved model artefacts (.pkl / .json)
├── data/                        # Raw and processed dataset files
└── README.md
```

---

## 📓 Notebook Walkthrough

The notebook is divided into **16 structured sections**, each building on the last to deliver a complete, professional ML workflow. Below is a detailed guide to what you will learn in each section and why it matters for real-world credit risk modelling.

---

### Section 1 — 🎯 Problem Definition & Business Context

This section frames the credit risk problem from a **lender's perspective**, explaining the financial cost of false negatives (approving a bad borrower) versus false positives (rejecting a good borrower). Understanding the asymmetric cost of errors is crucial because it directly shapes which evaluation metric you optimise for — in credit risk, **Recall** for the defaulter class is often more important than raw Accuracy.

---

### Section 2 — 📦 Imports & Environment Setup

All required libraries are imported with version checks, random seeds are fixed for reproducibility (`numpy`, `random`, `sklearn`), and display settings are configured for clean output. This section teaches best practices for **notebook hygiene** — a skill that separates professional projects from casual experiments, and one that ensures anyone who clones the repo can reproduce your results identically.

---

### Section 3 — 📥 Data Loading & Initial Inspection

The **German Credit Dataset** (1,000 records, 20 features) is loaded directly from the UCI repository or a local CSV. Initial inspection includes `.shape`, `.dtypes`, `.head()`, `.info()`, and `.describe()` to quickly surface data types, ranges, and obvious anomalies. This section reinforces the discipline of **always understanding your data before touching a model**, a principle that prevents costly mistakes downstream.

---

### Section 4 — 🔍 Exploratory Data Analysis (EDA) — Univariate

Each feature is analysed in isolation. Continuous variables are visualised with **histograms and KDE plots** to reveal skewness, bimodality, and outliers; categorical variables are shown as **bar charts with frequency counts**. You will learn how the distribution of a single feature (e.g., credit amount, loan duration) can already hint at which customer segments are higher risk.

---

### Section 5 — 🔗 Bivariate & Multivariate Analysis

Relationships between features and the target variable (`default = 1`) are explored using **grouped box plots**, **violin plots**, and a **Pearson / Cramér's V correlation matrix**. A **heatmap of feature correlations** highlights multicollinearity, which matters for models like Logistic Regression. You will understand how cross-feature patterns (e.g., high loan duration *and* low credit history) compound default risk.

---

### Section 6 — 🧹 Data Cleaning & Missing Value Treatment

Although the German Credit Dataset is relatively clean, this section demonstrates a **generalised cleaning pipeline**: identifying missing values, handling them with median/mode imputation (via `SimpleImputer`), detecting duplicate rows, and sanity-checking data types. These are transferable skills that apply to every real-world dataset you will ever encounter in finance.

---

### Section 7 — 🏗️ Feature Engineering

New predictive features are derived from existing ones — for example, the **debt-to-income ratio**, **loan-to-income ratio**, and **credit utilisation proxies**. Ordinal variables (e.g., credit history quality) are mapped to meaningful integer scales. This section teaches you that **raw features are rarely optimal** and that domain knowledge — understanding what drives default — is a competitive advantage in financial ML.

---

### Section 8 — 🔢 Encoding & Preprocessing Pipeline

Categorical features are encoded (one-hot or ordinal) and numerical features are scaled using `StandardScaler` or `RobustScaler`. All steps are wrapped in a **scikit-learn `ColumnTransformer` + `Pipeline`** to prevent **data leakage** — one of the most common mistakes in ML projects. You will learn why fitting transformers only on training data is non-negotiable for trustworthy model evaluation.

---

### Section 9 — ⚖️ Class Imbalance & SMOTE

The target class distribution is visualised, revealing that defaults are a minority (~30%). Without correction, a naïve classifier can achieve 70% accuracy by predicting "no default" for everyone — which is useless for a lender. **SMOTE** (Synthetic Minority Over-sampling Technique) is applied to the training set only, and the before/after distributions are compared. You will understand both the theory and practical application of resampling strategies.

---

### Section 10 — 🤖 Model Training — Baseline

A **Logistic Regression** baseline is trained and evaluated, providing a simple, interpretable benchmark. Baseline metrics (Accuracy, Precision, Recall, F1, AUC-ROC) are recorded in a comparison table. This section establishes the importance of always starting simple — complex models are only justified if they meaningfully outperform a strong baseline.

---

### Section 11 — 🌲 Ensemble Models — Random Forest & Gradient Boosting

**Random Forest**, **XGBoost**, and **LightGBM** are trained and hyperparameter-tuned using `GridSearchCV` or `RandomizedSearchCV`. The section covers ensemble theory — bagging vs. boosting — and explains why these methods dominate tabular data competitions. Learning curves and validation curves are plotted to diagnose **overfitting vs. underfitting**.

---

### Section 12 — 📊 Model Evaluation & Comparison

All trained models are compared on a **unified leaderboard** using multiple metrics. The **ROC curve** and **Precision-Recall curve** are plotted for every model on the same axes, making it visually clear which model best serves the business objective. You will learn why the **AUC-PR curve is preferred over AUC-ROC** when classes are imbalanced — a subtle but important distinction for credit risk.

---

### Section 13 — 🧠 Explainability with SHAP

**SHAP summary plots**, **waterfall plots**, and **force plots** are generated for the best-performing model. Global feature importance reveals which variables drive default risk across the entire portfolio; local explanations show why a *specific* applicant was flagged. This section is critical for **regulatory compliance** (e.g., EU AI Act, GDPR Article 22) and for building stakeholder trust in automated credit decisions.

---

### Section 14 — 💼 Business Interpretation & Threshold Tuning

The optimal classification threshold is tuned by plotting **F1 score vs. threshold** and **cost-sensitive loss curves** (assigning a higher penalty to missed defaults). You will learn to translate a model's probabilistic output into a **lending policy** — for example, "approve all applicants with default probability < 0.25" — and measure the financial impact of that policy on the loan portfolio.

---

### Section 15 — 💾 Model Saving & Deployment Preparation

The best model pipeline is serialised using `joblib` and saved to the `models/` directory. A simple **prediction function** is written to accept a raw applicant dictionary and return a risk score and recommendation. This section bridges the gap between experimentation and deployment, showing you exactly what artefacts a downstream API or batch scoring job would consume.

---

### Section 16 — 📝 Summary, Conclusions & Next Steps

Key findings are summarised: which features matter most, which model performed best and why, and what business value could be realised. Concrete **next steps** are suggested (API deployment, fairness analysis, additional datasets), giving you a roadmap for extending the project. You will leave with a clear mental model of a complete, professional ML project lifecycle in financial services.

---

## 📈 Sample Results

> *(Plots are saved to the `plots/` directory and rendered inline in the notebook.)*

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 74.0% | 0.71 | 0.68 | 0.69 | 0.79 |
| Random Forest | 78.5% | 0.76 | 0.74 | 0.75 | 0.84 |
| XGBoost | 81.0% | 0.79 | 0.77 | 0.78 | 0.87 |
| **LightGBM** | **82.3%** | **0.81** | **0.79** | **0.80** | **0.88** |

![SHAP Summary Plot](plots/shap/shap_summary.png)
![ROC Curves](plots/model/roc_curves.png)

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/jameskoero/loan-risk-assessment.git
cd loan-risk-assessment

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter lab loan_risk_assessment.ipynb
```

Or run it instantly in the cloud — no setup required:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/loan-risk-assessment/blob/main/loan_risk_assessment.ipynb)

---

## 🗺️ Future Improvements (Roadmap)

This project is actively evolving. The following improvements are planned or under exploration, spanning additional data sources, advanced modelling techniques, deployment infrastructure, and responsible AI considerations.

**📂 Additional Datasets**
Integrating the [Lending Club Loan Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) and the [Home Credit Default Risk dataset](https://www.kaggle.com/competitions/home-credit-default-risk) would allow cross-domain evaluation and help test whether models trained on German credit data generalise to other lending contexts. Combining datasets could also enable transfer learning experiments.

**🤖 Deep Learning Models**
Tabular deep learning architectures such as **TabNet**, **NODE** (Neural Oblivious Decision Ensembles), and **FT-Transformer** have shown competitive performance against tree ensembles on structured financial data. Adding these as additional benchmarks — with attention to training time and interpretability trade-offs — is a natural extension.

**🌐 API Deployment**
Wrapping the best model in a **FastAPI** or **Flask** REST API would make it consumable by fintech applications. A `/predict` endpoint could accept applicant data as JSON and return a risk score with SHAP-based explanation. Containerising the service with **Docker** and deploying to **Render**, **Railway**, or **AWS Lambda** would complete the MLOps loop.

**🔄 MLOps & Experiment Tracking**
Integrating **MLflow** for experiment tracking, **DVC** for data and model versioning, and **GitHub Actions** for CI/CD (automated retraining on new data pushes) would bring this project to production-grade MLOps standards. These tools are increasingly expected in ML engineering roles at banks and fintech startups.

**⚖️ Fairness & Bias Analysis**
Credit scoring models can encode historical biases related to gender, age, or foreign worker status — all of which are present as features in the German Credit Dataset. Using **Fairlearn** or **AI Fairness 360 (AIF360)** to audit the model for disparate impact across protected groups is a critical ethical step and increasingly a regulatory requirement (e.g., Equal Credit Opportunity Act in the US, EU AI Act).

**📊 Interactive Dashboard**
Building a **Streamlit** or **Gradio** dashboard that lets non-technical stakeholders explore applicant risk scores, adjust decision thresholds, and view SHAP explanations in real time would dramatically improve the project's business value and communicability.

**📐 Calibration**
Applying **Platt Scaling** or **Isotonic Regression** to calibrate predicted probabilities ensures that "30% default probability" actually means 30% of such applicants default — a requirement for risk-based pricing models.

---

## 🤝 How to Contribute

Contributions are warmly welcomed! Whether you're fixing a bug, improving documentation, or adding a new model — your input makes this project better.

### Steps

1. **Fork** the repository  
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`  
3. **Make** your changes with clear, descriptive commits  
4. **Test** your changes in the notebook or relevant scripts  
5. **Push** to your fork: `git push origin feature/your-feature-name`  
6. **Open** a Pull Request with a clear description of what you changed and why  

### Contribution Ideas

- 🐛 Bug fixes or notebook cell errors  
- 📊 New visualisation types (e.g., calibration plots, LIME explanations)  
- 🤖 Additional model implementations (TabNet, CatBoost)  
- 📝 Documentation improvements  
- 🌍 Translations of the notebook or README  
- 📂 Integration of new datasets  

> Please follow the existing code style, add comments where helpful, and keep PRs focused on a single concern.

---

## ⭐ Star History

If this project has been useful to you, please consider giving it a ⭐ — it helps others discover it and motivates continued development!

[![Star History Chart](https://api.star-history.com/svg?repos=jameskoero/loan-risk-assessment&type=Date)](https://star-history.com/#jameskoero/loan-risk-assessment&Date)

---

## 🔗 Related Projects

Explore these similar repositories for complementary perspectives on credit risk and financial ML:

| Repository | Description |
|---|---|
| [**Lending Club Analysis**](https://github.com/topics/lending-club) | End-to-end EDA and default prediction on the Lending Club peer-to-peer lending dataset |
| [**Home Credit Default Risk**](https://github.com/topics/home-credit) | Kaggle competition solutions for predicting loan repayment ability from alternative data |
| [**Credit Risk Scorecard**](https://github.com/topics/credit-scorecard) | Traditional scorecard development with Weight of Evidence (WoE) and Information Value (IV) |
| [**FinML Awesome List**](https://github.com/georgezouq/awesome-ai-in-finance) | Curated list of machine learning papers, libraries, and datasets for finance |

---

## 🙏 Acknowledgements

This project builds on the work of researchers, open-source contributors, and dataset curators whose contributions are foundational to modern credit risk ML.

---

**🗃️ German Credit Dataset**
> Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. [https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

One of the most widely used benchmark datasets in credit risk research, containing 1,000 loan applicants described by 20 attributes. Its longevity (30+ years) makes it ideal for validating new modelling techniques against a well-understood baseline, and its inclusion of sensitive attributes (age, gender, foreign worker status) makes it a standard testbed for fairness research.

---

**📐 SMOTE — Synthetic Minority Over-sampling Technique**
> Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic Minority Over-sampling Technique*. Journal of Artificial Intelligence Research, 16, 321–357.

SMOTE addressed a fundamental challenge in credit modelling: real-world default rates are low (1–10%), creating severe class imbalance that causes naïve classifiers to ignore the minority class entirely. By interpolating synthetic samples between nearest neighbours, SMOTE enables models to learn the decision boundary more accurately without the pitfalls of simple duplication.

---

**🔍 SHAP — A Unified Approach to Interpreting Model Predictions**
> Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. Advances in Neural Information Processing Systems, 30.

SHAP brought rigorous game-theoretic foundations (Shapley values from cooperative game theory) to ML interpretability, enabling both global feature importance and per-prediction explanations. For credit risk, where regulators increasingly require "right to explanation" for automated decisions, SHAP has become the de facto standard for model transparency.

---

**⚡ XGBoost**
> Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

XGBoost's combination of regularisation, sparsity awareness, and cache-efficient tree construction made it the dominant algorithm in tabular ML competitions for nearly a decade. Its ability to handle missing values natively and output calibrated probabilities makes it especially valuable in credit scoring.

---

**🚀 LightGBM**
> Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. Advances in Neural Information Processing Systems, 30.

LightGBM's leaf-wise tree growth and histogram-based splitting make it significantly faster than XGBoost on large datasets, while achieving comparable or superior accuracy. It is widely adopted in production credit scoring systems where training on millions of loan records must complete within hours.

---

**🔬 scikit-learn**
> Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.

The foundation of this project's preprocessing and evaluation infrastructure. scikit-learn's `Pipeline` abstraction is particularly important for preventing data leakage — a critical concern in any production ML system.

---

## 👨‍💻 Author

<div align="center">

### James Koero

**Junior ML Engineer · Kisumu, Kenya 🇰🇪**

</div>

James is a self-taught Machine Learning Engineer with a **B.Sc. in Physics and Mathematics from Moi University**, Eldoret, Kenya. After graduating with a strong quantitative foundation, he pivoted into applied machine learning in 2023, driven by a fascination with how data-driven systems can solve complex, high-stakes problems in underserved markets.

His primary focus is **financial inclusion** — building credit and risk assessment systems that extend access to capital for individuals and small businesses across Sub-Saharan Africa who are underserved by traditional banking infrastructure. He believes that transparent, fair ML models can democratise financial services in the same way mobile money (M-Pesa) democratised payments.

Since beginning his self-taught ML journey, James has built expertise in the full ML engineering stack: data wrangling with Pandas, classical ML with scikit-learn, gradient boosting with XGBoost and LightGBM, and model explainability with SHAP. He documents his learning publicly on GitHub to help other self-taught engineers in emerging markets follow a similar path.

**Areas of Interest:** Credit Risk Modelling · Financial Inclusion · Explainable AI · MLOps · Responsible AI

---

📧 **Email:** [your.email@example.com](mailto:your.email@example.com)  
💼 **LinkedIn:** [linkedin.com/in/jameskoero](https://linkedin.com/in/jameskoero)  
🐦 **Twitter/X:** [@jameskoero](https://twitter.com/jameskoero)  
🌐 **Portfolio:** [jameskoero.github.io](https://jameskoero.github.io)  

---

## 📄 License

```
MIT License

Copyright (c) 2024 James Koero

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<div align="center">

*Made with ❤️ from Kisumu, Kenya*  
*If this project helped you, please ⭐ the repo and share it with others!*

</div>

