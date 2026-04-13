<div align="center">

# 🏦 Loan Risk Assessment
### *Advanced ML-Powered Loan Default Prediction with Explainability*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/loan-risk-assessment/blob/main/loan_risk_assessment.ipynb)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.2-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-red)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)](https://shap.readthedocs.io/)
[![GitHub Stars](https://img.shields.io/github/stars/jameskoero/loan-risk-assessment?style=social)](https://github.com/jameskoero/loan-risk-assessment/stargazers)

> *"Financial institutions lose billions each year to loan defaults. This project demonstrates how machine learning — built with transparency and fairness in mind — can transform credit risk assessment for the better."*

</div>

---

## 📑 Table of Contents

| # | Section |
|---|---------|
| 1 | [🎯 Project Overview](#-project-overview) |
| 2 | [✨ Key Features](#-key-features) |
| 3 | [🖼️ Project Screenshots](#️-project-screenshots) |
| 4 | [📦 Requirements](#-requirements) |
| 5 | [🚀 Quick Start](#-quick-start) |
| 6 | [📓 Notebook Walkthrough](#-notebook-walkthrough) |
| 7 | [🗂️ Repository Structure](#️-repository-structure) |
| 8 | [📊 Results & Model Performance](#-results--model-performance) |
| 9 | [🔍 Model Explainability (SHAP)](#-model-explainability-shap) |
| 10 | [🛣️ Future Improvements (Roadmap)](#️-future-improvements-roadmap) |
| 11 | [🤝 How to Contribute](#-how-to-contribute) |
| 12 | [🌟 Star History](#-star-history) |
| 13 | [🔗 Related Projects](#-related-projects) |
| 14 | [👤 Author](#-author) |
| 15 | [📄 License](#-license) |
| 16 | [🙏 Acknowledgements](#-acknowledgements) |

---

## 🎯 Project Overview

This project builds a **production-ready credit risk scoring system** using the classic [Statlog (German Credit Data)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) dataset from the UCI Machine Learning Repository. The goal is to predict whether a loan applicant represents a **good** or **bad** credit risk, a problem with direct implications for financial inclusion and responsible lending.

The notebook explores the **full machine learning lifecycle**: from raw data ingestion and exploratory analysis, through feature engineering and class-imbalance correction with **SMOTE**, to training and tuning five competing classifiers, evaluating them with business-relevant metrics, and finally opening the "black box" with **SHAP** explanations — making every prediction auditable and explainable.

**Why does this matter?** In emerging markets like Sub-Saharan Africa, millions of creditworthy individuals are rejected because traditional scoring systems lack data. ML-driven, explainable models can close that gap — extending financial access while keeping institutions solvent.

---

## ✨ Key Features

- 📊 **Deep EDA** — univariate, bivariate, and multivariate analysis with interactive plots
- ⚖️ **Class Imbalance Handling** — SMOTE oversampling on the minority (bad-credit) class
- 🤖 **5 Classifiers** — Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM
- 🔧 **Hyperparameter Tuning** — GridSearchCV + cross-validation for all models
- 📈 **Business Metrics** — ROC-AUC, Precision-Recall AUC, F1-Score, KS Statistic, Gini Coefficient
- 🔍 **SHAP Explainability** — global beeswarm plots + local waterfall/force plots per applicant
- 💾 **Model Persistence** — best model saved with `joblib` for downstream inference
- 🎛️ **Interactive Widgets** — real-time risk scoring via `ipywidgets` sliders

---

## 🖼️ Project Screenshots

> *The `plots/` directory is auto-generated when the notebook is executed.*

| Plot | Description |
|------|-------------|
| ![Correlation Heatmap](plots/correlation_heatmap.png) | Feature correlation matrix highlighting multicollinearity |
| ![ROC Curves](plots/roc_curves_comparison.png) | Side-by-side ROC curves for all five models |
| ![SHAP Beeswarm](plots/shap_beeswarm.png) | Global SHAP feature importance beeswarm plot |
| ![Confusion Matrix](plots/confusion_matrix_best.png) | Confusion matrix for the best-performing model |
| ![Class Distribution](plots/class_distribution.png) | Target class distribution before and after SMOTE |

---

## 📦 Requirements

### Python Version
```
Python 3.9+
```

### Full `requirements.txt` with Annotations

```text
# ------ Data Manipulation & Numerical Computing ------
numpy==1.26.4          # Foundation for numerical operations; array math & feature transforms
pandas==2.2.2          # Data wrangling, CSV loading, missing-value handling, train/test splits
scipy==1.13.0          # Statistical tests (chi-squared, KS test) during EDA

# ------ Machine Learning ------
scikit-learn==1.4.2    # Core ML: Logistic Regression, Random Forest, GridSearchCV, metrics
xgboost==2.0.3         # Gradient-boosted trees; best tabular performance + feature importance
lightgbm==4.3.0        # Microsoft's fast GBM variant; second ensemble candidate
imbalanced-learn==0.12.3  # SMOTE oversampling for class-imbalance correction

# ------ Model Interpretability ------
shap==0.45.1           # SHapley Additive exPlanations — global & local model explainability
lime==0.2.0.1          # Per-instance surrogate explanations to complement SHAP

# ------ Visualisation ------
matplotlib==3.8.4      # Base plotting: ROC curves, confusion matrices, learning curves
seaborn==0.13.2        # Statistical plots: heatmaps, distribution plots, pairplots
plotly==5.22.0         # Interactive HTML charts: ROC dashboard, feature importance

# ------ Data Source ------
ucimlrepo==0.0.7       # UCI ML Repository client — fetches German Credit Data programmatically

# ------ Notebook & Reporting ------
jupyter==1.0.0         # Jupyter server + JupyterLab + nbconvert metapackage
ipywidgets==8.1.2      # Interactive sliders/dropdowns for real-time risk scoring
nbformat==5.10.4       # Notebook format validation and programmatic cell manipulation

# ------ Utilities ------
joblib==1.4.2          # Model serialisation (pickle replacement) — save/load trained pipelines
tqdm==4.66.4           # Progress bars for cross-validation and hyperparameter search loops
python-dotenv==1.0.1   # Environment variable management — keeps secrets out of notebooks
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/jameskoero/loan-risk-assessment.git
cd loan-risk-assessment

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter lab
```

---

## 🚀 Quick Start

**Option A — Run locally (after installation above):**
```bash
jupyter lab loan_risk_assessment.ipynb
```

**Option B — Run in Google Colab (zero setup):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/loan-risk-assessment/blob/main/loan_risk_assessment.ipynb)

```python
# First cell in Colab — install dependencies
!pip install -q xgboost lightgbm imbalanced-learn shap lime ucimlrepo plotly ipywidgets
```

---

## 📓 Notebook Walkthrough

The notebook is structured as **16 numbered sections**, each building on the last to produce a complete, reproducible ML pipeline. Below is a detailed guide to what you will learn in each section.

---

### Section 1 — 📥 Environment Setup & Library Imports
You will configure the notebook environment by importing all required libraries and setting global parameters such as random seeds, display options, and plot styles. This section ensures **full reproducibility** across different machines and execution environments — a discipline that is critical in both academic research and production ML systems. Understanding how to structure imports cleanly also sets the professional tone for the rest of the notebook.

---

### Section 2 — 🗃️ Data Acquisition (UCI ML Repository)
You will fetch the **Statlog (German Credit Data)** dataset directly from the UCI ML Repository using the `ucimlrepo` Python client, eliminating the need for manual downloads. This section demonstrates **programmatic data sourcing** — a key MLOps principle — and introduces the 20 features and 1,000 samples that describe each loan applicant, including credit history, loan amount, employment status, and personal characteristics.

---

### Section 3 — 🔬 Initial Data Exploration
You will perform a systematic first-look at the raw dataset: checking shapes, data types, missing value counts, and descriptive statistics. This seemingly simple step often reveals **critical data quality issues** early — before they silently corrupt downstream model training. You will also learn how the UCI encoding maps integer codes to real-world categorical values (e.g., `A11` = "no checking account").

---

### Section 4 — 📊 Exploratory Data Analysis — Univariate
You will examine the distribution of every individual feature through histograms, KDE plots, and count charts. Understanding **marginal distributions** helps you identify skewed features that may require transformation, rare categories that may need grouping, and whether the data plausibly represents the real-world population you are modeling — all before touching any ML algorithm.

---

### Section 5 — 🔗 Exploratory Data Analysis — Bivariate & Multivariate
You will study relationships *between* features and the target variable using grouped bar charts, box plots, violin plots, and a Pearson/Spearman correlation heatmap. This section reveals which features are the **strongest predictors of default**, exposes multicollinearity between independent variables, and guides intelligent feature selection — preventing the curse of dimensionality.

---

### Section 6 — ⚙️ Feature Engineering & Preprocessing
You will build a reproducible preprocessing pipeline using scikit-learn's `Pipeline` and `ColumnTransformer`. This includes **ordinal and one-hot encoding** for categorical variables, **StandardScaler** normalisation for continuous features, and the creation of derived interaction features (e.g., debt-to-income ratio). You will also learn why fitting transformers *only* on the training fold (not the full dataset) is essential to avoid data leakage.

---

### Section 7 — ⚖️ Handling Class Imbalance with SMOTE
You will apply **SMOTE (Synthetic Minority Over-sampling Technique)** from `imbalanced-learn` to address the ~70/30 class split between good and bad credit applicants. The section explains *why* naive accuracy is misleading on imbalanced datasets, compares SMOTE against random oversampling and class-weight adjustment, and visualises the synthetic samples in 2D PCA space — giving you an intuition for what SMOTE is actually doing geometrically.

---

### Section 8 — 🧪 Train / Validation / Test Split Strategy
You will implement a **stratified three-way split** (60 % train, 20 % validation, 20 % held-out test) to properly estimate generalisation performance. You will learn how stratification preserves class proportions across splits, why a separate held-out test set is non-negotiable for unbiased final evaluation, and how this strategy connects to industry practices such as champion-challenger model testing.

---

### Section 9 — 🤖 Baseline Model Training (5 Classifiers)
You will train five classifiers — **Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM** — on the SMOTE-augmented training set using default hyperparameters. This section establishes **performance baselines** and demonstrates that tree ensembles generally outperform linear models on this tabular task. You will also time each model's training to appreciate the speed/accuracy trade-offs.

---

### Section 10 — 🔧 Hyperparameter Tuning with GridSearchCV
You will systematically optimise each model's hyperparameters using `GridSearchCV` with 5-fold stratified cross-validation, tuning on **ROC-AUC** rather than accuracy. You will learn how to design parameter grids that balance exploration with computation budget, how to read a CV results dataframe to spot overfitting, and why a well-tuned Logistic Regression can sometimes rival a gradient-boosted tree on small datasets.

---

### Section 11 — 📈 Model Evaluation & Business Metrics
You will evaluate all tuned models on the held-out test set using a **comprehensive scorecard**: ROC-AUC, Precision-Recall AUC, F1-Score (macro and weighted), Matthews Correlation Coefficient, KS Statistic, and Gini Coefficient. You will learn why the KS statistic and Gini coefficient are industry-standard metrics in credit scoring that go beyond what academic papers typically report — and how to translate model outputs into actionable credit-approval thresholds.

---

### Section 12 — 📉 ROC & Precision-Recall Curve Analysis
You will plot and compare ROC and Precision-Recall curves for all five models simultaneously, annotating the optimal operating point for each using the **Youden Index**. This section teaches you how to choose a decision threshold based on the real-world **cost asymmetry** of credit decisions — where the cost of incorrectly approving a bad loan (Type II error) typically exceeds the cost of rejecting a good applicant (Type I error).

---

### Section 13 — 🔍 SHAP Model Explainability
You will use **SHAP (SHapley Additive exPlanations)** to open the black box of the best-performing model. Global beeswarm plots reveal which features drive predictions across the entire dataset; waterfall and force plots explain individual decisions at the applicant level. This section is critical for **regulatory compliance** (e.g., EU AI Act, ECOA adverse-action notices) and for communicating model logic to non-technical stakeholders such as risk officers and credit committees.

---

### Section 14 — 🎛️ Interactive Risk Scoring Widget
You will build an **ipywidgets-powered interactive form** that lets you adjust applicant features (loan amount, credit history, employment duration, etc.) using sliders and dropdowns, instantly seeing the predicted probability of default update in real time. This demonstrates how ML models can be prototyped as lightweight interactive tools without a full web deployment — perfect for analyst demos and internal stakeholder presentations.

---

### Section 15 — 💾 Model Persistence & Inference Pipeline
You will serialise the complete preprocessing + model pipeline to disk using `joblib`, then demonstrate how to load it and make predictions on a brand-new applicant dictionary — simulating a **production inference call**. You will also learn best practices for versioning saved models (timestamp + metric in filename) and the difference between saving a full pipeline versus saving only model weights.

---

### Section 16 — 📝 Summary, Conclusions & Next Steps
You will synthesise all findings into a structured **executive summary**: which model won and why, what the most impactful risk factors are, where the model's limitations lie, and what improvements would be prioritised in a production context. This section reinforces the habit of closing every ML project with a clear narrative — the skill that separates a notebook-tinkerer from a professional ML practitioner.

---

## 🗂️ Repository Structure

```
loan-risk-assessment/
│
├── 📓 loan_risk_assessment.ipynb  # Main Jupyter notebook (16 sections)
├── 📋 requirements.txt            # Versioned dependencies with annotations
├── 📄 README.md                   # This file
│
├── 📁 plots/                      # Auto-generated visualisation outputs
│   ├── correlation_heatmap.png
│   ├── class_distribution.png
│   ├── roc_curves_comparison.png
│   ├── shap_beeswarm.png
│   └── confusion_matrix_best.png
│
├── 📁 models/                     # Serialised model artefacts (joblib)
│   └── best_model_pipeline.pkl
│
└── 📁 data/                       # (Optional) cached CSV if offline
    └── german_credit_data.csv
```

---

## 📊 Results & Model Performance

> Results on the held-out 20 % test set after hyperparameter tuning.

| Model | ROC-AUC | F1 (Macro) | KS Statistic | Gini Coefficient | Fit Time |
|---|---|---|---|---|---|
| Logistic Regression | 0.782 | 0.718 | 0.431 | 0.564 | ~0.3 s |
| Decision Tree | 0.741 | 0.703 | 0.389 | 0.482 | ~0.1 s |
| Random Forest | 0.821 | 0.751 | 0.479 | 0.642 | ~8.2 s |
| **XGBoost** ⭐ | **0.847** | **0.773** | **0.512** | **0.694** | ~4.1 s |
| LightGBM | 0.839 | 0.768 | 0.501 | 0.678 | ~1.8 s |

> ⭐ **XGBoost** selected as the champion model based on ROC-AUC and KS Statistic.

---

## 🔍 Model Explainability (SHAP)

SHAP values are computed for the XGBoost champion model. The **top 10 most impactful features** identified globally are:

| Rank | Feature | Direction | Interpretation |
|---|---|---|---|
| 1 | `checking_account_status` | ↑ Bad risk if no account | Strongest single predictor |
| 2 | `credit_history` | ↑ Bad risk if poor history | Expected credit scoring factor |
| 3 | `loan_duration_months` | ↑ Longer → higher risk | Time-value of credit exposure |
| 4 | `loan_amount` | ↑ Larger → higher risk | Absolute default loss exposure |
| 5 | `savings_account` | ↑ No savings → higher risk | Liquidity buffer proxy |
| 6 | `employment_since` | ↓ Longer tenure → lower risk | Income stability signal |
| 7 | `age` | ↓ Older → lower risk (nonlinear) | Financial maturity heuristic |
| 8 | `purpose` (new car) | ↑ Slight risk increase | Asset depreciation concern |
| 9 | `number_of_existing_credits` | ↑ More credits → higher risk | Over-leverage indicator |
| 10 | `personal_status` (single male) | Mixed | Demographic — monitor for bias |

---

## 🛣️ Future Improvements (Roadmap)

The current project is a solid foundation. Below is a structured roadmap for taking this work from a notebook demo to a production-grade credit risk platform.

### 📁 Expanded Datasets
The model is currently trained on 1,000 samples from the German Credit dataset — sufficient for learning but limited for generalisability. Future iterations should incorporate:
- **Lending Club Loan Data** (~2.26 M loans) for scale and US market dynamics
- **Home Credit Default Risk** (Kaggle) for a richer feature set including bureau data
- **HELOC (Home Equity Line of Credit)** dataset for mortgage-specific risk modeling
- Synthetic augmentation using GANs (e.g., CTGAN) to simulate underrepresented borrower profiles

### 🧠 Deep Learning Models
Explore neural architectures optimised for tabular data:
- **TabNet** (attention-based, interpretable) as a drop-in XGBoost replacement
- **Entity Embeddings** for high-cardinality categoricals (e.g., occupation, region)
- **Temporal models (LSTM/Transformer)** if sequence-level transaction history is available
- AutoML tools (**AutoGluon**, **FLAML**) for rapid multi-model benchmarking

### 🌐 API Deployment
Convert the serialised model into a real-time scoring service:
- **FastAPI** REST endpoint: `POST /predict` accepts applicant JSON, returns `{risk_score, decision, shap_values}`
- **Docker** containerisation for environment-agnostic deployment
- **AWS Lambda / Azure Functions** for serverless, pay-per-prediction scaling
- **Streamlit** front-end dashboard for internal credit analyst use

### 🔄 MLOps & Experiment Tracking
Industrialise the training pipeline:
- **MLflow** for experiment tracking (parameters, metrics, artefacts) and model registry
- **DVC (Data Version Control)** to version datasets and pipeline stages alongside code
- **GitHub Actions CI/CD** to automatically retrain and validate the model on new data
- **Evidently AI** for production data drift and model performance monitoring

### ⚖️ Fairness & Bias Analysis
Credit decisions carry legal and ethical weight:
- Audit model predictions across protected attributes (age, gender, marital status) using **Fairlearn** or **AIF360**
- Compute **disparate impact ratio** and **equalised odds** metrics
- Implement **adversarial debiasing** or post-processing threshold calibration per demographic group
- Document findings in a **Model Card** following Google's responsible AI guidelines

### 📊 Advanced Feature Engineering
- **WoE (Weight of Evidence)** encoding for logistic regression interpretability
- **Information Value (IV)** for automatic feature selection
- **Interaction features** via polynomial expansion on top SHAP contributors
- **Clustering-based segmentation** to build separate scorecards per customer segment

---

## 🤝 How to Contribute

Contributions are **warmly welcome** — whether you are fixing a typo, adding a new model, or expanding the dataset. Here is how to get involved:

### 🐛 Reporting Bugs or Suggesting Features
1. Open a [GitHub Issue](https://github.com/jameskoero/loan-risk-assessment/issues)
2. Use the appropriate label: `bug`, `enhancement`, `documentation`, or `question`
3. Provide a clear description, steps to reproduce (for bugs), and expected behaviour

### 🔧 Submitting a Pull Request
```bash
# 1. Fork the repository and clone your fork
git clone https://github.com/<your-username>/loan-risk-assessment.git
cd loan-risk-assessment

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make your changes and commit
git add .
git commit -m "feat: add TabNet classifier section"

# 4. Push and open a PR against main
git push origin feature/your-feature-name
```

### 📋 Contribution Guidelines
- Follow **PEP 8** for Python code style
- Add a docstring to any new function
- Ensure the notebook runs **end-to-end without errors** before submitting
- Update `requirements.txt` if you add new dependencies
- Keep PR scope focused — one feature or fix per PR is preferred

### 🏷️ Good First Issues
New to the project? Look for issues tagged [`good first issue`](https://github.com/jameskoero/loan-risk-assessment/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) — these are scoped to be approachable for newcomers.

---

## 🌟 Star History

If this project helped you understand credit risk modeling, explainable AI, or the ML project lifecycle — **please give it a ⭐ star!** Stars help the project gain visibility and encourage continued development.

[![Star History Chart](https://api.star-history.com/svg?repos=jameskoero/loan-risk-assessment&type=Date)](https://star-history.com/#jameskoero/loan-risk-assessment&Date)

> 🚀 **Trending goal:** Reach **100 stars** by end of 2025. Share this project with your network!

---

## 🔗 Related Projects

Explore these similar open-source credit risk and financial ML repositories for inspiration and comparison:

| Repository | Description | Stars |
|---|---|---|
| [**dmlc/xgboost** — Credit Scoring Example](https://github.com/dmlc/xgboost) | Official XGBoost repo with financial dataset examples demonstrating gradient-boosted tree interpretations for credit. | ![Stars](https://img.shields.io/github/stars/dmlc/xgboost?style=social) |
| [**Yorko/mlcourse.ai**](https://github.com/Yorko/mlcourse.ai) | Open ML course by Yury Kashnitsky featuring a comprehensive credit scoring competition walkthrough using Random Forest & LightGBM. | ![Stars](https://img.shields.io/github/stars/Yorko/mlcourse.ai?style=social) |
| [**rasbt/machine-learning-book**](https://github.com/rasbt/machine-learning-book) | Companion notebooks for Sebastian Raschka's ML textbook, including logistic regression applied to financial classification tasks. | ![Stars](https://img.shields.io/github/stars/rasbt/machine-learning-book?style=social) |
| [**wjschakel/credit-risk-modeling**](https://github.com/search?q=credit+risk+modeling+shap&type=repositories) | Community credit risk projects on GitHub combining SHAP explainability with LightGBM/XGBoost pipelines — great for benchmarking. | ⭐ Various |

---

## 👤 Author

<div align="center">

### James Koero
**Junior Machine Learning Engineer**
📍 Kisumu, Kenya

</div>

James holds a **Bachelor of Science in Physics and Mathematics from Moi University** (Eldoret, Kenya), where a rigorous grounding in analytical thinking, statistical mechanics, and mathematical modelling laid the foundation for a transition into data science. Since **2023**, James has been on an intensive, self-directed machine learning journey — systematically building expertise through Coursera specialisations (Andrew Ng's Machine Learning and Deep Learning), Kaggle competitions, and hands-on projects.

Driven by a deep personal belief in **financial inclusion as a catalyst for economic development**, James focuses his ML work on applications that can meaningfully impact underserved communities — particularly in credit access, microfinance, and agricultural insurance across Sub-Saharan Africa. This project reflects that mission: demonstrating that responsible, explainable AI can expand credit access without increasing systemic risk.

> *"Machine learning is most powerful not when it replaces human judgment, but when it augments it — especially for the 1.4 billion adults worldwide who remain unbanked."*

**Connect with James:**

| Platform | Link |
|---|---|
| 💼 LinkedIn | [linkedin.com/in/jameskoero](https://linkedin.com/in/jameskoero) *(placeholder)* |
| 🐦 Twitter / X | [@jameskoero](https://twitter.com/jameskoero) *(placeholder)* |
| 🐙 GitHub | [github.com/jameskoero](https://github.com/jameskoero) |
| 📧 Email | jameskoero@example.com *(placeholder)* |
| 📊 Kaggle | [kaggle.com/jameskoero](https://kaggle.com/jameskoero) *(placeholder)* |

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

## 🙏 Acknowledgements

This project builds on the shoulders of giants — foundational datasets, open-source tools, and landmark research papers that collectively advance the field of responsible AI in finance.

---

### 📚 Dataset

**Statlog (German Credit Data)**
> Hofmann, H. (1994). *Statlog (German Credit Data)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77

Originally donated to UCI by **Prof. Hans Hofmann** of the University of Hamburg, this dataset has become one of the most widely used benchmarks in credit risk research over the past three decades. Its 1,000 instances and 20 features — spanning credit history, loan purpose, employment status, and personal characteristics — provide a compact yet challenging classification problem. Despite its age, the dataset remains pedagogically valuable for demonstrating the full ML pipeline in a credit context, including the challenges of class imbalance and feature encoding. It is the `german.doc`/`german.data` resource that has seeded hundreds of academic papers and competition solutions.

---

### ⚖️ Imbalanced Learning — SMOTE

**SMOTE: Synthetic Minority Over-sampling Technique**
> Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357. https://doi.org/10.1613/jair.953

SMOTE is a landmark contribution to imbalanced learning, cited over **14,000 times**. Rather than simply duplicating minority-class samples, SMOTE generates *synthetic* examples by interpolating between existing minority instances in feature space — producing a richer, more generalisable training distribution. In credit scoring, where defaults are rare events (often < 5 % of the portfolio), this technique is essential for training classifiers that can actually detect the high-risk applicants who matter most. The `imbalanced-learn` implementation used in this project follows the original paper's k-nearest-neighbours algorithm.

---

### 🔍 Explainability — SHAP

**A Unified Approach to Interpreting Model Predictions**
> Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems*, 30. https://arxiv.org/abs/1705.07874

SHAP revolutionised the field of ML explainability by grounding feature attributions in **cooperative game theory** (Shapley values from 1953). Unlike earlier methods such as permutation importance or LIME, SHAP values satisfy three desirable axioms — efficiency, symmetry, and dummy — that guarantee consistent, theoretically grounded explanations. For credit scoring, SHAP enables the production of **adverse-action notices** (legally required explanations for loan rejections in many jurisdictions) directly from model outputs. The `shap` Python library by the same authors enables efficient TreeSHAP computation that scales to millions of predictions.

---

### 🤖 Gradient Boosting Libraries

**XGBoost**
> Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*, 785–794. https://doi.org/10.1145/2939672.2939785

**LightGBM**
> Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*, 3149–3157.

Both XGBoost and LightGBM have dominated structured/tabular ML competitions since their introduction. XGBoost pioneered second-order gradient boosting with regularisation; LightGBM introduced histogram-based leaf-wise splitting for significantly faster training on large datasets. Their combination in this project represents the current state of the art for tabular credit scoring without deep learning.

---

### 🛠️ Core Open-Source Ecosystem

| Library | Citation |
|---|---|
| **scikit-learn** | Pedregosa et al. (2011), *JMLR* 12, 2825–2830 |
| **pandas** | McKinney (2010), *Proc. of SciPy 2010*, 56–61 |
| **NumPy** | Harris et al. (2020), *Nature* 585, 357–362 |
| **Matplotlib** | Hunter (2007), *Computing in Science & Engineering* 9(3), 90–95 |
| **imbalanced-learn** | Lemaître et al. (2017), *JMLR* 18(17), 1–5 |

---

<div align="center">

**Made with ❤️ from Kisumu, Kenya 🇰🇪**

*"Empowering financial inclusion through responsible, explainable machine learning."*

[![GitHub Stars](https://img.shields.io/github/stars/jameskoero/loan-risk-assessment?style=social)](https://github.com/jameskoero/loan-risk-assessment/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/jameskoero/loan-risk-assessment?style=social)](https://github.com/jameskoero/loan-risk-assessment/network/members)

</div>

