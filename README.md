# Loan Default Risk Assessment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jameskoero/loan-risk-assessment/blob/main/loan_risk_assessment.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Credit risk model trained on the [German Credit Dataset (OpenML)](https://www.openml.org/d/31). The goal is to predict whether a loan applicant is likely to default, using the full ML pipeline from raw data to a deployable model.

## What it does

The notebook and script walk through:

- Exploratory data analysis (distributions, correlations, categorical breakdowns)
- Feature engineering — 6 new features derived from existing columns (leverage ratio, age bands, loan term flags)
- Preprocessing with a `ColumnTransformer` pipeline (imputation, encoding, scaling)
- Class imbalance handling with SMOTE (dataset is ~70/30)
- Training and comparing 5 classifiers: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost
- Hyperparameter tuning via `GridSearchCV` on the best performer
- SHAP explainability — global feature importance and beeswarm plots
- Threshold optimisation using a business cost matrix (false negatives cost more than false positives)
- 4-tier risk scorecard: LOW / MEDIUM-LOW / MEDIUM-HIGH / HIGH
- Model export via `joblib` + a `predict_loan_risk()` inference function

All 16 plots are saved automatically to `plots/`.

## Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.72 | ~0.60 | ~0.56 | ~0.58 | ~0.76 |
| Decision Tree | ~0.70 | ~0.57 | ~0.60 | ~0.58 | ~0.69 |
| Random Forest | ~0.76 | ~0.64 | ~0.60 | ~0.62 | ~0.80 |
| Gradient Boosting | ~0.75 | ~0.65 | ~0.61 | ~0.63 | ~0.78 |
| **GB Tuned** | **~0.76** | **~0.67** | **~0.62** | **~0.64** | **~0.79** |

ROC-AUC is the main metric given the class imbalance. Scores will vary slightly depending on random seed.

## Running the project

**Option 1 — Colab (easiest)**

Click the badge above. Data loads automatically from OpenML, no uploads needed.

**Option 2 — Local**

```bash
git clone https://github.com/jameskoero/loan-risk-assessment.git
cd loan-risk-assessment
pip install -r requirements.txt

# run the script directly
python loan_risk_assessment.py

# or rebuild and open the notebook
python generate_notebook.py
jupyter notebook loan_risk_assessment.ipynb
```

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

## Files

```
loan-risk-assessment/
├── loan_risk_assessment.ipynb   — main notebook
├── loan_risk_assessment.py      — standalone script
├── generate_notebook.py         — rebuilds the .ipynb
├── README.md
├── LICENSE
├── model/
│   ├── loan_risk_model.joblib
│   └── model_metadata.json
└── plots/                       — 16 auto-generated plots
```

## About

Built by James Koero — self-taught ML practitioner based in Kisumu, Kenya. Physics/Mathematics graduate (Moi University, 2012), working in ML since 2023.

Contact: jmskoero@gmail.com · github.com/jameskoero

## Acknowledgements

- Hans Hofmann — original German Credit Dataset (1994)
- OpenML — dataset hosting
- Lundberg & Lee (2017) — SHAP
- Chawla et al. (2002) — SMOTE

## License

MIT — see [LICENSE](LICENSE).