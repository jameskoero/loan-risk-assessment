# Architecture

## Overview

The Loan Risk Assessment system is a **single-file ML pipeline** (`src/loan_risk_assessment.py`) that is also exposed as a Python package via `src/__init__.py`. It follows a **functional design** — each pipeline stage is an independent function that can be called in isolation or composed into an end-to-end run via `main()`.

```
load_data()
    │
    ▼
run_eda()
    │
    ▼
engineer_features()
    │
    ▼
build_preprocessor()  ──►  ColumnTransformer (impute + scale/encode)
    │
    ▼
SMOTE (optional, requires imbalanced-learn)
    │
    ▼
train_models()  ──►  5 classifiers trained and evaluated
    │
    ▼
tune_best_model()  ──►  GridSearchCV over GradientBoosting
    │
    ▼
evaluation_plots()  ──►  ROC, PR, confusion matrices, bar chart
    │
    ▼
optimise_threshold()  ──►  Business cost minimisation
    │
    ▼
plot_feature_importance()
    │
    ▼
save_model()  ──►  models/loan_risk_model.joblib + model_metadata.json
```

## Component Details

### Data Layer

- **Source**: German Credit Data via `sklearn.datasets.fetch_openml` (auto-downloaded).
- **Target**: Binary `default` column derived from the original `class` column.

### Preprocessing

A `ColumnTransformer` applies separate pipelines to numeric and categorical columns:

- **Numeric**: `SimpleImputer(strategy="median")` → `StandardScaler()`
- **Categorical**: `SimpleImputer(strategy="constant", fill_value="missing")` → `OneHotEncoder(handle_unknown="ignore")`

### Models

| Model | Library | Notes |
|-------|---------|-------|
| Logistic Regression | scikit-learn | Baseline linear model |
| Decision Tree | scikit-learn | Depth-limited to avoid overfitting |
| Random Forest | scikit-learn | 300 trees, balanced class weights |
| Gradient Boosting | scikit-learn | Primary model |
| XGBoost / HGB | xgboost (optional) | Falls back to `GradientBoostingClassifier` |

### Outputs

| Artifact | Location | Description |
|----------|----------|-------------|
| Plots | `plots/*.png` | 10+ diagnostic charts |
| Trained pipeline | `models/loan_risk_model.joblib` | Preprocessing + model |
| Metadata | `models/model_metadata.json` | Params, AUC, threshold |
