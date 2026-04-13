# API Reference

This document describes all public functions exposed by `src/loan_risk_assessment.py`.

---

## `load_data() → pd.DataFrame`

Downloads the German Credit dataset from OpenML (ID 31) and returns a clean DataFrame.

**Returns**

| Column | Type | Description |
|--------|------|-------------|
| all original 20 columns | mixed | Raw feature columns |
| `default` | int | Target — 1 = bad credit risk, 0 = good |

**Example**

```python
from src.loan_risk_assessment import load_data

df = load_data()
print(df.shape)   # (1000, 21)
```

---

## `run_eda(df) → tuple[list, list, pd.Series]`

Runs Exploratory Data Analysis and saves plots to `plots/`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `df` | `pd.DataFrame` | Raw dataframe from `load_data()` |

**Returns** `(num_cols, cat_cols, target_corr)` — lists of numeric/categorical column names and a Series of correlations with the target.

---

## `engineer_features(df) → pd.DataFrame`

Adds 6 domain-inspired features to the dataframe:

- `debt_to_credit_ratio`
- `loan_income_proxy`
- `long_term_loan` (binary)
- `high_credit` (binary)
- `senior_applicant` (binary)
- `age_group` (categorical)

---

## `build_preprocessor(X) → tuple[ColumnTransformer, list, list]`

Builds a sklearn `ColumnTransformer` that imputes and scales numeric features, and imputes + one-hot encodes categorical features.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `X` | `pd.DataFrame` | Feature matrix (without target) |

**Returns** `(preprocessor, num_features, cat_features)`

---

## `train_models(X_tr, y_tr, X_te, y_te) → tuple[dict, dict]`

Trains Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and XGBoost (or fallback HGB), then evaluates each on the test set.

**Returns** `(results, models)` — dicts keyed by model name.

---

## `tune_best_model(X_tr, y_tr, X_te, y_te) → tuple`

Runs `GridSearchCV` over Gradient Boosting hyperparameters and returns the best estimator.

**Returns** `(best_model, y_pred, y_proba, grid_search_cv)`

---

## `evaluation_plots(results, y_te, best_model, X_te, y_proba_best) → None`

Saves ROC curves, PR curves, confusion matrices, and metrics bar chart to `plots/`.

---

## `optimise_threshold(y_te, y_proba) → float`

Searches for the classification threshold that minimises a business cost function
(configurable via `COST_FN` and `COST_FP` module-level constants).

**Returns** Optimal threshold value.

---

## `plot_feature_importance(model, feature_names, top_n=20) → None`

Plots and saves the top-N feature importances from a fitted tree ensemble.

---

## `save_model(pipeline, best_params, best_cv_auc, y_te, y_pred, y_proba, threshold) → None`

Serialises the full `Pipeline` (preprocessor + model) to `models/loan_risk_model.joblib`
and writes `models/model_metadata.json`.
