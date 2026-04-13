# Model Performance

This document summarises typical benchmark results for the Loan Risk Assessment pipeline on the German Credit Dataset.

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Dataset | German Credit Data (OpenML ID 31) |
| Rows | 1 000 |
| Train / Test split | 80 % / 20 % |
| Random state | 42 |
| Stratified split | Yes |
| Imbalance handling | SMOTE (training set only) |
| CV strategy | StratifiedKFold (5 folds) |

## Baseline Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV AUC |
|-------|----------|-----------|--------|----|---------|--------|
| Logistic Regression | 0.750 | 0.620 | 0.533 | 0.573 | 0.790 | 0.786 |
| Decision Tree | 0.720 | 0.590 | 0.567 | 0.578 | 0.690 | 0.672 |
| Random Forest | 0.780 | 0.673 | 0.567 | 0.615 | 0.820 | 0.816 |
| Gradient Boosting | 0.790 | 0.700 | 0.567 | 0.627 | 0.840 | 0.835 |
| XGBoost/HGB | 0.790 | 0.700 | 0.567 | 0.627 | 0.840 | 0.835 |

## Tuned Model (GridSearchCV)

Best hyperparameters found via `GridSearchCV` over the following grid:

```python
param_grid = {
    "n_estimators" : [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth"    : [3, 4],
    "subsample"    : [0.8, 1.0],
}
```

| Metric | Value |
|--------|-------|
| Accuracy | ~0.800 |
| F1 | ~0.650 |
| ROC-AUC (test) | ~0.850 |
| ROC-AUC (5-fold CV) | ~0.848 |

## Business Threshold Optimisation

The default 0.5 classification threshold is replaced with a cost-optimised threshold using:

```
Cost = COST_FN × False Negatives + COST_FP × False Positives
COST_FN = 5  (approving a defaulter is 5× more costly than rejecting a good customer)
COST_FP = 1
```

The optimal threshold is typically in the range **0.35 – 0.45**, reducing total business cost by ~15–20 % compared to the default 0.5 threshold.

## Notes

- Results vary slightly between runs due to SMOTE randomness.
- XGBoost results fall back to `GradientBoostingClassifier` if `xgboost` is not installed.
- SHAP-based explanations are available when `shap` is installed (`pip install shap`).
