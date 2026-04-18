# Model Performance

## Baseline metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.87 |
| ROC-AUC | 0.92 |
| Precision | 0.84 |
| Recall | 0.81 |
| F1 | 0.82 |

## Training strategy

- Stratified split (80/20)
- Optional SMOTE balancing
- Multi-model comparison
- Gradient Boosting hyperparameter tuning with GridSearchCV
- Threshold optimization with cost function (`COST_FN`, `COST_FP`)

## Explainability

SHAP generates:

- Global feature importance summary
- Per-prediction waterfall explanation

Outputs are saved under `plots/` and `images/`.
