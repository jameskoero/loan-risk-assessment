# Architecture

## Pipeline flow

1. Load German Credit dataset from OpenML
2. Engineer derived risk features
3. Build preprocessing pipeline (numeric + categorical)
4. Train/evaluate candidate models
5. Tune Gradient Boosting with GridSearchCV
6. Optimize decision threshold by business cost
7. Export model and metadata to `models/`
8. Serve predictions via CLI/API

## Layout decisions

- `src/` contains core package code
- `tests/` validates feature engineering, preprocessing, and prediction helpers
- `docs/` centralizes project documentation
- `notebooks/` contains exploration notebook

## Artifacts

Generated at runtime:

- `models/loan_risk_model.joblib`
- `models/model_metadata.json`
- `plots/` and `images/` visual diagnostics
