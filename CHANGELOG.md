# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] — 2026-04-18

### Added
- **`requirements.txt`** — pinned dependency file listing all packages needed to
  reproduce the project environment. Install with `pip install -r requirements.txt`.
- **`loan_risk_assessment.ipynb`** — Jupyter Notebook for step-by-step interactive
  walkthrough. (Replaces the incorrectly named `loan_risk_assessment-2.ipynb.txt`.)
- **SHAP explainability** (`run_shap_analysis()`) — global summary plot and
  per-prediction waterfall plot are now generated during the pipeline run and
  saved to both `plots/` and `images/`.
- **Learning curve plot** (`plot_learning_curve()`) — visualises training vs.
  cross-validation AUC as a function of training-set size, helping diagnose
  bias/variance trade-offs.
- **Calibration curve** (`plot_calibration()`) — checks whether the model's
  predicted probabilities match observed default rates (critical for finance).
- **`predict.py`** — command-line inference script. Loads the saved model and
  scores new loan applications from a CSV file or JSON string.
- **`app.py`** — minimal Flask REST API with `/health` and `/predict` endpoints
  for serving the model as a web service.
- **`tests/test_pipeline.py`** — `pytest` unit tests covering `engineer_features()`,
  `build_preprocessor()`, `predict.py` helpers, and an optional saved-model
  round-trip test.
- **`Makefile`** — one-command shortcuts: `make setup`, `make run`, `make test`,
  `make api`, `make clean`.
- **`images/` directory** — key plots (ROC curve, confusion matrix, feature
  importance, SHAP summary/waterfall) are now also saved here so they render
  correctly in the GitHub README.

### Changed
- `COST_FN` and `COST_FP` constants now include detailed inline comments
  explaining their business meaning and how to tune them for different lending
  strategies.
- The pipeline's final print statement now mentions both `plots/` and `images/`
  as output destinations.

### Removed
- `loan_risk_assessment-2.ipynb.txt` — replaced by `loan_risk_assessment.ipynb`
  with the correct file extension.
- `README-16.md` — duplicate of `README.md`; deleted to keep the repository clean.

---

## [1.0.0] — Initial Release

### Added
- End-to-end ML pipeline: data loading (OpenML), EDA, feature engineering,
  SMOTE resampling, multi-model training (Logistic Regression, Decision Tree,
  Random Forest, Gradient Boosting, XGBoost), GridSearchCV tuning, evaluation
  plots, business-cost threshold optimisation, feature importance, and model
  export via `joblib`.
- `generate_notebook.py` — script to auto-generate the Jupyter Notebook from
  the main Python script.
- `DESCRIPTION.md` — extended project description.
- `README.md` — project overview with mermaid diagrams, performance table, and
  usage instructions.
