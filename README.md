# Loan Default Risk Assessment

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-pytest-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Production-ready ML project for predicting loan default risk with explainable outputs.

## Why this project

Lenders need default predictions that are both accurate and explainable. This repository provides an end-to-end pipeline with training, evaluation, explainability (SHAP), and serving interfaces.

## Features

- End-to-end training pipeline (`src/loan_risk_assessment.py`)
- Feature engineering + preprocessing utilities
- Model artifact + metadata export to `models/`
- CLI inference (`predict.py`) and Flask API (`app.py`)
- Jupyter walkthrough (`notebooks/loan_risk_assessment.ipynb`)
- Unit tests with `pytest`

## Installation

```bash
git clone https://github.com/jameskoero/loan-risk-assessment.git
cd loan-risk-assessment
pip install -r requirements.txt
pip install -e .
```

## Quick start

Train and save model artifacts:

```bash
python src/loan_risk_assessment.py
```

Run tests:

```bash
pytest tests/ -v
```

Score new records from CSV:

```bash
python predict.py --input input.csv --output scores.csv
```

Run API:

```bash
python app.py
```

## API overview

- `GET /health` → service readiness
- `POST /predict` → score one applicant (JSON object) or many (JSON array)

See [docs/API.md](docs/API.md) for full request/response details.

## Model performance (latest baseline)

- Accuracy: **0.87**
- ROC-AUC: **0.92**
- Precision: **0.84**
- Recall: **0.81**
- F1: **0.82**

See [docs/MODEL_PERFORMANCE.md](docs/MODEL_PERFORMANCE.md) for benchmark details.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Model Performance](docs/MODEL_PERFORMANCE.md)
- [Contributing](docs/CONTRIBUTING.md)

## Project structure

```text
loan-risk-assessment/
├── docs/
│   ├── API.md
│   ├── ARCHITECTURE.md
│   ├── CONTRIBUTING.md
│   └── MODEL_PERFORMANCE.md
├── notebooks/
│   └── loan_risk_assessment.ipynb
├── src/
│   ├── __init__.py
│   └── loan_risk_assessment.py
├── tests/
│   ├── __init__.py
│   └── test_loan_risk_assessment.py
├── app.py
├── predict.py
├── setup.py
├── requirements.txt
├── .gitignore
├── CHANGELOG.md
├── LICENSE
└── README.md
```

## Contributing

Please read [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
