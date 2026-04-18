# Makefile — convenient commands for the loan-risk-assessment project
# Usage: make <target>   e.g.  make setup  |  make run  |  make test

.PHONY: setup run test api clean help

## Install all Python dependencies
setup:
	pip install -r requirements.txt

## Run the full ML pipeline (downloads data, trains, evaluates, saves model + plots)
run:
	python src/loan_risk_assessment.py

## Run the test suite
test:
	pytest tests/ -v

## Start the Flask prediction API (model must be trained first)
api:
	python app.py

## Remove generated artefacts (plots, model files, __pycache__)
clean:
	rm -rf plots/ images/ models/ __pycache__ .pytest_cache tests/__pycache__

## Show this help message
help:
	@echo ""
	@echo "Available targets:"
	@echo "  make setup   — install dependencies from requirements.txt"
	@echo "  make run     — train and evaluate the full ML pipeline"
	@echo "  make test    — run pytest unit tests"
	@echo "  make api     — start the Flask prediction API on port 5000"
	@echo "  make clean   — remove generated plots, model files, caches"
	@echo ""
