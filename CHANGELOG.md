# Changelog

All notable changes to this project are documented in this file.

## [1.2.0] - 2026-04-18

### Changed
- Reorganized repository into a production-style layout with `src/`, `docs/`, `tests/`, and `notebooks/`.
- Moved training pipeline to `src/loan_risk_assessment.py` and added package exports in `src/__init__.py`.
- Standardized artifact directory naming from `model/` to `models/` across training, inference, API, tests, and notebook.
- Updated tests file to `tests/test_loan_risk_assessment.py`.
- Replaced root README with consolidated documentation and corrected paths.

### Added
- `docs/API.md`
- `docs/ARCHITECTURE.md`
- `docs/MODEL_PERFORMANCE.md`
- `docs/CONTRIBUTING.md`
- `setup.py`
- `.gitignore`
- `LICENSE`

### Removed
- `generate_notebook.py`
- `description`
- `DESCRIPTION.md`
- duplicate legacy license file (`LICENSE (4).txt`)

## [1.1.0] - 2026-04-18

### Added
- Pipeline enhancements, inference scripts, tests, and automation helpers.
