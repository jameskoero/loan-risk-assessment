# Contributing to Loan Risk Assessment

Thank you for your interest in contributing! 🎉

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/loan-risk-assessment.git
   cd loan-risk-assessment
   ```
3. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install pytest
   ```

## Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes.
3. Run the tests to confirm nothing is broken:
   ```bash
   pytest tests/ -v
   ```
4. Commit with a clear message:
   ```bash
   git commit -m "feat: add XYZ feature"
   ```
5. Push to your fork and open a **Pull Request** against `main`.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Add docstrings (Google style) to all public functions.
- Keep lines ≤ 100 characters.

## Reporting Issues

- Use the [GitHub issue tracker](https://github.com/jameskoero/loan-risk-assessment/issues).
- Include a minimal reproducible example where possible.

## License

By contributing you agree that your contributions will be licensed under the [MIT License](../LICENSE).
