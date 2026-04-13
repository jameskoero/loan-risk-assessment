"""
generate_notebook.py
Reads loan_risk_assessment.py and splits it into logical notebook cells,
then writes a valid Jupyter notebook JSON to loan_risk_assessment.ipynb.
Uses only Python's built-in json module (no nbformat required).
"""

import json
import re
import os


SCRIPT_PATH = 'loan_risk_assessment.py'
NOTEBOOK_PATH = 'loan_risk_assessment.ipynb'

SECTION_MARKER = re.compile(r'^# ─+')

COLAB_BADGE = (
    '[![Open In Colab]'
    '(https://colab.research.google.com/assets/colab-badge.svg)]'
    '(https://colab.research.google.com/github/jameskoero/loan-risk-assessment'
    '/blob/main/loan_risk_assessment.ipynb)'
)


def make_markdown_cell(source):
    lines = source if isinstance(source, list) else source.splitlines(keepends=True)
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': lines,
    }


def make_code_cell(source):
    lines = source if isinstance(source, list) else source.splitlines(keepends=True)
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': lines,
    }


def split_script_into_sections(text):
    """Split script on section-comment markers (# ─── ...) or double blank lines."""
    sections = []
    current = []
    for line in text.splitlines(keepends=True):
        if SECTION_MARKER.match(line) and current:
            sections.append(''.join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(''.join(current).strip())
    return [s for s in sections if s]


def build_notebook(cells):
    return {
        'nbformat': 4,
        'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {
                'name': 'python',
                'version': '3.10.0',
            },
            'colab': {
                'provenance': [],
                'collapsed_sections': [],
            },
        },
        'cells': cells,
    }


def main():
    if not os.path.exists(SCRIPT_PATH):
        raise FileNotFoundError(f"Script not found: {SCRIPT_PATH}")

    with open(SCRIPT_PATH, 'r') as f:
        script_text = f.read()

    sections = split_script_into_sections(script_text)

    cells = []

    # Title markdown cell
    cells.append(make_markdown_cell(
        '# 🏦 Advanced Loan Default Risk Assessment\n\n'
        'A comprehensive ML pipeline for credit risk scoring using the '
        'German Credit dataset. Includes SMOTE balancing, 5 models, SHAP '
        'explainability, business cost optimisation, and a 4-tier risk scorecard.\n'
    ))

    # Colab badge cell
    cells.append(make_markdown_cell(COLAB_BADGE + '\n'))

    # Install cell
    cells.append(make_code_cell(
        '!pip install imbalanced-learn xgboost lightgbm shap -q\n'
    ))

    # Code cells from sections
    for section in sections:
        cells.append(make_code_cell(section + '\n'))

    nb = build_notebook(cells)

    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"Notebook saved to {NOTEBOOK_PATH} ({len(cells)} cells)")


if __name__ == '__main__':
    main()
