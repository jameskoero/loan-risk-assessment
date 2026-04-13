"""
generate_notebook.py — Rebuild loan_risk_assessment.ipynb from the pipeline script.

Splits loan_risk_assessment.py on '# ─── ...' section markers; each section
becomes one code cell.  A Colab-badge markdown cell and a pip-install cell are
prepended automatically.

Usage::

    python generate_notebook.py
"""

import json
import re

SCRIPT = "loan_risk_assessment.py"
NOTEBOOK = "loan_risk_assessment.ipynb"

COLAB_BADGE = (
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/jameskoero/loan-risk-assessment"
    "/blob/main/loan_risk_assessment.ipynb)"
)

PIP_CELL = "!pip install -q numpy pandas scikit-learn imbalanced-learn xgboost lightgbm shap matplotlib seaborn joblib"

SECTION_RE = re.compile(r"^# ─── .+", re.MULTILINE)


def make_cell(cell_type: str, source: list[str]) -> dict:
    base = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source,
    }
    if cell_type == "code":
        base.update({"outputs": [], "execution_count": None})
    return base


def split_script(text: str) -> list[str]:
    """Return a list of source-code blocks split on section-header comments."""
    boundaries = [m.start() for m in SECTION_RE.finditer(text)]
    if not boundaries:
        return [text]
    segments: list[str] = []
    # Everything before the first section marker
    preamble = text[: boundaries[0]].strip()
    if preamble:
        segments.append(preamble)
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        segments.append(text[start:end].strip())
    return segments


def to_source_lines(block: str) -> list[str]:
    lines = block.splitlines(keepends=True)
    # Strip trailing blank lines at end
    while lines and lines[-1].strip() == "":
        lines.pop()
    return lines


def build_notebook(segments: list[str]) -> dict:
    cells = [
        make_cell("markdown", [COLAB_BADGE]),
        make_cell("code", [PIP_CELL]),
    ]
    for seg in segments:
        cells.append(make_cell("code", to_source_lines(seg)))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
            "colab": {
                "provenance": [],
            },
        },
        "cells": cells,
    }


def main() -> None:
    with open(SCRIPT, encoding="utf-8") as fh:
        text = fh.read()

    segments = split_script(text)
    nb = build_notebook(segments)

    with open(NOTEBOOK, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")

    print(f"Generated {NOTEBOOK} with {len(nb['cells'])} cells.")


if __name__ == "__main__":
    main()
