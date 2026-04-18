"""
predict.py — Inference script for the Loan Default Risk Assessment model.

Usage
-----
  # Score a CSV of new loan applications:
  python predict.py --input new_applications.csv --output scores.csv

  # Score a single applicant supplied as JSON (useful for quick tests):
  python predict.py --json '{"duration":24,"credit_amount":5000,"age":35,...}'

Output columns added to the input CSV (or printed for --json):
  default_probability : float [0, 1]  — model's estimated default probability
  risk_label          : str           — "Low" / "Medium" / "High"
  predicted_default   : int 0/1       — hard prediction at the optimal threshold

Requirements
------------
  pip install scikit-learn joblib pandas
  The trained model must exist at:  model/loan_risk_model.joblib
  Metadata (including the optimal threshold) must exist at:
                                     model/model_metadata.json
  Run loan_risk_assessment.py first to generate these files.
"""

import argparse
import json
import sys
import joblib
import pandas as pd

MODEL_PATH    = "model/loan_risk_model.joblib"
METADATA_PATH = "model/model_metadata.json"

# Default threshold used when metadata is not available
DEFAULT_THRESHOLD = 0.5


def load_pipeline():
    try:
        pipeline = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        sys.exit(
            f"❌  Model not found at '{MODEL_PATH}'.\n"
            "    Run  python loan_risk_assessment.py  first to train and save the model."
        )
    return pipeline


def load_threshold():
    try:
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        return meta.get("optimal_threshold", DEFAULT_THRESHOLD)
    except FileNotFoundError:
        print(f"⚠️  Metadata not found at '{METADATA_PATH}'. Using threshold={DEFAULT_THRESHOLD}")
        return DEFAULT_THRESHOLD


def label_risk(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    if prob < 0.60:
        return "Medium"
    return "High"


def score(df: pd.DataFrame, pipeline, threshold: float) -> pd.DataFrame:
    # Drop the target column if it was accidentally included in the input
    X = df.drop(columns=["default"], errors="ignore")

    probabilities = pipeline.predict_proba(X)[:, 1]
    result = df.copy()
    result["default_probability"] = probabilities.round(4)
    result["risk_label"]          = [label_risk(p) for p in probabilities]
    result["predicted_default"]   = (probabilities >= threshold).astype(int)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Score loan applications using the trained risk model."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", "-i",
        help="Path to a CSV file containing loan application features."
    )
    group.add_argument(
        "--json", "-j",
        help="Single application as a JSON string."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to write output CSV. If omitted, results are printed to stdout."
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Classification threshold (overrides the value stored in metadata)."
    )
    args = parser.parse_args()

    pipeline  = load_pipeline()
    threshold = args.threshold if args.threshold is not None else load_threshold()
    print(f"ℹ️  Using classification threshold: {threshold:.4f}")

    if args.input:
        df = pd.read_csv(args.input)
        print(f"✅ Loaded {len(df)} application(s) from '{args.input}'")
    else:
        record = json.loads(args.json)
        df     = pd.DataFrame([record])

    scored = score(df, pipeline, threshold)

    if args.output:
        scored.to_csv(args.output, index=False)
        print(f"✅ Scores written to '{args.output}'")
    else:
        print(scored[["default_probability", "risk_label", "predicted_default"]].to_string(index=False))


if __name__ == "__main__":
    main()
