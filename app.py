"""
app.py — Minimal Flask REST API for the Loan Default Risk Assessment model.

Endpoints
---------
  GET  /health
       Returns {"status": "ok"} — use for readiness/liveness probes.

  POST /predict
       Accepts a JSON body containing loan application features.
       Returns the default probability, risk label, and hard prediction.

       Example request body (single applicant):
       {
           "duration": 24,
           "credit_amount": 5000,
           "installment_commitment": 4,
           "age": 35,
           "existing_credits": 1,
           "num_dependents": 1,
           "residence_since": 2,
           "checking_status": "no checking",
           "credit_history": "existing paid",
           "purpose": "furniture/equipment",
           "savings_status": "<100",
           "employment": "1<=X<4",
           "personal_status": "male single",
           "other_parties": "none",
           "property_magnitude": "real estate",
           "other_payment_plans": "none",
           "housing": "own",
           "job": "skilled",
           "own_telephone": "yes",
           "foreign_worker": "yes"
       }

       Example response:
       {
           "default_probability": 0.23,
           "risk_label": "Low",
           "predicted_default": 0
       }

       You may also send a list of applicants (JSON array) to score in bulk.

Usage
-----
  pip install flask
  python app.py

  The server starts on http://0.0.0.0:5000 by default.
  Set PORT environment variable to override.

Requirements
------------
  Trained model at:  models/loan_risk_model.joblib
  Metadata at:       models/model_metadata.json
  Run src/loan_risk_assessment.py first to generate these files.
"""

import json
import os
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH    = "models/loan_risk_model.joblib"
METADATA_PATH = "models/model_metadata.json"
DEFAULT_THRESHOLD = 0.5

app = Flask(__name__)

# ── Load model once at startup ────────────────────────────────────────────────
try:
    _pipeline = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from '{MODEL_PATH}'")
except FileNotFoundError:
    print(
        f"❌  Model not found at '{MODEL_PATH}'.\n"
        "    Run  python src/loan_risk_assessment.py  first.",
        file=sys.stderr,
    )
    _pipeline = None

try:
    with open(METADATA_PATH) as _f:
        _meta = json.load(_f)
    _threshold = _meta.get("optimal_threshold", DEFAULT_THRESHOLD)
    print(f"ℹ️  Using classification threshold: {_threshold:.4f}")
except FileNotFoundError:
    _threshold = DEFAULT_THRESHOLD
    print(f"⚠️  Metadata not found. Using default threshold={DEFAULT_THRESHOLD}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _label_risk(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    if prob < 0.60:
        return "Medium"
    return "High"


def _score_df(df: pd.DataFrame):
    X = df.drop(columns=["default"], errors="ignore")
    probabilities = _pipeline.predict_proba(X)[:, 1]
    return [
        {
            "default_probability": round(float(p), 4),
            "risk_label": _label_risk(p),
            "predicted_default": int(p >= _threshold),
        }
        for p in probabilities
    ]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Liveness / readiness probe."""
    if _pipeline is None:
        return jsonify({"status": "error", "detail": "Model not loaded"}), 503
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Score one or many loan applications."""
    if _pipeline is None:
        return jsonify({"error": "Model not loaded. Run the training script first."}), 503

    payload = request.get_json(force=True, silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400

    # Accept either a single object or a list of objects
    records = payload if isinstance(payload, list) else [payload]

    try:
        df     = pd.DataFrame(records)
        result = _score_df(df)
    except Exception as exc:
        app.logger.error("Prediction error: %s", exc)
        return jsonify({"error": "Failed to score the provided input. Check that all required fields are present and correctly formatted."}), 422

    # Return a single object if the request was a single object
    if not isinstance(payload, list):
        return jsonify(result[0]), 200
    return jsonify(result), 200


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
