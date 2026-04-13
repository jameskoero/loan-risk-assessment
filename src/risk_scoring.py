"""
Risk scoring: translate model probabilities into actionable risk categories.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


class RiskScorer:
    """
    Wrap a trained model + preprocessor to provide human-readable risk assessments.
    """

    THRESHOLDS = {
        'low':       0.20,
        'medium':    0.40,
        'high':      0.60,
        'very_high': 1.0,
    }

    RISK_LABELS = ['Low', 'Medium', 'High', 'Very High']

    def __init__(self, model, preprocessor, feature_engineer_fn=None):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_engineer_fn = feature_engineer_fn
        self.decision_threshold: float = 0.5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_single(self, loan_data: dict) -> dict:
        """
        Score a single loan application.

        Parameters
        ----------
        loan_data : dict with loan feature values (same keys as training columns).

        Returns
        -------
        dict with 'probability', 'risk_category', 'recommendation'.
        """
        df = pd.DataFrame([loan_data])
        result = self.score_batch(df)
        return {
            'probability':    float(result['default_probability'].iloc[0]),
            'risk_category':  str(result['risk_category'].iloc[0]),
            'recommendation': str(result['recommendation'].iloc[0]),
        }

    def score_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a batch of loan applications.

        Returns *df* with three new columns:
        default_probability, risk_category, recommendation.
        """
        df_out = df.copy()

        # Optional feature engineering step
        if self.feature_engineer_fn is not None:
            df_out = self.feature_engineer_fn(df_out)

        X = self.preprocessor.transform(df_out)
        probs = self.model.predict_proba(X)[:, 1]

        df_out['default_probability'] = probs
        df_out['risk_category']  = [self._categorise(p) for p in probs]
        df_out['recommendation'] = [self.get_recommendation(p) for p in probs]
        return df_out

    def get_recommendation(self, probability: float) -> str:
        """Map a default probability to a lending recommendation."""
        if probability < self.THRESHOLDS['low']:
            return 'Approve'
        if probability < self.THRESHOLDS['medium']:
            return 'Review'
        if probability < self.THRESHOLDS['high']:
            return 'Decline'
        return 'Decline - High Risk'

    def find_optimal_threshold(self, X_val, y_val, metric: str = 'f1') -> float:
        """
        Search [0.05, 0.95] for the classification threshold that maximises *metric*.

        Supported metrics: 'f1', 'precision', 'recall'.

        Returns the optimal threshold and stores it in self.decision_threshold.
        """
        probs = self.model.predict_proba(X_val)[:, 1]
        thresholds = np.linspace(0.05, 0.95, 91)
        best_score = -1.0
        best_threshold = 0.5

        scorer_map = {
            'f1':        lambda y, p: f1_score(y, p, zero_division=0),
            'precision': lambda y, p: precision_score(y, p, zero_division=0),
            'recall':    lambda y, p: recall_score(y, p, zero_division=0),
        }
        score_fn = scorer_map.get(metric, scorer_map['f1'])

        for t in thresholds:
            preds = (probs >= t).astype(int)
            score = score_fn(y_val, preds)
            if score > best_score:
                best_score = score
                best_threshold = float(t)

        self.decision_threshold = best_threshold
        return best_threshold

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _categorise(self, probability: float) -> str:
        if probability < self.THRESHOLDS['low']:
            return 'Low'
        if probability < self.THRESHOLDS['medium']:
            return 'Medium'
        if probability < self.THRESHOLDS['high']:
            return 'High'
        return 'Very High'
