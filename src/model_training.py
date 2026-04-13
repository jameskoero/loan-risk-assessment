"""
Model training for loan default risk prediction.
Trains multiple classifiers, applies SMOTE for class imbalance, and selects the best model.
"""
import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False


class LoanRiskModelTrainer:
    """Train, compare and persist multiple loan default classifiers."""

    MODELS = {
        'logistic_regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        'xgboost': XGBClassifier(
            n_estimators=100, max_depth=6, random_state=42,
            eval_metric='logloss', verbosity=0,
        ),
    }

    def __init__(self, models: dict | None = None):
        self.models = {k: v for k, v in (models or self.MODELS).items()}
        self.trained_models: dict = {}
        self.cv_scores: dict[str, float] = {}
        self.best_model_name: str | None = None
        self.best_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_all(self, X_train, y_train, use_smote: bool = True) -> None:
        """
        Train every configured model.

        Parameters
        ----------
        X_train, y_train : training data.
        use_smote : if True and imbalanced-learn is available, oversample the
                    minority class before fitting each model.
        """
        X_res, y_res = self._apply_smote(X_train, y_train, use_smote)

        for name, model in self.models.items():
            print(f"  Training {name}…")
            self.train_single(name, model, X_res, y_res)

        self.best_model_name = max(self.cv_scores, key=self.cv_scores.get)
        self.best_model = self.trained_models[self.best_model_name]
        print(f"\n  Best model: {self.best_model_name} "
              f"(CV AUC-ROC = {self.cv_scores[self.best_model_name]:.4f})")

    def train_single(self, name: str, model, X_train, y_train) -> None:
        """Fit one model and evaluate via 5-fold CV AUC-ROC."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        model.fit(X_train, y_train)
        self.trained_models[name] = model
        self.cv_scores[name] = float(np.mean(scores))

    def get_best_model(self):
        """Return the model with the highest CV AUC-ROC."""
        if self.best_model is None:
            raise RuntimeError("No models trained yet. Call train_all() first.")
        return self.best_model

    def save_models(self, dir_path: str) -> None:
        """Persist all trained models to *dir_path* using joblib."""
        os.makedirs(dir_path, exist_ok=True)
        for name, model in self.trained_models.items():
            joblib.dump(model, os.path.join(dir_path, f"{name}.pkl"))
        joblib.dump(self.cv_scores, os.path.join(dir_path, "cv_scores.pkl"))
        joblib.dump(self.best_model_name, os.path.join(dir_path, "best_model_name.pkl"))

    @classmethod
    def load_models(cls, dir_path: str) -> 'LoanRiskModelTrainer':
        """Reload a previously saved trainer from *dir_path*."""
        trainer = cls(models={})
        cv_scores_path = os.path.join(dir_path, "cv_scores.pkl")
        best_name_path = os.path.join(dir_path, "best_model_name.pkl")

        if os.path.exists(cv_scores_path):
            trainer.cv_scores = joblib.load(cv_scores_path)
        if os.path.exists(best_name_path):
            trainer.best_model_name = joblib.load(best_name_path)

        for fname in os.listdir(dir_path):
            if fname.endswith('.pkl') and fname not in ('cv_scores.pkl', 'best_model_name.pkl'):
                name = fname[:-4]
                trainer.trained_models[name] = joblib.load(os.path.join(dir_path, fname))

        if trainer.best_model_name and trainer.best_model_name in trainer.trained_models:
            trainer.best_model = trainer.trained_models[trainer.best_model_name]

        return trainer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_smote(X_train, y_train, use_smote: bool):
        if not use_smote or not _SMOTE_AVAILABLE:
            return X_train, y_train

        minority_count = int((y_train == 1).sum())
        if minority_count < 10:
            return X_train, y_train

        try:
            smote = SMOTE(random_state=42)
            return smote.fit_resample(X_train, y_train)
        except Exception:
            return X_train, y_train
