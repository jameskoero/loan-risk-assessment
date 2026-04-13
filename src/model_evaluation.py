"""
Model evaluation utilities: metrics, plots, comparison table, and report generation.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    roc_curve, precision_recall_curve, confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.calibration import CalibrationDisplay


class ModelEvaluator:
    """Evaluate and visualise a binary classifier for loan default prediction."""

    def __init__(self, model, model_name: str = 'model'):
        self.model = model
        self.model_name = model_name

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def evaluate(self, X_test, y_test) -> dict:
        """
        Compute a comprehensive set of binary-classification metrics.

        Returns
        -------
        dict with keys: accuracy, precision, recall, f1, auc_roc, auc_pr,
                        brier_score, ks_statistic.
        """
        y_pred = self.model.predict(X_test)
        y_prob = self._predict_proba(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ks = float(np.max(np.abs(tpr - fpr)))

        return {
            'accuracy':     float(accuracy_score(y_test, y_pred)),
            'precision':    float(precision_score(y_test, y_pred, zero_division=0)),
            'recall':       float(recall_score(y_test, y_pred, zero_division=0)),
            'f1':           float(f1_score(y_test, y_pred, zero_division=0)),
            'auc_roc':      float(roc_auc_score(y_test, y_prob)),
            'auc_pr':       float(average_precision_score(y_test, y_prob)),
            'brier_score':  float(brier_score_loss(y_test, y_prob)),
            'ks_statistic': ks,
        }

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def plot_roc_curve(self, X_test, y_test, ax=None):
        y_prob = self._predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve — {self.model_name}')
        ax.legend(loc='lower right')
        return ax

    def plot_precision_recall_curve(self, X_test, y_test, ax=None):
        y_prob = self._predict_proba(X_test)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(recall, precision, lw=2, label=f'AP = {ap:.3f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve — {self.model_name}')
        ax.legend(loc='upper right')
        return ax

    def plot_confusion_matrix(self, X_test, y_test, ax=None):
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['No Default', 'Default'])

        if ax is None:
            _, ax = plt.subplots()
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(f'Confusion Matrix — {self.model_name}')
        return ax

    def plot_feature_importance(self, feature_names, ax=None, top_n: int = 20):
        importances = self._get_feature_importances(feature_names)
        if importances is None:
            return ax

        series = pd.Series(importances, index=feature_names).abs().nlargest(top_n)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
        series[::-1].plot(kind='barh', ax=ax)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top-{top_n} Feature Importances — {self.model_name}')
        plt.tight_layout()
        return ax

    def plot_calibration_curve(self, X_test, y_test, ax=None):
        y_prob = self._predict_proba(X_test)

        if ax is None:
            _, ax = plt.subplots()
        CalibrationDisplay.from_predictions(
            y_test, y_prob, n_bins=10, ax=ax, name=self.model_name
        )
        ax.set_title(f'Calibration Curve — {self.model_name}')
        return ax

    # ------------------------------------------------------------------
    # Full report
    # ------------------------------------------------------------------

    def generate_full_report(
        self, X_test, y_test, feature_names, output_dir: str = 'reports'
    ) -> dict:
        """
        Generate all diagnostic plots, print a metrics table, and save
        metrics to JSON inside *output_dir*.
        """
        os.makedirs(output_dir, exist_ok=True)

        metrics = self.evaluate(X_test, y_test)

        # Print metrics
        print(f"\n{'='*55}")
        print(f"  Evaluation Report — {self.model_name}")
        print(f"{'='*55}")
        for k, v in metrics.items():
            print(f"  {k:<22}: {v:.4f}")
        print(f"{'='*55}\n")

        # Save metrics JSON
        with open(os.path.join(output_dir, f'{self.model_name}_metrics.json'), 'w') as f:
            json.dump({'model': self.model_name, 'metrics': metrics}, f, indent=2)

        # Plots
        plot_specs = [
            ('roc_curve',               self.plot_roc_curve),
            ('precision_recall_curve',  self.plot_precision_recall_curve),
            ('confusion_matrix',        self.plot_confusion_matrix),
            ('calibration_curve',       self.plot_calibration_curve),
        ]
        for tag, plot_fn in plot_specs:
            fig, ax = plt.subplots(figsize=(6, 5))
            plot_fn(X_test, y_test, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'{self.model_name}_{tag}.png'), dpi=100)
            plt.close(fig)

        # Feature importance
        fig, ax = plt.subplots(figsize=(8, 7))
        self.plot_feature_importance(feature_names, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'{self.model_name}_feature_importance.png'), dpi=100)
        plt.close(fig)

        print(f"  Report saved to: {output_dir}/")
        return metrics

    # ------------------------------------------------------------------
    # Static comparison helper
    # ------------------------------------------------------------------

    @staticmethod
    def compare_models(evaluators: dict, X_test, y_test) -> pd.DataFrame:
        """
        Evaluate every model in *evaluators* and return a comparison DataFrame.

        Parameters
        ----------
        evaluators : dict mapping model_name → ModelEvaluator instance.
        """
        rows = []
        for name, evaluator in evaluators.items():
            metrics = evaluator.evaluate(X_test, y_test)
            metrics['model'] = name
            rows.append(metrics)
        df = pd.DataFrame(rows).set_index('model')
        return df.sort_values('auc_roc', ascending=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_proba(self, X) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        # Linear models with decision_function
        return self.model.decision_function(X)

    def _get_feature_importances(self, feature_names):
        model = self.model
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        if hasattr(model, 'coef_'):
            coef = model.coef_
            return coef.ravel() if coef.ndim > 1 else coef
        return None
