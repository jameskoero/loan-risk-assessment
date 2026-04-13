import matplotlib
matplotlib.use('Agg')

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_curve
)
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')

PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── 1. DATA LOADING ──────────────────────────────────────────────────────────

def load_data():
    """Download German Credit (credit-g) dataset from OpenML."""
    print("Loading credit-g dataset from OpenML...")
    data = fetch_openml(name='credit-g', version=1, as_frame=True)
    X = data.frame.drop(columns=['class'])
    y = (data.frame['class'] == 'bad').astype(int)
    print(f"Dataset shape: {X.shape}, Target balance: {y.value_counts().to_dict()}")
    return X, y


# ─── 2. PREPROCESSING ────────────────────────────────────────────────────────

def preprocess(X):
    """Encode categoricals and scale numerics; return transformed array + encoders."""
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    label_encoders = {}
    X_encoded = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', 'passthrough', categorical_cols),
    ])

    X_proc = preprocessor.fit_transform(X_encoded)
    feature_names = numeric_cols + categorical_cols
    return X_proc, feature_names, preprocessor, label_encoders


# ─── 3. SMOTE CLASS BALANCING ────────────────────────────────────────────────

def apply_smote(X_train, y_train):
    """Oversample minority class with SMOTE."""
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


# ─── 4. MODEL DEFINITIONS ────────────────────────────────────────────────────

def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=200, use_label_encoder=False,
                                  eval_metric='logloss', random_state=42, verbosity=0),
        'LightGBM': LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
    }


# ─── 5. UNIFIED EVALUATION ───────────────────────────────────────────────────

def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    """Return dict of evaluation metrics for a fitted model."""
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    return {
        'model': name,
        'accuracy': accuracy_score(y_test, preds),
        'roc_auc': roc_auc_score(y_test, proba),
        'f1': f1_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall': recall_score(y_test, preds),
        'confusion_matrix': confusion_matrix(y_test, preds),
        'proba': proba,
    }


# ─── 6. BUSINESS COST MATRIX & OPTIMAL THRESHOLD ────────────────────────────

FN_COST = 5  # False negative (missed default) costs 5x
FP_COST = 1  # False positive (declined good loan) costs 1x

def optimal_threshold(y_true, proba):
    """Find threshold that minimises business cost."""
    thresholds = np.linspace(0.01, 0.99, 200)
    best_thresh, best_cost = 0.5, np.inf
    for t in thresholds:
        preds = (proba >= t).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        cost = FN_COST * fn + FP_COST * fp
        if cost < best_cost:
            best_cost = cost
            best_thresh = t
    return best_thresh, best_cost


# ─── 7. RISK SCORECARD ───────────────────────────────────────────────────────

def assign_risk_tier(proba):
    """Map probability to 4-tier risk label."""
    tiers = pd.cut(
        proba,
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    return tiers


# ─── 8. PLOT: CLASS DISTRIBUTION ────────────────────────────────────────────

def plot_class_distribution(y):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = y.value_counts()
    ax.bar(['Good (0)', 'Bad (1)'], counts.values, color=['steelblue', 'tomato'])
    ax.set_title('Class Distribution — German Credit Dataset')
    ax.set_ylabel('Count')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/01_class_distribution.png', dpi=150)
    plt.close(fig)


# ─── 9. PLOT: CORRELATION HEATMAP ───────────────────────────────────────────

def plot_correlation_heatmap(X, feature_names):
    df = pd.DataFrame(X, columns=feature_names)
    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    sns.heatmap(df.corr(), mask=mask, annot=False, cmap='coolwarm',
                center=0, ax=ax, linewidths=0.5)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/02_correlation_heatmap.png', dpi=150)
    plt.close(fig)


# ─── 10. PLOT: ROC CURVES ────────────────────────────────────────────────────

def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']
    for res, color in zip(results, colors):
        fpr, tpr, _ = roc_curve(y_test, res['proba'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{res['model']} (AUC={res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves — All Models')
    ax.legend(loc='lower right')
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/03_roc_curves.png', dpi=150)
    plt.close(fig)


# ─── 11. PLOT: MODEL COMPARISON BAR ─────────────────────────────────────────

def plot_model_comparison(results):
    df = pd.DataFrame(results)[['model', 'accuracy', 'roc_auc', 'f1', 'precision', 'recall']]
    df_melt = df.melt(id_vars='model', var_name='Metric', value_name='Score')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_melt, x='model', y='Score', hue='Metric', ax=ax)
    ax.set_title('Model Performance Comparison')
    ax.set_ylim(0, 1)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=15)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/04_model_comparison.png', dpi=150)
    plt.close(fig)


# ─── 12. PLOT: CONFUSION MATRICES ────────────────────────────────────────────

def plot_confusion_matrices(results):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, res in zip(axes, results):
        sns.heatmap(res['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    ax=ax, cbar=False)
        ax.set_title(res['model'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.suptitle('Confusion Matrices', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/05_confusion_matrices.png', dpi=150)
    plt.close(fig)


# ─── 13. PLOT: FEATURE IMPORTANCE ────────────────────────────────────────────

def plot_feature_importance(model, feature_names, model_name='Random Forest'):
    if not hasattr(model, 'feature_importances_'):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[-20:]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(np.array(feature_names)[idx], importances[idx], color='steelblue')
    ax.set_title(f'Top 20 Feature Importances — {model_name}')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/06_feature_importance.png', dpi=150)
    plt.close(fig)


# ─── 14. PLOT: PRECISION-RECALL CURVES ───────────────────────────────────────

def plot_precision_recall(results, y_test):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']
    for res, color in zip(results, colors):
        prec, rec, _ = precision_recall_curve(y_test, res['proba'])
        ap = average_precision_score(y_test, res['proba'])
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{res['model']} (AP={ap:.3f})")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves')
    ax.legend(loc='upper right')
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/07_precision_recall_curves.png', dpi=150)
    plt.close(fig)


# ─── 15. PLOT: PROBABILITY DISTRIBUTIONS ────────────────────────────────────

def plot_probability_distributions(results, y_test):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    for ax, res in zip(axes, results):
        for label, color in [(0, 'steelblue'), (1, 'tomato')]:
            mask = y_test == label
            ax.hist(res['proba'][mask], bins=20, alpha=0.6, color=color,
                    label=f'Class {label}', density=True)
        ax.set_title(res['model'])
        ax.set_xlabel('P(default)')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Density')
    plt.suptitle('Predicted Probability Distributions', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/08_probability_distributions.png', dpi=150)
    plt.close(fig)


# ─── 16. PLOT: COST THRESHOLD CURVE ─────────────────────────────────────────

def plot_cost_threshold(y_test, proba, model_name, best_thresh):
    thresholds = np.linspace(0.01, 0.99, 200)
    costs = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        costs.append(FN_COST * fn + FP_COST * fp)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, costs, color='steelblue', lw=2)
    ax.axvline(best_thresh, color='red', linestyle='--', label=f'Optimal θ={best_thresh:.2f}')
    ax.set_xlabel('Classification Threshold')
    ax.set_ylabel('Total Business Cost')
    ax.set_title(f'Cost vs. Threshold — {model_name}')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/09_cost_threshold.png', dpi=150)
    plt.close(fig)


# ─── 17. PLOT: RISK SCORECARD ────────────────────────────────────────────────

def plot_risk_scorecard(proba, y_test):
    tiers = assign_risk_tier(proba)
    df = pd.DataFrame({'tier': tiers, 'actual': y_test.values})
    summary = df.groupby('tier', observed=True)['actual'].agg(['mean', 'count'])
    summary.columns = ['Default Rate', 'Count']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['seagreen', 'gold', 'darkorange', 'tomato']
    axes[0].bar(summary.index, summary['Count'], color=colors)
    axes[0].set_title('Loan Count by Risk Tier')
    axes[0].set_xlabel('Risk Tier')
    axes[0].set_ylabel('Count')

    axes[1].bar(summary.index, summary['Default Rate'], color=colors)
    axes[1].set_title('Default Rate by Risk Tier')
    axes[1].set_xlabel('Risk Tier')
    axes[1].set_ylabel('Default Rate')
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/10_risk_scorecard.png', dpi=150)
    plt.close(fig)


# ─── 18. PLOT: CALIBRATION CURVES ────────────────────────────────────────────

def plot_calibration_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['steelblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple']
    for (name, model), color in zip(models.items(), colors):
        CalibrationDisplay.from_estimator(
            model, X_test, y_test, n_bins=10, ax=ax,
            name=name, color=color
        )
    ax.set_title('Calibration Curves — All Models')
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/11_calibration_curves.png', dpi=150)
    plt.close(fig)


# ─── 19. PLOT: LEARNING CURVES ───────────────────────────────────────────────

def plot_learning_curves(model, X_train_full, y_train_full, model_name):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train_full, y_train_full,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='roc_auc', train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='steelblue', label='Train AUC')
    ax.fill_between(train_sizes,
                    train_scores.mean(axis=1) - train_scores.std(axis=1),
                    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.2, color='steelblue')
    ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', color='tomato', label='CV AUC')
    ax.fill_between(train_sizes,
                    val_scores.mean(axis=1) - val_scores.std(axis=1),
                    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2, color='tomato')
    ax.set_title(f'Learning Curves — {model_name}')
    ax.set_xlabel('Training Samples')
    ax.set_ylabel('ROC-AUC')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/12_learning_curves.png', dpi=150)
    plt.close(fig)


# ─── 20. PLOT: SHAP BAR ──────────────────────────────────────────────────────

def plot_shap_bar(shap_values, feature_names, model_name):
    fig, ax = plt.subplots(figsize=(8, 7))
    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(mean_abs)[-20:]
    ax.barh(np.array(feature_names)[idx], mean_abs[idx], color='steelblue')
    ax.set_title(f'SHAP Mean |Value| — {model_name}')
    ax.set_xlabel('Mean |SHAP Value|')
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/13_shap_bar.png', dpi=150)
    plt.close(fig)


# ─── 21. PLOT: SHAP BEESWARM ─────────────────────────────────────────────────

def plot_shap_beeswarm(shap_values, X_sample, feature_names, model_name):
    explanation = shap.Explanation(
        values=shap_values,
        data=X_sample,
        feature_names=feature_names
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title(f'SHAP Beeswarm — {model_name}')
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/14_shap_beeswarm.png', dpi=150)
    plt.close(fig)


# ─── 22. PLOT: SCORE DISTRIBUTION ───────────────────────────────────────────

def plot_score_distribution(proba, y_test, model_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(proba[y_test == 0], bins=30, alpha=0.6, color='steelblue',
            label='Good (actual)', density=True)
    ax.hist(proba[y_test == 1], bins=30, alpha=0.6, color='tomato',
            label='Bad (actual)', density=True)
    ax.set_xlabel('Risk Score (P(default))')
    ax.set_ylabel('Density')
    ax.set_title(f'Risk Score Distribution — {model_name}')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/15_score_distribution.png', dpi=150)
    plt.close(fig)


# ─── 23. PLOT: THRESHOLD METRICS ─────────────────────────────────────────────

def plot_threshold_metrics(y_test, proba, model_name, best_thresh):
    thresholds = np.linspace(0.01, 0.99, 200)
    accs, f1s, precs, recs = [], [], [], []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        accs.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds, zero_division=0))
        precs.append(precision_score(y_test, preds, zero_division=0))
        recs.append(recall_score(y_test, preds, zero_division=0))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, accs, label='Accuracy', lw=2)
    ax.plot(thresholds, f1s, label='F1', lw=2)
    ax.plot(thresholds, precs, label='Precision', lw=2)
    ax.plot(thresholds, recs, label='Recall', lw=2)
    ax.axvline(best_thresh, color='red', linestyle='--', label=f'Optimal θ={best_thresh:.2f}')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title(f'Metrics vs. Threshold — {model_name}')
    ax.legend()
    plt.tight_layout()
    fig.savefig(f'{PLOTS_DIR}/16_threshold_metrics.png', dpi=150)
    plt.close(fig)


# ─── 24. EXPORT MODEL ────────────────────────────────────────────────────────

def export_model(model, model_name, best_metrics, preprocessor):
    joblib.dump(model, 'loan_risk_model.joblib')
    metadata = {
        'model_name': model_name,
        'roc_auc': round(best_metrics['roc_auc'], 4),
        'accuracy': round(best_metrics['accuracy'], 4),
        'f1': round(best_metrics['f1'], 4),
        'precision': round(best_metrics['precision'], 4),
        'recall': round(best_metrics['recall'], 4),
        'fn_cost_multiplier': FN_COST,
        'fp_cost': FP_COST,
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model exported: loan_risk_model.joblib")
    print(f"Metadata: model_metadata.json")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 1. Load data
    X, y = load_data()

    # 2. Preprocessing
    X_proc, feature_names, preprocessor, label_encoders = preprocess(X)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_proc, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. SMOTE on training set
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # 5. Train all models
    models = get_models()
    fitted_models = {}
    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train_res, y_train_res)
        fitted_models[name] = model
        print(f"  ✓ {name}")

    # 6. Evaluate all models
    results = [evaluate_model(n, m, X_test, y_test)
               for n, m in fitted_models.items()]
    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ('confusion_matrix', 'proba')}
        for r in results
    ])
    print("\n── Model Performance ──")
    print(results_df.to_string(index=False))

    # 7. Identify best model by ROC-AUC
    best_result = max(results, key=lambda r: r['roc_auc'])
    best_name = best_result['model']
    best_model = fitted_models[best_name]
    best_proba = best_result['proba']
    print(f"\nBest model: {best_name} (AUC={best_result['roc_auc']:.4f})")

    # 8. Optimal threshold
    best_thresh, best_cost = optimal_threshold(y_test, best_proba)
    print(f"Optimal threshold: {best_thresh:.3f} (cost={best_cost})")

    # 9. Generate all 16 plots
    print("\nGenerating plots...")
    plot_class_distribution(y)
    plot_correlation_heatmap(X_proc, feature_names)
    plot_roc_curves(results, y_test)
    plot_model_comparison(results)
    plot_confusion_matrices(results)
    plot_feature_importance(best_model, feature_names, best_name)
    plot_precision_recall(results, y_test)
    plot_probability_distributions(results, y_test)
    plot_cost_threshold(y_test, best_proba, best_name, best_thresh)
    plot_risk_scorecard(best_proba, y_test)
    plot_calibration_curves(fitted_models, X_test, y_test)
    plot_learning_curves(best_model, X_train, y_train, best_name)

    # SHAP analysis
    explainer = shap.TreeExplainer(best_model)
    X_sample = X_test[:200]
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    plot_shap_bar(shap_values, feature_names, best_name)
    plot_shap_beeswarm(shap_values, X_sample, feature_names, best_name)

    plot_score_distribution(best_proba, y_test, best_name)
    plot_threshold_metrics(y_test, best_proba, best_name, best_thresh)
    print(f"  ✓ 16 plots saved to '{PLOTS_DIR}/'")

    # 10. Risk scorecard summary
    tiers = assign_risk_tier(best_proba)
    print("\n── Risk Scorecard (test set) ──")
    print(pd.DataFrame({'tier': tiers, 'actual': y_test.values})
          .groupby('tier', observed=True)['actual']
          .agg(count='count', default_rate='mean')
          .to_string())

    # 11. Export model
    export_model(best_model, best_name, best_result, preprocessor)
    print("\nDone.")
