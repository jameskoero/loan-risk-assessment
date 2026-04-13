"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         LOAN DEFAULT RISK ASSESSMENT — Advanced ML Pipeline                ║
║         Author  : James Koero | Junior ML Engineer | Kisumu, Kenya         ║
║         Dataset : German Credit Data (OpenML ID 31) — auto-downloaded      ║
║         GitHub  : github.com/YOUR_GITHUB_USERNAME/loan-risk-assessment     ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE
-----
  Local  : python loan_risk_assessment.py
  Colab  : Upload file and run, or use the .ipynb notebook
  PyramIDE: Run directly — uses Agg backend, blocking plots

DEPENDENCIES
------------
  pip install scikit-learn pandas numpy matplotlib seaborn imbalanced-learn
              shap xgboost joblib
"""

# ── Backend (non-interactive — safe for PyramIDE / headless) ─────────────────
import matplotlib
matplotlib.use("Agg")

import os, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

# ── Sklearn ───────────────────────────────────────────────────────────────────
from sklearn.datasets     import fetch_openml
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    GridSearchCV, learning_curve
)
from sklearn.preprocessing   import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose          import ColumnTransformer
from sklearn.pipeline         import Pipeline
from sklearn.impute           import SimpleImputer
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.calibration      import calibration_curve
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
import joblib

# ── Optional packages ─────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_OK = True
except ImportError:
    SMOTE_OK = False
    print("⚠️  imbalanced-learn not found. Run: pip install imbalanced-learn")

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs("plots", exist_ok=True)
os.makedirs("model", exist_ok=True)

COST_FN = 5   # Relative cost of approving a defaulter
COST_FP = 1   # Relative cost of rejecting a good customer

plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                     "axes.spines.right": False})
sns.set_theme(style="darkgrid")

# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
def load_data():
    print("\n" + "="*60)
    print("STEP 1 — DATA LOADING")
    print("="*60)
    print("⏳ Downloading German Credit dataset from OpenML...")
    credit  = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    df      = credit.frame.copy()
    df["default"] = (df["class"] == "bad").astype(int)
    df.drop(columns=["class"], inplace=True)

    # String-ify categories for downstream processing
    for col in df.select_dtypes("category").columns:
        df[col] = df[col].astype(str)

    print(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    vc = df["default"].value_counts()
    print(f"   No Default : {vc[0]} ({vc[0]/len(df)*100:.1f}%)")
    print(f"   Default    : {vc[1]} ({vc[1]/len(df)*100:.1f}%)")
    return df

# ═════════════════════════════════════════════════════════════════════════════
# 2. EDA
# ═════════════════════════════════════════════════════════════════════════════
def run_eda(df):
    print("\n" + "="*60)
    print("STEP 2 — EXPLORATORY DATA ANALYSIS")
    print("="*60)

    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c != "default"]
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # 2.1 Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Target Variable — Default Distribution", fontweight="bold")
    counts = df["default"].value_counts()
    axes[0].bar(["No Default", "Default"], counts.values, color=["#2ECC71", "#E74C3C"],
                width=0.5, edgecolor="white")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")
    axes[0].set_title("Count")
    axes[1].pie(counts.values, labels=["No Default", "Default"],
                colors=["#2ECC71", "#E74C3C"], autopct="%1.1f%%",
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Percentage")
    plt.tight_layout()
    plt.savefig("plots/01_target_distribution.png", bbox_inches="tight")
    plt.show(block=True)

    # 2.2 Numeric distributions
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle("Numeric Features by Default Status", fontweight="bold")
    for i, col in enumerate(num_cols[:6]):
        for label, color in [(0, "#2ECC71"), (1, "#E74C3C")]:
            axes[i].hist(df[df["default"] == label][col], bins=20, alpha=0.6,
                         color=color, label="No Def" if label == 0 else "Default",
                         density=True, edgecolor="none")
        axes[i].set_title(col); axes[i].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("plots/02_numeric_distributions.png", bbox_inches="tight")
    plt.show(block=True)

    # 2.3 Correlation heatmap
    corr = df[num_cols + ["default"]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn_r",
                center=0, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Matrix", fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/03_correlation_heatmap.png", bbox_inches="tight")
    plt.show(block=True)

    target_corr = corr["default"].drop("default").abs().sort_values(ascending=False)
    print("\nTop correlations with default:")
    print(target_corr.head(5).to_string())
    return num_cols, cat_cols, target_corr

# ═════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════
def engineer_features(df):
    print("\n" + "="*60)
    print("STEP 3 — FEATURE ENGINEERING")
    print("="*60)
    df = df.copy()
    df["debt_to_credit_ratio"] = df["installment_commitment"] / (df["credit_amount"] + 1)
    df["loan_income_proxy"]    = df["credit_amount"] / (df["duration"] + 1)
    df["long_term_loan"]       = (df["duration"] > 24).astype(int)
    df["high_credit"]          = (df["credit_amount"] > df["credit_amount"].median()).astype(int)
    df["senior_applicant"]     = (df["age"] >= 60).astype(int)
    df["age_group"]            = pd.cut(df["age"], bins=[0, 25, 35, 50, 100],
                                        labels=["u25", "25_35", "35_50", "o50"]).astype(str)
    new_feats = ["debt_to_credit_ratio", "loan_income_proxy",
                 "long_term_loan", "high_credit", "senior_applicant", "age_group"]
    print(f"✅ Added {len(new_feats)} new features: {new_feats}")
    return df

# ═════════════════════════════════════════════════════════════════════════════
# 4. PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════
def build_preprocessor(X):
    num_f = X.select_dtypes(include=np.number).columns.tolist()
    cat_f = X.select_dtypes(include="object").columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    prep = ColumnTransformer([
        ("num", num_pipe, num_f),
        ("cat", cat_pipe, cat_f)
    ], verbose_feature_names_out=True)
    return prep, num_f, cat_f

# ═════════════════════════════════════════════════════════════════════════════
# 5. TRAIN MODELS
# ═════════════════════════════════════════════════════════════════════════════
def train_models(X_tr, y_tr, X_te, y_te):
    print("\n" + "="*60)
    print("STEP 5 — MODEL TRAINING")
    print("="*60)

    booster = (xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                                  subsample=0.8, use_label_encoder=False,
                                  eval_metric="logloss", random_state=RANDOM_STATE)
               if XGB_OK else
               GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                          max_depth=4, random_state=RANDOM_STATE))

    MODELS = {
        "Logistic Regression" : LogisticRegression(max_iter=2000, class_weight="balanced",
                                                    random_state=RANDOM_STATE),
        "Decision Tree"       : DecisionTreeClassifier(max_depth=6, min_samples_leaf=10,
                                                        class_weight="balanced",
                                                        random_state=RANDOM_STATE),
        "Random Forest"       : RandomForestClassifier(n_estimators=300, max_depth=8,
                                                        class_weight="balanced",
                                                        random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting"   : GradientBoostingClassifier(n_estimators=200,
                                                            learning_rate=0.05,
                                                            max_depth=4,
                                                            random_state=RANDOM_STATE),
        "XGBoost/HGB"         : booster,
    }

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    print(f"\n{'Model':<25} {'Acc':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 45)

    for name, model in MODELS.items():
        model.fit(X_tr, y_tr)
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        acc     = accuracy_score(y_te, y_pred)
        f1      = f1_score(y_te, y_pred)
        aucv    = roc_auc_score(y_te, y_proba)
        cv_auc  = cross_val_score(model, X_tr, y_tr, cv=skf,
                                   scoring="roc_auc", n_jobs=-1).mean()
        results[name] = {"model": model, "y_pred": y_pred, "y_proba": y_proba,
                         "accuracy": acc, "f1": f1, "auc": aucv, "cv_auc": cv_auc,
                         "precision": precision_score(y_te, y_pred, zero_division=0),
                         "recall": recall_score(y_te, y_pred)}
        print(f"{name:<25} {acc:>6.3f} {f1:>6.3f} {aucv:>6.3f}")

    return results, MODELS

# ═════════════════════════════════════════════════════════════════════════════
# 6. HYPERPARAMETER TUNING
# ═════════════════════════════════════════════════════════════════════════════
def tune_best_model(X_tr, y_tr, X_te, y_te):
    print("\n" + "="*60)
    print("STEP 6 — HYPERPARAMETER TUNING (GridSearchCV)")
    print("="*60)
    param_grid = {
        "n_estimators" : [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth"    : [3, 4],
        "subsample"    : [0.8, 1.0],
    }
    gs = GridSearchCV(
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", n_jobs=-1, verbose=0
    )
    gs.fit(X_tr, y_tr)
    best = gs.best_estimator_
    y_pred  = best.predict(X_te)
    y_proba = best.predict_proba(X_te)[:, 1]
    print(f"Best params   : {gs.best_params_}")
    print(f"Best CV AUC   : {gs.best_score_:.4f}")
    print(f"Test AUC      : {roc_auc_score(y_te, y_proba):.4f}")
    print(f"Test F1       : {f1_score(y_te, y_pred):.4f}")
    return best, y_pred, y_proba, gs

# ═════════════════════════════════════════════════════════════════════════════
# 7. EVALUATION PLOTS
# ═════════════════════════════════════════════════════════════════════════════
def evaluation_plots(results, y_te, best_model, X_te, y_proba_best):
    print("\n" + "="*60)
    print("STEP 7 — EVALUATION PLOTS")
    print("="*60)

    # ROC Curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Comparison — ROC & PR Curves", fontweight="bold")
    colors = plt.cm.Set1(np.linspace(0, 0.8, len(results)))

    for (name, res), col in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_te, res["y_proba"])
        axes[0].plot(fpr, tpr, lw=2, color=col, label=f"{name} ({res['auc']:.3f})")
        pr, rc, _   = precision_recall_curve(y_te, res["y_proba"])
        ap          = average_precision_score(y_te, res["y_proba"])
        axes[1].plot(rc, pr, lw=2, color=col, label=f"{name} (AP={ap:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--"); axes[0].set_title("ROC Curves")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend(fontsize=8)
    axes[1].set_title("Precision-Recall Curves")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("plots/07_roc_pr_curves.png", bbox_inches="tight")
    plt.show(block=True)

    # Confusion matrices
    n  = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes_flat = axes.flatten()
    fig.suptitle("Confusion Matrices", fontweight="bold")
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_te, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes_flat[i],
                    xticklabels=["No Def", "Default"],
                    yticklabels=["No Def", "Default"])
        axes_flat[i].set_title(f"{name}\nAUC={res['auc']:.3f}  F1={res['f1']:.3f}", fontsize=9)
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.tight_layout()
    plt.savefig("plots/08_confusion_matrices.png", bbox_inches="tight")
    plt.show(block=True)

    # Metrics bar chart
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    df_r = pd.DataFrame({nm: {m: res[m] for m in metrics}
                         for nm, res in results.items()}).T
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df_r)); w = 0.15
    for i, (m, c) in enumerate(zip(metrics, ["#3498DB","#2ECC71","#E74C3C","#F39C12","#9B59B6"])):
        ax.bar(x + i*w, df_r[m], w, label=m.title(), color=c, alpha=0.85, edgecolor="white")
    ax.set_xticks(x + w*2); ax.set_xticklabels(df_r.index, rotation=15, ha="right")
    ax.set_title("Model Performance Comparison", fontweight="bold")
    ax.legend(ncol=5, fontsize=9); ax.set_ylim(0, 1.12)
    plt.tight_layout()
    plt.savefig("plots/09_metrics_comparison.png", bbox_inches="tight")
    plt.show(block=True)

    print("✅ Evaluation plots saved to plots/")

# ═════════════════════════════════════════════════════════════════════════════
# 8. THRESHOLD OPTIMISATION
# ═════════════════════════════════════════════════════════════════════════════
def optimise_threshold(y_te, y_proba):
    print("\n" + "="*60)
    print("STEP 8 — BUSINESS THRESHOLD OPTIMISATION")
    print("="*60)
    thresholds = np.linspace(0.1, 0.9, 161)
    costs = []
    for t in thresholds:
        y_t = (y_proba >= t).astype(int)
        if len(np.unique(y_t)) < 2:
            costs.append(1e9)
            continue
        cm  = confusion_matrix(y_te, y_t)
        tn, fp, fn, tp = cm.ravel()
        costs.append(COST_FN * fn + COST_FP * fp)

    opt_idx   = np.argmin(costs)
    opt_thresh = thresholds[opt_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, costs, color="#E74C3C", lw=2)
    ax.axvline(opt_thresh, color="#2ECC71", linestyle="--", lw=2,
               label=f"Optimal: {opt_thresh:.2f}")
    ax.axvline(0.5, color="#3498DB", linestyle=":", lw=1.5, label="Default: 0.50")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Business Cost")
    ax.set_title("Cost-Optimised Threshold", fontweight="bold"); ax.legend()
    plt.tight_layout()
    plt.savefig("plots/14_threshold_optimisation.png", bbox_inches="tight")
    plt.show(block=True)

    print(f"Optimal threshold : {opt_thresh:.3f}")
    y_opt = (y_proba >= opt_thresh).astype(int)
    print(classification_report(y_te, y_opt, target_names=["No Default", "Default"]))
    return opt_thresh

# ═════════════════════════════════════════════════════════════════════════════
# 9. FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════════
def plot_feature_importance(model, feature_names, top_n=20):
    fi = pd.Series(model.feature_importances_, index=feature_names) \
           .sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(fi.index[::-1], fi.values[::-1],
            color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, top_n)),
            edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances", fontweight="bold")
    ax.set_xlabel("Importance (Gain)")
    plt.tight_layout()
    plt.savefig("plots/16_feature_importance.png", bbox_inches="tight")
    plt.show(block=True)
    print("\n🔝 Top 10 features:")
    for i, (f, v) in enumerate(fi.head(10).items(), 1):
        print(f"  {i:2}. {f:<45} {v:.5f}")

# ═════════════════════════════════════════════════════════════════════════════
# 10. SAVE MODEL
# ═════════════════════════════════════════════════════════════════════════════
def save_model(pipeline, best_params, best_cv_auc, y_te, y_pred, y_proba, threshold):
    joblib.dump(pipeline, "model/loan_risk_model.joblib")
    meta = {
        "author"           : "James Koero",
        "dataset"          : "German Credit Data — OpenML ID 31",
        "model_type"       : "GradientBoostingClassifier (sklearn)",
        "best_params"      : best_params,
        "cv_roc_auc"       : round(float(best_cv_auc), 4),
        "test_roc_auc"     : round(roc_auc_score(y_te, y_proba), 4),
        "test_f1"          : round(f1_score(y_te, y_pred), 4),
        "optimal_threshold": round(float(threshold), 4),
    }
    with open("model/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\n✅ Model saved → model/loan_risk_model.joblib")
    print("✅ Metadata saved → model/model_metadata.json")

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "█"*60)
    print("  LOAN DEFAULT RISK ASSESSMENT — James Koero")
    print("  German Credit Data | Gradient Boosting | SMOTE")
    print("█"*60)

    # 1. Load
    df = load_data()

    # 2. EDA
    num_cols, cat_cols, target_corr = run_eda(df)

    # 3. Feature Engineering
    df = engineer_features(df)

    # 4. Split & Preprocess
    print("\n" + "="*60)
    print("STEP 4 — PREPROCESSING & SPLIT")
    print("="*60)
    X = df.drop(columns=["default"])
    y = df["default"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    prep, num_f, cat_f = build_preprocessor(X_train)
    prep.fit(X_train)
    X_train_p = prep.transform(X_train)
    X_test_p  = prep.transform(X_test)
    feature_names = prep.get_feature_names_out()
    print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")
    print(f"Features after encoding: {len(feature_names)}")

    # SMOTE
    if SMOTE_OK:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train_sm, y_train_sm = sm.fit_resample(X_train_p, y_train)
        print(f"After SMOTE → {len(y_train_sm)} training samples (balanced)")
    else:
        X_train_sm, y_train_sm = X_train_p, y_train
        print("⚠️  SMOTE skipped (imbalanced-learn not installed)")

    # 5. Train
    results, _ = train_models(X_train_sm, y_train_sm, X_test_p, y_test)

    # 6. Tune
    best_model, y_pred_best, y_proba_best, gs = tune_best_model(
        X_train_sm, y_train_sm, X_test_p, y_test
    )
    results["GB (Tuned)"] = {
        "model": best_model, "y_pred": y_pred_best, "y_proba": y_proba_best,
        "accuracy": accuracy_score(y_test, y_pred_best),
        "precision": precision_score(y_test, y_pred_best, zero_division=0),
        "recall": recall_score(y_test, y_pred_best),
        "f1": f1_score(y_test, y_pred_best),
        "auc": roc_auc_score(y_test, y_proba_best),
        "cv_auc": gs.best_score_
    }

    # 7. Evaluation
    evaluation_plots(results, y_test, best_model, X_test_p, y_proba_best)

    # 8. Threshold
    opt_thresh = optimise_threshold(y_test, y_proba_best)

    # 9. Feature Importance
    plot_feature_importance(best_model, feature_names)

    # 10. Save
    final_pipe = Pipeline([("prep", prep), ("model", best_model)])
    save_model(final_pipe, gs.best_params_, gs.best_score_,
               y_test, y_pred_best, y_proba_best, opt_thresh)

    print("\n" + "█"*60)
    print("  ✅  ALL STEPS COMPLETE — plots saved to plots/")
    print("  ✅  Model saved → model/loan_risk_model.joblib")
    print("█"*60)

if __name__ == "__main__":
    main()
