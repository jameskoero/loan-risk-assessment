"""
Loan Default Risk Assessment System
====================================
Full end-to-end pipeline:
  data generation → feature engineering → train/test split →
  preprocessing → model training → evaluation → risk scoring demo
"""
import os
import sys
import argparse

import matplotlib
matplotlib.use('Agg')

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Allow running from the repo root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from src.data_generator      import generate_loan_dataset, save_dataset
from src.feature_engineering import engineer_features
from src.data_preprocessing  import LoanDataPreprocessor
from src.model_training      import LoanRiskModelTrainer
from src.model_evaluation    import ModelEvaluator
from src.risk_scoring        import RiskScorer


# ---------------------------------------------------------------------------
# Sample loans for the scoring demo
# ---------------------------------------------------------------------------
DEMO_LOANS = [
    {   # Low-risk applicant
        'loan_amount': 15000, 'loan_term': 36, 'interest_rate': 7.5,
        'annual_income': 90000, 'debt_to_income_ratio': 10.0, 'credit_score': 780,
        'employment_length': 8, 'num_credit_lines': 10, 'num_derogatory_marks': 0,
        'home_ownership': 'MORTGAGE', 'loan_purpose': 'home_improvement',
        'verification_status': 'Verified',
    },
    {   # Medium-risk applicant
        'loan_amount': 25000, 'loan_term': 48, 'interest_rate': 14.0,
        'annual_income': 55000, 'debt_to_income_ratio': 28.0, 'credit_score': 650,
        'employment_length': 3, 'num_credit_lines': 8, 'num_derogatory_marks': 1,
        'home_ownership': 'RENT', 'loan_purpose': 'debt_consolidation',
        'verification_status': 'Source Verified',
    },
    {   # High-risk applicant
        'loan_amount': 40000, 'loan_term': 60, 'interest_rate': 22.0,
        'annual_income': 35000, 'debt_to_income_ratio': 45.0, 'credit_score': 520,
        'employment_length': 1, 'num_credit_lines': 5, 'num_derogatory_marks': 4,
        'home_ownership': 'RENT', 'loan_purpose': 'other',
        'verification_status': 'Not Verified',
    },
    {   # Very high-risk applicant
        'loan_amount': 60000, 'loan_term': 60, 'interest_rate': 28.0,
        'annual_income': 28000, 'debt_to_income_ratio': 58.0, 'credit_score': 350,
        'employment_length': 0, 'num_credit_lines': 2, 'num_derogatory_marks': 8,
        'home_ownership': 'RENT', 'loan_purpose': 'other',
        'verification_status': 'Not Verified',
    },
    {   # Strong applicant — car loan
        'loan_amount': 20000, 'loan_term': 24, 'interest_rate': 6.0,
        'annual_income': 120000, 'debt_to_income_ratio': 8.0, 'credit_score': 820,
        'employment_length': 15, 'num_credit_lines': 15, 'num_derogatory_marks': 0,
        'home_ownership': 'OWN', 'loan_purpose': 'car',
        'verification_status': 'Verified',
    },
]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    n_samples: int = 10000,
    output_dir: str = 'output',
    random_state: int = 42,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    reports_dir = os.path.join(output_dir, 'reports')
    models_dir  = os.path.join(output_dir, 'models')

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Loan Risk Assessment Pipeline  (n_samples={n_samples})")
    print(f"{'='*60}")
    print("\n[1/8] Generating synthetic loan dataset…")
    df = generate_loan_dataset(n_samples=n_samples, random_state=random_state)
    save_dataset(df, os.path.join(output_dir, 'loan_data_raw.csv'))
    default_rate = df['default'].mean()
    print(f"      Dataset shape : {df.shape}")
    print(f"      Default rate  : {default_rate:.1%}")

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    print("\n[2/8] Engineering features…")
    df_eng = engineer_features(df)
    save_dataset(df_eng, os.path.join(output_dir, 'loan_data_engineered.csv'))
    print(f"      Engineered shape: {df_eng.shape}")

    # ------------------------------------------------------------------
    # 3. Train / test split (stratified 80/20)
    # ------------------------------------------------------------------
    print("\n[3/8] Splitting data (80% train / 20% test, stratified)…")
    train_df, test_df = train_test_split(
        df_eng, test_size=0.20, random_state=random_state, stratify=df_eng['default']
    )
    print(f"      Train size: {len(train_df)}   Test size: {len(test_df)}")

    # ------------------------------------------------------------------
    # 4. Preprocessing  (fit on train only)
    # ------------------------------------------------------------------
    print("\n[4/8] Preprocessing (fit on train, transform both)…")
    preprocessor = LoanDataPreprocessor()
    X_train, y_train = preprocessor.fit_transform(train_df)
    X_test  = preprocessor.transform(test_df)
    y_test  = test_df['default'].reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    preprocessor.save(os.path.join(output_dir, 'preprocessor.pkl'))
    print(f"      X_train shape: {X_train.shape}   X_test shape: {X_test.shape}")

    # ------------------------------------------------------------------
    # 5. Model training
    # ------------------------------------------------------------------
    print("\n[5/8] Training models (with SMOTE oversampling)…")
    trainer = LoanRiskModelTrainer()
    trainer.train_all(X_train, y_train, use_smote=True)
    trainer.save_models(models_dir)

    print("\n  Cross-validation AUC-ROC scores:")
    for name, score in sorted(trainer.cv_scores.items(), key=lambda x: -x[1]):
        marker = ' ← best' if name == trainer.best_model_name else ''
        print(f"    {name:<30}: {score:.4f}{marker}")

    # ------------------------------------------------------------------
    # 6. Model comparison
    # ------------------------------------------------------------------
    print("\n[6/8] Evaluating all models on test set…")
    evaluators = {
        name: ModelEvaluator(model, model_name=name)
        for name, model in trainer.trained_models.items()
    }
    comparison_df = ModelEvaluator.compare_models(evaluators, X_test, y_test)
    print("\n  Model Comparison (sorted by AUC-ROC):")
    print(comparison_df.to_string(float_format='{:.4f}'.format))
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'))

    # ------------------------------------------------------------------
    # 7. Full report for best model
    # ------------------------------------------------------------------
    print(f"\n[7/8] Generating full report for best model: {trainer.best_model_name}…")
    best_evaluator = ModelEvaluator(trainer.best_model, model_name=trainer.best_model_name)
    best_evaluator.generate_full_report(
        X_test, y_test,
        feature_names=preprocessor.feature_names_,
        output_dir=reports_dir,
    )

    # ------------------------------------------------------------------
    # 8. Risk scoring demo
    # ------------------------------------------------------------------
    print("\n[8/8] Demonstrating risk scoring on sample loans…")
    scorer = RiskScorer(
        model=trainer.best_model,
        preprocessor=preprocessor,
        feature_engineer_fn=engineer_features,
    )

    print(f"\n  {'#':<4} {'Risk':<12} {'P(default)':<14} {'Recommendation'}")
    print(f"  {'-'*60}")
    for i, loan in enumerate(DEMO_LOANS, 1):
        result = scorer.score_single(loan)
        print(
            f"  {i:<4} {result['risk_category']:<12} "
            f"{result['probability']:<14.3f} {result['recommendation']}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"  Best model  : {trainer.best_model_name}")
    best_auc = comparison_df.loc[trainer.best_model_name, 'auc_roc']
    print(f"  Test AUC-ROC: {best_auc:.4f}")
    print(f"  Output dir  : {os.path.abspath(output_dir)}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loan Risk Assessment Pipeline')
    parser.add_argument('--samples',      type=int, default=10000, help='Number of loan records')
    parser.add_argument('--output-dir',   type=str, default='output', help='Output directory')
    parser.add_argument('--random-state', type=int, default=42,    help='Random seed')
    args = parser.parse_args()

    run_pipeline(
        n_samples=args.samples,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )
