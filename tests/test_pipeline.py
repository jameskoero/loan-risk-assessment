"""
Unit tests for the loan default risk assessment pipeline.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

# Ensure the repo root is on sys.path so 'src' is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_generator      import generate_loan_dataset
from src.feature_engineering import engineer_features
from src.data_preprocessing  import LoanDataPreprocessor
from src.model_training      import LoanRiskModelTrainer
from src.model_evaluation    import ModelEvaluator
from src.risk_scoring        import RiskScorer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def raw_df():
    return generate_loan_dataset(n_samples=600, random_state=0)


@pytest.fixture(scope='module')
def engineered_df(raw_df):
    return engineer_features(raw_df)


@pytest.fixture(scope='module')
def split_data(engineered_df):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(
        engineered_df, test_size=0.20, random_state=0, stratify=engineered_df['default']
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


@pytest.fixture(scope='module')
def preprocessed(split_data):
    train_df, test_df = split_data
    pre = LoanDataPreprocessor()
    X_train, y_train = pre.fit_transform(train_df)
    X_test = pre.transform(test_df)
    y_test  = test_df['default'].reset_index(drop=True)
    return pre, X_train.reset_index(drop=True), y_train.reset_index(drop=True), \
                X_test.reset_index(drop=True),  y_test


@pytest.fixture(scope='module')
def trained_trainer(preprocessed):
    _, X_train, y_train, _, _ = preprocessed
    trainer = LoanRiskModelTrainer(models={
        'logistic_regression': __import__(
            'sklearn.linear_model', fromlist=['LogisticRegression']
        ).LogisticRegression(max_iter=500, C=1.0, random_state=42),
    })
    trainer.train_all(X_train, y_train, use_smote=True)
    return trainer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDataGenerator:
    def test_shape(self, raw_df):
        assert raw_df.shape == (600, 13), f"Unexpected shape: {raw_df.shape}"

    def test_columns_present(self, raw_df):
        expected = {
            'loan_amount', 'loan_term', 'interest_rate', 'annual_income',
            'debt_to_income_ratio', 'credit_score', 'employment_length',
            'num_credit_lines', 'num_derogatory_marks', 'home_ownership',
            'loan_purpose', 'verification_status', 'default',
        }
        assert expected.issubset(set(raw_df.columns))

    def test_default_rate(self, raw_df):
        rate = raw_df['default'].mean()
        assert 0.10 <= rate <= 0.35, f"Default rate out of expected range: {rate:.2%}"

    def test_no_nulls(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0


class TestFeatureEngineering:
    def test_new_columns_present(self, engineered_df):
        for col in [
            'monthly_payment', 'debt_burden', 'credit_utilization_risk',
            'loan_to_income', 'payment_to_income', 'risk_score_raw',
        ]:
            assert col in engineered_df.columns, f"Missing column: {col}"

    def test_no_nan(self, engineered_df):
        derived = [
            'monthly_payment', 'debt_burden', 'credit_utilization_risk',
            'loan_to_income', 'payment_to_income', 'risk_score_raw',
        ]
        assert engineered_df[derived].isnull().sum().sum() == 0

    def test_no_inf(self, engineered_df):
        numeric = engineered_df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()


class TestPreprocessor:
    def test_fit_transform_output_shape(self, preprocessed):
        _, X_train, y_train, _, _ = preprocessed
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert len(y_train) == X_train.shape[0]

    def test_no_nan_after_transform(self, preprocessed):
        _, X_train, _, X_test, _ = preprocessed
        assert X_train.isnull().sum().sum() == 0
        assert X_test.isnull().sum().sum()  == 0

    def test_transform_consistency(self, preprocessed):
        """Train and test sets must have identical feature columns."""
        _, X_train, _, X_test, _ = preprocessed
        assert list(X_train.columns) == list(X_test.columns)


class TestRiskScorer:
    @pytest.fixture(scope='class')
    def scorer(self, preprocessed, trained_trainer):
        pre, _, _, _, _ = preprocessed
        model = trained_trainer.best_model
        return RiskScorer(model=model, preprocessor=pre, feature_engineer_fn=engineer_features)

    def test_categories(self, scorer):
        loan = {
            'loan_amount': 20000, 'loan_term': 36, 'interest_rate': 10.0,
            'annual_income': 60000, 'debt_to_income_ratio': 20.0, 'credit_score': 700,
            'employment_length': 5, 'num_credit_lines': 8, 'num_derogatory_marks': 0,
            'home_ownership': 'RENT', 'loan_purpose': 'car', 'verification_status': 'Verified',
        }
        result = scorer.score_single(loan)
        assert result['risk_category'] in RiskScorer.RISK_LABELS, \
            f"Unexpected category: {result['risk_category']}"

    def test_probability_range(self, scorer):
        loans = [
            {
                'loan_amount': 10000 * (i + 1), 'loan_term': 36, 'interest_rate': 8.0,
                'annual_income': 50000, 'debt_to_income_ratio': 15.0, 'credit_score': 720,
                'employment_length': 4, 'num_credit_lines': 7, 'num_derogatory_marks': 0,
                'home_ownership': 'OWN', 'loan_purpose': 'car', 'verification_status': 'Verified',
            }
            for i in range(3)
        ]
        df = pd.DataFrame(loans)
        results = scorer.score_batch(df)
        assert (results['default_probability'] >= 0).all()
        assert (results['default_probability'] <= 1).all()


class TestModelEvaluation:
    def test_metrics_range(self, preprocessed, trained_trainer):
        _, _, _, X_test, y_test = preprocessed
        evaluator = ModelEvaluator(trained_trainer.best_model, 'logistic_regression')
        metrics = evaluator.evaluate(X_test, y_test)

        for key in ('accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr'):
            assert 0.0 <= metrics[key] <= 1.0, f"{key} = {metrics[key]} out of [0, 1]"
        assert metrics['brier_score'] >= 0.0
        assert metrics['ks_statistic'] >= 0.0


class TestFullPipeline:
    def test_smoke(self, tmp_path):
        """Run the entire pipeline with a tiny dataset and assert it completes."""
        import main as pipeline_module
        pipeline_module.run_pipeline(
            n_samples=500,
            output_dir=str(tmp_path / 'smoke_output'),
            random_state=7,
        )
        # Best model must have been selected
        # Verify output files exist
        assert (tmp_path / 'smoke_output' / 'loan_data_raw.csv').exists()
        assert (tmp_path / 'smoke_output' / 'model_comparison.csv').exists()
