from setuptools import find_packages, setup

setup(
    name="loan-risk-assessment",
    version="1.1.0",
    description="Loan default risk assessment with explainable ML",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn==1.8.0",
        "pandas==3.0.2",
        "numpy==2.4.4",
        "matplotlib==3.10.8",
        "seaborn==0.13.2",
        "shap==0.51.0",
        "xgboost==3.2.0",
        "imbalanced-learn==0.14.1",
        "joblib==1.5.3",
        "flask==3.1.3",
    ],
)
