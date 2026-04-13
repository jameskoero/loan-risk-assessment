"""
Setup script for the loan-risk-assessment package.
"""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="loan-risk-assessment",
    version="1.0.0",
    author="James Koero",
    author_email="jmskoero@gmail.com",
    description="Advanced Loan Default Risk Assessment — ML system using GradientBoosting, SHAP, and Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jameskoero/loan-risk-assessment",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
