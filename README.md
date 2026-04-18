# Loan Risk Assessment

## Overview
This project aims to assess the risk of loan defaults using machine learning techniques, providing lenders with better insights into the risk associated with prospective borrowers. The model helps in making informed lending decisions.

## Problem Statement
Lenders are often faced with uncertainty regarding the risk of default on loans. This project seeks to mitigate this issue by leveraging historical loan data to predict the likelihood of default using machine learning algorithms.

## ML Pipeline
The pipeline consists of data preprocessing, feature selection, model training, evaluation, and deployment phases. Each stage is designed to optimize the model's performance and ensure robustness in real-world scenarios.

## Key Features
- Predictive modeling to assess loan default risk
- User-friendly interface for input and visualization
- Comprehensive reporting and analytics

## Tech Stack
- Programming Language: Python
- Frameworks: Scikit-learn, Flask
- Data Handling: Pandas, NumPy
- Visualization: Matplotlib, Seaborn

## Model Performance
The model's performance metrics are evaluated using accuracy, precision, recall, and F1-score to ensure it meets the expected standards for credit risk assessment.

## SHAP Explainability
SHAP values are used to interpret the model's predictions, providing insights into the features that drive the predictions. This enhances trust and transparency in the model's decision-making process.

## Project Structure
```
├── images/
├── data/
│   └── loan_data.csv
├��─ notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── model.py
│   ├���─ data_preprocessing.py
│   └── app.py
└── README.md
```

## How to Run
1. Clone the repository.
2. Install the required packages from `requirements.txt`.
3. Run the application using `python src/app.py`.

## Dataset
The dataset used for training and evaluation is located in the `data/` folder. It contains historical loan application records.

## Business Impact
By accurately assessing loan risks, this project aids lenders in minimizing defaults, improving overall financial health, and making data-driven lending decisions.

## Author
James Koero  
[jameskoero](https://github.com/jameskoero)

## License
This project is licensed under the MIT License.

## Visualizations
All visualizations are stored in the `images/` folder. Make sure to route your paths correctly to access the respective images for better insights during analysis.