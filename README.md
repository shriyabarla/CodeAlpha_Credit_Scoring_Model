## Credit Scoring Model

## Overview
This repository contains a machine learning pipeline designed to predict an individual's creditworthiness and probability of default within two years.

## Technical Architecture
The pipeline processes raw financial data and builds predictive models using the following workflow:
1. **Feature Engineering:** Extracted actionable signals from raw data, including `TotalPastDue`, `MonthlyDebt`, `IncomePerPerson`, and `HighRevolvingUtil` flags.
2. **Data Leakage Prevention:** Implemented strict train/test splitting prior to applying median imputation and standard scaling.
3. **Model Training:** Built baseline models using Logistic Regression and Random Forest Classifiers (handling class imbalance via class weighting).
4. **Threshold Optimization:** Shifted the classification threshold from the default 0.5 to an optimized mathematical balance to maximize the F1-Score, aligning the model with realistic banking risk tolerance.

## Evaluation Metrics
The final Random Forest model evaluates performance based on precision, recall, F1-score, and ROC-AUC, ensuring the model successfully separates high-risk from low-risk applicants.

## Files
* `credit_model.py`: The complete end-to-end training and evaluation script.
* `random_forest_model.pkl`: The serialized, production-ready predictive model.
* `scaler.pkl` & `imputer.pkl`: Saved preprocessing artifacts for inference.
* `cs-training.csv`: The Kaggle financial dataset.
