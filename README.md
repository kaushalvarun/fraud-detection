# Credit Card Fraud Detection

A machine learning project to automatically detect fraudulent credit card transactions.

## Business Context
Banks can experience significant losses due to fraud, with non-ml solutions such as possibly manual review process too slow and costly to scale. Business is required an 
automated solution capable of identifying suspicious transactions in real time, without 
disrupting legitimate customers through unnecessary card blocks. With explainability to 
regulators as a hard requirement.

## Problem Statement

| | |
|---|---|
| **Business Objective** | Identify fraudulent transactions to minimize monetary losses for the bank and customers, while reducing unnecessary disruption to legitimate customers |
| **ML Task** | Supervised Learning — Binary Classification |
| **Input (X)** | 2 years of anonymized transaction data (V1-V28 PCA components, Amount, Time) |
| **Output (y)** | fraud = 1, legit = 0 |
| **Primary Metric** | F2 Score — weights recall twice as heavily as precision, reflecting that missed fraud costs more than a false alarm |
| **Secondary Metric** | Confusion Matrix — full picture of TP, TN, FP, FN |

## Constraints
- **Latency** — model must not significantly increase transaction processing time
- **Explainability** — must be interpretable for regulators and compliance teams
- **Customer disruption** — minimize false positives, unnecessary card blocks damage trust
- **Model drift** — fraud patterns evolve, model requires regular retraining cadence


## Dataset
- **Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions over 48 hours
- **Class imbalance:** 0.17% fraud (473 cases) — severe imbalance
- **Features:** V1-V28 (PCA anonymized), Amount, Time

## Approach & Results

| Model | Recall | Precision | F2 Score | False Negatives | False Positives |
|---|---|---|---|---|---|
| Baseline Logistic Regression | 0.54 | 0.88 | 0.59 | 41 | 7 |
| + class_weight='balanced' | 0.89 | 0.06 | 0.22 | 10 | 1262 |
| + SMOTE + Threshold Tuning (0.99) | 0.79 | 0.57 | 0.73 | 19 | 54 |

**Best model:** SMOTE + threshold tuning achieving F2 of 0.73

## Key Findings
- Fraud transactions tend to be **smaller amounts** — consistent with transaction testing behavior
- **Time alone is a weak predictor** — fraud occurs across all hours with slight preference for off-peak
- **V17, V14, V12, V10** are the strongest negative predictors of fraud
- Logistic regression hits a ceiling due to its linear decision boundary — fraud patterns are non-linear

## Next Steps
- Rebuild with **XGBoost** — industry standard for fraud detection on tabular data
- Target F2 > 0.85
- Package best model into a **FastAPI endpoint** for production deployment

## Tech Stack
- Python, Jupyter Notebook
- pandas, numpy, matplotlib, seaborn
- scikit-learn, imbalanced-learn