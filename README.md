# Customer Churn Prediction — End-to-End ML Pipeline

## Overview

This project predicts whether a telecom customer is likely to churn using a machine learning classification pipeline.

The project demonstrates data preparation, leakage-safe preprocessing, model training, evaluation, and saving a reusable scikit-learn pipeline.

## Problem Statement

Customer churn occurs when customers stop using a service. Predicting churn can help businesses identify customers who may need retention support.

**Objective:** Predict whether a customer will churn (`Yes`) or stay (`No`) based on customer and service information.

## Dataset

The project uses the Telco Customer Churn dataset.

- Total records: 7,043
- Input features after removing the identifier and target: 19
- Target: `Churn`
- Retained customers: 5,174
- Churned customers: 1,869

The raw CSV is stored locally in `data/raw/` and is not included in this repository. Download the dataset separately and place it at the path specified in the project instructions. Review the original dataset's terms before redistributing it.

## Tech Stack

- Python
- pandas
- scikit-learn
- NumPy
- joblib

## Machine Learning Workflow

1. Load and inspect the dataset.
2. Convert `TotalCharges` to numeric, treating invalid or blank values as missing.
3. Remove the `customerID` identifier from the input features.
4. Encode the target as `0 = No` and `1 = Yes`.
5. Split the data into training and test sets using stratified sampling.
6. Impute and scale numeric features.
7. Impute and one-hot encode categorical features.
8. Train a Logistic Regression classifier.
9. Evaluate performance on the held-out test set.
10. Save the complete preprocessing and prediction pipeline.

Preprocessing is fitted on the training data through a scikit-learn `Pipeline` and `ColumnTransformer`.

## Model Performance

Held-out test-set results:

| Metric | Score |
|---|---:|
| Accuracy | 0.738 |
| ROC-AUC | 0.841 |
| Churn precision | 0.50 |
| Churn recall | 0.78 |
| Churn F1-score | 0.61 |

Confusion matrix (rows = actual, columns = predicted):

```text
[[747, 288],
 [ 81, 293]]
```

The model detects many churners but also produces false positives. These results represent a baseline, not a production-ready churn system.

## Project Structure

```text
customer-churn-pipeline/
├── data/
│   └── raw/                 # Local dataset; not committed
├── models/                  # Generated model artifacts
├── notebooks/
│   └── 01_churn_pipeline.ipynb
├── src/
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup and Run

Create and activate a virtual environment, then install the dependencies.

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the dataset separately and place it in `data/raw/` using the filename expected by `src/train.py`.

Run training from the repository root:

```bash
python src/train.py
```

The script prints evaluation metrics and saves the complete pipeline to:

```text
models/churn_pipeline.joblib
```

## Future Improvements

- Compare Logistic Regression with tree-based models.
- Tune hyperparameters using cross-validation on the training set.
- Compare precision, recall, F1-score, and ROC-AUC.
- Evaluate classification thresholds based on business costs.
- Add a prediction interface and automated tests.

## Disclaimer

This project is for educational and portfolio purposes. Model predictions should not be treated as guarantees of customer behavior.
