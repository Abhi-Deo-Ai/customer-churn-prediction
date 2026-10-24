"""Train a leakage-safe customer churn baseline pipeline."""
from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COLUMNS = ["customerID"]


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to Telco Customer Churn CSV",
    )
    parser.add_argument("--model-out", default="models/churn_pipeline.joblib")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Download the CSV and pass its path with --data."
        )

    df = pd.read_csv(data_path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column {TARGET!r}; found {df.columns.tolist()}")

    # In common versions of this dataset, blank TotalCharges values are strings.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    y = df[TARGET].astype(str).str.strip().map({"Yes": 1, "No": 0})
    valid = y.notna()
    df, y = df.loc[valid].copy(), y.loc[valid].astype(int)

    X = df.drop(columns=[TARGET] + [c for c in ID_COLUMNS if c in df.columns])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Test-set results ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred, target_names=["No churn", "Churn"]))
    print("Confusion matrix [rows=true, columns=predicted]:")
    print(confusion_matrix(y_test, y_pred))
    if hasattr(model, "predict_proba"):
        print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.3f}")

    output = Path(args.model_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print(f"Saved complete pipeline to: {output}")


if __name__ == "__main__":
    main()
