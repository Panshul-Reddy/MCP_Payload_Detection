"""
Model Evaluation — loads a saved model and evaluates it on new feature CSVs.
Outputs classification metrics and per-flow predictions with probabilities.

Usage:
    python -m model.evaluate --model models/best_model.pkl --features data/test_features.csv
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# Columns to drop (same as training)
DROP_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "label"]


def load_model(model_path: str) -> tuple:
    """Load the model bundle."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["label_encoder"], bundle["feature_names"]


def evaluate(model_path: str, features_csv: str) -> None:
    """Evaluate a saved model on a features CSV."""
    model, le, feature_names = load_model(model_path)

    df = pd.read_csv(features_csv)
    print(f"{'='*60}")
    print(f"Evaluation on: {features_csv}")
    print(f"Model:         {model_path}")
    print(f"{'='*60}")

    # Encode labels
    y_true = le.transform(df["label"])

    # Prepare features (same columns as training)
    X = df[[c for c in feature_names if c in df.columns]].copy()

    # Add missing columns with 0
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0

    X = X[feature_names]  # Ensure correct order
    X = X.fillna(X.median())
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Predict
    y_pred = model.predict(X)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    misclassified = (y_true != y_pred).sum()

    print(f"\n  Accuracy:      {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision:     {prec:.4f}")
    print(f"  Recall:        {rec:.4f}")
    print(f"  F1-score:      {f1:.4f}")
    print(f"  Misclassified: {misclassified} / {len(y_true)}")

    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    print(f"Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Per-flow predictions with probabilities
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        pred_df = df.copy()
        pred_df["predicted_label"] = le.inverse_transform(y_pred)
        for i, cls in enumerate(le.classes_):
            pred_df[f"prob_{cls}"] = proba[:, i]

        out_path = features_csv.replace(".csv", "_predictions.csv")
        pred_df.to_csv(out_path, index=False)
        print(f"\n[Evaluate] Per-flow predictions saved to {out_path}")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/evaluation_results.txt", "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Data: {features_csv}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Misclassified: {misclassified} / {len(y_true)}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=le.classes_))


def main():
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument("--model", default="models/best_model.pkl")
    parser.add_argument("--features", required=True)
    args = parser.parse_args()

    evaluate(args.model, args.features)


if __name__ == "__main__":
    main()
