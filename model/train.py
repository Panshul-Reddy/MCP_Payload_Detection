"""
Model Training — trains 3 classifiers (Random Forest, XGBoost, Logistic
Regression) on the feature CSV, evaluates with cross-validation and a
held-out test split, and saves the best model by F1-score.

Usage:
    python -m model.train --features data/features.csv --output models/
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Columns to drop before training (identifiers + leaky features)
DROP_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "label"]


def load_and_prepare(csv_path: str) -> tuple:
    """Load CSV, encode labels, return (X, y, label_encoder, feature_names)."""
    df = pd.read_csv(csv_path)
    print(f"[Train] Loaded {len(df)} samples from {csv_path}")
    print(f"  Class distribution:")
    for cls, count in df["label"].value_counts().items():
        print(f"    {cls}: {count} ({count/len(df)*100:.1f}%)")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["label"])

    # Drop identifier/leaky columns
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].copy()

    # Fill NaN with median
    X = X.fillna(X.median())

    # Replace inf with column max
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    return X, y, le, feature_cols


def _build_classifiers() -> dict:
    """Build the classifier candidates."""
    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
    }

    # Try XGBoost, fall back to GradientBoosting
    try:
        from xgboost import XGBClassifier
        classifiers["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        )
    except ImportError:
        classifiers["Gradient Boosting"] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )

    return classifiers


def train(
    csv_path: str,
    output_dir: str = "models",
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> None:
    """Train all classifiers, evaluate, save the best."""
    X, y, le, feature_names = load_and_prepare(csv_path)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    classifiers = _build_classifiers()
    results = {}

    print(f"\n{'='*60}")
    print("TRAINING AND EVALUATION")
    print(f"{'='*60}")

    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")

        # Cross-validation
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="f1_weighted")
        print(f"  CV F1 (mean +/- std): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # Full training
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        misclassified = (y_test != y_pred).sum()

        print(f"  Accuracy:      {acc:.4f}  ({acc*100:.1f}%)")
        print(f"  Precision:     {prec:.4f}")
        print(f"  Recall:        {rec:.4f}")
        print(f"  F1-score:      {f1:.4f}")
        print(f"  Misclassified: {misclassified} / {len(y_test)}")

        results[name] = {"clf": clf, "f1": f1, "acc": acc}

    # Find best model
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_clf = results[best_name]["clf"]
    best_f1 = results[best_name]["f1"]
    best_acc = results[best_name]["acc"]

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name}")
    print(f"  Accuracy:  {best_acc:.4f}  ({best_acc*100:.1f}%)")
    print(f"  F1-score:  {best_f1:.4f}")
    print(f"{'='*60}")

    # Feature importances
    if hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
        feat_imp = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        print(f"\nTop-10 feature importances:")
        for fname, imp in feat_imp[:10]:
            print(f"  {fname:<25s} {imp:.4f}")

    # Classification report on test set
    y_pred_best = best_clf.predict(X_test)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))

    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_best))

    # Save model bundle
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "best_model.pkl")
    bundle = {
        "model": best_clf,
        "label_encoder": le,
        "feature_names": feature_names,
        "model_name": best_name,
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"\n[Train] Model saved to {model_path}")

    # Save results to file
    os.makedirs("results", exist_ok=True)
    with open("results/training_results.txt", "w") as f:
        f.write(f"Best Model: {best_name}\n")
        f.write(f"Accuracy: {best_acc:.4f}\n")
        f.write(f"F1-score: {best_f1:.4f}\n")
        f.write(f"Samples: {len(X)}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred_best, target_names=le.classes_))

    print(f"[Train] Results saved to results/training_results.txt")


def main():
    parser = argparse.ArgumentParser(description="Model Training")
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--output", default="models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    train(args.features, args.output, args.test_size, args.cv_folds)


if __name__ == "__main__":
    main()
