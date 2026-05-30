"""
Model Training v2 -- trains multiple classifiers with proper methodology:
  - Stratified 5-fold cross-validation with per-fold reporting
  - Class-weight balancing for imbalanced datasets
  - Feature importance analysis
  - Confidence calibration
  - Ablation study support

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
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Columns to drop before training (identifiers + label)
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

    # Report zero-variance features
    zero_var = [c for c in feature_cols if X[c].nunique() <= 1]
    if zero_var:
        print(f"\n  WARNING: Zero-variance features (will be ignored by tree models):")
        for c in zero_var:
            print(f"    {c}: constant value = {X[c].iloc[0]}")

    print(f"\n  Total features: {len(feature_cols)}")

    return X, y, le, feature_cols


def _build_classifiers() -> dict:
    """Build the classifier candidates with class-weight balancing."""
    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            )),
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


def _run_ablation(X, y, feature_names, best_clf_class, best_params, cv_folds):
    """Run ablation study: remove feature groups and measure impact."""
    print(f"\n{'='*60}")
    print("ABLATION STUDY")
    print(f"{'='*60}")

    # Define feature groups
    feature_groups = {
        "Timing (IAT)": [c for c in feature_names if c.startswith("iat_")],
        "Packet Size": [c for c in feature_names if c.startswith("pkt_size_") and not c.startswith("pkt_size_entropy")],
        "TLS Record": [c for c in feature_names if c.startswith("tls_")],
        "First-N Sizes": [c for c in feature_names if c.startswith("pkt_size_") and c[-1].isdigit()],
        "First-N Dirs": [c for c in feature_names if c.startswith("pkt_dir_")],
        "Turn-Taking": [c for c in feature_names if "turn" in c or "direction" in c],
        "Burst": [c for c in feature_names if "burst" in c],
        "TCP Flags": [c for c in feature_names if c.startswith("flag_")],
        "Directional Bytes": [c for c in feature_names if c in ["fwd_bytes", "bwd_bytes", "byte_asymmetry"]],
    }

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Baseline: all features
    clf_all = best_clf_class(**best_params)
    y_pred_all = cross_val_predict(clf_all, X, y, cv=skf)
    baseline_f1 = f1_score(y, y_pred_all, average="weighted")
    baseline_acc = accuracy_score(y, y_pred_all)

    print(f"\n  Baseline (all features):  Acc={baseline_acc:.4f}  F1={baseline_f1:.4f}")
    print(f"  {'Group':<25s} {'Removed':<8s} {'Acc':<10s} {'F1':<10s} {'Delta F1':<10s}")
    print(f"  {'-'*63}")

    results = []
    for group_name, group_cols in feature_groups.items():
        # Find actual columns present
        present = [c for c in group_cols if c in feature_names]
        if not present:
            continue

        # Remove the group
        remaining = [c for c in feature_names if c not in present]
        X_ablated = X[remaining]

        clf_ablated = best_clf_class(**best_params)
        try:
            y_pred_abl = cross_val_predict(clf_ablated, X_ablated, y, cv=skf)
            abl_f1 = f1_score(y, y_pred_abl, average="weighted")
            abl_acc = accuracy_score(y, y_pred_abl)
            delta = abl_f1 - baseline_f1
            print(f"  {group_name:<25s} {len(present):<8d} {abl_acc:<10.4f} {abl_f1:<10.4f} {delta:+.4f}")
            results.append((group_name, len(present), abl_acc, abl_f1, delta))
        except Exception as e:
            print(f"  {group_name:<25s} ERROR: {e}")

    return results


def train(
    csv_path: str,
    output_dir: str = "models",
    test_size: float = 0.2,
    cv_folds: int = 5,
    run_ablation: bool = True,
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
    print(f"TRAINING AND EVALUATION ({cv_folds}-fold Stratified CV)")
    print(f"{'='*60}")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for name, clf in classifiers.items():
        print(f"\n--- {name} ---")

        # Per-fold CV
        fold_f1s = []
        fold_accs = []
        for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train[val_idx]

            clf_fold = type(clf)(**clf.get_params()) if not isinstance(clf, Pipeline) else clf
            clf_fold.fit(X_fold_train, y_fold_train)
            y_fold_pred = clf_fold.predict(X_fold_val)

            fold_acc = accuracy_score(y_fold_val, y_fold_pred)
            fold_f1 = f1_score(y_fold_val, y_fold_pred, average="weighted")
            fold_f1s.append(fold_f1)
            fold_accs.append(fold_acc)
            print(f"  Fold {fold_i+1}: Acc={fold_acc:.4f}  F1={fold_f1:.4f}")

        print(f"  CV Mean: Acc={np.mean(fold_accs):.4f}+/-{np.std(fold_accs):.4f}  "
              f"F1={np.mean(fold_f1s):.4f}+/-{np.std(fold_f1s):.4f}")

        # Full training on train split
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        misclassified = (y_test != y_pred).sum()

        print(f"\n  Test Set Results:")
        print(f"    Accuracy:      {acc:.4f}  ({acc*100:.1f}%)")
        print(f"    Precision:     {prec:.4f}")
        print(f"    Recall:        {rec:.4f}")
        print(f"    F1-score:      {f1:.4f}")
        print(f"    Misclassified: {misclassified} / {len(y_test)}")

        results[name] = {"clf": clf, "f1": f1, "acc": acc, "cv_f1_mean": np.mean(fold_f1s)}

    # Find best model by CV F1 (more robust than test F1)
    best_name = max(results, key=lambda k: results[k]["cv_f1_mean"])
    best_clf = results[best_name]["clf"]
    best_f1 = results[best_name]["f1"]
    best_acc = results[best_name]["acc"]
    best_cv_f1 = results[best_name]["cv_f1_mean"]

    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_name}")
    print(f"  CV F1:     {best_cv_f1:.4f}")
    print(f"  Test Acc:  {best_acc:.4f}  ({best_acc*100:.1f}%)")
    print(f"  Test F1:   {best_f1:.4f}")
    print(f"{'='*60}")

    # Feature importances
    imp_model = best_clf
    if isinstance(imp_model, Pipeline):
        imp_model = imp_model.named_steps.get("clf", imp_model)

    if hasattr(imp_model, "feature_importances_"):
        importances = imp_model.feature_importances_
        feat_imp = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        print(f"\nTop-15 feature importances:")
        for fname, imp in feat_imp[:15]:
            bar = "#" * int(imp * 100)
            print(f"  {fname:<30s} {imp:.4f} {bar}")

    # Classification report on test set
    y_pred_best = best_clf.predict(X_test)
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred_best, target_names=le.classes_))

    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best)
    print(cm)

    # Confidence analysis
    if hasattr(best_clf, "predict_proba"):
        proba = best_clf.predict_proba(X_test)
        max_proba = proba.max(axis=1)
        uncertain = (max_proba < 0.8).sum()
        print(f"\nConfidence Analysis:")
        print(f"  Mean confidence:     {max_proba.mean():.4f}")
        print(f"  Min confidence:      {max_proba.min():.4f}")
        print(f"  Uncertain (<0.8):    {uncertain} / {len(y_test)} ({uncertain/len(y_test)*100:.1f}%)")

    # Ablation study
    if run_ablation and hasattr(imp_model, "feature_importances_"):
        best_params = imp_model.get_params()
        _run_ablation(X, y, feature_names, type(imp_model), best_params, cv_folds)

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
        f.write(f"CV F1:    {best_cv_f1:.4f}\n")
        f.write(f"Test Acc: {best_acc:.4f}\n")
        f.write(f"Test F1:  {best_f1:.4f}\n")
        f.write(f"Samples:  {len(X)}\n")
        f.write(f"Features: {len(feature_names)}\n")
        f.write(f"\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred_best, target_names=le.classes_))

    print(f"[Train] Results saved to results/training_results.txt")


def main():
    parser = argparse.ArgumentParser(description="Model Training v2")
    parser.add_argument("--features", default="data/features.csv")
    parser.add_argument("--output", default="models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()

    train(
        args.features, args.output, args.test_size, args.cv_folds,
        run_ablation=not args.no_ablation,
    )


if __name__ == "__main__":
    main()
