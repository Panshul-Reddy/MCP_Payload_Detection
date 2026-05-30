"""
Overfitting Analysis — comprehensive check for overfitting using:
  1. Train vs. Test accuracy gap (should be <5%)
  2. Learning curves (accuracy vs dataset size)
  3. Session-based cross-validation (train on some sessions, test on others)
  4. Per-class confidence distribution
  5. Feature correlation with label (detect shortcut features)

Usage:
    python model/overfit_check.py --features data/features_mega.csv
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder

DROP_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "label"]
PARAMS = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced",
              n_jobs=-1, random_state=42)


def _load(csv_path):
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    return df, X, y, le, feature_cols


def check_train_test_gap(X, y):
    """Test 1: Train accuracy vs CV accuracy gap."""
    print("=" * 60)
    print("TEST 1: TRAIN vs. CV ACCURACY GAP")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_accs = []
    test_accs = []
    train_f1s = []
    test_f1s = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(**PARAMS)
        clf.fit(X_train, y_train)

        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        tr_acc = accuracy_score(y_train, y_pred_train)
        te_acc = accuracy_score(y_test, y_pred_test)
        tr_f1 = f1_score(y_train, y_pred_train, average="weighted")
        te_f1 = f1_score(y_test, y_pred_test, average="weighted")

        train_accs.append(tr_acc)
        test_accs.append(te_acc)
        train_f1s.append(tr_f1)
        test_f1s.append(te_f1)

        print(f"  Fold {fold+1}: Train Acc={tr_acc:.4f}  Test Acc={te_acc:.4f}  "
              f"Gap={tr_acc - te_acc:+.4f}")

    mean_gap = np.mean(train_accs) - np.mean(test_accs)
    print(f"\n  Mean Train Acc: {np.mean(train_accs):.4f}")
    print(f"  Mean Test Acc:  {np.mean(test_accs):.4f}")
    print(f"  Mean Gap:       {mean_gap:+.4f}")

    if mean_gap > 0.05:
        print(f"\n  ** WARNING: Gap > 5% suggests OVERFITTING")
    elif mean_gap > 0.02:
        print(f"\n  ** CAUTION: Gap > 2% suggests mild overfitting")
    else:
        print(f"\n  ** GOOD: Gap < 2% suggests no significant overfitting")

    return mean_gap


def check_learning_curves(X, y):
    """Test 2: Learning curves -- does more data help?"""
    print(f"\n{'=' * 60}")
    print("TEST 2: LEARNING CURVES")
    print("=" * 60)

    clf = RandomForestClassifier(**PARAMS)
    train_sizes = np.linspace(0.1, 1.0, 8)

    train_sizes_abs, train_scores, test_scores = learning_curve(
        clf, X, y,
        train_sizes=train_sizes,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        random_state=42,
    )

    print(f"\n  {'N_train':<10s} {'Train F1':<12s} {'CV F1':<12s} {'Gap':<10s}")
    print(f"  {'-'*44}")

    for n, tr, te in zip(train_sizes_abs, train_scores, test_scores):
        tr_mean = np.mean(tr)
        te_mean = np.mean(te)
        gap = tr_mean - te_mean
        print(f"  {n:<10d} {tr_mean:<12.4f} {te_mean:<12.4f} {gap:+.4f}")

    # Check if curve is still rising (more data would help)
    last_3_cv = [np.mean(test_scores[i]) for i in range(-3, 0)]
    improvement = last_3_cv[-1] - last_3_cv[0]

    if improvement > 0.005:
        print(f"\n  ** NOTE: CV F1 still improving ({improvement:+.4f} over last 3 points)")
        print(f"     -> More data would likely help")
    else:
        print(f"\n  ** GOOD: CV F1 has plateaued (change={improvement:+.4f})")
        print(f"     -> Adding more data may not significantly improve results")


def check_session_based_cv(df, X, y, le):
    """Test 3: Session-based CV -- train on some pcap sessions, test on others."""
    print(f"\n{'=' * 60}")
    print("TEST 3: SESSION-BASED CROSS-VALIDATION")
    print("=" * 60)

    # Identify sessions by src_port ranges (each orchestrator run uses different ports)
    # Group flows by their server port (src_port for server-side)
    server_ports = df['src_port'].unique()
    print(f"  Unique server ports: {len(server_ports)} ({sorted(server_ports)[:10]}...)")

    # Group by server port as proxy for session
    sessions = {}
    for port in server_ports:
        mask = df['src_port'] == port
        indices = df.index[mask].tolist()
        if len(indices) >= 3:  # Only sessions with enough flows
            sessions[port] = indices

    if len(sessions) < 3:
        print("  Not enough distinct sessions for session-based CV")
        print("  (Need at least 3 sessions with 3+ flows each)")
        return

    session_keys = list(sessions.keys())
    print(f"  Usable sessions: {len(session_keys)}")

    # Leave-one-session-out CV
    results = []
    for i, test_port in enumerate(session_keys):
        test_idx = sessions[test_port]
        train_idx = []
        for port in session_keys:
            if port != test_port:
                train_idx.extend(sessions[port])

        if len(test_idx) < 2 or len(train_idx) < 5:
            continue

        X_train = X.iloc[train_idx]
        y_train = y[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y[test_idx]

        # Check we have both classes in train
        if len(np.unique(y_train)) < 2:
            continue

        clf = RandomForestClassifier(**PARAMS)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Get class label of test set
        test_labels = le.inverse_transform(y_test)
        dominant_class = pd.Series(test_labels).mode().iloc[0]

        results.append((test_port, len(test_idx), acc, f1, dominant_class))

    if not results:
        print("  Could not perform session-based CV")
        return

    print(f"\n  {'Session Port':<15s} {'N_test':<8s} {'Acc':<10s} {'F1':<10s} {'Class'}")
    print(f"  {'-'*53}")
    for port, n, acc, f1, cls in results:
        marker = " **" if acc < 0.9 else ""
        print(f"  {port:<15d} {n:<8d} {acc:<10.4f} {f1:<10.4f} {cls}{marker}")

    accs = [r[2] for r in results]
    f1s = [r[3] for r in results]
    print(f"\n  Mean Session Acc: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Mean Session F1:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    if np.std(accs) > 0.1:
        print(f"\n  ** WARNING: High variance across sessions (std={np.std(accs):.4f})")
        print(f"     -> Model may not generalize well to new traffic patterns")
    else:
        print(f"\n  ** GOOD: Consistent performance across sessions")


def check_confidence_distribution(X, y, le):
    """Test 4: Confidence distribution per class."""
    print(f"\n{'=' * 60}")
    print("TEST 4: PREDICTION CONFIDENCE DISTRIBUTION")
    print("=" * 60)

    clf = RandomForestClassifier(**PARAMS)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    all_proba = []
    all_true = []
    all_pred = []

    for train_idx, test_idx in skf.split(X, y):
        clf_fold = RandomForestClassifier(**PARAMS)
        clf_fold.fit(X.iloc[train_idx], y[train_idx])
        proba = clf_fold.predict_proba(X.iloc[test_idx])
        pred = clf_fold.predict(X.iloc[test_idx])
        all_proba.extend(proba.max(axis=1))
        all_true.extend(y[test_idx])
        all_pred.extend(pred)

    all_proba = np.array(all_proba)
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    for cls_idx, cls_name in enumerate(le.classes_):
        mask = all_true == cls_idx
        cls_proba = all_proba[mask]
        cls_correct = all_pred[mask] == all_true[mask]
        print(f"\n  {cls_name}:")
        print(f"    N samples:      {mask.sum()}")
        print(f"    Mean confidence: {cls_proba.mean():.4f}")
        print(f"    Min confidence:  {cls_proba.min():.4f}")
        print(f"    Accuracy:        {cls_correct.mean():.4f}")
        uncertain = (cls_proba < 0.8).sum()
        print(f"    Uncertain (<0.8): {uncertain} ({uncertain/mask.sum()*100:.1f}%)")

    # Overall
    total_uncertain = (all_proba < 0.8).sum()
    total_wrong = (all_pred != all_true).sum()
    print(f"\n  Overall:")
    print(f"    Total uncertain (<0.8): {total_uncertain}/{len(all_proba)} ({total_uncertain/len(all_proba)*100:.1f}%)")
    print(f"    Total misclassified:    {total_wrong}/{len(all_pred)} ({total_wrong/len(all_pred)*100:.1f}%)")


def check_shortcut_features(df, feature_cols):
    """Test 5: Check for features that perfectly correlate with the label."""
    print(f"\n{'=' * 60}")
    print("TEST 5: SHORTCUT FEATURE DETECTION")
    print("=" * 60)

    label_numeric = (df['label'] == 'mcp').astype(int)
    correlations = []
    for col in feature_cols:
        try:
            corr = abs(df[col].corr(label_numeric))
            correlations.append((col, corr))
        except Exception:
            pass

    correlations.sort(key=lambda x: -x[1])

    print(f"\n  Features most correlated with label (potential shortcuts):")
    print(f"  {'Feature':<30s} {'|Correlation|'}")
    print(f"  {'-'*45}")
    for name, corr in correlations[:15]:
        marker = " ** SHORTCUT" if corr > 0.9 else (" * HIGH" if corr > 0.7 else "")
        print(f"  {name:<30s} {corr:.4f}{marker}")

    high_corr = [name for name, corr in correlations if corr > 0.9]
    if high_corr:
        print(f"\n  ** WARNING: {len(high_corr)} features have |correlation| > 0.9 with label")
        print(f"     These may be shortcut features: {high_corr[:5]}")
    else:
        print(f"\n  ** GOOD: No features have |correlation| > 0.9 with label")


def main():
    parser = argparse.ArgumentParser(description="Overfitting Analysis")
    parser.add_argument("--features", default="data/features_mega.csv")
    args = parser.parse_args()

    df, X, y, le, feature_cols = _load(args.features)
    print(f"Dataset: {len(df)} samples, {len(feature_cols)} features")
    print(f"Classes: {dict(zip(le.classes_, np.bincount(y)))}")

    gap = check_train_test_gap(X, y)
    check_learning_curves(X, y)
    check_session_based_cv(df, X, y, le)
    check_confidence_distribution(X, y, le)
    check_shortcut_features(df, feature_cols)

    # Final verdict
    print(f"\n{'=' * 60}")
    print("FINAL VERDICT")
    print("=" * 60)
    if gap < 0.02:
        print("  Train-Test Gap: PASS (< 2%)")
    else:
        print(f"  Train-Test Gap: FAIL ({gap:.1%})")
    print(f"  Recommendation: See detailed results above for each test")


if __name__ == "__main__":
    main()
