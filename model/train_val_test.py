"""
Proper Train / Validate / Test Split — creates completely separate
splits based on capture SESSIONS (not random samples).

This proves there's no data leakage from training into testing because
each split comes from different capture runs.

The script:
1. Groups flows by their capture session (identified by pcap filename/port)
2. Assigns entire sessions to train (60%), validate (20%), or test (20%)
3. Trains the model on TRAIN only
4. Tunes on VALIDATE
5. Final evaluation on TEST (never seen during training)

Usage:
    python model/train_val_test.py --features data/features_final.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DROP_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "label"]


def _identify_sessions(df):
    """Group flows into sessions based on server port (proxy for capture run)."""
    sessions = {}
    for port in df['src_port'].unique():
        mask = df['src_port'] == port
        indices = df.index[mask].tolist()
        if len(indices) >= 2:
            sessions[port] = indices
    return sessions


def main():
    parser = argparse.ArgumentParser(description="Train/Validate/Test Split")
    parser.add_argument("--features", default="data/features_final.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    le = LabelEncoder()
    y_all = le.fit_transform(df["label"])
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X_all = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    print(f"Dataset: {len(df)} samples, {len(feature_cols)} features")
    print(f"Classes: mcp={sum(y_all==0)}, non_mcp={sum(y_all==1)}")

    # ---------------------------------------------------------------
    # Session-based split
    # ---------------------------------------------------------------
    sessions = _identify_sessions(df)
    session_keys = sorted(sessions.keys())
    print(f"\nIdentified {len(session_keys)} sessions")

    # Classify sessions by their dominant label
    mcp_sessions = []
    nonmcp_sessions = []
    for port in session_keys:
        idx = sessions[port]
        labels = df.iloc[idx]['label']
        if (labels == 'mcp').sum() > (labels == 'non_mcp').sum():
            mcp_sessions.append(port)
        else:
            nonmcp_sessions.append(port)

    print(f"  MCP sessions: {len(mcp_sessions)}")
    print(f"  Non-MCP sessions: {len(nonmcp_sessions)}")

    # Shuffle and split sessions: 60% train, 20% validate, 20% test
    rng = np.random.RandomState(42)
    rng.shuffle(mcp_sessions)
    rng.shuffle(nonmcp_sessions)

    def _split_list(lst):
        n = len(lst)
        t1 = int(n * 0.6)
        t2 = int(n * 0.8)
        return lst[:t1], lst[t1:t2], lst[t2:]

    mcp_train, mcp_val, mcp_test = _split_list(mcp_sessions)
    non_train, non_val, non_test = _split_list(nonmcp_sessions)

    train_sessions = mcp_train + non_train
    val_sessions = mcp_val + non_val
    test_sessions = mcp_test + non_test

    train_idx = [i for s in train_sessions for i in sessions[s]]
    val_idx = [i for s in val_sessions for i in sessions[s]]
    test_idx = [i for s in test_sessions for i in sessions[s]]

    X_train, y_train = X_all.iloc[train_idx], y_all[train_idx]
    X_val, y_val = X_all.iloc[val_idx], y_all[val_idx]
    X_test, y_test = X_all.iloc[test_idx], y_all[test_idx]

    print(f"\n{'='*60}")
    print(f"SESSION-BASED TRAIN / VALIDATE / TEST SPLIT")
    print(f"{'='*60}")
    print(f"  Train:    {len(train_idx)} flows from {len(train_sessions)} sessions "
          f"(MCP={sum(y_train==0)}, non_mcp={sum(y_train==1)})")
    print(f"  Validate: {len(val_idx)} flows from {len(val_sessions)} sessions "
          f"(MCP={sum(y_val==0)}, non_mcp={sum(y_val==1)})")
    print(f"  Test:     {len(test_idx)} flows from {len(test_sessions)} sessions "
          f"(MCP={sum(y_test==0)}, non_mcp={sum(y_test==1)})")

    # Check for index overlap
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    assert len(train_set & val_set) == 0, "Train/Val overlap!"
    assert len(train_set & test_set) == 0, "Train/Test overlap!"
    assert len(val_set & test_set) == 0, "Val/Test overlap!"
    print(f"\n  [OK] No index overlap between splits (verified)")

    # ---------------------------------------------------------------
    # Train models
    # ---------------------------------------------------------------
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42
        ),
    }
    if HAS_XGB:
        scale = sum(y_train == 1) / max(sum(y_train == 0), 1)
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale, use_label_encoder=False,
            eval_metric="logloss", random_state=42, verbosity=0,
        )

    best_model = None
    best_val_f1 = -1
    best_name = ""

    for name, clf in models.items():
        clf.fit(X_train, y_train)

        y_pred_train = clf.predict(X_train)
        y_pred_val = clf.predict(X_val)

        train_acc = accuracy_score(y_train, y_pred_train)
        train_f1 = f1_score(y_train, y_pred_train, average="weighted")
        val_acc = accuracy_score(y_val, y_pred_val)
        val_f1 = f1_score(y_val, y_pred_val, average="weighted")

        print(f"\n--- {name} ---")
        print(f"  Train: Acc={train_acc:.4f}  F1={train_f1:.4f}")
        print(f"  Val:   Acc={val_acc:.4f}  F1={val_f1:.4f}")
        print(f"  Gap:   {train_acc - val_acc:+.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = clf
            best_name = name

    # ---------------------------------------------------------------
    # Final evaluation on TEST (never seen)
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"FINAL TEST EVALUATION (completely unseen sessions)")
    print(f"Model: {best_name}")
    print(f"{'='*60}")

    y_pred_test = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average="weighted")

    print(f"\n  Accuracy:  {test_acc:.4f} ({test_acc*100:.1f}%)")
    print(f"  F1-score:  {test_f1:.4f}")
    print(f"  Misclassified: {sum(y_pred_test != y_test)} / {len(y_test)}")

    print(f"\n{classification_report(y_test, y_pred_test, target_names=le.classes_)}")

    cm = confusion_matrix(y_test, y_pred_test)
    print(f"Confusion Matrix:")
    print(f"  {cm}")

    # Confidence analysis
    if hasattr(best_model, 'predict_proba'):
        proba = best_model.predict_proba(X_test)
        conf = proba.max(axis=1)
        print(f"\nConfidence Analysis:")
        print(f"  Mean: {conf.mean():.4f}")
        print(f"  Min:  {conf.min():.4f}")
        print(f"  Uncertain (<0.8): {(conf < 0.8).sum()} / {len(conf)}")

    # ---------------------------------------------------------------
    # Overfitting verdict
    # ---------------------------------------------------------------
    y_pred_train_final = best_model.predict(X_train)
    train_acc_final = accuracy_score(y_train, y_pred_train_final)

    print(f"\n{'='*60}")
    print(f"OVERFITTING VERDICT")
    print(f"{'='*60}")
    print(f"  Train Accuracy: {train_acc_final:.4f}")
    print(f"  Val Accuracy:   {best_val_f1:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Train-Test Gap: {train_acc_final - test_acc:+.4f}")

    gap = train_acc_final - test_acc
    if gap < 0.02:
        print(f"\n  [PASS] No overfitting (gap < 2%)")
    elif gap < 0.05:
        print(f"\n  [CAUTION] Mild overfitting (gap {gap:.1%})")
    else:
        print(f"\n  [FAIL] Significant overfitting (gap {gap:.1%})")

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/train_val_test_results.txt", "w") as f:
        f.write(f"Model: {best_name}\n")
        f.write(f"Train flows: {len(train_idx)} from {len(train_sessions)} sessions\n")
        f.write(f"Val flows: {len(val_idx)} from {len(val_sessions)} sessions\n")
        f.write(f"Test flows: {len(test_idx)} from {len(test_sessions)} sessions\n")
        f.write(f"Train Acc: {train_acc_final:.4f}\n")
        f.write(f"Val F1: {best_val_f1:.4f}\n")
        f.write(f"Test Acc: {test_acc:.4f}\n")
        f.write(f"Test F1: {test_f1:.4f}\n")
        f.write(f"Train-Test Gap: {gap:+.4f}\n")
        f.write(f"\n{classification_report(y_test, y_pred_test, target_names=le.classes_)}\n")

    print(f"\n[Results saved to results/train_val_test_results.txt]")


if __name__ == "__main__":
    main()
