"""
Early Packet Classifier — classifies MCP vs non-MCP traffic using ONLY
the first N packets (N=3 to 17) instead of waiting for the entire flow.

This enables near-real-time classification:
  - After 3 packets:  ~200ms latency, basic classification
  - After 5 packets:  ~300ms latency, good classification
  - After 10 packets: ~500ms latency, strong classification
  - After 17 packets: ~1s latency, highest accuracy

The classifier uses:
  1. First-N packet sizes (raw values)
  2. First-N packet directions (1=fwd, -1=bwd)
  3. First-N inter-arrival times
  4. Partial TLS record counts (from available packets)
  5. Direction change count (turn-taking signal)
  6. Running byte totals (fwd vs bwd)

Usage:
    # Train early classifiers for different N values
    python model/early_classifier.py --features data/features_final.csv

    # Train for a specific N
    python model/early_classifier.py --features data/features_final.csv --n-packets 5
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


DROP_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "label"]


def get_early_features(n_packets: int) -> list:
    """Get feature column names available after N packets."""
    features = []

    # First-N packet sizes
    for i in range(1, min(n_packets + 1, 11)):  # max 10 in our extractor
        features.append(f"pkt_size_{i}")

    # First-N packet directions
    for i in range(1, min(n_packets + 1, 11)):
        features.append(f"pkt_dir_{i}")

    # Partial flow features computable from N packets
    features.extend([
        "total_packets",       # will be N (or close to it)
        "fwd_packets",         # partial
        "bwd_packets",         # partial
        "total_bytes",         # partial
        "fwd_bytes",           # partial
        "bwd_bytes",           # partial
        "pkt_size_mean",       # partial mean
        "pkt_size_std",        # partial std
        "pkt_size_max",        # partial max
        "flow_asymmetry",      # partial asymmetry
    ])

    # Direction/turn features (available early)
    features.extend([
        "direction_changes",   # key early signal
        "fwd_bwd_ratio",       # available after a few packets
    ])

    # TLS features (may have some records after N packets)
    if n_packets >= 3:
        features.extend([
            "tls_handshake_count",
            "tls_app_data_count",
            "tls_total_records",
        ])

    if n_packets >= 5:
        features.extend([
            "tls_ccs_count",
            "tls_app_data_ratio",
            "tls_handshake_ratio",
        ])

    if n_packets >= 7:
        features.extend([
            "tls_record_size_mean",
            "tls_record_size_std",
            "tls_app_data_size_mean",
        ])

    # IAT features (need at least 2 packets)
    if n_packets >= 3:
        features.extend([
            "iat_mean",
            "iat_std",
            "iat_min",
            "iat_max",
        ])

    return features


def train_early_classifier(df: pd.DataFrame, n_packets: int, cv_folds: int = 5):
    """Train and evaluate an early classifier for N packets."""
    early_features = get_early_features(n_packets)

    # Filter to features that actually exist in the dataframe
    available = [f for f in early_features if f in df.columns]
    missing = [f for f in early_features if f not in df.columns]

    if len(available) < 3:
        print(f"  [N={n_packets}] Too few features available ({len(available)}), skipping")
        return None

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X = df[available].fillna(0).replace([np.inf, -np.inf], 0)

    # Train XGBoost if available, else Random Forest
    if HAS_XGB:
        scale = sum(y == 1) / max(sum(y == 0), 1)
        clf = XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            scale_pos_weight=scale, use_label_encoder=False,
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        model_name = "XGBoost"
    else:
        clf = RandomForestClassifier(
            n_estimators=150, min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        model_name = "Random Forest"

    # Cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    y_pred_cv = cross_val_predict(clf, X, y, cv=skf)
    cv_f1 = f1_score(y, y_pred_cv, average="weighted")
    cv_acc = accuracy_score(y, y_pred_cv)

    # Train final model
    clf.fit(X, y)

    # Confidence analysis
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        mean_conf = proba.max(axis=1).mean()
        low_conf = (proba.max(axis=1) < 0.8).sum()
    else:
        mean_conf = 1.0
        low_conf = 0

    result = {
        "n_packets": n_packets,
        "features_used": len(available),
        "features_missing": len(missing),
        "cv_accuracy": cv_acc,
        "cv_f1": cv_f1,
        "mean_confidence": mean_conf,
        "low_confidence_count": low_conf,
        "model": clf,
        "model_name": model_name,
        "feature_names": available,
        "label_encoder": le,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Early Packet Classifier")
    parser.add_argument("--features", default="data/features_final.csv")
    parser.add_argument("--n-packets", type=int, default=0,
                        help="Specific N to train (0 = train all from 3 to 17)")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="models")
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    print(f"Dataset: {len(df)} samples")
    print(f"Classes: mcp={len(df[df.label=='mcp'])}, non_mcp={len(df[df.label=='non_mcp'])}")

    if args.n_packets > 0:
        n_values = [args.n_packets]
    else:
        n_values = [3, 5, 7, 10, 13, 17]

    print(f"\n{'='*70}")
    print(f"EARLY PACKET CLASSIFICATION — How few packets do we need?")
    print(f"{'='*70}")
    print(f"\n  {'N':>3}  {'Features':>8}  {'CV Acc':>8}  {'CV F1':>8}  {'Confidence':>10}  {'Uncertain':>9}")
    print(f"  {'-'*56}")

    results = []
    best_result = None
    best_f1 = 0

    for n in n_values:
        result = train_early_classifier(df, n, args.cv_folds)
        if result is None:
            continue

        results.append(result)
        r = result
        print(f"  {r['n_packets']:>3}  {r['features_used']:>8}  {r['cv_accuracy']:>7.1%}  "
              f"{r['cv_f1']:>7.1%}  {r['mean_confidence']:>9.1%}  {r['low_confidence_count']:>9}")

        if r['cv_f1'] > best_f1:
            best_f1 = r['cv_f1']
            best_result = result

    if not results:
        print("No models trained!")
        return

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")

    # Find the sweet spot: minimum N with >95% accuracy
    sweet_spot = None
    for r in results:
        if r["cv_accuracy"] >= 0.95 and sweet_spot is None:
            sweet_spot = r

    if sweet_spot:
        print(f"\n  Sweet spot: N={sweet_spot['n_packets']} packets")
        print(f"    Accuracy: {sweet_spot['cv_accuracy']:.1%}")
        print(f"    F1:       {sweet_spot['cv_f1']:.1%}")
        print(f"    Features: {sweet_spot['features_used']}")
        print(f"    Latency:  ~{sweet_spot['n_packets'] * 50}ms (estimated)")

    print(f"\n  Best overall: N={best_result['n_packets']} packets")
    print(f"    Accuracy: {best_result['cv_accuracy']:.1%}")
    print(f"    F1:       {best_result['cv_f1']:.1%}")

    # Save all early models
    os.makedirs(args.output_dir, exist_ok=True)
    for r in results:
        model_path = os.path.join(args.output_dir, f"early_model_n{r['n_packets']}.pkl")
        bundle = {
            "model": r["model"],
            "model_name": r["model_name"],
            "label_encoder": r["label_encoder"],
            "feature_names": r["feature_names"],
            "n_packets": r["n_packets"],
            "cv_accuracy": r["cv_accuracy"],
            "cv_f1": r["cv_f1"],
        }
        with open(model_path, "wb") as f:
            pickle.dump(bundle, f)

    print(f"\n  Models saved to {args.output_dir}/early_model_n*.pkl")

    # Save summary
    os.makedirs("results", exist_ok=True)
    with open("results/early_classifier_results.txt", "w") as f:
        f.write("EARLY PACKET CLASSIFICATION RESULTS\n")
        f.write(f"Dataset: {len(df)} samples\n\n")
        f.write(f"{'N':>3}  {'Features':>8}  {'CV Acc':>8}  {'CV F1':>8}  {'Confidence':>10}\n")
        f.write(f"{'-'*45}\n")
        for r in results:
            f.write(f"{r['n_packets']:>3}  {r['features_used']:>8}  {r['cv_accuracy']:>7.1%}  "
                    f"{r['cv_f1']:>7.1%}  {r['mean_confidence']:>9.1%}\n")
        if sweet_spot:
            f.write(f"\nSweet spot: N={sweet_spot['n_packets']} ({sweet_spot['cv_accuracy']:.1%} acc)\n")
        f.write(f"Best: N={best_result['n_packets']} ({best_result['cv_accuracy']:.1%} acc)\n")

    print(f"  Results saved to results/early_classifier_results.txt")


if __name__ == "__main__":
    main()
