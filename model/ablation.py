"""
Standalone Ablation Study -- systematically removes feature groups
from the Random Forest model and measures the impact on accuracy.

This proves which features are actually driving classification
performance vs. which are noise.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

DROP_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "label"]

# Load data
df = pd.read_csv('data/features_combined.csv')
le = LabelEncoder()
y = le.fit_transform(df["label"])
feature_cols = [c for c in df.columns if c not in DROP_COLS]
X = df[feature_cols].fillna(0)

print(f"Dataset: {len(df)} samples, {len(feature_cols)} features")
print(f"Classes: mcp={sum(y==0)}, non_mcp={sum(y==1)}")

# Define feature groups
feature_groups = {
    "Timing (IAT)":       [c for c in feature_cols if c.startswith("iat_")],
    "Packet Sizes":       [c for c in feature_cols if c.startswith("pkt_size_") and not c[-1].isdigit()],
    "TLS Record":         [c for c in feature_cols if c.startswith("tls_")],
    "First-10 Sizes":     [c for c in feature_cols if c.startswith("pkt_size_") and c[-1].isdigit()],
    "First-10 Dirs":      [c for c in feature_cols if c.startswith("pkt_dir_")],
    "Turn-Taking":        [c for c in feature_cols if any(kw in c for kw in ["turn", "direction"])],
    "Burst":              [c for c in feature_cols if "burst" in c],
    "TCP Flags":          [c for c in feature_cols if c.startswith("flag_")],
    "Directional Bytes":  [c for c in feature_cols if c in ["fwd_bytes", "bwd_bytes", "byte_asymmetry"]],
    "Payload Stats":      [c for c in feature_cols if c.startswith("payload") or c == "data_pkt_ratio"],
    "Flow Basics":        [c for c in feature_cols if c in ["flow_duration", "total_packets", "total_bytes"]],
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
params = dict(n_estimators=200, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=42)

# Baseline
clf_all = RandomForestClassifier(**params)
y_pred_all = cross_val_predict(clf_all, X, y, cv=skf)
baseline_f1 = f1_score(y, y_pred_all, average="weighted")
baseline_acc = accuracy_score(y, y_pred_all)

print(f"\n{'='*70}")
print(f"ABLATION STUDY (5-fold CV)")
print(f"{'='*70}")
print(f"\nBaseline (all {len(feature_cols)} features):  Acc={baseline_acc:.4f}  F1={baseline_f1:.4f}")
print(f"\n{'Feature Group':<25s} {'#Removed':<10s} {'Acc':<10s} {'F1':<10s} {'Delta F1':<10s} {'Impact'}")
print(f"{'-'*75}")

results = []
for group_name, group_cols in feature_groups.items():
    present = [c for c in group_cols if c in feature_cols]
    if not present:
        continue
    remaining = [c for c in feature_cols if c not in present]
    X_abl = X[remaining]
    clf_abl = RandomForestClassifier(**params)
    y_pred_abl = cross_val_predict(clf_abl, X_abl, y, cv=skf)
    abl_f1 = f1_score(y, y_pred_abl, average="weighted")
    abl_acc = accuracy_score(y, y_pred_abl)
    delta = abl_f1 - baseline_f1
    
    if delta < -0.02:
        impact = "*** CRITICAL"
    elif delta < -0.005:
        impact = "** Important"
    elif delta < 0:
        impact = "* Minor"
    else:
        impact = "Negligible"
    
    print(f"  {group_name:<23s} {len(present):<10d} {abl_acc:<10.4f} {abl_f1:<10.4f} {delta:+.4f}     {impact}")
    results.append((group_name, len(present), abl_acc, abl_f1, delta, impact))

# Feature importances from full model
print(f"\n{'='*70}")
print("TOP-20 INDIVIDUAL FEATURE IMPORTANCES")
print(f"{'='*70}")
clf_full = RandomForestClassifier(**params)
clf_full.fit(X, y)
importances = sorted(zip(feature_cols, clf_full.feature_importances_), key=lambda x: -x[1])
for i, (name, imp) in enumerate(importances[:20]):
    bar = "#" * int(imp * 150)
    print(f"  {i+1:2d}. {name:<30s} {imp:.4f} {bar}")

# Check for zero-importance features
zero_imp = [name for name, imp in importances if imp < 0.001]
if zero_imp:
    print(f"\n  Features with near-zero importance ({len(zero_imp)}):")
    for name in zero_imp:
        print(f"    - {name}")
