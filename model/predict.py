"""
Inference Script — loads the trained model and classifies network flows
from pcap files or pre-extracted feature CSVs.

This is the production-use script that makes the project usable beyond training.

Usage:
    # Classify flows from a pcap file
    python -m model.predict --pcap data/pcap_big/mcp_1234.pcap

    # Classify flows from a directory of pcaps
    python -m model.predict --pcap data/pcap_big/

    # Classify from a pre-extracted features CSV
    python -m model.predict --features data/features_final.csv

    # Use a specific model
    python -m model.predict --pcap capture.pcap --model models/best_model.pkl
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

# Ensure project root is in path for imports
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Also add cwd in case running from project root
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Columns that are identifiers, not features — must be dropped before prediction
DROP_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "label"]


def load_model(model_path: str):
    """Load a trained model from a pickle file."""
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    # Handle both direct model and dict bundle formats
    if isinstance(bundle, dict):
        model = bundle.get("model", bundle.get("clf", bundle.get("pipeline")))
        if model is None:
            raise ValueError(f"Cannot find model in bundle keys: {list(bundle.keys())}")
        return model, bundle
    else:
        return bundle, {"model": bundle}


def extract_features_from_pcap(pcap_path: str, label: str = "unknown") -> pd.DataFrame:
    """Extract features from a single pcap file using the project extractor."""
    # Import the extractor module
    try:
        from feature_extraction.extractor import (
            _flow_key, _compute_flow_features, rdpcap
        )
    except ImportError:
        print("[Error] Cannot import feature_extraction.extractor")
        print("  Make sure you are running from the project root directory")
        sys.exit(1)

    packets = rdpcap(pcap_path)

    # Group packets by flow
    flows = {}
    for pkt in packets:
        key = _flow_key(pkt)
        if key is None:
            continue
        if key not in flows:
            flows[key] = []
        flows[key].append(pkt)

    # Extract features per flow
    rows = []
    for key, pkts in flows.items():
        if len(pkts) < 3:
            continue
        try:
            row = _compute_flow_features(pkts, label, key)
            rows.append(row)
        except Exception:
            pass

    return pd.DataFrame(rows)


def predict_from_features(model, df: pd.DataFrame) -> pd.DataFrame:
    """Run prediction on a DataFrame of features."""
    # Keep identifier columns for output
    id_cols = {}
    for col in DROP_COLS:
        if col in df.columns:
            id_cols[col] = df[col].values

    # Prepare feature matrix
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

    # Predict
    predictions = model.predict(X)

    # Confidence scores
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        confidences = proba.max(axis=1)
    else:
        confidences = np.ones(len(predictions))

    # Build results
    results = pd.DataFrame({
        "flow_id": range(len(predictions)),
        "src_ip": id_cols.get("src_ip", ["?"] * len(predictions)),
        "dst_ip": id_cols.get("dst_ip", ["?"] * len(predictions)),
        "src_port": id_cols.get("src_port", [0] * len(predictions)),
        "dst_port": id_cols.get("dst_port", [0] * len(predictions)),
        "prediction": predictions,
        "confidence": np.round(confidences, 4),
    })

    # Map numeric predictions to labels if needed
    if results["prediction"].dtype in [np.int64, np.int32, int]:
        label_map = {0: "mcp", 1: "non_mcp"}
        results["prediction"] = results["prediction"].map(label_map)

    # Add true label if available
    if "label" in id_cols:
        results["true_label"] = id_cols["label"]

    return results


def main():
    parser = argparse.ArgumentParser(description="MCP Traffic Classifier - Inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pcap", help="Path to pcap file or directory of pcaps")
    group.add_argument("--features", help="Path to pre-extracted features CSV")
    parser.add_argument("--model", default="models/best_model.pkl",
                        help="Path to trained model (default: models/best_model.pkl)")
    parser.add_argument("--output", default=None,
                        help="Save predictions to CSV (optional)")
    args = parser.parse_args()

    # Load model
    print(f"[Predict] Loading model: {args.model}")
    model, bundle = load_model(args.model)
    print(f"[Predict] Model type: {type(model).__name__}")

    # Get features
    if args.features:
        print(f"[Predict] Loading features: {args.features}")
        df = pd.read_csv(args.features)
    elif args.pcap:
        if os.path.isdir(args.pcap):
            # Process all pcaps in directory
            dfs = []
            for fname in sorted(os.listdir(args.pcap)):
                if not fname.endswith(".pcap"):
                    continue
                fpath = os.path.join(args.pcap, fname)
                label = "mcp" if fname.startswith("mcp_") else "non_mcp" if fname.startswith("non_mcp_") else "unknown"
                print(f"  Extracting: {fname} (label={label})")
                sub_df = extract_features_from_pcap(fpath, label)
                if not sub_df.empty:
                    dfs.append(sub_df)
                    print(f"    -> {len(sub_df)} flows")
            if not dfs:
                print("[Predict] No flows extracted from any pcap!")
                return
            df = pd.concat(dfs, ignore_index=True)
        else:
            # Single pcap file
            print(f"[Predict] Extracting features from: {args.pcap}")
            df = extract_features_from_pcap(args.pcap)
            if df.empty:
                print("[Predict] No flows extracted!")
                return

    print(f"[Predict] Total flows: {len(df)}")

    # Run predictions
    results = predict_from_features(model, df)

    # Print results
    print(f"\n{'='*70}")
    print(f"CLASSIFICATION RESULTS")
    print(f"{'='*70}")
    print(f"\n  {'Flow':<6} {'Src IP':<18} {'Dst IP':<18} {'Ports':<14} {'Prediction':<12} {'Conf':>6}")
    print(f"  {'-'*74}")

    for _, row in results.iterrows():
        ports = f"{int(row['src_port'])}>{int(row['dst_port'])}"
        pred = row['prediction']
        conf = row['confidence']
        marker = " *" if conf < 0.8 else ""
        print(f"  {int(row['flow_id']):<6} {row['src_ip']:<18} {row['dst_ip']:<18} "
              f"{ports:<14} {pred:<12} {conf:>5.1%}{marker}")

    # Summary
    mcp_count = (results["prediction"] == "mcp").sum()
    non_mcp_count = (results["prediction"] == "non_mcp").sum()
    uncertain = (results["confidence"] < 0.8).sum()

    print(f"\n  Summary:")
    print(f"    MCP detected:     {mcp_count}")
    print(f"    Non-MCP detected: {non_mcp_count}")
    print(f"    Uncertain (<80%): {uncertain}")
    print(f"    Mean confidence:  {results['confidence'].mean():.1%}")

    # Accuracy if true labels available
    if "true_label" in results.columns:
        # Map true labels to match prediction format
        correct = (results["prediction"] == results["true_label"]).sum()
        total = len(results)
        print(f"    Accuracy:         {correct}/{total} ({correct/total:.1%})")

    # Save results
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\n[Predict] Results saved to: {args.output}")


if __name__ == "__main__":
    main()
