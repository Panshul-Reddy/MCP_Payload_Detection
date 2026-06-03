"""
Full Dataset Generator -- runs multiple rounds of BOTH:
  1. Loopback pipeline (MCP + non-MCP local traffic with random ports)
  2. External HTTPS capture (real-world TLS to public APIs)

Then combines everything, extracts features, and trains the final model.

Usage:
    python full_generate.py --loopback-rounds 3 --external-rounds 2 --requests 30
"""

import argparse
import os
import subprocess
import sys
import time
import random


def _py():
    return sys.executable


def _run(cmd, label, timeout=180):
    """Run a command and return success status."""
    print(f"\n  [{label}] Running: {' '.join(cmd[:8])}...")
    try:
        result = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
        if result.stdout:
            # Print last few lines of output
            lines = result.stdout.strip().split("\n")
            for line in lines[-5:]:
                print(f"  [{label}] {line}")
        if result.returncode != 0 and result.stderr:
            print(f"  [{label}] STDERR: {result.stderr[-200:]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  [{label}] TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"  [{label}] ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Full Dataset Generator")
    parser.add_argument("--loopback-rounds", type=int, default=3,
                        help="Number of loopback pipeline runs (MCP + local non-MCP)")
    parser.add_argument("--external-rounds", type=int, default=2,
                        help="Number of external HTTPS capture runs")
    parser.add_argument("--requests", type=int, default=30,
                        help="Requests per generator per round")
    parser.add_argument("--duration", type=int, default=30,
                        help="Seconds per loopback capture")
    parser.add_argument("--pcap-dir", default="data/pcap_full",
                        help="Output directory for all pcaps")
    parser.add_argument("--output", default="data/features_full.csv",
                        help="Output features CSV")
    args = parser.parse_args()

    py = _py()
    pcap_dir = args.pcap_dir
    os.makedirs(pcap_dir, exist_ok=True)

    print("=" * 60)
    print("FULL DATASET GENERATION")
    print(f"  Loopback rounds:  {args.loopback_rounds}")
    print(f"  External rounds:  {args.external_rounds}")
    print(f"  Requests/round:   {args.requests}")
    print(f"  Output dir:       {pcap_dir}")
    print("=" * 60)

    # ---------------------------------------------------------------
    # Phase 1: Loopback pipeline runs (MCP + non-MCP local traffic)
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 1: LOOPBACK PIPELINE (MCP + local non-MCP)")
    print(f"{'='*60}")

    for i in range(1, args.loopback_rounds + 1):
        # Randomize parameters for diversity
        reqs = random.randint(max(15, args.requests - 10), args.requests + 15)
        dur = random.randint(max(20, args.duration - 10), args.duration + 10)
        mcp_sess = random.randint(2, 5)
        ws_sess = random.randint(2, 4)
        tcp_conn = random.randint(2, 5)

        print(f"\n--- Loopback Round {i}/{args.loopback_rounds} ---")
        print(f"  Params: reqs={reqs}, dur={dur}s, mcp_sess={mcp_sess}, ws={ws_sess}, tcp={tcp_conn}")

        cmd = [
            py, "-m", "traffic_capture.orchestrator",
            "--duration", str(dur),
            "--requests", str(reqs),
            "--mcp-sessions", str(mcp_sess),
            "--ws-sessions", str(ws_sess),
            "--tcp-connections", str(tcp_conn),
            "--tls",
            "--output-dir", pcap_dir,
        ]
        _run(cmd, f"Loopback {i}", timeout=dur + 120)

    # ---------------------------------------------------------------
    # Phase 2: External HTTPS captures (real-world TLS traffic)
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 2: EXTERNAL HTTPS (real-world TLS traffic)")
    print(f"{'='*60}")

    for i in range(1, args.external_rounds + 1):
        reqs = random.randint(max(15, args.requests - 5), args.requests + 10)
        print(f"\n--- External Round {i}/{args.external_rounds} ---")
        print(f"  Params: reqs={reqs}")

        cmd = [
            py, "-m", "traffic_capture.capture_external",
            "--requests", str(reqs),
            "--duration", str(120),
            "--output-dir", pcap_dir,
        ]
        _run(cmd, f"External {i}", timeout=180)

    # ---------------------------------------------------------------
    # Phase 3: Extract features from all pcaps
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 3: FEATURE EXTRACTION")
    print(f"{'='*60}")

    cmd = [
        py, "-m", "feature_extraction.extractor",
        "--pcap-dir", pcap_dir,
        "--output", args.output,
    ]
    _run(cmd, "Extraction", timeout=120)

    # ---------------------------------------------------------------
    # Phase 4: Train model
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("PHASE 4: MODEL TRAINING")
    print(f"{'='*60}")

    cmd = [
        py, "-m", "model.train",
        "--features", args.output,
        "--output", "models",
        "--cv-folds", "5",
        "--no-ablation",
    ]
    _run(cmd, "Training", timeout=120)

    # ---------------------------------------------------------------
    # Check class balance
    # ---------------------------------------------------------------
    import pandas as pd
    if os.path.exists(args.output):
        df = pd.read_csv(args.output)
        mcp_count = len(df[df.label == 'mcp'])
        non_mcp_count = len(df[df.label == 'non_mcp'])
        ratio = max(mcp_count, non_mcp_count) / max(min(mcp_count, non_mcp_count), 1)
        print(f"\n  Class Balance:")
        print(f"    MCP:     {mcp_count}")
        print(f"    Non-MCP: {non_mcp_count}")
        print(f"    Ratio:   1:{ratio:.1f}")
        if ratio > 3.0:
            print(f"    WARNING: Class imbalance ratio exceeds 3:1!")
            print(f"    Consider generating more MCP traffic.")
        else:
            print(f"    OK: Ratio within acceptable range")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")

    # Count pcap files
    pcaps = [f for f in os.listdir(pcap_dir) if f.endswith(".pcap")]
    mcp_pcaps = [f for f in pcaps if f.startswith("mcp_")]
    nonmcp_pcaps = [f for f in pcaps if f.startswith("non_mcp_")]
    print(f"  Pcap files:     {len(pcaps)} ({len(mcp_pcaps)} MCP, {len(nonmcp_pcaps)} non-MCP)")
    print(f"  Features CSV:   {args.output}")
    print(f"  Model:          models/best_model.pkl")


if __name__ == "__main__":
    main()
