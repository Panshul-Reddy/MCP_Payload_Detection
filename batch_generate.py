"""
Batch Dataset Generator — loops the orchestrator pipeline to accumulate
pcap files until a target row count is reached.

Randomizes parameters each iteration for traffic diversity.

Usage:
    python batch_generate.py --target-rows 2000 --tls
"""

import argparse
import os
import random
import subprocess
import sys


def _python_exe():
    return sys.executable


def _count_csv_rows(csv_path: str) -> int:
    """Count rows in a CSV file (excluding header)."""
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r") as f:
        return max(0, sum(1 for _ in f) - 1)


def main():
    parser = argparse.ArgumentParser(description="Batch Dataset Generator")
    parser.add_argument("--target-rows", type=int, default=2000)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--pcap-dir", default="data/pcap")
    parser.add_argument("--output", default="data/features.csv")
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--tls", action="store_true")
    parser.add_argument("--cert", default="certs/server.crt")
    parser.add_argument("--key", default="certs/server.key")
    args = parser.parse_args()

    py = _python_exe()
    os.makedirs(args.pcap_dir, exist_ok=True)

    for iteration in range(1, args.max_iterations + 1):
        current_rows = _count_csv_rows(args.output)
        if current_rows >= args.target_rows:
            print(f"\n[Batch] Target reached: {current_rows} >= {args.target_rows}")
            break

        # Randomize parameters for diversity
        duration = random.randint(max(30, args.duration - 30), args.duration + 30)
        requests = random.randint(max(20, args.requests - 20), args.requests + 30)
        mcp_sessions = random.randint(2, 8)
        ws_sessions = random.randint(2, 6)
        tcp_connections = random.randint(2, 6)

        print(f"\n{'='*60}")
        print(f"[Batch] Iteration {iteration}/{args.max_iterations}")
        print(f"  Current rows: {current_rows} / {args.target_rows}")
        print(f"  Params: duration={duration}s, requests={requests}, "
              f"mcp_sessions={mcp_sessions}, ws={ws_sessions}, tcp={tcp_connections}")
        print(f"{'='*60}")

        # Run orchestrator
        orch_cmd = [
            py, "-m", "traffic_capture.orchestrator",
            "--duration", str(duration),
            "--requests", str(requests),
            "--mcp-sessions", str(mcp_sessions),
            "--ws-sessions", str(ws_sessions),
            "--tcp-connections", str(tcp_connections),
            "--output-dir", args.pcap_dir,
        ]
        if args.tls:
            orch_cmd += ["--tls", "--cert", args.cert, "--key", args.key]

        try:
            subprocess.run(orch_cmd, check=True, timeout=duration + 120)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"[Batch] Orchestrator error: {e}")
            continue

        # Extract features
        extract_cmd = [
            py, "-m", "feature_extraction.extractor",
            "--pcap-dir", args.pcap_dir,
            "--output", args.output,
        ]
        try:
            subprocess.run(extract_cmd, check=True, timeout=120)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"[Batch] Feature extraction error: {e}")
            continue

        new_rows = _count_csv_rows(args.output)
        print(f"[Batch] Rows after extraction: {new_rows}")

    final_rows = _count_csv_rows(args.output)
    print(f"\n[Batch] Done. Final dataset: {final_rows} rows in {args.output}")


if __name__ == "__main__":
    main()
