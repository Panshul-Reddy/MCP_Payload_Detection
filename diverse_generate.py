"""
Diverse Traffic Orchestrator — runs multiple MCP server TYPES simultaneously
alongside non-MCP traffic to generate highly diverse training data.

Runs 5 different MCP server types (GitHub, Filesystem, Fetch, Memory, Database)
on separate random ports + non-MCP traffic, captures everything.

Usage:
    python diverse_generate.py --requests 15 --duration 30
    python diverse_generate.py --requests 20 --duration 40 --rounds 3
"""

import argparse
import os
import random
import subprocess
import sys
import time


SERVER_TYPES = ["github", "filesystem", "fetch", "memory", "database"]


def _py():
    return sys.executable


def _run_bg(cmd, label):
    """Start a background process."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    print(f"  [{label}] PID {proc.pid}: {' '.join(cmd[:6])}...")
    return proc


def _kill(proc, label):
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    print(f"  [{label}] Stopped")


def run_diverse_round(args, round_num, pcap_dir):
    """Run one round of diverse traffic generation."""
    py = _py()

    # Random ports for each MCP server type
    base = random.randint(7000, 9000)
    mcp_ports = {st: base + i for i, st in enumerate(SERVER_TYPES)}
    original_mcp_port = base + 6  # Original MCP server as reliable fallback
    non_mcp_http = base + 10
    non_mcp_ws = base + 11
    non_mcp_tcp = base + 12

    all_ports = list(mcp_ports.values()) + [original_mcp_port, non_mcp_http, non_mcp_ws, non_mcp_tcp]

    print(f"\n{'='*60}")
    print(f"ROUND {round_num} — Ports: MCP={list(mcp_ports.values())}, "
          f"HTTP={non_mcp_http}, WS={non_mcp_ws}, TCP={non_mcp_tcp}")
    print(f"{'='*60}")

    servers = []
    clients = []

    try:
        # --- Start diverse MCP servers ---
        for stype, port in mcp_ports.items():
            cmd = [py, "-m", "mcp_server.diverse_servers",
                   "--type", stype, "--port", str(port), "--tls"]
            servers.append((_run_bg(cmd, f"MCP-{stype}"), f"MCP-{stype}"))

        # --- Start original MCP server (reliable fallback) ---
        cmd = [py, "-m", "mcp_server.server",
               "--port", str(original_mcp_port), "--tls"]
        servers.append((_run_bg(cmd, "MCP-original"), "MCP-original"))

        # --- Start non-MCP server ---
        cmd = [py, "-m", "non_mcp_traffic.server",
               "--http-port", str(non_mcp_http),
               "--ws-port", str(non_mcp_ws)]
        servers.append((_run_bg(cmd, "NonMCP-Server"), "NonMCP-Server"))

        # --- Start TCP echo server ---
        cmd = [py, "-c",
               f"import sys; sys.path.insert(0, '.'); "
               f"from traffic_capture.orchestrator import _run_tcp_echo_server; "
               f"_run_tcp_echo_server('0.0.0.0', {non_mcp_tcp})"]
        servers.append((_run_bg(cmd, "TCP-Echo"), "TCP-Echo"))

        print(f"\n  Waiting for servers to start...")
        time.sleep(8)  # More time for TLS servers

        # --- Start packet capture ---
        from scapy.all import AsyncSniffer, TCP as ScapyTCP, wrpcap, get_if_list

        iface = "\\Device\\NPF_Loopback"
        for i in get_if_list():
            if "loopback" in i.lower() or "NPF_Loopback" in i:
                iface = i
                break

        port_filter = " or ".join([f"port {p}" for p in all_ports])
        bpf = f"tcp and ({port_filter})"
        print(f"  [Capture] Interface: {iface}")
        print(f"  [Capture] BPF: {bpf}")

        sniffer = AsyncSniffer(iface=iface, filter=bpf, store=True)
        sniffer.start()
        time.sleep(2)

        # --- Start traffic generators ---
        capture_duration = args.duration + 5

        # Original MCP client (reliable — always generates MCP traffic)
        cmd = [py, "-m", "mcp_client.client",
               "--url", f"https://localhost:{original_mcp_port}/sse",
               "--sessions", str(random.randint(2, 4)),
               "--requests", str(args.requests)]
        clients.append((_run_bg(cmd, "Client-original"), "Client-original"))

        # Diverse MCP clients for each server type
        for stype, port in mcp_ports.items():
            reqs = random.randint(max(8, args.requests - 5), args.requests + 5)
            sessions = random.randint(1, 2)
            cmd = [py, "-m", "mcp_client.diverse_client",
                   "--url", f"https://localhost:{port}/sse",
                   "--type", stype,
                   "--requests", str(reqs),
                   "--sessions", str(sessions)]
            clients.append((_run_bg(cmd, f"Client-{stype}"), f"Client-{stype}"))

        # Non-MCP HTTP traffic
        cmd = [py, "-m", "non_mcp_traffic.http_traffic",
               "--url", f"https://localhost:{non_mcp_http}",
               "--requests", str(args.requests),
               "--cert", "certs/server.crt"]
        clients.append((_run_bg(cmd, "HTTP-Traffic"), "HTTP-Traffic"))

        # Non-MCP WebSocket traffic
        cmd = [py, "-m", "non_mcp_traffic.websocket_traffic",
               "--url", f"ws://localhost:{non_mcp_ws}",
               "--sessions", "2", "--messages", str(args.requests)]
        clients.append((_run_bg(cmd, "WS-Traffic"), "WS-Traffic"))

        # Non-MCP TCP traffic
        cmd = [py, "-m", "non_mcp_traffic.tcp_traffic",
               "--host", "localhost", "--port", str(non_mcp_tcp),
               "--connections", "2", "--messages", str(args.requests)]
        clients.append((_run_bg(cmd, "TCP-Traffic"), "TCP-Traffic"))

        # Adversarial traffic (JSON-RPC, GraphQL, gRPC that mimics MCP)
        cmd = [py, "-m", "non_mcp_traffic.adversarial",
               "--url", f"https://localhost:{non_mcp_http}",
               "--requests", str(args.requests)]
        clients.append((_run_bg(cmd, "Adversarial"), "Adversarial"))

        # Wait for traffic
        print(f"\n  Generating traffic for {args.duration}s...")
        time.sleep(args.duration)

        # Kill clients
        for proc, label in clients:
            _kill(proc, label)

        # Wait for trailing packets
        time.sleep(3)

        # Stop capture
        results = sniffer.stop()
        if results is None:
            results = []
        print(f"\n  [Capture] Total packets: {len(results)}")

        if results:
            # Split into MCP and non-MCP by port
            mcp_pkts = []
            non_mcp_pkts = []
            mcp_port_set = set(mcp_ports.values()) | {original_mcp_port}

            for pkt in results:
                if pkt.haslayer(ScapyTCP):
                    sp = pkt[ScapyTCP].sport
                    dp = pkt[ScapyTCP].dport
                    if sp in mcp_port_set or dp in mcp_port_set:
                        mcp_pkts.append(pkt)
                    else:
                        non_mcp_pkts.append(pkt)

            ts = int(time.time())
            if mcp_pkts:
                mcp_path = os.path.join(pcap_dir, f"mcp_{ts}.pcap")
                wrpcap(mcp_path, mcp_pkts)
                print(f"  [Capture] MCP: {len(mcp_pkts)} pkts -> {mcp_path}")
            if non_mcp_pkts:
                non_path = os.path.join(pcap_dir, f"non_mcp_{ts}.pcap")
                wrpcap(non_path, non_mcp_pkts)
                print(f"  [Capture] Non-MCP: {len(non_mcp_pkts)} pkts -> {non_path}")

    finally:
        # Stop all servers
        for proc, label in servers:
            _kill(proc, label)


def main():
    parser = argparse.ArgumentParser(description="Diverse Traffic Generator")
    parser.add_argument("--requests", type=int, default=15,
                        help="Requests per generator per round")
    parser.add_argument("--duration", type=int, default=30,
                        help="Seconds per round")
    parser.add_argument("--rounds", type=int, default=1,
                        help="Number of generation rounds")
    parser.add_argument("--pcap-dir", default="data/pcap_diverse_v2",
                        help="Output directory for pcaps")
    parser.add_argument("--extract", action="store_true", default=True,
                        help="Extract features after generation")
    parser.add_argument("--train-early", action="store_true", default=True,
                        help="Train early classifiers after extraction")
    args = parser.parse_args()

    py = _py()
    os.makedirs(args.pcap_dir, exist_ok=True)

    print("=" * 60)
    print("DIVERSE MCP TRAFFIC GENERATION")
    print(f"  Server types: {', '.join(SERVER_TYPES)}")
    print(f"  Rounds:       {args.rounds}")
    print(f"  Requests:     ~{args.requests} per generator per round")
    print(f"  Duration:     {args.duration}s per round")
    print(f"  Output:       {args.pcap_dir}")
    print("=" * 60)

    for i in range(1, args.rounds + 1):
        run_diverse_round(args, i, args.pcap_dir)

    # --- Feature extraction ---
    if args.extract:
        print(f"\n{'='*60}")
        print("FEATURE EXTRACTION")
        print(f"{'='*60}")
        output_csv = "data/features_diverse_v2.csv"
        cmd = [py, "-m", "feature_extraction.extractor",
               "--pcap-dir", args.pcap_dir, "--output", output_csv]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-5:]:
                print(f"  {line}")

        # Combine with existing data
        import pandas as pd
        existing = "data/features_final.csv"
        if os.path.exists(existing) and os.path.exists(output_csv):
            df_old = pd.read_csv(existing)
            df_new = pd.read_csv(output_csv)
            # Align columns
            common_cols = list(set(df_old.columns) & set(df_new.columns))
            df_combined = pd.concat([df_old[common_cols], df_new[common_cols]], ignore_index=True)
            combined_path = "data/features_final_v2.csv"
            df_combined.to_csv(combined_path, index=False)
            mcp_count = len(df_combined[df_combined.label == 'mcp'])
            non_mcp = len(df_combined[df_combined.label == 'non_mcp'])
            print(f"\n  Combined dataset: {len(df_combined)} flows "
                  f"(MCP={mcp_count}, non_mcp={non_mcp})")
            print(f"  Saved to: {combined_path}")
        else:
            combined_path = output_csv

        # --- Train early classifiers ---
        if args.train_early:
            print(f"\n{'='*60}")
            print("TRAINING EARLY CLASSIFIERS")
            print(f"{'='*60}")
            csv_to_use = combined_path if os.path.exists(combined_path) else output_csv
            cmd = [py, "model/early_classifier.py", "--features", csv_to_use]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.stdout:
                print(result.stdout)
            if result.returncode != 0 and result.stderr:
                print(f"  STDERR: {result.stderr[-300:]}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
