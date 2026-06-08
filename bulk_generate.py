"""
Bulk Traffic Generator — high-throughput flow generation for building
large datasets (10k-20k+ flows).

Key optimization: Servers start ONCE, clients cycle in fast batches.
Each batch spawns new MCP sessions (= new TCP flows) and new HTTP
requests (= new TCP flows). No server restart overhead.

Usage:
    # Generate ~10k flows (~15 minutes)
    python bulk_generate.py --batches 80 --batch-duration 8

    # Generate ~20k flows (~30 minutes)
    python bulk_generate.py --batches 160 --batch-duration 8

    # Quick test (500 flows, ~2 minutes)
    python bulk_generate.py --batches 5 --batch-duration 8
"""

import argparse
import os
import random
import subprocess
import sys
import time

from scapy.all import AsyncSniffer, TCP as ScapyTCP, wrpcap, get_if_list


SERVER_TYPES = ["github", "filesystem", "fetch", "memory", "database"]


def _py():
    return sys.executable


def _run_bg(cmd, label=""):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if label:
        print(f"  [{label}] PID {proc.pid}")
    return proc


def _kill(proc):
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Bulk Traffic Generator")
    parser.add_argument("--batches", type=int, default=80,
                        help="Number of client batches to run (each ~60-120 flows)")
    parser.add_argument("--batch-duration", type=int, default=8,
                        help="Seconds per client batch")
    parser.add_argument("--mcp-sessions", type=int, default=4,
                        help="MCP client sessions per server per batch")
    parser.add_argument("--mcp-requests", type=int, default=3,
                        help="Requests per MCP session (lower=more flows)")
    parser.add_argument("--http-requests", type=int, default=20,
                        help="HTTP requests per batch (each=1 flow)")
    parser.add_argument("--adversarial-requests", type=int, default=10,
                        help="Adversarial requests per batch")
    parser.add_argument("--pcap-dir", default="data/pcap_bulk")
    parser.add_argument("--save-interval", type=int, default=20,
                        help="Save pcap every N batches")
    args = parser.parse_args()

    py = _py()
    os.makedirs(args.pcap_dir, exist_ok=True)

    # Randomized ports
    base = random.randint(7000, 8500)
    mcp_ports = {st: base + i for i, st in enumerate(SERVER_TYPES)}
    original_mcp_port = base + 6
    http_port = base + 10
    ws_port = base + 11
    tcp_port = base + 12
    all_mcp_ports = set(mcp_ports.values()) | {original_mcp_port}
    all_ports = list(all_mcp_ports) + [http_port, ws_port, tcp_port]

    est_flows = args.batches * (
        len(SERVER_TYPES) * args.mcp_sessions * 2 +  # MCP: ~2 flows per session
        args.mcp_sessions * 2 +                       # original MCP
        args.http_requests +                           # HTTP (1 flow per request)
        4 +                                            # WS + TCP
        args.adversarial_requests                      # adversarial
    )

    print("=" * 65)
    print("BULK TRAFFIC GENERATOR")
    print("=" * 65)
    print(f"  Batches:        {args.batches}")
    print(f"  Batch duration: {args.batch_duration}s")
    print(f"  MCP sessions:   {args.mcp_sessions} per server per batch")
    print(f"  MCP requests:   {args.mcp_requests} per session")
    print(f"  HTTP requests:  {args.http_requests} per batch")
    print(f"  Adversarial:    {args.adversarial_requests} per batch")
    print(f"  Estimated flows: ~{est_flows}")
    print(f"  Estimated time:  ~{args.batches * (args.batch_duration + 3) // 60} minutes")
    print(f"  Ports: MCP={list(mcp_ports.values())}+{original_mcp_port}")
    print(f"         HTTP={http_port} WS={ws_port} TCP={tcp_port}")
    print("=" * 65)

    servers = []

    try:
        # ============================================================
        # PHASE 1: Start all servers ONCE
        # ============================================================
        print("\n[Phase 1] Starting servers...")

        for stype, port in mcp_ports.items():
            cmd = [py, "-m", "mcp_server.diverse_servers",
                   "--type", stype, "--port", str(port), "--tls"]
            servers.append(_run_bg(cmd, f"MCP-{stype}:{port}"))

        cmd = [py, "-m", "mcp_server.server",
               "--port", str(original_mcp_port), "--tls"]
        servers.append(_run_bg(cmd, f"MCP-original:{original_mcp_port}"))

        cmd = [py, "-m", "non_mcp_traffic.server",
               "--http-port", str(http_port), "--ws-port", str(ws_port)]
        servers.append(_run_bg(cmd, f"NonMCP:{http_port},{ws_port}"))

        # TCP echo server
        cmd = [py, "-c",
               f"import sys; sys.path.insert(0, '.'); "
               f"from traffic_capture.orchestrator import _run_tcp_echo_server; "
               f"_run_tcp_echo_server('0.0.0.0', {tcp_port})"]
        servers.append(_run_bg(cmd, f"TCP-Echo:{tcp_port}"))

        print("  Waiting for servers to initialize...")
        time.sleep(8)

        # ============================================================
        # PHASE 2: Start capture
        # ============================================================
        iface = "\\Device\\NPF_Loopback"
        for i in get_if_list():
            if "loopback" in i.lower() or "NPF_Loopback" in i:
                iface = i
                break

        port_filter = " or ".join([f"port {p}" for p in all_ports])
        bpf = f"tcp and ({port_filter})"
        print(f"\n[Phase 2] Starting capture on {iface}")

        sniffer = AsyncSniffer(iface=iface, filter=bpf, store=True)
        sniffer.start()
        time.sleep(2)

        # ============================================================
        # PHASE 3: Cycle client batches
        # ============================================================
        print(f"\n[Phase 3] Running {args.batches} client batches...")
        total_start = time.time()
        total_mcp = 0
        total_non_mcp = 0
        save_count = 0

        for batch in range(1, args.batches + 1):
            batch_clients = []

            # --- MCP clients ---
            for stype, port in mcp_ports.items():
                cmd = [py, "-m", "mcp_client.diverse_client",
                       "--url", f"https://localhost:{port}/sse",
                       "--type", stype,
                       "--requests", str(args.mcp_requests),
                       "--sessions", str(args.mcp_sessions)]
                batch_clients.append(_run_bg(cmd))

            # Original MCP client
            cmd = [py, "-m", "mcp_client.client",
                   "--url", f"https://localhost:{original_mcp_port}/sse",
                   "--sessions", str(args.mcp_sessions),
                   "--requests", str(args.mcp_requests)]
            batch_clients.append(_run_bg(cmd))

            # --- Non-MCP HTTP ---
            cmd = [py, "-m", "non_mcp_traffic.http_traffic",
                   "--url", f"https://localhost:{http_port}",
                   "--requests", str(args.http_requests),
                   "--cert", "certs/server.crt"]
            batch_clients.append(_run_bg(cmd))

            # --- Non-MCP WebSocket ---
            cmd = [py, "-m", "non_mcp_traffic.websocket_traffic",
                   "--url", f"ws://localhost:{ws_port}",
                   "--sessions", "2", "--messages", str(max(5, args.http_requests // 3))]
            batch_clients.append(_run_bg(cmd))

            # --- Non-MCP TCP ---
            cmd = [py, "-m", "non_mcp_traffic.tcp_traffic",
                   "--host", "localhost", "--port", str(tcp_port),
                   "--connections", "2", "--messages", str(max(5, args.http_requests // 3))]
            batch_clients.append(_run_bg(cmd))

            # --- Adversarial ---
            if args.adversarial_requests > 0:
                cmd = [py, "-m", "non_mcp_traffic.adversarial",
                       "--url", f"https://localhost:{http_port}",
                       "--requests", str(args.adversarial_requests)]
                batch_clients.append(_run_bg(cmd))

            # Wait for batch
            time.sleep(args.batch_duration)

            # Kill all clients
            for proc in batch_clients:
                _kill(proc)

            # Brief pause for packets to settle
            time.sleep(1)

            # Progress
            elapsed = time.time() - total_start
            eta = (elapsed / batch) * (args.batches - batch)
            print(f"  Batch {batch}/{args.batches} done "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

            # Periodic save
            if batch % args.save_interval == 0:
                save_count += 1
                try:
                    interim = sniffer.results if hasattr(sniffer, 'results') else []
                    if interim:
                        mcp_pkts = []
                        non_mcp_pkts = []
                        for pkt in interim:
                            if pkt.haslayer(ScapyTCP):
                                sp = pkt[ScapyTCP].sport
                                dp = pkt[ScapyTCP].dport
                                if sp in all_mcp_ports or dp in all_mcp_ports:
                                    mcp_pkts.append(pkt)
                                else:
                                    non_mcp_pkts.append(pkt)
                        print(f"    [Checkpoint] {len(interim)} pkts "
                              f"(MCP:{len(mcp_pkts)}, non-MCP:{len(non_mcp_pkts)})")
                except Exception:
                    pass

        # ============================================================
        # PHASE 4: Stop capture and save
        # ============================================================
        print(f"\n[Phase 4] Stopping capture...")
        time.sleep(3)
        results = sniffer.stop()
        if results is None:
            results = []

        print(f"  Total packets captured: {len(results)}")

        if results:
            mcp_pkts = []
            non_mcp_pkts = []
            for pkt in results:
                if pkt.haslayer(ScapyTCP):
                    sp = pkt[ScapyTCP].sport
                    dp = pkt[ScapyTCP].dport
                    if sp in all_mcp_ports or dp in all_mcp_ports:
                        mcp_pkts.append(pkt)
                    else:
                        non_mcp_pkts.append(pkt)

            ts = int(time.time())

            # Save in chunks to avoid memory issues
            chunk_size = 50000
            if mcp_pkts:
                for i in range(0, len(mcp_pkts), chunk_size):
                    chunk = mcp_pkts[i:i+chunk_size]
                    suffix = f"_{i//chunk_size}" if len(mcp_pkts) > chunk_size else ""
                    path = os.path.join(args.pcap_dir, f"mcp_{ts}{suffix}.pcap")
                    wrpcap(path, chunk)
                print(f"  MCP packets: {len(mcp_pkts)} saved")

            if non_mcp_pkts:
                for i in range(0, len(non_mcp_pkts), chunk_size):
                    chunk = non_mcp_pkts[i:i+chunk_size]
                    suffix = f"_{i//chunk_size}" if len(non_mcp_pkts) > chunk_size else ""
                    path = os.path.join(args.pcap_dir, f"non_mcp_{ts}{suffix}.pcap")
                    wrpcap(path, chunk)
                print(f"  Non-MCP packets: {len(non_mcp_pkts)} saved")

    finally:
        # ============================================================
        # PHASE 5: Shutdown servers
        # ============================================================
        print("\n[Phase 5] Stopping servers...")
        for proc in servers:
            _kill(proc)

    # ============================================================
    # PHASE 6: Feature extraction
    # ============================================================
    print(f"\n[Phase 6] Extracting features...")
    output_csv = "data/features_bulk.csv"
    cmd = [py, "-m", "feature_extraction.extractor",
           "--pcap-dir", args.pcap_dir, "--output", output_csv]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-8:]:
            print(f"  {line}")
    if result.returncode != 0 and result.stderr:
        print(f"  ERROR: {result.stderr[-500:]}")

    # Combine with existing data
    import pandas as pd
    existing = "data/features_final_v2.csv"
    if os.path.exists(existing) and os.path.exists(output_csv):
        df_old = pd.read_csv(existing)
        df_new = pd.read_csv(output_csv)
        common_cols = sorted(set(df_old.columns) & set(df_new.columns))
        df_combined = pd.concat([df_old[common_cols], df_new[common_cols]], ignore_index=True)
        combined_path = "data/features_final_v3.csv"
        df_combined.to_csv(combined_path, index=False)
        mcp_count = len(df_combined[df_combined.label == "mcp"])
        non_mcp_count = len(df_combined[df_combined.label == "non_mcp"])
        print(f"\n  Combined dataset: {len(df_combined)} flows "
              f"(MCP={mcp_count}, non_mcp={non_mcp_count})")
        print(f"  Saved to: {combined_path}")
    else:
        combined_path = output_csv

    # ============================================================
    # PHASE 7: Train models
    # ============================================================
    print(f"\n[Phase 7] Training full model...")
    cmd = [py, "-m", "model.train",
           "--features", combined_path, "--output", "models", "--no-ablation"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-15:]:
            print(f"  {line}")

    print(f"\n[Phase 7b] Training early classifiers...")
    cmd = [py, "model/early_classifier.py", "--features", combined_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-20:]:
            print(f"  {line}")

    total_time = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"BULK GENERATION COMPLETE")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Dataset: {combined_path}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
