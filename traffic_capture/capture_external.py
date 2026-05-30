"""
External Traffic Capture -- captures real-world HTTPS traffic on the main
network interface while the external HTTPS generator runs.

This produces non_mcp pcap files from real TLS traffic to public APIs,
giving the classifier realistic "hard negatives" to train on.

Usage:
    python -m traffic_capture.capture_external --requests 30 --duration 60 --output-dir data/pcap_external
"""

import argparse
import os
import platform
import subprocess
import sys
import threading
import time

from scapy.all import AsyncSniffer, TCP, wrpcap, conf, get_if_list, get_if_addr


def _find_main_interface() -> str:
    """Find the main (non-loopback) network interface."""
    system = platform.system()
    ifaces = get_if_list()

    if system == "Windows":
        # On Windows, find interface with a real IP (not loopback)
        for iface in ifaces:
            if "loopback" in iface.lower() or "NPF_Loopback" in iface:
                continue
            try:
                addr = get_if_addr(iface)
                if addr and addr != "0.0.0.0" and addr != "127.0.0.1":
                    return iface
            except Exception:
                continue
        # Fallback: try conf.iface
        return str(conf.iface)
    else:
        # Linux/macOS
        for candidate in ["eth0", "wlan0", "en0", "enp0s3"]:
            if candidate in ifaces:
                return candidate
        return str(conf.iface)


def capture_external_traffic(
    num_requests: int = 30,
    duration: int = 60,
    output_dir: str = "data/pcap_external",
) -> str:
    """
    Run external HTTPS generator + packet capture simultaneously.
    Saves the captured traffic as a non_mcp pcap file.
    """
    os.makedirs(output_dir, exist_ok=True)
    py = sys.executable
    interface = _find_main_interface()

    print(f"[External Capture] Interface: {interface}")
    print(f"[External Capture] Requests: {num_requests}")
    print(f"[External Capture] Duration: {duration}s")

    # Start packet capture (HTTPS = port 443)
    bpf = "tcp port 443"
    print(f"[External Capture] BPF filter: {bpf}")

    sniffer = AsyncSniffer(
        iface=interface,
        filter=bpf,
        store=True,
    )
    sniffer.start()

    # Longer delay for sniffer to fully initialize (round 1 failed with 1s)
    time.sleep(3)

    # Pre-warm: make one request to trigger DNS + TLS setup before real capture
    import requests as _req
    try:
        _req.get("https://jsonplaceholder.typicode.com/posts/1", timeout=10)
        print("[External Capture] Pre-warm request succeeded")
    except Exception:
        print("[External Capture] Pre-warm request failed (continuing)")
    time.sleep(1)

    # Run external HTTPS generator as subprocess
    gen_proc = subprocess.Popen(
        [py, "-m", "non_mcp_traffic.external_https", "--requests", str(num_requests)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    print(f"[External Capture] Generator started (PID {gen_proc.pid})")

    # Wait for generator to finish (or timeout)
    try:
        stdout, stderr = gen_proc.communicate(timeout=duration)
        if stdout:
            print(stdout.decode(errors="replace"))
    except subprocess.TimeoutExpired:
        gen_proc.kill()
        print("[External Capture] Generator timed out")

    # Extra wait for trailing packets
    time.sleep(3)

    # Stop capture
    results = sniffer.stop()
    if results is None:
        results = []

    print(f"[External Capture] Total packets: {len(results)}")

    if not results:
        print("[External Capture] WARNING: No packets captured")
        return ""

    # Save as non_mcp pcap
    ts = int(time.time())
    pcap_path = os.path.join(output_dir, f"non_mcp_{ts}.pcap")
    wrpcap(pcap_path, results)
    print(f"[External Capture] Saved: {pcap_path} ({len(results)} packets)")

    return pcap_path


def main():
    parser = argparse.ArgumentParser(description="External HTTPS Capture")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--output-dir", default="data/pcap_external")
    args = parser.parse_args()

    capture_external_traffic(args.requests, args.duration, args.output_dir)


if __name__ == "__main__":
    main()
