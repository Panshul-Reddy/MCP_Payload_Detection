"""
External Traffic Capture v2 -- captures real-world HTTPS traffic on the main
network interface while the external HTTPS generator runs.

Fixed: Uses Scapy's conf.iface as default (proven to work) instead of
trying to auto-detect which often picks the wrong adapter on Windows.

Usage:
    python -m traffic_capture.capture_external --requests 30 --duration 60 --output-dir data/pcap_external
"""

import argparse
import os
import subprocess
import sys
import time

from scapy.all import AsyncSniffer, TCP, wrpcap, conf, get_if_list, get_if_addr


def _find_main_interface() -> str:
    """Find the main network interface for external traffic capture."""
    # Use Scapy's default interface (it picks the one with the default route)
    default = str(conf.iface)

    # Verify it has a real IP
    try:
        addr = get_if_addr(default)
        if addr and addr not in ("0.0.0.0", "127.0.0.1"):
            return default
    except Exception:
        pass

    # Fallback: find interface with a routable IP (192.168.x.x, 10.x.x.x, etc.)
    for iface in get_if_list():
        if "loopback" in iface.lower() or "NPF_Loopback" in iface:
            continue
        try:
            addr = get_if_addr(iface)
            if addr and addr.startswith(("192.168.", "10.", "172.")):
                return iface
        except Exception:
            continue

    return default


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

    # Show what we're capturing on
    try:
        addr = get_if_addr(interface)
    except Exception:
        addr = "unknown"

    print(f"[External Capture] Interface: {interface}")
    print(f"[External Capture] Interface IP: {addr}")
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

    # Wait for sniffer to fully initialize
    time.sleep(3)

    # Pre-warm: make one request to verify connectivity and warm DNS/TLS
    import requests as _req
    try:
        r = _req.get("https://jsonplaceholder.typicode.com/posts/1", timeout=10)
        print(f"[External Capture] Pre-warm: status={r.status_code}")
    except Exception as e:
        print(f"[External Capture] Pre-warm failed: {e}")
    time.sleep(2)

    # Check if sniffer caught the pre-warm packets
    # (This verifies the interface is correct)
    try:
        interim = sniffer.results if hasattr(sniffer, 'results') and sniffer.results else []
        print(f"[External Capture] Pre-warm packets captured: {len(interim)}")
        if len(interim) == 0:
            print("[External Capture] WARNING: No pre-warm packets - interface may be wrong")
    except Exception:
        pass

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
