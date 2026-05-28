"""
Packet Capture — Scapy-based packet sniffer that captures network
traffic, splits it by port into MCP and non-MCP categories, and saves
labelled pcap files.

Usage:
    python -m traffic_capture.capture \
        --interface lo --mcp-ports 8000 --non-mcp-ports 5000,5001,5002 \
        --duration 60 --output-dir data/pcap
"""

import argparse
import os
import platform
import sys
import time

from scapy.all import (
    AsyncSniffer,
    TCP,
    UDP,
    wrpcap,
    get_if_list,
)


def _default_loopback():
    """Auto-detect the loopback interface."""
    system = platform.system()
    if system == "Darwin":
        return "lo0"
    elif system == "Linux":
        return "lo"
    else:  # Windows
        ifaces = get_if_list()
        for iface in ifaces:
            if "loopback" in iface.lower() or "NPF_Loopback" in iface:
                return iface
        # Fallback
        return "\\Device\\NPF_Loopback"


def capture_traffic(
    interface: str,
    mcp_ports: list[int],
    non_mcp_ports: list[int],
    duration: int,
    output_dir: str,
    label: str = "",
) -> tuple[str, str]:
    """
    Capture packets on the given interface for `duration` seconds.
    Split by port into MCP and non-MCP pcap files.

    Returns (mcp_pcap_path, non_mcp_pcap_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    all_ports = mcp_ports + non_mcp_ports
    bpf = " or ".join(f"port {p}" for p in all_ports)
    bpf = f"tcp and ({bpf})"

    print(f"[Capture] Interface: {interface}")
    print(f"[Capture] BPF filter: {bpf}")
    print(f"[Capture] Duration: {duration}s")

    sniffer = AsyncSniffer(
        iface=interface,
        filter=bpf,
        store=True,
    )

    sniffer.start()
    time.sleep(duration)
    results = sniffer.stop()

    if results is None:
        results = []

    print(f"[Capture] Total packets captured: {len(results)}")

    # Split packets by port
    mcp_pkts = []
    non_mcp_pkts = []

    mcp_set = set(mcp_ports)
    non_mcp_set = set(non_mcp_ports)

    for pkt in results:
        if pkt.haslayer(TCP):
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif pkt.haslayer(UDP):
            sport = pkt[UDP].sport
            dport = pkt[UDP].dport
        else:
            continue

        if sport in mcp_set or dport in mcp_set:
            mcp_pkts.append(pkt)
        elif sport in non_mcp_set or dport in non_mcp_set:
            non_mcp_pkts.append(pkt)

    # Save pcap files with timestamp
    ts = int(time.time())
    mcp_path = os.path.join(output_dir, f"mcp_{ts}.pcap")
    non_mcp_path = os.path.join(output_dir, f"non_mcp_{ts}.pcap")

    if mcp_pkts:
        wrpcap(mcp_path, mcp_pkts)
        print(f"[Capture] MCP packets: {len(mcp_pkts)} -> {mcp_path}")
    else:
        print("[Capture] WARNING: No MCP packets captured")
        mcp_path = ""

    if non_mcp_pkts:
        wrpcap(non_mcp_path, non_mcp_pkts)
        print(f"[Capture] Non-MCP packets: {len(non_mcp_pkts)} -> {non_mcp_path}")
    else:
        print("[Capture] WARNING: No non-MCP packets captured")
        non_mcp_path = ""

    return mcp_path, non_mcp_path


def main():
    parser = argparse.ArgumentParser(description="Traffic Capture")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--mcp-ports", default="8000", help="Comma-separated MCP ports")
    parser.add_argument("--non-mcp-ports", default="5000,5001,5002",
                        help="Comma-separated non-MCP ports")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--output-dir", default="data/pcap")
    args = parser.parse_args()

    interface = args.interface or _default_loopback()
    mcp_ports = [int(p) for p in args.mcp_ports.split(",")]
    non_mcp_ports = [int(p) for p in args.non_mcp_ports.split(",")]

    capture_traffic(interface, mcp_ports, non_mcp_ports, args.duration, args.output_dir)


if __name__ == "__main__":
    main()
