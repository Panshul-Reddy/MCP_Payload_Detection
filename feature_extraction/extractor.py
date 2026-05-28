"""
Feature Extraction — reads labelled pcap files, groups packets into
bidirectional flows (5-tuple), and extracts a rich set of metadata-only
network features (no payload inspection).

Outputs a CSV with one row per flow and 35+ numeric features.

Usage:
    python -m feature_extraction.extractor --pcap-dir data/pcap --output data/features.csv
"""

import argparse
import math
import os
import re

import numpy as np
import pandas as pd
from scapy.all import TCP, UDP, rdpcap


# Constants
BURST_GAP_THRESHOLD = 0.5  # seconds between bursts
IDLE_GAP_THRESHOLD = 1.0   # seconds to count as idle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flow_key(pkt) -> tuple | None:
    """Compute normalized bidirectional flow key (5-tuple)."""
    if pkt.haslayer(TCP):
        proto = 6
        layer = pkt[TCP]
    elif pkt.haslayer(UDP):
        proto = 17
        layer = pkt[UDP]
    else:
        return None

    try:
        src_ip = pkt["IP"].src
        dst_ip = pkt["IP"].dst
    except Exception:
        return None

    src_port = layer.sport
    dst_port = layer.dport

    # Normalize: sort endpoints so both directions map to the same flow
    ep1 = (src_ip, src_port)
    ep2 = (dst_ip, dst_port)
    if ep1 > ep2:
        ep1, ep2 = ep2, ep1
    return (ep1[0], ep2[0], ep1[1], ep2[1], proto)


def _pkt_size(pkt) -> int:
    """Total packet size in bytes."""
    return len(pkt)


def _payload_size(pkt) -> int:
    """TCP/UDP payload size."""
    if pkt.haslayer(TCP):
        return len(bytes(pkt[TCP].payload))
    elif pkt.haslayer(UDP):
        return len(bytes(pkt[UDP].payload))
    return 0


def _tcp_flags(pkt) -> dict:
    """Extract TCP flag counts."""
    flags = {"SYN": 0, "ACK": 0, "PSH": 0, "FIN": 0, "RST": 0}
    if pkt.haslayer(TCP):
        f = pkt[TCP].flags
        if f & 0x02: flags["SYN"] = 1
        if f & 0x10: flags["ACK"] = 1
        if f & 0x08: flags["PSH"] = 1
        if f & 0x01: flags["FIN"] = 1
        if f & 0x04: flags["RST"] = 1
    return flags


def _is_forward(pkt, fwd_src: str, fwd_sport: int) -> bool:
    """Check if packet is in the forward direction."""
    try:
        return pkt["IP"].src == fwd_src and pkt[TCP].sport == fwd_sport
    except Exception:
        return True


def _safe_stats(arr) -> dict:
    """Compute mean/std/min/max/median, handle empty arrays."""
    if len(arr) == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def _shannon_entropy(values, n_bins: int = 8) -> float:
    """Shannon entropy of value distribution (discretized into bins)."""
    if len(values) < 2:
        return 0.0
    values = np.array(values, dtype=float)
    vmin, vmax = values.min(), values.max()
    if vmin == vmax:
        return 0.0
    # Discretize into n_bins equal-width bins
    bins = np.linspace(vmin, vmax + 1e-10, n_bins + 1)
    counts, _ = np.histogram(values, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ---------------------------------------------------------------------------
# Per-flow feature computation
# ---------------------------------------------------------------------------

def _compute_flow_features(packets: list, label: str, flow_key: tuple) -> dict:
    """Extract all features for one flow."""
    src_ip, dst_ip, src_port, dst_port, proto = flow_key

    # Determine forward direction (first packet's src is "forward")
    first_pkt = packets[0]
    try:
        fwd_src = first_pkt["IP"].src
        fwd_sport = first_pkt[TCP].sport if first_pkt.haslayer(TCP) else first_pkt[UDP].sport
    except Exception:
        fwd_src = src_ip
        fwd_sport = src_port

    # Timestamps
    timestamps = [float(pkt.time) for pkt in packets]
    flow_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0

    # Packet sizes
    all_sizes = [_pkt_size(pkt) for pkt in packets]
    payload_sizes = [_payload_size(pkt) for pkt in packets]

    # Direction split
    fwd_sizes = []
    bwd_sizes = []
    fwd_payloads = []
    bwd_payloads = []
    for pkt in packets:
        if _is_forward(pkt, fwd_src, fwd_sport):
            fwd_sizes.append(_pkt_size(pkt))
            fwd_payloads.append(_payload_size(pkt))
        else:
            bwd_sizes.append(_pkt_size(pkt))
            bwd_payloads.append(_payload_size(pkt))

    # Inter-arrival times
    iats = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]

    # TCP flags
    flag_counts = {"SYN": 0, "ACK": 0, "PSH": 0, "FIN": 0, "RST": 0}
    for pkt in packets:
        for flag, val in _tcp_flags(pkt).items():
            flag_counts[flag] += val

    # Burst analysis
    bursts = []
    current_burst = 1
    for iat in iats:
        if iat < BURST_GAP_THRESHOLD:
            current_burst += 1
        else:
            bursts.append(current_burst)
            current_burst = 1
    bursts.append(current_burst)

    burst_stats = _safe_stats(bursts)

    # Idle time analysis
    idle_times = [iat for iat in iats if iat >= IDLE_GAP_THRESHOLD]
    idle_stats = _safe_stats(idle_times) if idle_times else {"mean": 0, "std": 0, "max": 0}

    # Small / large packet counts
    small_pkts = sum(1 for s in all_sizes if s < 100)
    large_pkts = sum(1 for s in all_sizes if s > 1000)

    # Flow asymmetry
    total_bytes = sum(all_sizes)
    fwd_bytes = sum(fwd_sizes)
    bwd_bytes = sum(bwd_sizes)
    asymmetry = fwd_bytes / max(total_bytes, 1)

    # Packet size stats
    pkt_stats = _safe_stats(all_sizes)
    iat_stats = _safe_stats(iats) if iats else _safe_stats([])

    # Entropy features
    pkt_size_entropy = _shannon_entropy(all_sizes)
    iat_entropy = _shannon_entropy(iats) if iats else 0.0
    payload_entropy = _shannon_entropy(payload_sizes)

    return {
        # Identifiers (dropped before training)
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": proto,
        # Flow-level features
        "flow_duration": round(flow_duration, 6),
        "total_packets": len(packets),
        "fwd_packets": len(fwd_sizes),
        "bwd_packets": len(bwd_sizes),
        "total_bytes": total_bytes,
        "fwd_bytes": fwd_bytes,
        "bwd_bytes": bwd_bytes,
        # Packet size statistics
        "pkt_size_mean": round(pkt_stats["mean"], 2),
        "pkt_size_std": round(pkt_stats["std"], 2),
        "pkt_size_min": pkt_stats["min"],
        "pkt_size_max": pkt_stats["max"],
        "pkt_size_median": pkt_stats["median"],
        # Inter-arrival time statistics
        "iat_mean": round(iat_stats["mean"], 6),
        "iat_std": round(iat_stats["std"], 6),
        "iat_min": round(iat_stats["min"], 6),
        "iat_max": round(iat_stats["max"], 6),
        # Direction asymmetry
        "flow_asymmetry": round(asymmetry, 4),
        # Burst features
        "burst_count": len(bursts),
        "burst_size_mean": round(burst_stats["mean"], 2),
        "burst_size_std": round(burst_stats["std"], 2),
        # TCP flags
        "flag_SYN": flag_counts["SYN"],
        "flag_ACK": flag_counts["ACK"],
        "flag_PSH": flag_counts["PSH"],
        "flag_FIN": flag_counts["FIN"],
        "flag_RST": flag_counts["RST"],
        # Packet size categories
        "small_packets": small_pkts,
        "large_packets": large_pkts,
        # Idle time
        "idle_time_mean": round(idle_stats.get("mean", 0), 6),
        "idle_time_std": round(idle_stats.get("std", 0), 6),
        "idle_time_max": round(idle_stats.get("max", 0), 6),
        # Entropy features
        "pkt_size_entropy": round(pkt_size_entropy, 4),
        "iat_entropy": round(iat_entropy, 4),
        "payload_size_entropy": round(payload_entropy, 4),
        # Label
        "label": label,
    }


# ---------------------------------------------------------------------------
# Pcap → DataFrame
# ---------------------------------------------------------------------------

def extract_features_from_pcap(pcap_path: str, label: str) -> pd.DataFrame:
    """Read one pcap file, group packets by flow, extract features."""
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"  [!] Cannot read {pcap_path}: {e}")
        return pd.DataFrame()

    # Group packets by flow
    flows = {}
    for pkt in packets:
        key = _flow_key(pkt)
        if key is None:
            continue
        if key not in flows:
            flows[key] = []
        flows[key].append(pkt)

    # Extract features per flow (skip very small flows)
    rows = []
    for key, pkts in flows.items():
        if len(pkts) < 3:  # Skip flows with < 3 packets
            continue
        try:
            row = _compute_flow_features(pkts, label, key)
            rows.append(row)
        except Exception as e:
            pass  # Skip problematic flows

    return pd.DataFrame(rows)


def extract_features_from_directory(
    pcap_dir: str,
    output_csv: str,
) -> pd.DataFrame:
    """Process all pcap files in a directory and output a single CSV."""
    all_dfs = []

    for filename in sorted(os.listdir(pcap_dir)):
        if not filename.endswith(".pcap"):
            continue

        filepath = os.path.join(pcap_dir, filename)

        # Determine label from filename
        if filename.startswith("mcp_"):
            label = "mcp"
        elif filename.startswith("non_mcp_"):
            label = "non_mcp"
        else:
            print(f"  [!] Skipping unknown file: {filename}")
            continue

        print(f"  Processing {filename} (label={label})...")
        df = extract_features_from_pcap(filepath, label)
        if not df.empty:
            all_dfs.append(df)
            print(f"    -> {len(df)} flows extracted")

    if not all_dfs:
        print("[!] No flows extracted from any pcap file")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Save CSV
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"\n[Feature Extraction] Total: {len(combined)} flows -> {output_csv}")
    print(f"  MCP flows:     {len(combined[combined['label'] == 'mcp'])}")
    print(f"  Non-MCP flows: {len(combined[combined['label'] == 'non_mcp'])}")

    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Feature Extractor")
    parser.add_argument("--pcap-dir", default="data/pcap")
    parser.add_argument("--output", default="data/features.csv")
    args = parser.parse_args()

    extract_features_from_directory(args.pcap_dir, args.output)


if __name__ == "__main__":
    main()
