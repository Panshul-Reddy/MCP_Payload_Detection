"""
Feature Extraction v2 -- reads labelled pcap files, groups packets into
bidirectional flows (5-tuple), and extracts a rich set of metadata-only
network features including TLS-specific and temporal sequence features.

Improvements over v1:
  - TLS record type analysis (handshake, app data, alert counts/ratios)
  - First-N packet size sequence (captures protocol fingerprint)
  - First-N packet direction sequence
  - Direction change / turn-taking features
  - Inter-turn latency statistics
  - Removed zero-variance features (pkt_size_min, flag_SYN)
  - Removed redundant payload_size_entropy (r=0.87 with pkt_size_entropy)
  - Payload size stats separated from total packet size stats

Outputs a CSV with one row per flow and 55+ numeric features.

Usage:
    python -m feature_extraction.extractor --pcap-dir data/pcap --output data/features.csv
"""

import argparse
import os
import struct

import numpy as np
import pandas as pd
from scapy.all import TCP, UDP, Raw, rdpcap


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BURST_GAP_THRESHOLD = 0.5   # seconds between bursts
IDLE_GAP_THRESHOLD = 1.0    # seconds to count as idle
FIRST_N_PACKETS = 10        # number of packets for sequence features


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


def _payload_bytes(pkt) -> bytes:
    """Extract raw TCP/UDP payload bytes."""
    if pkt.haslayer(Raw):
        return bytes(pkt[Raw].load)
    if pkt.haslayer(TCP):
        return bytes(pkt[TCP].payload)
    if pkt.haslayer(UDP):
        return bytes(pkt[UDP].payload)
    return b""


def _payload_size(pkt) -> int:
    """TCP/UDP payload size."""
    return len(_payload_bytes(pkt))


def _tcp_flags(pkt) -> dict:
    """Extract TCP flag bits."""
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
    """Check if packet is in the forward direction (first-packet sender)."""
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
    bins = np.linspace(vmin, vmax + 1e-10, n_bins + 1)
    counts, _ = np.histogram(values, bins=bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ---------------------------------------------------------------------------
# TLS record parsing helpers
# ---------------------------------------------------------------------------

# TLS content types
TLS_CHANGE_CIPHER_SPEC = 20  # 0x14
TLS_ALERT = 21               # 0x15
TLS_HANDSHAKE = 22           # 0x16
TLS_APPLICATION_DATA = 23    # 0x17

TLS_CONTENT_TYPES = {
    TLS_CHANGE_CIPHER_SPEC: "change_cipher_spec",
    TLS_ALERT: "alert",
    TLS_HANDSHAKE: "handshake",
    TLS_APPLICATION_DATA: "application_data",
}


def _parse_tls_records(payload: bytes) -> list[tuple[int, int]]:
    """
    Parse TLS record headers from a TCP payload.
    Returns list of (content_type, record_length) tuples.

    TLS record format: [type:1B][version:2B][length:2B][data:length]
    """
    records = []
    offset = 0
    while offset + 5 <= len(payload):
        content_type = payload[offset]
        # Validate: content type must be 20-23 (known TLS types)
        if content_type not in TLS_CONTENT_TYPES:
            break
        # Version should be 0x0301 (TLS 1.0) through 0x0304 (TLS 1.3)
        version_major = payload[offset + 1]
        version_minor = payload[offset + 2]
        if version_major != 3 or version_minor > 4:
            break
        record_length = struct.unpack("!H", payload[offset + 3:offset + 5])[0]
        if record_length > 16384 + 256:  # TLS max record + overhead
            break
        records.append((content_type, record_length))
        offset += 5 + record_length
    return records


def _extract_tls_features(packets: list) -> dict:
    """Extract TLS-specific features from flow packets."""
    all_records = []
    tls_type_counts = {
        "tls_handshake_count": 0,
        "tls_app_data_count": 0,
        "tls_alert_count": 0,
        "tls_ccs_count": 0,
    }
    tls_record_sizes = []
    app_data_sizes = []
    handshake_sizes = []
    packets_with_tls = 0

    for pkt in packets:
        payload = _payload_bytes(pkt)
        if len(payload) < 5:
            continue
        records = _parse_tls_records(payload)
        if not records:
            continue
        packets_with_tls += 1
        for ctype, rlen in records:
            all_records.append((ctype, rlen))
            tls_record_sizes.append(rlen)
            if ctype == TLS_HANDSHAKE:
                tls_type_counts["tls_handshake_count"] += 1
                handshake_sizes.append(rlen)
            elif ctype == TLS_APPLICATION_DATA:
                tls_type_counts["tls_app_data_count"] += 1
                app_data_sizes.append(rlen)
            elif ctype == TLS_ALERT:
                tls_type_counts["tls_alert_count"] += 1
            elif ctype == TLS_CHANGE_CIPHER_SPEC:
                tls_type_counts["tls_ccs_count"] += 1

    total_records = len(all_records)

    # TLS record size statistics
    rec_stats = _safe_stats(tls_record_sizes)
    app_stats = _safe_stats(app_data_sizes)

    features = {
        # TLS record counts
        **tls_type_counts,
        "tls_total_records": total_records,
        "tls_packets_ratio": packets_with_tls / max(len(packets), 1),
        # TLS record type ratios (out of total records)
        "tls_handshake_ratio": tls_type_counts["tls_handshake_count"] / max(total_records, 1),
        "tls_app_data_ratio": tls_type_counts["tls_app_data_count"] / max(total_records, 1),
        # TLS record size statistics
        "tls_record_size_mean": round(rec_stats["mean"], 2),
        "tls_record_size_std": round(rec_stats["std"], 2),
        "tls_record_size_max": rec_stats["max"],
        # TLS app data size statistics
        "tls_app_data_size_mean": round(app_stats["mean"], 2),
        "tls_app_data_size_std": round(app_stats["std"], 2),
        # TLS record size entropy
        "tls_record_size_entropy": round(_shannon_entropy(tls_record_sizes), 4),
    }

    # Sanity check: If no handshake was seen but app_data was parsed,
    # the records are likely false positives from encrypted payload bytes
    if features["tls_handshake_count"] == 0 and features["tls_app_data_count"] > 0:
        # Zero out all TLS features - can't trust the parsing
        for key in features:
            if isinstance(features[key], float):
                features[key] = 0.0
            else:
                features[key] = 0

    return features


# ---------------------------------------------------------------------------
# Temporal / Sequence features
# ---------------------------------------------------------------------------

def _extract_sequence_features(
    packets: list,
    fwd_src: str,
    fwd_sport: int,
) -> dict:
    """
    Extract first-N packet sequence features and turn-taking features.

    These capture the protocol's request-response fingerprint:
    - MCP over SSE has a distinctive pattern of small client requests
      followed by server-sent event streams
    - HTTP REST has short req-resp pairs
    - WebSocket has interleaved small frames
    """
    n = FIRST_N_PACKETS

    # First-N packet sizes (0-padded if flow has fewer packets)
    sizes = [_pkt_size(pkt) for pkt in packets[:n]]
    while len(sizes) < n:
        sizes.append(0)

    # First-N packet directions: +1 forward, -1 backward
    directions = []
    for pkt in packets[:n]:
        if _is_forward(pkt, fwd_src, fwd_sport):
            directions.append(1)
        else:
            directions.append(-1)
    while len(directions) < n:
        directions.append(0)

    # Direction changes (turn-taking analysis over entire flow)
    all_directions = []
    for pkt in packets:
        if _is_forward(pkt, fwd_src, fwd_sport):
            all_directions.append(1)
        else:
            all_directions.append(-1)

    direction_changes = 0
    for i in range(1, len(all_directions)):
        if all_directions[i] != all_directions[i - 1]:
            direction_changes += 1

    # Turn count: number of contiguous runs in the same direction
    turn_count = 1
    for i in range(1, len(all_directions)):
        if all_directions[i] != all_directions[i - 1]:
            turn_count += 1

    # Inter-turn latency: time between direction changes
    timestamps = [float(pkt.time) for pkt in packets]
    turn_latencies = []
    for i in range(1, len(all_directions)):
        if all_directions[i] != all_directions[i - 1]:
            turn_latencies.append(timestamps[i] - timestamps[i - 1])

    turn_lat_stats = _safe_stats(turn_latencies) if turn_latencies else _safe_stats([])

    # Packets per turn: how many packets in each contiguous direction run
    turn_sizes = []
    current_turn_size = 1
    for i in range(1, len(all_directions)):
        if all_directions[i] == all_directions[i - 1]:
            current_turn_size += 1
        else:
            turn_sizes.append(current_turn_size)
            current_turn_size = 1
    turn_sizes.append(current_turn_size)
    turn_size_stats = _safe_stats(turn_sizes)

    # Forward/backward packet ratio (direction-agnostic)
    fwd_count = sum(1 for d in all_directions if d == 1)
    bwd_count = sum(1 for d in all_directions if d == -1)
    dir_ratio = fwd_count / max(fwd_count + bwd_count, 1)

    features = {}

    # First-N sizes
    for i in range(n):
        features[f"pkt_size_{i}"] = sizes[i]

    # First-N directions
    for i in range(n):
        features[f"pkt_dir_{i}"] = directions[i]

    # Turn-taking features
    features["direction_changes"] = direction_changes
    features["turn_count"] = turn_count
    features["direction_ratio"] = round(dir_ratio, 4)

    # Inter-turn latency
    features["turn_latency_mean"] = round(turn_lat_stats["mean"], 6)
    features["turn_latency_std"] = round(turn_lat_stats["std"], 6)
    features["turn_latency_max"] = round(turn_lat_stats["max"], 6)

    # Packets per turn
    features["turn_size_mean"] = round(turn_size_stats["mean"], 2)
    features["turn_size_std"] = round(turn_size_stats["std"], 2)
    features["turn_size_max"] = turn_size_stats["max"]

    return features


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

    # -----------------------------------------------------------------------
    # Basic flow features
    # -----------------------------------------------------------------------

    timestamps = [float(pkt.time) for pkt in packets]
    flow_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0

    # Packet sizes
    all_sizes = [_pkt_size(pkt) for pkt in packets]
    payload_sizes = [_payload_size(pkt) for pkt in packets]
    non_zero_payloads = [s for s in payload_sizes if s > 0]

    # Direction split
    fwd_sizes = []
    bwd_sizes = []
    for pkt in packets:
        if _is_forward(pkt, fwd_src, fwd_sport):
            fwd_sizes.append(_pkt_size(pkt))
        else:
            bwd_sizes.append(_pkt_size(pkt))

    # Inter-arrival times
    iats = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]

    # TCP flags
    flag_counts = {"ACK": 0, "PSH": 0, "FIN": 0, "RST": 0}
    for pkt in packets:
        flags = _tcp_flags(pkt)
        for flag in flag_counts:
            flag_counts[flag] += flags[flag]

    # -----------------------------------------------------------------------
    # Burst analysis
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Aggregated statistics
    # -----------------------------------------------------------------------

    total_bytes = sum(all_sizes)
    fwd_bytes = sum(fwd_sizes)
    bwd_bytes = sum(bwd_sizes)

    # Byte-based asymmetry (not directional -- just ratio of larger to smaller)
    byte_asymmetry = abs(fwd_bytes - bwd_bytes) / max(total_bytes, 1)

    pkt_stats = _safe_stats(all_sizes)
    iat_stats = _safe_stats(iats) if iats else _safe_stats([])
    payload_stats = _safe_stats(non_zero_payloads) if non_zero_payloads else _safe_stats([])

    # Packet size categories
    small_pkts = sum(1 for s in all_sizes if s < 100)
    large_pkts = sum(1 for s in all_sizes if s > 1000)
    small_large_ratio = small_pkts / max(large_pkts, 1)

    # Payload ratio: how many packets actually carry data
    data_pkt_ratio = len(non_zero_payloads) / max(len(packets), 1)

    # Entropy features
    pkt_size_entropy = _shannon_entropy(all_sizes)
    iat_entropy = _shannon_entropy(iats) if iats else 0.0

    # -----------------------------------------------------------------------
    # TLS features
    # -----------------------------------------------------------------------

    tls_features = _extract_tls_features(packets)

    # -----------------------------------------------------------------------
    # Sequence / Temporal features
    # -----------------------------------------------------------------------

    seq_features = _extract_sequence_features(packets, fwd_src, fwd_sport)

    # -----------------------------------------------------------------------
    # Assemble output
    # -----------------------------------------------------------------------

    result = {
        # Identifiers (dropped before training)
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": src_port,
        "dst_port": dst_port,
        "protocol": proto,

        # Flow-level features
        "flow_duration": round(flow_duration, 6),
        "total_packets": len(packets),
        "total_bytes": total_bytes,
        "fwd_bytes": fwd_bytes,
        "bwd_bytes": bwd_bytes,
        "byte_asymmetry": round(byte_asymmetry, 4),

        # Packet size statistics
        "pkt_size_mean": round(pkt_stats["mean"], 2),
        "pkt_size_std": round(pkt_stats["std"], 2),
        "pkt_size_max": pkt_stats["max"],
        "pkt_size_median": pkt_stats["median"],

        # Payload size statistics (non-zero payloads only)
        "payload_mean": round(payload_stats["mean"], 2),
        "payload_std": round(payload_stats["std"], 2),
        "payload_max": payload_stats["max"],
        "data_pkt_ratio": round(data_pkt_ratio, 4),

        # Inter-arrival time statistics
        "iat_mean": round(iat_stats["mean"], 6),
        "iat_std": round(iat_stats["std"], 6),
        "iat_min": round(iat_stats["min"], 6),
        "iat_max": round(iat_stats["max"], 6),

        # Burst features
        "burst_count": len(bursts),
        "burst_size_mean": round(burst_stats["mean"], 2),
        "burst_size_std": round(burst_stats["std"], 2),

        # TCP flags (removed SYN -- near-constant)
        "flag_ACK": flag_counts["ACK"],
        "flag_PSH": flag_counts["PSH"],
        "flag_FIN": flag_counts["FIN"],
        "flag_RST": flag_counts["RST"],

        # Packet size categories
        "small_packets": small_pkts,
        "large_packets": large_pkts,
        "small_large_ratio": round(small_large_ratio, 4),

        # Idle time
        "idle_time_mean": round(idle_stats.get("mean", 0), 6),
        "idle_time_max": round(idle_stats.get("max", 0), 6),

        # Entropy features (removed redundant payload_size_entropy)
        "pkt_size_entropy": round(pkt_size_entropy, 4),
        "iat_entropy": round(iat_entropy, 4),
    }

    # Add TLS features
    result.update(tls_features)

    # Add sequence/temporal features
    result.update(seq_features)

    # Label (must be last for readability)
    result["label"] = label

    return result


# ---------------------------------------------------------------------------
# Pcap -> DataFrame
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
        except Exception:
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

    # Print feature count
    feature_cols = [c for c in combined.columns if c not in
                    ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "label"]]
    print(f"  Features:      {len(feature_cols)}")

    return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Feature Extractor v2")
    parser.add_argument("--pcap-dir", default="data/pcap")
    parser.add_argument("--output", default="data/features.csv")
    args = parser.parse_args()

    extract_features_from_directory(args.pcap_dir, args.output)


if __name__ == "__main__":
    main()
