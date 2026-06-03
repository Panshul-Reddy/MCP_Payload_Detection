"""
Live Traffic Capture & Classification — captures packets in real-time,
groups them into flows, and classifies each completed flow using the
trained model.

A flow is considered "complete" when no new packets arrive for --timeout
seconds.

Usage:
    # Monitor loopback interface (default)
    python -m model.live_capture --model models/best_model.pkl

    # Monitor a specific interface
    python -m model.live_capture --interface "\\Device\\NPF_{018A21DB-...}" --model models/best_model.pkl

    # Custom timeout and BPF filter
    python -m model.live_capture --timeout 5 --filter "tcp port 8443 or tcp port 5443"
"""

import argparse
import os
import pickle
import platform
import sys
import threading
import time

import numpy as np
import pandas as pd
from scapy.all import AsyncSniffer, TCP, UDP, get_if_list


# Import feature extraction
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DROP_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "label"]


def _default_loopback():
    """Auto-detect the loopback interface."""
    system = platform.system()
    if system == "Darwin":
        return "lo0"
    elif system == "Linux":
        return "lo"
    else:
        ifaces = get_if_list()
        for iface in ifaces:
            if "loopback" in iface.lower() or "NPF_Loopback" in iface:
                return iface
        return "\\Device\\NPF_Loopback"


def _flow_key_from_pkt(pkt):
    """Compute normalized bidirectional flow key from a packet."""
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

    ep1 = (src_ip, src_port)
    ep2 = (dst_ip, dst_port)
    if ep1 > ep2:
        ep1, ep2 = ep2, ep1
    return (ep1[0], ep2[0], ep1[1], ep2[1], proto)


class LiveClassifier:
    """Accumulates packets into flows and classifies when complete."""

    def __init__(self, model, timeout: float = 3.0):
        self.model = model
        self.timeout = timeout
        self.flows = {}           # key -> list of packets
        self.last_seen = {}       # key -> timestamp of last packet
        self.classified = set()   # keys already classified
        self.lock = threading.Lock()
        self.total_classified = 0
        self.total_mcp = 0
        self.total_non_mcp = 0

    def on_packet(self, pkt):
        """Called for each captured packet."""
        key = _flow_key_from_pkt(pkt)
        if key is None:
            return

        with self.lock:
            if key not in self.flows:
                self.flows[key] = []
            self.flows[key].append(pkt)
            self.last_seen[key] = time.time()

    def check_completed_flows(self):
        """Check for flows that haven't seen packets recently."""
        now = time.time()
        completed = []

        with self.lock:
            for key in list(self.last_seen.keys()):
                if key in self.classified:
                    continue
                if now - self.last_seen[key] >= self.timeout:
                    packets = self.flows.get(key, [])
                    if len(packets) >= 3:
                        completed.append((key, packets[:]))
                    self.classified.add(key)
                    # Clean up
                    if key in self.flows:
                        del self.flows[key]
                    if key in self.last_seen:
                        del self.last_seen[key]

        for key, packets in completed:
            self._classify_flow(key, packets)

    def _classify_flow(self, key, packets):
        """Extract features and classify a single flow."""
        try:
            from feature_extraction.extractor import _compute_flow_features
            row = _compute_flow_features(packets, "unknown", key)
        except Exception as e:
            return

        # Prepare feature vector
        feature_cols = [c for c in row.keys() if c not in DROP_COLS]
        X = pd.DataFrame([row])[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)

        # Predict
        prediction = self.model.predict(X)[0]
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = proba.max()
        else:
            confidence = 1.0

        # Map numeric to label
        label_map = {0: "mcp", 1: "non_mcp"}
        if isinstance(prediction, (int, np.integer)):
            prediction = label_map.get(prediction, str(prediction))

        self.total_classified += 1
        if prediction == "mcp":
            self.total_mcp += 1
        else:
            self.total_non_mcp += 1

        # Print result
        src_ip, dst_ip, src_port, dst_port, proto = key
        marker = " [UNCERTAIN]" if confidence < 0.8 else ""
        tag = " <<<" if prediction == "mcp" else ""

        print(f"  [{self.total_classified:>4}] "
              f"{src_ip}:{src_port} <-> {dst_ip}:{dst_port}  "
              f"pkts={len(packets):<4}  "
              f"=> {prediction:<8} ({confidence:.1%}){marker}{tag}")


def main():
    parser = argparse.ArgumentParser(description="Live MCP Traffic Classifier")
    parser.add_argument("--interface", default=None,
                        help="Network interface (default: auto-detect loopback)")
    parser.add_argument("--model", default="models/best_model.pkl",
                        help="Path to trained model")
    parser.add_argument("--timeout", type=float, default=3.0,
                        help="Seconds of silence before classifying a flow (default: 3)")
    parser.add_argument("--filter", default="tcp",
                        help="BPF filter (default: 'tcp')")
    parser.add_argument("--duration", type=int, default=0,
                        help="Capture duration in seconds (0 = infinite, Ctrl+C to stop)")
    args = parser.parse_args()

    # Load model
    print(f"[Live] Loading model: {args.model}")
    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    if isinstance(bundle, dict):
        model = bundle.get("model", bundle.get("clf"))
    else:
        model = bundle
    print(f"[Live] Model type: {type(model).__name__}")

    # Setup
    interface = args.interface or _default_loopback()
    classifier = LiveClassifier(model, timeout=args.timeout)

    print(f"\n{'='*70}")
    print(f"LIVE MCP TRAFFIC CLASSIFIER")
    print(f"{'='*70}")
    print(f"  Interface:   {interface}")
    print(f"  BPF filter:  {args.filter}")
    print(f"  Flow timeout: {args.timeout}s")
    print(f"  Duration:    {'infinite (Ctrl+C to stop)' if args.duration == 0 else f'{args.duration}s'}")
    print(f"{'='*70}\n")
    print(f"  {'#':>6} {'Flow':^45} {'Pkts':>5}  {'Result':^20}")
    print(f"  {'-'*80}")

    # Start sniffer
    sniffer = AsyncSniffer(
        iface=interface,
        filter=args.filter,
        prn=classifier.on_packet,
        store=False,
    )
    sniffer.start()

    # Periodic check for completed flows
    start_time = time.time()
    try:
        while True:
            time.sleep(1.0)
            classifier.check_completed_flows()

            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                break

    except KeyboardInterrupt:
        print("\n\n[Live] Stopping capture...")

    # Final check
    sniffer.stop()
    time.sleep(args.timeout + 1)
    classifier.check_completed_flows()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total flows classified: {classifier.total_classified}")
    print(f"  MCP detected:           {classifier.total_mcp}")
    print(f"  Non-MCP detected:       {classifier.total_non_mcp}")
    elapsed = time.time() - start_time
    print(f"  Duration:               {elapsed:.0f}s")


if __name__ == "__main__":
    main()
