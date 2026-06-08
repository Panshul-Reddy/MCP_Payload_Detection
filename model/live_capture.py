"""
Live Traffic Capture & Classification v2 — captures packets in real-time
and classifies flows using EARLY PACKET FEATURES (first N packets).

Two classification modes:
  1. EARLY MODE (default): Classify after N packets arrive (not waiting for flow end)
     - Uses early_model_n{N}.pkl trained on first-N packet features
     - If confidence >= threshold, classify immediately (~200-500ms latency)
     - If confidence < threshold, wait for more packets and retry
  2. FLOW MODE: Classify after flow timeout (original behavior)

Usage:
    # Early classification after 5 packets
    python -m model.live_capture --mode early --n-packets 5 --confidence 0.85

    # Early classification after 10 packets (more accurate)
    python -m model.live_capture --mode early --n-packets 10 --confidence 0.80

    # Traditional flow-based classification
    python -m model.live_capture --mode flow --timeout 3

    # Monitor specific ports
    python -m model.live_capture --mode early --filter "tcp port 8443 or tcp port 5443"
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
from scapy.all import AsyncSniffer, TCP, UDP, Raw, get_if_list

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

DROP_COLS = ["src_ip", "dst_ip", "src_port", "dst_port", "protocol", "label"]


def _default_loopback():
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


def _flow_key(pkt):
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


def _extract_early_features(packets, key, n_packets):
    """Extract features from the first N packets of a flow."""
    src_ip, dst_ip, src_port, dst_port, proto = key
    pkts = packets[:n_packets]

    # Determine forward direction
    try:
        fwd_src = pkts[0]["IP"].src
        fwd_sport = pkts[0][TCP].sport if pkts[0].haslayer(TCP) else pkts[0][UDP].sport
    except Exception:
        fwd_src = src_ip
        fwd_sport = src_port

    # Basic per-packet features
    sizes = [len(pkt) for pkt in pkts]
    directions = []
    fwd_sizes = []
    bwd_sizes = []

    for pkt in pkts:
        try:
            is_fwd = pkt["IP"].src == fwd_src and pkt[TCP].sport == fwd_sport
        except Exception:
            is_fwd = True
        directions.append(1 if is_fwd else -1)
        if is_fwd:
            fwd_sizes.append(len(pkt))
        else:
            bwd_sizes.append(len(pkt))

    # Timestamps and IATs
    timestamps = [float(pkt.time) for pkt in pkts]
    iats = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]

    # TLS record parsing (from first N packets)
    import struct
    tls_handshake = 0
    tls_app_data = 0
    tls_ccs = 0
    tls_total = 0
    tls_record_sizes = []
    tls_app_sizes = []

    for pkt in pkts:
        payload = b""
        if pkt.haslayer(Raw):
            payload = bytes(pkt[Raw].load)
        elif pkt.haslayer(TCP):
            payload = bytes(pkt[TCP].payload)
        if len(payload) < 5:
            continue
        offset = 0
        while offset + 5 <= len(payload):
            ct = payload[offset]
            if ct not in (20, 21, 22, 23):
                break
            vm = payload[offset + 1]
            if vm != 3:
                break
            rec_len = struct.unpack("!H", payload[offset+3:offset+5])[0]
            if rec_len > 16384 + 256:
                break
            tls_total += 1
            tls_record_sizes.append(rec_len)
            if ct == 22:
                tls_handshake += 1
            elif ct == 23:
                tls_app_data += 1
                tls_app_sizes.append(rec_len)
            elif ct == 20:
                tls_ccs += 1
            offset += 5 + rec_len

    # Sanity check
    if tls_handshake == 0 and tls_app_data > 0:
        tls_handshake = 0
        tls_app_data = 0
        tls_ccs = 0
        tls_total = 0
        tls_record_sizes = []
        tls_app_sizes = []

    # Direction changes
    dir_changes = sum(1 for i in range(1, len(directions)) if directions[i] != directions[i-1])

    # Build feature dict
    features = {}

    # First-N packet sizes (pad with 0)
    for i in range(10):
        features[f"pkt_size_{i+1}"] = sizes[i] if i < len(sizes) else 0
        features[f"pkt_dir_{i+1}"] = directions[i] if i < len(directions) else 0

    total_bytes = sum(sizes)
    fwd_bytes = sum(fwd_sizes)
    bwd_bytes = sum(bwd_sizes)

    features["total_packets"] = len(pkts)
    features["fwd_packets"] = len(fwd_sizes)
    features["bwd_packets"] = len(bwd_sizes)
    features["total_bytes"] = total_bytes
    features["fwd_bytes"] = fwd_bytes
    features["bwd_bytes"] = bwd_bytes
    features["pkt_size_mean"] = float(np.mean(sizes)) if sizes else 0
    features["pkt_size_std"] = float(np.std(sizes)) if sizes else 0
    features["pkt_size_max"] = max(sizes) if sizes else 0
    features["flow_asymmetry"] = fwd_bytes / max(total_bytes, 1)
    features["direction_changes"] = dir_changes
    features["fwd_bwd_ratio"] = len(fwd_sizes) / max(len(bwd_sizes), 1)

    # TLS features
    features["tls_handshake_count"] = tls_handshake
    features["tls_app_data_count"] = tls_app_data
    features["tls_total_records"] = tls_total
    features["tls_ccs_count"] = tls_ccs
    features["tls_app_data_ratio"] = tls_app_data / max(tls_total, 1)
    features["tls_handshake_ratio"] = tls_handshake / max(tls_total, 1)
    features["tls_record_size_mean"] = float(np.mean(tls_record_sizes)) if tls_record_sizes else 0
    features["tls_record_size_std"] = float(np.std(tls_record_sizes)) if tls_record_sizes else 0
    features["tls_app_data_size_mean"] = float(np.mean(tls_app_sizes)) if tls_app_sizes else 0

    # IAT features
    features["iat_mean"] = float(np.mean(iats)) if iats else 0
    features["iat_std"] = float(np.std(iats)) if iats else 0
    features["iat_min"] = float(np.min(iats)) if iats else 0
    features["iat_max"] = float(np.max(iats)) if iats else 0

    return features


class LiveClassifier:
    """Accumulates packets and classifies using early or flow-based features."""

    def __init__(self, model, mode="early", n_packets=10, confidence_threshold=0.85,
                 flow_timeout=3.0, flow_model=None):
        self.model = model  # early model
        self.flow_model = flow_model  # full flow model (optional)
        self.mode = mode
        self.n_packets = n_packets
        self.confidence_threshold = confidence_threshold
        self.flow_timeout = flow_timeout

        self.flows = {}
        self.last_seen = {}
        self.classified = set()
        self.early_classified = set()  # flows classified early
        self.lock = threading.Lock()

        self.stats = {"total": 0, "mcp": 0, "non_mcp": 0, "early": 0, "late": 0,
                      "high_conf": 0, "low_conf": 0}

    def on_packet(self, pkt):
        key = _flow_key(pkt)
        if key is None:
            return

        with self.lock:
            if key in self.classified:
                return
            if key not in self.flows:
                self.flows[key] = []
            self.flows[key].append(pkt)
            self.last_seen[key] = time.time()
            pkt_count = len(self.flows[key])

        # Early classification check
        if self.mode == "early" and key not in self.early_classified and pkt_count >= self.n_packets:
            self._try_early_classify(key)

    def check_completed_flows(self):
        now = time.time()
        completed = []
        with self.lock:
            for key in list(self.last_seen.keys()):
                if key in self.classified:
                    continue
                if now - self.last_seen[key] >= self.flow_timeout:
                    packets = self.flows.get(key, [])
                    if len(packets) >= 3:
                        completed.append((key, packets[:]))
                    self.classified.add(key)
                    if key in self.flows:
                        del self.flows[key]
                    if key in self.last_seen:
                        del self.last_seen[key]

        for key, packets in completed:
            if key not in self.early_classified:
                self._classify_flow(key, packets, "flow-timeout")

    def _try_early_classify(self, key):
        with self.lock:
            packets = self.flows.get(key, [])[:self.n_packets]

        if len(packets) < self.n_packets:
            return

        features = _extract_early_features(packets, key, self.n_packets)

        # Get the features the model expects
        if hasattr(self.model, '_feature_names') or hasattr(self, '_model_features'):
            pass

        # Build feature vector using model's expected features
        try:
            bundle_features = self._model_feature_names
        except AttributeError:
            bundle_features = list(features.keys())

        available = {k: features.get(k, 0) for k in bundle_features}
        X = pd.DataFrame([available])[bundle_features].fillna(0).replace([np.inf, -np.inf], 0)

        prediction = self.model.predict(X)[0]
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = proba.max()
        else:
            confidence = 1.0

        label_map = {0: "mcp", 1: "non_mcp"}
        if isinstance(prediction, (int, np.integer)):
            prediction = label_map.get(prediction, str(prediction))

        if confidence >= self.confidence_threshold:
            self.early_classified.add(key)
            with self.lock:
                self.classified.add(key)
                if key in self.flows:
                    del self.flows[key]
                if key in self.last_seen:
                    del self.last_seen[key]

            self.stats["total"] += 1
            self.stats["early"] += 1
            self.stats["high_conf"] += 1
            if prediction == "mcp":
                self.stats["mcp"] += 1
            else:
                self.stats["non_mcp"] += 1

            src_ip, dst_ip, src_port, dst_port, _ = key
            tag = " <<<" if prediction == "mcp" else ""
            print(f"  [{self.stats['total']:>4}] EARLY({self.n_packets}pkts) "
                  f"{src_ip}:{src_port} <-> {dst_ip}:{dst_port}  "
                  f"=> {prediction:<8} ({confidence:.1%}){tag}")

    def _classify_flow(self, key, packets, reason="flow-timeout"):
        if self.flow_model:
            try:
                from feature_extraction.extractor import _compute_flow_features
                row = _compute_flow_features(packets, "unknown", key)
                feature_cols = [c for c in row.keys() if c not in DROP_COLS]
                X = pd.DataFrame([row])[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
                prediction = self.flow_model.predict(X)[0]
                if hasattr(self.flow_model, "predict_proba"):
                    confidence = self.flow_model.predict_proba(X)[0].max()
                else:
                    confidence = 1.0
            except Exception:
                return
        else:
            features = _extract_early_features(packets, key, min(len(packets), 17))
            try:
                bundle_features = self._model_feature_names
            except AttributeError:
                bundle_features = list(features.keys())
            available = {k: features.get(k, 0) for k in bundle_features}
            X = pd.DataFrame([available])[bundle_features].fillna(0).replace([np.inf, -np.inf], 0)
            prediction = self.model.predict(X)[0]
            if hasattr(self.model, "predict_proba"):
                confidence = self.model.predict_proba(X)[0].max()
            else:
                confidence = 1.0

        label_map = {0: "mcp", 1: "non_mcp"}
        if isinstance(prediction, (int, np.integer)):
            prediction = label_map.get(prediction, str(prediction))

        self.stats["total"] += 1
        self.stats["late"] += 1
        if confidence >= 0.8:
            self.stats["high_conf"] += 1
        else:
            self.stats["low_conf"] += 1
        if prediction == "mcp":
            self.stats["mcp"] += 1
        else:
            self.stats["non_mcp"] += 1

        src_ip, dst_ip, src_port, dst_port, _ = key
        tag = " <<<" if prediction == "mcp" else ""
        marker = " [LOW-CONF]" if confidence < 0.8 else ""
        print(f"  [{self.stats['total']:>4}] FLOW({len(packets)}pkts)  "
              f"{src_ip}:{src_port} <-> {dst_ip}:{dst_port}  "
              f"=> {prediction:<8} ({confidence:.1%}){marker}{tag}")


def main():
    parser = argparse.ArgumentParser(description="Live MCP Traffic Classifier v2")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--mode", choices=["early", "flow"], default="early",
                        help="Classification mode: early (first N packets) or flow (wait for timeout)")
    parser.add_argument("--n-packets", type=int, default=10,
                        help="Packets needed for early classification (default: 10)")
    parser.add_argument("--confidence", type=float, default=0.85,
                        help="Confidence threshold for early classification (default: 0.85)")
    parser.add_argument("--model", default=None,
                        help="Model path (auto-selects based on mode)")
    parser.add_argument("--flow-model", default="models/best_model.pkl",
                        help="Full flow model for fallback")
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--filter", default="tcp")
    parser.add_argument("--duration", type=int, default=0)
    args = parser.parse_args()

    # Load early model
    if args.model:
        model_path = args.model
    elif args.mode == "early":
        model_path = f"models/early_model_n{args.n_packets}.pkl"
        if not os.path.exists(model_path):
            model_path = "models/best_model.pkl"
            print(f"[Live] Early model not found, using full model: {model_path}")
    else:
        model_path = "models/best_model.pkl"

    print(f"[Live] Loading model: {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle.get("model", bundle) if isinstance(bundle, dict) else bundle
    feature_names = bundle.get("feature_names", []) if isinstance(bundle, dict) else []

    # Load flow model for fallback
    flow_model = None
    if args.flow_model and os.path.exists(args.flow_model) and args.flow_model != model_path:
        with open(args.flow_model, "rb") as f:
            fb = pickle.load(f)
        flow_model = fb.get("model", fb) if isinstance(fb, dict) else fb

    interface = args.interface or _default_loopback()
    classifier = LiveClassifier(
        model, mode=args.mode, n_packets=args.n_packets,
        confidence_threshold=args.confidence, flow_timeout=args.timeout,
        flow_model=flow_model,
    )
    classifier._model_feature_names = feature_names

    print(f"\n{'='*70}")
    print(f"LIVE MCP TRAFFIC CLASSIFIER v2")
    print(f"{'='*70}")
    print(f"  Mode:        {args.mode.upper()}")
    if args.mode == "early":
        print(f"  N packets:   {args.n_packets} (classify after this many)")
        print(f"  Confidence:  {args.confidence:.0%} (threshold for early decision)")
    print(f"  Interface:   {interface}")
    print(f"  BPF filter:  {args.filter}")
    print(f"  Timeout:     {args.timeout}s (fallback for incomplete flows)")
    print(f"  Duration:    {'infinite (Ctrl+C)' if args.duration == 0 else f'{args.duration}s'}")
    print(f"{'='*70}\n")

    sniffer = AsyncSniffer(iface=interface, filter=args.filter, prn=classifier.on_packet, store=False)
    sniffer.start()

    start_time = time.time()
    try:
        while True:
            time.sleep(1.0)
            classifier.check_completed_flows()
            if args.duration > 0 and (time.time() - start_time) >= args.duration:
                break
    except KeyboardInterrupt:
        print("\n\n[Live] Stopping...")

    sniffer.stop()
    time.sleep(args.timeout + 1)
    classifier.check_completed_flows()

    s = classifier.stats
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total classified:  {s['total']}")
    print(f"  MCP detected:      {s['mcp']}")
    print(f"  Non-MCP detected:  {s['non_mcp']}")
    print(f"  Early decisions:   {s['early']} (classified after {args.n_packets} packets)")
    print(f"  Late decisions:    {s['late']} (classified after flow timeout)")
    print(f"  High confidence:   {s['high_conf']}")
    print(f"  Low confidence:    {s['low_conf']}")
    print(f"  Duration:          {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    main()
