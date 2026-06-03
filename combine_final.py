"""Combine ALL existing datasets into a final mega set."""
import pandas as pd
import os

files = [
    'data/features_v2.csv',        # Original 134 flows (fixed ports)
    'data/features_diverse.csv',   # 167 flows (random ports)
    'data/features_full.csv',      # 239 flows (loopback + some external)
    'data/features_big.csv',       # 623 flows (6 loopback rounds)
]

dfs = []
for path in files:
    if os.path.exists(path):
        df = pd.read_csv(path)
        dfs.append(df)
        mcp = len(df[df.label == 'mcp'])
        non = len(df[df.label == 'non_mcp'])
        print(f"  {path}: {len(df)} flows (MCP={mcp}, non_mcp={non})")

combined = pd.concat(dfs, ignore_index=True)
before = len(combined)

# Deduplicate by flow signature
combined = combined.drop_duplicates(
    subset=['src_port', 'dst_port', 'flow_duration', 'total_packets', 'total_bytes'],
    keep='first',
)
after = len(combined)
print(f"\n  Dedup: {before} -> {after} ({before - after} removed)")

combined.to_csv('data/features_final.csv', index=False)

mcp = len(combined[combined.label == 'mcp'])
non = len(combined[combined.label == 'non_mcp'])
print(f"\n  FINAL DATASET: {after} flows")
print(f"    MCP:     {mcp} ({mcp/after*100:.1f}%)")
print(f"    Non-MCP: {non} ({non/after*100:.1f}%)")
print(f"    Unique src_ports: {combined.src_port.nunique()}")
print(f"    Unique dst_ports: {combined.dst_port.nunique()}")
print(f"    Unique IPs: {combined.dst_ip.nunique()}")
