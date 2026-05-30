"""Combine ALL feature CSVs into one mega dataset, deduplicate, and show stats."""
import pandas as pd
import os

files = [
    ('data/features_v2.csv', 'Original (fixed ports)'),
    ('data/features_diverse.csv', 'Diverse (random ports)'),
    ('data/features_full.csv', 'Full (loopback + external HTTPS)'),
]

dfs = []
for path, label in files:
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['source'] = label
        dfs.append(df)
        print(f"  {label}: {len(df)} flows (MCP={len(df[df.label=='mcp'])}, non_mcp={len(df[df.label=='non_mcp'])})")
    else:
        print(f"  {label}: NOT FOUND")

combined = pd.concat(dfs, ignore_index=True)

# Deduplicate: same flow key = same src_port, dst_port, flow_duration, total_packets
before = len(combined)
combined = combined.drop_duplicates(
    subset=['src_port', 'dst_port', 'flow_duration', 'total_packets', 'total_bytes'],
    keep='first',
)
after = len(combined)
print(f"\n  Deduplication: {before} -> {after} ({before - after} duplicates removed)")

# Drop source column before saving
combined = combined.drop(columns=['source'], errors='ignore')
combined.to_csv('data/features_mega.csv', index=False)

print(f"\n  MEGA DATASET: {len(combined)} flows")
print(f"    MCP:     {len(combined[combined.label=='mcp'])}")
print(f"    Non-MCP: {len(combined[combined.label=='non_mcp'])}")
print(f"    Unique src_ports: {combined.src_port.nunique()}")
print(f"    Unique dst_ports: {combined.dst_port.nunique()}")

# Check IP diversity
print(f"    Unique src_ips: {combined.src_ip.nunique()}")
print(f"    Unique dst_ips: {combined.dst_ip.nunique()}")
if combined.dst_ip.nunique() > 1:
    print(f"    Non-loopback IPs: {[ip for ip in combined.dst_ip.unique() if ip != '127.0.0.1']}")
