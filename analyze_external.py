"""Analyze external HTTPS flows vs loopback flows."""
import pandas as pd
df = pd.read_csv('data/features_final.csv')

ext = df[df.src_port == 443]
mcp = df[df.label == "mcp"]
local_non = df[(df.label == "non_mcp") & (df.src_port != 443)]

print(f"External flows (port 443): {len(ext)}")
print(f"Labels: {ext.label.value_counts().to_dict()}")

print(f"\nKey features comparison:")
cols = ['tls_total_records', 'tls_app_data_count', 'tls_app_data_ratio',
        'tls_handshake_count', 'flow_duration', 'total_packets', 'total_bytes',
        'turn_size_mean', 'direction_changes']
print(f"  {'Feature':<25s} {'MCP':>10s} {'Local non-MCP':>14s} {'External':>10s}")
print(f"  {'-'*60}")
for col in cols:
    m = mcp[col].mean()
    l = local_non[col].mean()
    e = ext[col].mean()
    print(f"  {col:<25s} {m:>10.2f} {l:>14.2f} {e:>10.2f}")

print(f"\nWhy external might be misclassified:")
print(f"  External tls_app_data_count: {ext.tls_app_data_count.describe().to_dict()}")
print(f"  MCP tls_app_data_count:      mean={mcp.tls_app_data_count.mean():.1f}, min={mcp.tls_app_data_count.min()}")
print(f"  Local non tls_app_data_cnt:  mean={local_non.tls_app_data_count.mean():.1f}, max={local_non.tls_app_data_count.max()}")
