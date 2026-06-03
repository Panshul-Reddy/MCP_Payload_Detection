import pandas as pd

# All datasets
dfs = []
for path in ['data/features_big.csv', 'data/features_v2.csv',
             'data/features_diverse.csv', 'data/features_full.csv',
             'data/features_ext_all.csv', 'data/features_ext_v2.csv']:
    df = pd.read_csv(path)
    dfs.append(df)

c = pd.concat(dfs, ignore_index=True)
c = c.drop_duplicates(
    subset=['src_port', 'dst_port', 'flow_duration', 'total_packets', 'total_bytes'],
    keep='first'
)
c.to_csv('data/features_final.csv', index=False)

mcp = len(c[c.label == "mcp"])
non = len(c[c.label == "non_mcp"])
ext = len(c[c.src_port == 443])
print(f"FINAL: {len(c)} flows (MCP={mcp}, non_mcp={non})")
print(f"External HTTPS (port 443): {ext}")
print(f"Unique dst_ips: {sorted(c.dst_ip.unique())}")
