"""Combine original and diverse feature CSVs into one combined dataset."""
import pandas as pd

df1 = pd.read_csv('data/features_v2.csv')
df2 = pd.read_csv('data/features_diverse.csv')

combined = pd.concat([df1, df2], ignore_index=True)
combined.to_csv('data/features_combined.csv', index=False)

print(f"Original:  {len(df1)} flows (MCP={len(df1[df1.label=='mcp'])}, non_mcp={len(df1[df1.label=='non_mcp'])})")
print(f"Diverse:   {len(df2)} flows (MCP={len(df2[df2.label=='mcp'])}, non_mcp={len(df2[df2.label=='non_mcp'])})")
print(f"Combined:  {len(combined)} flows (MCP={len(combined[combined.label=='mcp'])}, non_mcp={len(combined[combined.label=='non_mcp'])})")

# Verify port diversity
print(f"\nPort diversity check:")
print(f"  Unique src_port values: {combined.src_port.nunique()}")
print(f"  Unique dst_port values: {combined.dst_port.nunique()}")
print(f"  src_port range: {combined.src_port.min()} - {combined.src_port.max()}")
