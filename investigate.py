"""Investigate all identified issues with the current model and data."""
import pandas as pd
import pickle
import numpy as np

df = pd.read_csv('data/features.csv')

print('='*60)
print('ISSUE 1: Data Leakage Check')
print('='*60)

with open('models/best_model.pkl', 'rb') as f:
    bundle = pickle.load(f)
features_used = bundle['feature_names']
print(f'Features used by model ({len(features_used)}):')
for f_name in features_used:
    print(f'  {f_name}')
leaky = [f for f in features_used if f in ['src_ip','dst_ip','src_port','dst_port','protocol']]
if leaky:
    print(f'\n** PROBLEM: Leaky features found: {leaky}')
else:
    print('\n** GOOD: No IP/port features in model inputs')

print()
print('='*60)
print('ISSUE 2: Dataset Homogeneity Check')
print('='*60)

print(f'Unique src_ip: {df.src_ip.unique()}')
print(f'Unique dst_ip: {df.dst_ip.unique()}')
print(f'Unique dst_port (MCP):     {sorted(df[df.label=="mcp"].dst_port.unique())}')
print(f'Unique dst_port (non_mcp): {sorted(df[df.label=="non_mcp"].dst_port.unique())}')
print(f'Unique src_port range: {df.src_port.min()} - {df.src_port.max()}')

for cls in ['mcp', 'non_mcp']:
    subset = df[df.label == cls]
    print(f'\n{cls} flows ({len(subset)}):')
    print(f'  pkt_size_min values: {sorted(subset.pkt_size_min.unique())}')
    print(f'  flag_RST values: {sorted(subset.flag_RST.unique())}')
    print(f'  flow_duration: mean={subset.flow_duration.mean():.3f}, std={subset.flow_duration.std():.3f}')
    print(f'  total_packets: mean={subset.total_packets.mean():.1f}, std={subset.total_packets.std():.1f}')

print()
print('='*60)
print('ISSUE 3: Feature Redundancy Check')  
print('='*60)
corr = df[['pkt_size_entropy','payload_size_entropy','iat_entropy']].corr()
print('Correlation between entropy features:')
print(corr.round(3).to_string())
identical = (df['pkt_size_entropy'] == df['payload_size_entropy']).all()
print(f'\npkt_size_entropy == payload_size_entropy in ALL rows? {identical}')
if not identical:
    diff = (df['pkt_size_entropy'] - df['payload_size_entropy']).abs()
    print(f'  Mean abs difference: {diff.mean():.4f}')
    print(f'  Max abs difference:  {diff.max():.4f}')

print()
print('='*60)
print('ISSUE 4: Feature Importance Analysis')
print('='*60)
model = bundle['model']
if hasattr(model, 'feature_importances_'):
    importances = sorted(zip(features_used, model.feature_importances_), key=lambda x: -x[1])
    print('Feature importances (all):')
    for name, imp in importances:
        bar = '#' * int(imp * 100)
        print(f'  {name:<25s} {imp:.4f} {bar}')
    
    # Check if any single feature dominates
    top_imp = importances[0][1]
    if top_imp > 0.3:
        print(f'\n** WARNING: Top feature {importances[0][0]} has {top_imp:.1%} importance - possible over-reliance')

print()
print('='*60)
print('SUMMARY')
print('='*60)
print(f'Total samples: {len(df)}')
print(f'Class balance: mcp={len(df[df.label=="mcp"])}, non_mcp={len(df[df.label=="non_mcp"])}')
print(f'All from loopback: {(df.src_ip == "127.0.0.1").all()}')
print(f'CV used: Yes (StratifiedKFold), but small dataset limits reliability')
