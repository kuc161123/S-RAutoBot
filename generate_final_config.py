import pandas as pd
import yaml

# Load data
df = pd.read_csv('market_wide_results.csv')
with open('safe_symbols.txt', 'r') as f:
    safe_symbols = set(f.read().splitlines())
    
# Filter dataframe for best configs of SAFE symbols only
df_safe = df[df['symbol'].isin(safe_symbols)].copy()

# Sort by R to get best config per symbol
df_sorted = df_safe.sort_values(by=['total_r', 'wr'], ascending=False)
best_configs = df_sorted.drop_duplicates(subset='symbol')

print(f"Generating config for {len(best_configs)} safe symbols...")

# Load base config to preserve settings
with open('config.yaml', 'r') as f:
    base_config = yaml.safe_load(f)

# Build new symbols dict
new_symbols_config = {}
total_expected_r = 0

for _, row in best_configs.iterrows():
    sym = row['symbol']
    new_symbols_config[sym] = {
        'enabled': True,
        'rr': float(row['rr']),
        'atr_mult': float(row['atr']),
        'divergence_type': row['div_type'],
        'expected_wr': float(row['wr']),
        'expected_avg_r': float(row['avg_r'])
    }
    total_expected_r += row['total_r']

# Update base config
base_config['symbols'] = new_symbols_config
base_config['strategy_description'] = f"1H Precision Divergence (Safe Liquid) - {len(new_symbols_config)} Symbols"
base_config['expected_90d_r'] = round(total_expected_r, 2)

# Save
with open('config_combined.yaml', 'w') as f:
    yaml.dump(base_config, f, sort_keys=False)

print(f"DONE. Saved to config_combined.yaml")
print(f"Total Projected 90d R: +{total_expected_r:.2f}R")
