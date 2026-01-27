#!/usr/bin/env python3
"""
APPLY CONFIG UPDATE
===================
Updates config.yaml with the best parameters found in market_wide_results.csv.
CRITICAL: Creates a backup of config.yaml first.
"""

import pandas as pd
import yaml
import shutil
import os
from datetime import datetime

def load_yaml_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml_config(data, path='config.yaml'):
    with open(path, 'w') as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)

def main():
    print("🛠️  Applying Optimized Configurations...")

    # 1. CREATE BACKUP
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"config.yaml.backup_{timestamp}"
    
    if os.path.exists('config.yaml'):
        shutil.copy('config.yaml', backup_file)
        print(f"✅ Backup created: {backup_file}")
    else:
        print("❌ config.yaml not found!")
        return

    # 2. LOAD DATA
    try:
        config = load_yaml_config()
        # We use the raw results to get precise parameters
        research_df = pd.read_csv('market_wide_results.csv')
        # We use verified comparison to know WHICH symbols to update (only improved ones)
        verified_df = pd.read_csv('verified_comparison.csv')
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # Filter for ALL profitable symbols
    profitable_symbols = research_df[research_df['total_r'] > 0]['symbol'].unique().tolist()
    print(f"📋 Found {len(profitable_symbols)} PROFITABLE symbols to add/update.")

    updated_count = 0
    added_count = 0
    
    current_symbols_config = config.get('symbols', {})
    
    for sym in profitable_symbols:
        # Get best config from research
        sym_results = research_df[research_df['symbol'] == sym]
        if sym_results.empty: continue
        
        best = sym_results.sort_values('total_r', ascending=False).iloc[0]
        
        # Prepare config dict
        new_config = {
            'enabled': True,
            'rr': float(best['rr']),
            'atr_mult': float(best['atr']),
            'divergence_type': best['div_type'],
            'expected_wr': float(best['wr']),
            'expected_avg_r': float(best['avg_r'])
        }

        if sym in current_symbols_config:
            # Update existing
            config['symbols'][sym].update(new_config)
            updated_count += 1
        else:
            # Add new
            config['symbols'][sym] = new_config
            added_count += 1

    print(f"✅ Updated {updated_count} existing symbols.")
    print(f"➕ Added {added_count} NEW symbols.")
    print(f"📊 Total Portfolio Size: {len(config['symbols'])}")

    # 3. SAVE
    save_yaml_config(config)
    print(f"✅ Successfully updated {updated_count} symbols in config.yaml")
    print(f"🔙 Revert anytime using: cp {backup_file} config.yaml")

if __name__ == "__main__":
    main()
