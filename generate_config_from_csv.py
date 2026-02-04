#!/usr/bin/env python3
"""
Generate new config.yaml symbols section from multi_div_validated.csv
"""
import pandas as pd
import yaml

def generate_config():
    # Read validated results
    df = pd.read_csv('/Users/lualakol/AutoTrading Bot/multi_div_validated.csv')
    
    # Read current config to preserve non-symbols sections
    with open('/Users/lualakol/AutoTrading Bot/config.yaml.backup_20260129_multi_div', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate new symbols section
    symbols = {}
    for _, row in df.iterrows():
        symbol = row['symbol']
        # If symbol already exists (multi-div), we need to handle it
        # For now, we'll use the config with highest test_r
        test_r = float(row['test_r'])
        if symbol in symbols:
            if test_r > symbols[symbol].get('_test_r', 0):
                symbols[symbol] = {
                    'enabled': True,
                    'rr': float(row['rr']),
                    'atr_mult': float(row['atr']),
                    'divergence_type': str(row['div_type']),
                    'expected_wr': float(row['test_wr']),
                    'expected_avg_r': round(float(row['test_r']) / float(row['test_trades']), 3),
                    '_test_r': test_r
                }
        else:
            symbols[symbol] = {
                'enabled': True,
                'rr': float(row['rr']),
                'atr_mult': float(row['atr']),
                'divergence_type': str(row['div_type']),
                'expected_wr': float(row['test_wr']),
                'expected_avg_r': round(float(row['test_r']) / float(row['test_trades']), 3),
                '_test_r': test_r
            }
    
    # Remove temporary _test_r field
    for sym in symbols:
        if '_test_r' in symbols[sym]:
            del symbols[sym]['_test_r']
    
    # Update config
    config['symbols'] = symbols
    config['strategy']['name'] = '1h_184_symbols_multi_div_validated'
    config['strategy']['description'] = f'Multi-Div Validated - {len(symbols)} Symbols (271 configs)'
    
    # Calculate expected 90d R - ensure Python float
    total_test_r = float(df['test_r'].sum())
    expected_90d_r = round(total_test_r * (90/91.25), 2)
    config['strategy']['expected_90d_r'] = expected_90d_r
    
    # Write new config
    with open('/Users/lualakol/AutoTrading Bot/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    print(f"✅ Updated config.yaml with {len(symbols)} symbols")
    print(f"📊 Total Test R: {total_test_r:.2f}")
    print(f"📈 Expected 90d R: {expected_90d_r:.2f}")

if __name__ == "__main__":
    generate_config()
