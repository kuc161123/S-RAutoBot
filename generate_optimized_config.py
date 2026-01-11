
import pandas as pd
import yaml
import os

INPUT_FILE = 'precision_optimization_deduped.csv'
CONFIG_FILE = 'config.yaml'
OUTPUT_CONFIG_FILE = 'config_optimized.yaml'

def generate_config():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Loading optimization results from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Filter for robustness
    # We want at least 10 trades to be statistically somewhat significant
    # (Note: In the high-precision mode, 10 trades over 90 days is acceptable)
    df = df[df['trades'] >= 10]
    
    # Sort by Efficiency (Avg R per trade) then Total R
    df = df.sort_values(by=['avg_r', 'total_r'], ascending=[False, False])
    
    # Deduplicate to keep only the BEST config per symbol
    best_configs = df.drop_duplicates(subset=['symbol'], keep='first')
    
    print(f"Found robust configurations for {len(best_configs)} symbols.")
    
    # Load existing config to preserve API keys etc.
    current_config = {}
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            current_config = yaml.safe_load(f)
    
    # Prepare new symbols dict
    new_symbols_config = {}
    
    # Create the structure matches what Bot4H expects (SymbolRRConfig)
    # The bot expects: 
    # symbols:
    #   BTCUSDT:
    #     rr: 3.0
    #     atr_mult: 1.0
    #     div_type: REG_BULL  (Main divergence focus, though bot can support list if we advanced it)
    
    total_expected_r = 0
    
    for _, row in best_configs.iterrows():
        symbol = row['symbol']
        new_symbols_config[symbol] = {
            'rr': float(row['rr']),
            'atr_mult': float(row['atr']),
            'divergence_type': row['div_type'], # The optimized divergence type
            'expected_wr': float(row['wr']),
            'expected_avg_r': float(row['avg_r'])
        }
        total_expected_r += row['total_r']
        
    # Update config
    current_config['symbols'] = new_symbols_config
    
    # Update strategy comments/metadata if possible
    if 'strategy' not in current_config:
        current_config['strategy'] = {}
        
    current_config['strategy']['description'] = f"Precision Optimized - {len(new_symbols_config)} Symbols"
    current_config['strategy']['expected_90d_r'] = float(round(total_expected_r, 2))
    
    # Save
    with open(OUTPUT_CONFIG_FILE, 'w') as f:
        yaml.dump(current_config, f, sort_keys=False)
        
    print(f"Successfully generated {OUTPUT_CONFIG_FILE}")
    print(f"Total Symbols: {len(new_symbols_config)}")
    print(f"Total Expected 90d R: {total_expected_r:.2f}R")

if __name__ == "__main__":
    generate_config()
