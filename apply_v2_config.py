import yaml
import shutil
import os

def run():
    # 1. Backup existing files
    if os.path.exists('symbol_overrides.yaml'):
        shutil.copy('symbol_overrides.yaml', 'symbol_overrides.yaml.bak')
        print("✅ Backed up symbol_overrides.yaml")
        
    if os.path.exists('config.yaml'):
        shutil.copy('config.yaml', 'config.yaml.bak')
        print("✅ Backed up config.yaml")
        
    # 2. Install V2 Overrides
    with open('symbol_overrides_v2.yaml', 'r') as f:
        v2_data = yaml.safe_load(f)
        
    with open('symbol_overrides.yaml', 'w') as f:
        yaml.dump(v2_data, f)
        print("✅ Installed V2 Overrides to symbol_overrides.yaml")
        
    # 3. Update Config Symbols
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    v2_symbols = list(v2_data.keys())
    # Sort for neatness
    v2_symbols.sort()
    
    config['trade']['symbols'] = v2_symbols
    
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f)
        print(f"✅ Updated config.yaml with {len(v2_symbols)} symbols: {v2_symbols}")
        
    print("\n🚀 V2 CONFIGURATION APPLIED SUCCESSFULLY!")
    print("Please RESTART the bot to activate.")

if __name__ == "__main__":
    run()
