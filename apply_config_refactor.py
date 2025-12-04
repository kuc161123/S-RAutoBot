import yaml
import shutil
import os

def run():
    print("🚀 Starting Configuration Refactor...")
    
    # 1. Load 115 Strategies
    overrides_path = 'symbol_overrides_400.yaml.bak'
    if not os.path.exists(overrides_path):
        overrides_path = 'symbol_overrides_400.yaml'
        
    if not os.path.exists(overrides_path):
        print(f"❌ Error: Could not find overrides file ({overrides_path})")
        return

    with open(overrides_path, 'r') as f:
        strategies = yaml.safe_load(f)
    
    print(f"✅ Loaded {len(strategies)} strategies from {overrides_path}")
    
    # 2. Load Config
    config_path = 'config.yaml'
    if os.path.exists(config_path):
        shutil.copy(config_path, 'config.yaml.bak_refactor')
        print("✅ Backed up config.yaml")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # 3. Inject Strategies
    config['strategies'] = strategies
    
    # 4. Update Symbols List
    all_symbols = list(strategies.keys())
    all_symbols.sort()
    config['trade']['symbols'] = all_symbols
    print(f"✅ Updated trade.symbols with {len(all_symbols)} symbols")
    
    # 5. Set Risk Percent
    if 'risk' not in config:
        config['risk'] = {}
    config['risk']['risk_percent'] = 0.5
    print("✅ Set risk.risk_percent to 0.5")
    
    # 6. Save Config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
        
    print("✅ Saved updated config.yaml")
    print("\n🎉 Refactor Complete! Config now contains all strategies.")

if __name__ == "__main__":
    run()
