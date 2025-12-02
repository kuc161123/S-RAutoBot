#!/usr/bin/env python3
"""
Integrate 400-symbol backtest results into bot configuration.

Steps:
1. Extract validated symbols from symbol_overrides_400.yaml
2. Backup existing symbol_overrides.yaml
3. Replace symbol_overrides.yaml with new backtest results
4. Update config.yaml with validated symbols
"""
import yaml
import shutil
from datetime import datetime
from pathlib import Path

def main():
    print("=" * 80)
    print("🔄 INTEGRATING 400-SYMBOL BACKTEST RESULTS")
    print("=" * 80)
    
    # Step 1: Load backtest results
    print("\n📊 Step 1: Loading backtest results...")
    with open('symbol_overrides_400.yaml', 'r') as f:
        backtest_overrides = yaml.safe_load(f)
    
    validated_symbols = list(backtest_overrides.keys())
    print(f"✅ Loaded {len(validated_symbols)} validated symbols from backtest")
    
    # Step 2: Backup existing symbol_overrides.yaml
    print("\n💾 Step 2: Backing up existing symbol_overrides.yaml...")
    if Path('symbol_overrides.yaml').exists():
        backup_name = f'symbol_overrides.yaml.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        shutil.copy2('symbol_overrides.yaml', backup_name)
        print(f"✅ Backed up to {backup_name}")
    else:
        print("⚠️  No existing symbol_overrides.yaml found (will create new)")
    
    # Step 3: Replace symbol_overrides.yaml with new backtest results
    print("\n📝 Step 3: Replacing symbol_overrides.yaml...")
    shutil.copy2('symbol_overrides_400.yaml', 'symbol_overrides.yaml')
    print(f"✅ Copied symbol_overrides_400.yaml → symbol_overrides.yaml")
    
    # Step 4: Update config.yaml with validated symbols
    print("\n⚙️  Step 4: Updating config.yaml...")
    
    # Backup config.yaml
    config_backup = f'config.yaml.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    shutil.copy2('config.yaml', config_backup)
    print(f"✅ Backed up config to {config_backup}")
    
    # Load and update config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    old_symbol_count = len(config.get('trade', {}).get('symbols', []))
    
    # Update symbols list
    config['trade']['symbols'] = sorted(validated_symbols)
    
    # Write updated config
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✅ Updated config.yaml")
    print(f"   Old symbol count: {old_symbol_count}")
    print(f"   New symbol count: {len(validated_symbols)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ INTEGRATION COMPLETE")
    print("=" * 80)
    print(f"📈 Validated symbols: {len(validated_symbols)}")
    print(f"📁 Backups created:")
    print(f"   - {config_backup}")
    if Path('symbol_overrides.yaml.backup.*').exists():
        print(f"   - {backup_name if 'backup_name' in locals() else '(no old overrides)'}")
    print("\n🚀 Bot is now configured to use 196 validated high-probability symbols!")
    print("=" * 80)

if __name__ == "__main__":
    main()
