import asyncio
import yaml
import os
from autobot.brokers.bybit import Bybit, BybitConfig

def replace_env_vars(config):
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, str) and config.startswith("${") and config.endswith("}"):
        var = config[2:-1]
        return os.getenv(var, config)
    return config

async def main():
    # Load config
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    cfg = replace_env_vars(cfg)
    
    # Initialize Bybit
    bybit = Bybit(BybitConfig(
        base_url=cfg['bybit']['base_url'],
        api_key=cfg['bybit']['api_key'],
        api_secret=cfg['bybit']['api_secret']
    ))
    
    print("🔍 Fetching all USDT perpetual symbols from Bybit...")
    
    # Get all instruments (returns list directly)
    try:
        all_symbols = bybit.get_instruments_info(category='linear')
        
        if not all_symbols:
            print("❌ No symbols returned from API")
            return
        
        print(f"📊 Total instruments found: {len(all_symbols)}")
        
        # Filter for USDT perpetuals that are tradeable
        usdt_symbols = []
        for sym in all_symbols:
            symbol = sym['symbol']
            
            # Must be USDT perpetual
            if not symbol.endswith('USDT'):
                continue
            
            # Must be in Trading status
            status = sym.get('status', '')
            if status != 'Trading':
                continue
            
            # Exclude indexes, spot margin, etc
            if 'PERP' in symbol or symbol.endswith('USD'):
                continue
            
            usdt_symbols.append({
                'symbol': symbol,
                'status': status,
                'max_leverage': sym.get('leverageFilter', {}).get('maxLeverage', '1'),
                'min_price': sym.get('priceFilter', {}).get('minPrice', '0'),
                'tick_size': sym.get('priceFilter', {}).get('tickSize', '0.01'),
            })
        
        print(f"✅ Found {len(usdt_symbols)} tradeable USDT perpetuals")
        
        # Sort by symbol name for consistency
        usdt_symbols.sort(key=lambda x: x['symbol'])
        
        # Take first 400
        target_count = min(400, len(usdt_symbols))
        selected = usdt_symbols[:target_count]
        
        print(f"\n📋 Selected {target_count} symbols:")
        print("="*60)
        
        # Group by first letter for display
        from collections import defaultdict
        grouped = defaultdict(list)
        for s in selected:
            first_letter = s['symbol'][0]
            grouped[first_letter].append(s['symbol'])
        
        for letter in sorted(grouped.keys()):
            symbols = grouped[letter]
            print(f"{letter}: {len(symbols)} symbols - {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
        
        # Save to file
        symbol_list = [s['symbol'] for s in selected]
        
        output = {
            'total_available': len(usdt_symbols),
            'selected_count': target_count,
            'symbols': symbol_list
        }
        
        with open('symbols_400.yaml', 'w') as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n✅ Saved to symbols_400.yaml")
        print(f"\nSymbol list preview:")
        print(symbol_list[:20])
        print(f"... and {len(symbol_list) - 20} more")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
