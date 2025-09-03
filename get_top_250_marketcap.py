#!/usr/bin/env python3
"""
Get top 250 coins by market cap that are available on Bybit
Enhanced version with better symbol coverage
"""
import requests
import yaml
import time

def get_bybit_symbols_with_specs():
    """Get all available USDT perpetuals on Bybit with their specs"""
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {
        "category": "linear",
        "status": "Trading"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['retCode'] != 0:
            print(f"Error: {data['retMsg']}")
            return None
        
        available = {}
        for item in data['result']['list']:
            symbol = item['symbol']
            if symbol.endswith('USDT') and not symbol.endswith('-'):
                available[symbol] = {
                    'qty_step': float(item['lotSizeFilter']['qtyStep']),
                    'min_qty': float(item['lotSizeFilter']['minOrderQty']),
                    'tick_size': float(item['priceFilter']['tickSize']),
                    'max_leverage': float(item['leverageFilter']['maxLeverage'])
                }
        
        return available
        
    except Exception as e:
        print(f"Failed to fetch from Bybit: {e}")
        return None

def get_all_bybit_symbols_sorted_by_volume():
    """Get all Bybit symbols sorted by 24h volume for better selection"""
    url = "https://api.bybit.com/v5/market/tickers"
    params = {
        "category": "linear"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['retCode'] != 0:
            return []
        
        symbols_by_volume = []
        for item in data['result']['list']:
            symbol = item['symbol']
            if symbol.endswith('USDT') and 'volume24h' in item:
                try:
                    volume = float(item['volume24h'])
                    symbols_by_volume.append((symbol, volume))
                except:
                    pass
        
        # Sort by volume (highest first)
        symbols_by_volume.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in symbols_by_volume]
        
    except Exception as e:
        print(f"Failed to fetch volumes: {e}")
        return []

# Extended list of top coins by market cap (manually curated + high volume coins)
TOP_MARKETCAP_PRIORITY = [
    # Top 50 by market cap (most important)
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "SUIUSDT",
    "LINKUSDT", "XLMUSDT", "HBARUSDT", "SHIBUSDT", "DOTUSDT",
    "BCHUSDT", "POLUSDT", "UNIUSDT", "LTCUSDT", "NEARUSDT",
    "KASUSDT", "APTUSDT", "ICPUSDT", "ENAUSDT", "PEPEUSDT",
    "FETUSDT", "RENDERUSDT", "FILUSDT", "ARBUSDT", "THETAUSDT",
    "FTMUSDT", "OPUSDT", "INJUSDT", "SEIUSDT", "WLDUSDT",
    "ATOMUSDT", "IMXUSDT", "TIAUSDT", "PYTHUSDT", "MKRUSDT",
    "ORDIUSDT", "GRTUSDT", "RUNEUSDT", "AAVEUSDT", "ALGOUSDT",
    "FLOWUSDT", "QNTUSDT", "JUPUSDT", "FLOKIUSDT", "ONDOUSDT",
    
    # 51-100
    "LDOUSDT", "CRVUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
    "GALAUSDT", "GMTUSDT", "APEUSDT", "CHZUSDT", "MASKUSDT",
    "ENJUSDT", "BLURUSDT", "CYBERUSDT", "ARKMUSDT", "PENDLEUSDT",
    "ETCUSDT", "MNTUSDT", "CROUSDT", "IPUSDT", "HYPEUSDT",
    "PENGUUSDT", "SKYUSDT", "BONKUSDT", "WIFUSDT", "TAUSDT",
    "MUSDT", "FARTCOINUSDT", "WLFIUSDT", "TRUMPUSDT", "PUMPFUNUSDT",
    "LRCUSDT", "MAVUSDT", "AI16ZUSDT", "DOLOUSDT", "JELLYJELLYUSDT",
    "POPCATUSDT", "RNDRUSDT", "EGLDUSDT", "SNXUSDT", "CFXUSDT",
    "ZECUSDT", "COMPUSDT", "DASHUSDT", "ZILLIUSDT", "GMXUSDT",
    "1INCHUSDT", "SUSHIUSDT", "BALANCERUSDT", "YFIUSDT", "UMAUSDT",
]

# Alternative names for some coins
SYMBOL_MAPPINGS = {
    "XRPUSDT": None,
    "SHIBUSDT": "1000SHIBUSDT",
    "PEPEUSDT": "1000PEPEUSDT",
    "FLOKIUSDT": "1000FLOKIUSDT",
    "BONKUSDT": "1000BONKUSDT",
    "POLUSDT": "POLUSDT",
    "TRXUSDT": None,
    "SUIUSDT": None,
    "XLMUSDT": None,
}

def update_config(symbols, specs):
    """Update config.yaml with new symbols"""
    
    # Read current config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update symbols list
    config['trade']['symbols'] = symbols
    
    # Update symbol_meta with specs
    if 'symbol_meta' not in config:
        config['symbol_meta'] = {}
    
    # Keep default
    config['symbol_meta']['default'] = {
        'qty_step': 0.001,
        'min_qty': 0.001,
        'tick_size': 0.1
    }
    
    # Update with fetched specs
    for symbol in symbols:
        if symbol in specs:
            config['symbol_meta'][symbol] = specs[symbol]
    
    # Write updated config
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Updated config.yaml with {len(symbols)} symbols")

if __name__ == "__main__":
    print("Fetching available symbols from Bybit...")
    available_specs = get_bybit_symbols_with_specs()
    
    if not available_specs:
        print("Failed to fetch Bybit symbols")
        exit(1)
    
    print(f"Found {len(available_specs)} total USDT perpetuals on Bybit")
    
    # Get symbols sorted by volume for better selection
    print("\nFetching symbols by volume...")
    symbols_by_volume = get_all_bybit_symbols_sorted_by_volume()
    
    # Build final list of 250 symbols
    final_symbols = []
    not_available = []
    added_from_priority = set()
    
    # First, add from priority list
    for symbol in TOP_MARKETCAP_PRIORITY:
        if symbol in SYMBOL_MAPPINGS:
            mapped = SYMBOL_MAPPINGS[symbol]
            if mapped and mapped in available_specs:
                if mapped not in final_symbols:
                    final_symbols.append(mapped)
                    added_from_priority.add(mapped)
            elif mapped is None:
                not_available.append(symbol)
        elif symbol in available_specs:
            if symbol not in final_symbols:
                final_symbols.append(symbol)
                added_from_priority.add(symbol)
        else:
            not_available.append(symbol)
    
    print(f"Added {len(final_symbols)} from priority list")
    
    # Then add high-volume symbols not in priority list
    if len(final_symbols) < 250:
        for symbol in symbols_by_volume:
            if symbol not in final_symbols and symbol in available_specs:
                # Skip stablecoins and weird tokens
                if any(x in symbol for x in ['USDC', 'BUSD', 'TUSD', 'DAI', 'EUR', 'GBP']):
                    continue
                    
                final_symbols.append(symbol)
                if len(final_symbols) >= 250:
                    break
    
    # If still not enough, add any remaining available symbols
    if len(final_symbols) < 250:
        for symbol in available_specs:
            if symbol not in final_symbols and symbol.endswith('USDT'):
                # Skip stablecoins
                if any(x in symbol for x in ['USDC', 'BUSD', 'TUSD', 'DAI', 'EUR', 'GBP']):
                    continue
                    
                final_symbols.append(symbol)
                if len(final_symbols) >= 250:
                    break
    
    # Limit to exactly 250
    final_symbols = final_symbols[:250]
    
    print(f"\n✅ Selected {len(final_symbols)} symbols")
    
    if not_available:
        print(f"\n⚠️  Not available on Bybit: {', '.join(not_available[:10])}")
    
    print("\n📊 Top 250 Coins (Available on Bybit):")
    print("=" * 80)
    
    # Display in 5 columns for compact view
    for i, symbol in enumerate(final_symbols, 1):
        spec = available_specs[symbol]
        display = symbol.replace("USDT", "").replace("1000", "")
        leverage = spec['max_leverage']
        
        # Show in 5 columns
        if i % 5 == 0:
            print(f"{i:3}. {display:10} ({leverage:3.0f}x)")
        else:
            print(f"{i:3}. {display:10} ({leverage:3.0f}x)", end="  ")
    
    if len(final_symbols) % 5 != 0:
        print()  # New line if not divisible by 5
    
    print("=" * 80)
    
    # Show statistics
    specs_for_final = {s: available_specs[s] for s in final_symbols}
    max_leverages = [s['max_leverage'] for s in specs_for_final.values()]
    
    print(f"\n📈 Statistics:")
    print(f"   • Total symbols: {len(final_symbols)}")
    print(f"   • From priority list: {len(added_from_priority)}")
    print(f"   • From volume ranking: {len(final_symbols) - len(added_from_priority)}")
    print(f"   • Highest leverage: {max(max_leverages):.0f}x")
    print(f"   • Lowest leverage: {min(max_leverages):.0f}x")
    print(f"   • Average leverage: {sum(max_leverages)/len(max_leverages):.1f}x")
    
    # Backup current config
    print("\n📦 Creating backup of current config...")
    with open('config.yaml', 'r') as f:
        backup_config = f.read()
    with open('config_backup_100symbols.yaml', 'w') as f:
        f.write(backup_config)
    print("✅ Backup saved to config_backup_100symbols.yaml")
    
    # Update config
    print("\n⚠️  WARNING: This will update from 100 to 250 symbols")
    print("WebSocket connections will be split to handle >200 subscriptions")
    
    update_config(final_symbols, specs_for_final)
    print("\n🚀 Bot will now monitor 250 symbols!")
    print("\n⚠️  IMPORTANT: If issues occur, restore with:")
    print("   cp config_backup_100symbols.yaml config.yaml")