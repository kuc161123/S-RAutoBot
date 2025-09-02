#!/usr/bin/env python3
"""
Get top 50 coins by market cap that are available on Bybit
Combines top market cap coins with Bybit availability
"""
import requests
import yaml

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
            return None, None
        
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

# Top coins by market cap (manually curated from CoinMarketCap/CoinGecko)
# Ordered by market cap as of January 2025
TOP_MARKETCAP_PRIORITY = [
    # Top 10 by market cap
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT",
    "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "SUIUSDT",
    
    # 11-20
    "LINKUSDT", "XLMUSDT", "HBARUSDT", "SHIBUSDT", "DOTUSDT",
    "BCHUSDT", "POLUSDT", "UNIUSDT", "LTCUSDT", "NEARUSDT",
    
    # 21-30
    "KASUSDT", "APTUSDT", "ICPUSDT", "ENAUSDT", "PEPEUSDT",
    "FETUSDT", "RENDERUSDT", "FILUSDT", "ARBUSDT", "THETAUSDT",
    
    # 31-40
    "FTMUSDT", "OPUSDT", "INJUSDT", "SEIUSDT", "WLDUSDT",
    "ATOMUSDT", "IMXUSDT", "TIAUSDT", "PYTHUSDT", "MKRUSDT",
    
    # 41-50
    "ORDIUSDT", "GRTUSDT", "RUNEUSDT", "AAVEUSDT", "ALGOUSDT",
    "FLOWUSDT", "QNTUSDT", "JUPUSDT", "FLOKIUSDT", "ONDOUSDT",
    
    # 51-60 (extras in case some aren't available)
    "LDOUSDT", "CRVUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
    "GALAUSDT", "GMTUSDT", "APEUSDT", "CHZUSDT", "MASKUSDT",
    
    # 61-70
    "ENJUSDT", "BLURUSDT", "CYBERUSDT", "ARKMUSDT", "PENDLEUSDT",
    "ETCUSDT", "MNTUSDT", "CROUSDT", "IPUSDT", "HYPEUSDT",
    
    # 71-80
    "PENGUUSDT", "SKYUSDT", "BONKUSDT", "WIFUSDT", "TAUSDT",
    "MUSDT", "FARTCOINUSDT", "WLFIUSDT", "TRUMPUSDT", "PUMPFUNUSDT"
]

# Alternative names for some coins
SYMBOL_MAPPINGS = {
    "XRPUSDT": None,  # Not available as perpetual on Bybit currently
    "SHIBUSDT": "1000SHIBUSDT",
    "PEPEUSDT": "1000PEPEUSDT",
    "FLOKIUSDT": "1000FLOKIUSDT",
    "BONKUSDT": "1000BONKUSDT",
    "POLUSDT": "POLUSDT",  # Polygon (MATIC renamed to POL)
    "TRXUSDT": None,  # Not available
    "SUIUSDT": None,  # Not available as perpetual
    "XLMUSDT": None,  # Not available
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
    
    # Build final list of 50 symbols
    final_symbols = []
    not_available = []
    
    for symbol in TOP_MARKETCAP_PRIORITY:
        # Check if we need to map the symbol
        if symbol in SYMBOL_MAPPINGS:
            mapped = SYMBOL_MAPPINGS[symbol]
            if mapped and mapped in available_specs:
                if mapped not in final_symbols:
                    final_symbols.append(mapped)
            elif mapped is None:
                not_available.append(symbol)
        elif symbol in available_specs:
            if symbol not in final_symbols:
                final_symbols.append(symbol)
        else:
            not_available.append(symbol)
        
        # Stop at 50
        if len(final_symbols) >= 50:
            break
    
    print(f"\n✅ Selected {len(final_symbols)} symbols from top market cap coins")
    
    if not_available:
        print(f"\n⚠️  Not available on Bybit (skipped): {', '.join(not_available[:10])}")
    
    print("\n📊 Top 50 Coins by Market Cap (Available on Bybit):")
    print("=" * 70)
    
    for i, symbol in enumerate(final_symbols, 1):
        spec = available_specs[symbol]
        # Clean display name
        display = symbol.replace("USDT", "").replace("1000", "")
        leverage = spec['max_leverage']
        
        # Show in columns for better readability
        if i % 2 == 1:
            print(f"{i:2}. {display:10} ({leverage:3.0f}x)", end="    ")
        else:
            print(f"{i:2}. {display:10} ({leverage:3.0f}x)")
    
    if len(final_symbols) % 2 == 1:
        print()  # New line if odd number
    
    print("=" * 70)
    
    # Show some stats
    specs_for_final = {s: available_specs[s] for s in final_symbols}
    max_leverages = [s['max_leverage'] for s in specs_for_final.values()]
    
    print(f"\n📈 Statistics:")
    print(f"   • Highest leverage: {max(max_leverages):.0f}x (BTC, ETH)")
    print(f"   • Lowest leverage: {min(max_leverages):.0f}x")
    print(f"   • Average leverage: {sum(max_leverages)/len(max_leverages):.1f}x")
    
    update_config(final_symbols, specs_for_final)
    print("\n🚀 Bot will now monitor top 50 coins by market cap!")