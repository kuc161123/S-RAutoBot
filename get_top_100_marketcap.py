#!/usr/bin/env python3
"""
Get top 100 coins by market cap that are available on Bybit
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

# Extended list - Top 150+ coins by market cap to ensure we get 100 available ones
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
    
    # 51-60
    "LDOUSDT", "CRVUSDT", "SANDUSDT", "MANAUSDT", "AXSUSDT",
    "GALAUSDT", "GMTUSDT", "APEUSDT", "CHZUSDT", "MASKUSDT",
    
    # 61-70
    "ENJUSDT", "BLURUSDT", "CYBERUSDT", "ARKMUSDT", "PENDLEUSDT",
    "ETCUSDT", "MNTUSDT", "CROUSDT", "IPUSDT", "HYPEUSDT",
    
    # 71-80
    "PENGUUSDT", "SKYUSDT", "BONKUSDT", "WIFUSDT", "TAUSDT",
    "MUSDT", "FARTCOINUSDT", "WLFIUSDT", "TRUMPUSDT", "PUMPFUNUSDT",
    
    # 81-90
    "LRCUSDT", "MAVUSDT", "AI16ZUSDT", "DOLOUSDT", "JELLYJELLYUSDT",
    "POPCATUSDT", "RNDRUSDT", "EGLDUSDT", "SNXUSDT", "CFXUSDT",
    
    # 91-100
    "ZECUSDT", "COMPUSDT", "DASHUSDT", "ZILLIUSDT", "GMXUSDT",
    "1INCHUSDT", "SUSHIUSDT", "BALANCERUSDT", "YFIUSDT", "UMAUSDT",
    
    # 101-110 (extras)
    "KSMUSDT", "LPTUSDT", "ANKRUSDT", "ZRXUSDT", "BATUSDT",
    "STORJUSDT", "CTCUSDT", "KNCUSDT", "BANDUSDT", "OCEANUSDT",
    
    # 111-120
    "IOSTUSDT", "CELOUSDT", "HOTUSDT", "SCUSDT", "ZENUSDT",
    "ONTUSDT", "RSRUSDT", "CKBUSDT", "HNTUSDT", "KLAYUSDT",
    
    # 121-130
    "WAVESUSDT", "FTXUSDT", "LUNAUSDT", "SKLUSDT", "SXPUSDT",
    "KAVASLPUSDT", "COTIUSDT", "ALICEUSDT", "DYDXUSDT", "GALUSDT",
    
    # 131-140
    "C98USDT", "TLMUSDT", "AXLUSDT", "QTUMUSDT", "OMGUSDT",
    "ICXUSDT", "BLZUSDT", "REEFUSDT", "SFPUSDT", "XVSUSDT",
    
    # 141-150
    "ENSUSDT", "PEOPLEUSDT", "ANTUSDT", "ROSEUSDT", "DUSKUSDT",
    "IMOUSDT", "MDTUSDT", "PUNDIXUSDT", "TRBUSDT", "ACHUSDT"
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
    
    # Build final list of 100 symbols
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
        
        # Stop at 100
        if len(final_symbols) >= 100:
            break
    
    # If we don't have 100 yet, add more available symbols
    if len(final_symbols) < 100:
        print(f"\n⚠️  Only found {len(final_symbols)} from priority list, adding more available symbols...")
        for symbol in available_specs:
            if symbol not in final_symbols and symbol.endswith('USDT'):
                final_symbols.append(symbol)
                if len(final_symbols) >= 100:
                    break
    
    print(f"\n✅ Selected {len(final_symbols)} symbols")
    
    if not_available:
        print(f"\n⚠️  Not available on Bybit (skipped): {', '.join(not_available[:10])}")
    
    print("\n📊 Top 100 Coins (Available on Bybit):")
    print("=" * 70)
    
    for i, symbol in enumerate(final_symbols, 1):
        spec = available_specs[symbol]
        # Clean display name
        display = symbol.replace("USDT", "").replace("1000", "")
        leverage = spec['max_leverage']
        
        # Show in columns for better readability (4 columns)
        if i % 4 == 0:
            print(f"{i:3}. {display:12} ({leverage:3.0f}x)")
        else:
            print(f"{i:3}. {display:12} ({leverage:3.0f}x)", end="  ")
    
    if len(final_symbols) % 4 != 0:
        print()  # New line if not divisible by 4
    
    print("=" * 70)
    
    # Show some stats
    specs_for_final = {s: available_specs[s] for s in final_symbols}
    max_leverages = [s['max_leverage'] for s in specs_for_final.values()]
    
    print(f"\n📈 Statistics:")
    print(f"   • Total symbols: {len(final_symbols)}")
    print(f"   • Highest leverage: {max(max_leverages):.0f}x")
    print(f"   • Lowest leverage: {min(max_leverages):.0f}x")
    print(f"   • Average leverage: {sum(max_leverages)/len(max_leverages):.1f}x")
    
    # Update config automatically
    update_config(final_symbols, specs_for_final)
    print("\n🚀 Bot will now monitor top 100 coins!")