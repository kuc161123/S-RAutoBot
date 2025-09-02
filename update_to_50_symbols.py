#!/usr/bin/env python3
"""
Update config with 50 verified Bybit USDT perpetual symbols
"""
import requests
import yaml

# Top 50 most liquid and established symbols on Bybit
TOP_50_SYMBOLS = [
    # Majors (Top 10)
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
    
    # Large Caps (10-20)
    "LINKUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT",
    "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT",
    
    # Mid Caps (20-30)
    "APTUSDT", "FTMUSDT", "FILUSDT", "SANDUSDT", "AXSUSDT",
    "MANAUSDT", "GALAUSDT", "ICPUSDT", "THETAUSDT", "ALGOUSDT",
    
    # Popular Trading Pairs (30-40)
    "CRVUSDT", "MKRUSDT", "AAVEUSDT", "GMTUSDT", "APEUSDT",
    "MASKUSDT", "ENJUSDT", "FLOWUSDT", "CHZUSDT", "GRTUSDT",
    
    # High Volume Alts (40-50)
    "1000PEPEUSDT", "BLURUSDT", "FLOKIUSDT", "WLDUSDT", "SSVUSDT",
    "PENDLEUSDT", "CYBERUSDT", "ARKMUSDT", "FETUSDT", "AGIXUSDT"
]

def fetch_all_specs():
    """Fetch specifications for all symbols"""
    
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {
        "category": "linear",
        "status": "Trading"
    }
    
    try:
        print("Fetching symbol specifications from Bybit...")
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['retCode'] != 0:
            print(f"Error: {data['retMsg']}")
            return None
        
        # Get all available symbols
        available = set()
        specs = {}
        
        for item in data['result']['list']:
            symbol = item['symbol']
            available.add(symbol)
            
            if symbol in TOP_50_SYMBOLS:
                specs[symbol] = {
                    'qty_step': float(item['lotSizeFilter']['qtyStep']),
                    'min_qty': float(item['lotSizeFilter']['minOrderQty']),
                    'tick_size': float(item['priceFilter']['tickSize']),
                    'max_leverage': float(item['leverageFilter']['maxLeverage'])
                }
        
        # Filter to only available symbols
        verified_symbols = [s for s in TOP_50_SYMBOLS if s in available]
        
        print(f"\n✅ Verified {len(verified_symbols)} symbols are available on Bybit")
        
        # Show which ones aren't available (if any)
        not_available = [s for s in TOP_50_SYMBOLS if s not in available]
        if not_available:
            print(f"⚠️  Not available: {', '.join(not_available)}")
        
        return verified_symbols, specs
        
    except Exception as e:
        print(f"Failed to fetch specs: {e}")
        return None, None

def update_config(symbols, specs):
    """Update config.yaml with verified symbols"""
    
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
    for symbol, spec in specs.items():
        config['symbol_meta'][symbol] = spec
    
    # Write updated config
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Updated config.yaml with {len(symbols)} verified symbols")

if __name__ == "__main__":
    verified_symbols, specs = fetch_all_specs()
    
    if verified_symbols and specs:
        print(f"\n📊 Symbol Specifications:")
        for symbol in verified_symbols[:10]:  # Show first 10
            if symbol in specs:
                s = specs[symbol]
                print(f"{symbol:12} Leverage: {s['max_leverage']:5.0f}x  Min: {s['min_qty']:8.3f}  Step: {s['qty_step']:8.3f}")
        
        update_config(verified_symbols, specs)
        print("\n🚀 Bot will now monitor 50 verified symbols!")
    else:
        print("Failed to verify symbols")