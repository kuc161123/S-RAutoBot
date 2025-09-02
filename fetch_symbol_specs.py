#!/usr/bin/env python3
"""
Fetch actual symbol specifications from Bybit API
Updates config.yaml with correct qty_step and min_qty values
"""
import requests
import yaml
import json

def fetch_symbol_specs():
    """Fetch symbol specifications from Bybit API"""
    
    url = "https://api.bybit.com/v5/market/instruments-info"
    params = {
        "category": "linear",  # USDT perpetual
        "status": "Trading"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['retCode'] != 0:
            print(f"Error: {data['retMsg']}")
            return None
        
        # Extract specifications for our symbols
        symbols_we_need = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
            "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "POLUSDT", "LINKUSDT",
            "DOTUSDT", "LTCUSDT", "UNIUSDT", "ATOMUSDT", "NEARUSDT",
            "ARBUSDT", "OPUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT"
        ]
        
        specs = {}
        for item in data['result']['list']:
            symbol = item['symbol']
            if symbol in symbols_we_need:
                specs[symbol] = {
                    'qty_step': float(item['lotSizeFilter']['qtyStep']),
                    'min_qty': float(item['lotSizeFilter']['minOrderQty']),
                    'tick_size': float(item['priceFilter']['tickSize']),
                    'max_leverage': float(item['leverageFilter']['maxLeverage'])
                }
                print(f"{symbol}:")
                print(f"  qty_step: {specs[symbol]['qty_step']}")
                print(f"  min_qty: {specs[symbol]['min_qty']}")
                print(f"  tick_size: {specs[symbol]['tick_size']}")
                print(f"  max_leverage: {specs[symbol]['max_leverage']}x")
        
        return specs
        
    except Exception as e:
        print(f"Failed to fetch specs: {e}")
        return None

def update_config(specs):
    """Update config.yaml with correct specifications"""
    
    # Read current config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update symbol_meta section
    if 'symbol_meta' not in config:
        config['symbol_meta'] = {}
    
    for symbol, spec in specs.items():
        config['symbol_meta'][symbol] = spec
    
    # Write updated config
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("\n✅ Updated config.yaml with correct specifications")

if __name__ == "__main__":
    print("Fetching symbol specifications from Bybit API...")
    specs = fetch_symbol_specs()
    
    if specs:
        print(f"\n📊 Found specifications for {len(specs)} symbols")
        
        # Auto-update config
        update_config(specs)
    else:
        print("Failed to fetch specifications")