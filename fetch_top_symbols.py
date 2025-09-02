#!/usr/bin/env python3
"""
Fetch top 50 most liquid USDT perpetual symbols from Bybit
"""
import requests
import yaml

def fetch_top_symbols():
    """Fetch top symbols by 24h volume"""
    
    url = "https://api.bybit.com/v5/market/tickers"
    params = {
        "category": "linear"  # USDT perpetual
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['retCode'] != 0:
            print(f"Error: {data['retMsg']}")
            return None
        
        # Filter USDT pairs and sort by 24h volume
        usdt_pairs = []
        for ticker in data['result']['list']:
            symbol = ticker['symbol']
            if symbol.endswith('USDT') and not symbol.endswith('USDT-'):  # Exclude dated futures
                usdt_pairs.append({
                    'symbol': symbol,
                    'volume24h': float(ticker['volume24h']),
                    'turnover24h': float(ticker['turnover24h'])
                })
        
        # Sort by 24h turnover (USD value)
        usdt_pairs.sort(key=lambda x: x['turnover24h'], reverse=True)
        
        # Get top 50
        top_50 = usdt_pairs[:50]
        
        print("Top 50 USDT Perpetual Symbols by Volume:\n")
        symbols_list = []
        for i, pair in enumerate(top_50, 1):
            print(f"{i:2}. {pair['symbol']:12} Volume: ${pair['turnover24h']/1e6:.1f}M")
            symbols_list.append(pair['symbol'])
        
        return symbols_list
        
    except Exception as e:
        print(f"Failed to fetch symbols: {e}")
        return None

def fetch_all_specs(symbols):
    """Fetch specifications for all symbols"""
    
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
        
        specs = {}
        for item in data['result']['list']:
            symbol = item['symbol']
            if symbol in symbols:
                specs[symbol] = {
                    'qty_step': float(item['lotSizeFilter']['qtyStep']),
                    'min_qty': float(item['lotSizeFilter']['minOrderQty']),
                    'tick_size': float(item['priceFilter']['tickSize']),
                    'max_leverage': float(item['leverageFilter']['maxLeverage'])
                }
        
        return specs
        
    except Exception as e:
        print(f"Failed to fetch specs: {e}")
        return None

def update_config(symbols, specs):
    """Update config.yaml with new symbols"""
    
    # Read current config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update symbols list
    config['trade']['symbols'] = symbols
    
    # Update symbol_meta with specs
    config['symbol_meta'] = {
        'default': {
            'qty_step': 0.001,
            'min_qty': 0.001,
            'tick_size': 0.1
        }
    }
    
    for symbol, spec in specs.items():
        config['symbol_meta'][symbol] = spec
    
    # Write updated config
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Updated config.yaml with {len(symbols)} symbols")

if __name__ == "__main__":
    print("Fetching top 50 symbols from Bybit...\n")
    
    symbols = fetch_top_symbols()
    if symbols:
        print(f"\n📊 Fetching specifications for {len(symbols)} symbols...")
        specs = fetch_all_specs(symbols)
        
        if specs:
            print(f"✅ Got specifications for {len(specs)} symbols")
            update_config(symbols, specs)
            print("\nBot will now monitor 50 symbols!")
        else:
            print("Failed to fetch specifications")
    else:
        print("Failed to fetch symbols")