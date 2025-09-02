#!/usr/bin/env python3
"""
Update config with top 50 coins by market cap that are available on Bybit
"""
import requests
import yaml
import time

def get_top_by_marketcap():
    """Get top 100 coins by market cap from CoinGecko"""
    print("Fetching top coins by market cap from CoinGecko...")
    
    # CoinGecko public API (no key needed for basic requests)
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "sparkline": False
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        # Map common symbols to Bybit format
        symbol_map = {
            "MATIC": "POL",  # Polygon rebrand
            "FTT": None,     # Delisted
            "LUNA": None,    # Old Luna
            "UST": None,     # Stablecoin
            "BUSD": None,    # Stablecoin
            "USDC": None,    # Stablecoin
            "USDT": None,    # Stablecoin
            "DAI": None,     # Stablecoin
            "TUSD": None,    # Stablecoin
            "FRAX": None,    # Stablecoin
            "GUSD": None,    # Stablecoin
            "USDP": None,    # Stablecoin
            "USDD": None,    # Stablecoin
            "WBTC": None,    # Wrapped
            "WETH": None,    # Wrapped
            "STETH": None,   # Liquid staking
            "LEO": None,     # Not on Bybit
            "TON": "TONUSDT", # Special format
            "PEPE": "1000PEPEUSDT",  # 1000x format
            "BONK": "1000BONKUSDT",  # 1000x format
            "FLOKI": "1000FLOKIUSDT", # 1000x format
            "SHIB": "1000SHIBUSDT",   # 1000x format
        }
        
        top_symbols = []
        for coin in data:
            symbol = coin['symbol'].upper()
            
            # Check if we need to map the symbol
            if symbol in symbol_map:
                mapped = symbol_map[symbol]
                if mapped:
                    top_symbols.append(mapped)
            else:
                # Standard USDT format
                top_symbols.append(f"{symbol}USDT")
        
        print(f"Got {len(top_symbols)} potential symbols from top 100 by market cap")
        return top_symbols[:80]  # Get extra to ensure we have 50 valid ones
        
    except Exception as e:
        print(f"Failed to fetch from CoinGecko: {e}")
        return None

def get_bybit_available():
    """Get all available USDT perpetuals on Bybit"""
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
        
        available = set()
        specs = {}
        
        for item in data['result']['list']:
            symbol = item['symbol']
            if symbol.endswith('USDT'):
                available.add(symbol)
                specs[symbol] = {
                    'qty_step': float(item['lotSizeFilter']['qtyStep']),
                    'min_qty': float(item['lotSizeFilter']['minOrderQty']),
                    'tick_size': float(item['priceFilter']['tickSize']),
                    'max_leverage': float(item['leverageFilter']['maxLeverage'])
                }
        
        return available, specs
        
    except Exception as e:
        print(f"Failed to fetch from Bybit: {e}")
        return None, None

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
    # Get top coins by market cap
    top_coins = get_top_by_marketcap()
    
    if not top_coins:
        # Fallback to predefined top 50 by market cap
        print("\nUsing predefined top 50 by market cap...")
        top_coins = [
            "BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT",
            "DOGEUSDT", "ADAUSDT", "TRXUSDT", "AVAXUSDT", "TONUSDT",
            "LINKUSDT", "SUIUSDT", "XLMUSDT", "HBARUSDT", "1000SHIBUSDT",
            "DOTUSDT", "BCHUSDT", "POLUSDT", "UNIUSDT", "LTCUSDT",
            "NEARUSDT", "KASUSDT", "APTUSDT", "ICPUSDT", "ENAUSDT",
            "1000PEPEUSDT", "FETUSDT", "RENDERUSDT", "FILUSDT", "ARBUSDT",
            "THETAUSDT", "FTMUSDT", "OPUSDT", "INJUSDT", "SEIUSDT",
            "WLDUSDT", "ATOMUSDT", "IMXUSDT", "TIAUSDT", "PYTH",
            "MKRUSDT", "ORDIUSDT", "GRTUSDT", "RUNEUSDT", "AAVEUSDT",
            "ALGOUSDT", "FLOWUSDT", "QNTUSDT", "JUPUSDT", "FLOKIUSDT"
        ]
    
    # Get available symbols from Bybit
    print("\nFetching available symbols from Bybit...")
    available, specs = get_bybit_available()
    
    if available and specs:
        # Filter to only available symbols
        valid_symbols = []
        not_available = []
        
        for symbol in top_coins:
            if symbol in available:
                valid_symbols.append(symbol)
            else:
                not_available.append(symbol)
            
            # Stop when we have 50
            if len(valid_symbols) >= 50:
                break
        
        print(f"\n✅ Found {len(valid_symbols)} valid symbols from top market cap coins")
        
        if not_available[:10]:  # Show first 10 not available
            print(f"⚠️  Not available on Bybit: {', '.join(not_available[:10])}")
        
        # Take exactly 50
        final_symbols = valid_symbols[:50]
        
        print("\n📊 Top 50 by Market Cap (available on Bybit):")
        for i, symbol in enumerate(final_symbols, 1):
            spec = specs.get(symbol, {})
            leverage = spec.get('max_leverage', 'N/A')
            print(f"{i:2}. {symbol:15} Max Leverage: {leverage}x")
        
        update_config(final_symbols, specs)
        print("\n🚀 Bot will now monitor top 50 coins by market cap!")
    else:
        print("Failed to verify symbols with Bybit")