#!/usr/bin/env python3
"""
Fetch all available USDT perpetual symbols from Bybit
"""
import requests
import json

def fetch_all_bybit_symbols():
    """Get all USDT perpetual symbols from Bybit API sorted by 24h Turnover (Liquidity)"""
    url = "https://api.bybit.com/v5/market/tickers"
    params = {
        "category": "linear",  # Linear perpetuals
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['retCode'] != 0:
            print(f"Error from Bybit API: {data['retMsg']}")
            return []
        
        # Filter for USDT perpetuals and sort by turnover
        tickers = []
        for item in data['result']['list']:
            symbol = item['symbol']
            if symbol.endswith('USDT'):
                turnover = float(item.get('turnover24h', 0))
                tickers.append({'symbol': symbol, 'turnover': turnover})
        
        # Sort by turnover descending (High Liquidity -> Low Liquidity)
        tickers.sort(key=lambda x: x['turnover'], reverse=True)
        
        sorted_symbols = [t['symbol'] for t in tickers]
        return sorted_symbols
        
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

if __name__ == "__main__":
    print("Fetching all USDT perpetual symbols from Bybit (sorted by 24h Turnover)...")
    symbols = fetch_all_bybit_symbols()
    
    print(f"\nFound {len(symbols)} tradable USDT perpetual symbols")
    
    # Save to file
    with open('all_bybit_symbols.json', 'w') as f:
        json.dump({'symbols': symbols, 'count': len(symbols)}, f, indent=2)
    
    print(f"✅ Saved to: all_bybit_symbols.json")
    
    # Print first 20 as sample
    print(f"\nTop 20 by Liquidity:")
    for i, sym in enumerate(symbols[:20], 1):
        print(f"  {i}. {sym}")
