#!/usr/bin/env python3
"""
Fetch data for a single symbol and run backtest
Usage: python3 process_one_symbol.py <SYMBOL>
"""
import sys
import json
import requests
from backtest_single_symbol import SingleSymbolOptimizer

def fetch_data(symbol, limit=100000):
    url = "https://api.bybit.com/v5/market/kline"
    base_params = {"category": "linear", "symbol": symbol, "interval": "3", "limit": 1000}
    
    all_candles = []
    end_time = None
    
    print(f"Fetching up to {limit} candles...")
    
    while len(all_candles) < limit:
        params = base_params.copy()
        if end_time:
            params['end'] = end_time
            
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['retCode'] != 0:
                print(f"Error from Bybit API: {data['retMsg']}")
                break
                
            batch = data['result']['list']
            if not batch:
                break
                
            # Process batch
            processed_batch = []
            for item in batch:
                processed_batch.append({
                    'timestamp': int(item[0]),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5])
                })
            
            # Append to all_candles
            all_candles.extend(processed_batch)
            
            # Update end_time for next batch (oldest timestamp - 1ms)
            last_ts = int(batch[-1][0])
            end_time = last_ts - 1
            
            print(f"  Fetched {len(all_candles)} candles...", end='\r')
            
            if len(batch) < 1000:
                break
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
            
    print(f"\nTotal fetched: {len(all_candles)}")
    all_candles.reverse() # Oldest first
    return all_candles

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 process_one_symbol.py <SYMBOL>")
        sys.exit(1)
    
    symbol = sys.argv[1]
    
    print(f"Fetching data for {symbol}...")
    candles = fetch_data(symbol)
    
    if not candles:
        print(f"Failed to fetch data for {symbol}")
        sys.exit(1)
    
    print(f"✓ Got {len(candles)} candles")
    
    # Save to temp file
    temp_file = f'temp_{symbol}.json'
    with open(temp_file, 'w') as f:
        json.dump({symbol: candles}, f)
    
    # Run backtest
    optimizer = SingleSymbolOptimizer(data_file=temp_file)
    result = optimizer.optimize_symbol(symbol)
    
    # Clean up
    import os
    os.remove(temp_file)
    
    if result:
        # Save result
        with open(f'result_{symbol}.json', 'w') as f:
            json.dump({
                'symbol': symbol,
                'win_rate': result['win_rate'],
                'trades': result['trades'],
                'params': result['params']
            }, f, indent=2)
        
        print(f"\n{'='*60}")
        if result['win_rate'] >= 40.0:
            print(f"✅ PROFITABLE: {symbol}")
            print(f"   WR: {result['win_rate']:.1f}%")
            print(f"   Trades: {result['trades']}")
            print(f"\n→ Ready to add to Pro Rules")
        else:
            print(f"❌ REJECTED: {symbol}")
            print(f"   WR: {result['win_rate']:.1f}% (< 40%)")
        print(f"{'='*60}")
