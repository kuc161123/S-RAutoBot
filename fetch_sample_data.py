import requests
import pandas as pd
import json
import os

def fetch_candles(symbol, interval='15', limit=1000):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    print(f"Fetching {symbol}...")
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['retCode'] != 0:
            print(f"Error fetching {symbol}: {data['retMsg']}")
            return None
            
        candles = []
        for item in data['result']['list']:
            # [startTime, open, high, low, close, volume, turnover]
            candles.append({
                'timestamp': int(item[0]),
                'open': float(item[1]),
                'high': float(item[2]),
                'low': float(item[3]),
                'close': float(item[4]),
                'volume': float(item[5])
            })
        
        # Sort by timestamp (oldest first)
        candles.sort(key=lambda x: x['timestamp'])
        return candles
        
    except Exception as e:
        print(f"Exception fetching {symbol}: {e}")
        return None

def main():
    # All 50+ tradeable symbols
    symbols = [
        # Top 10 Major Coins
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
        'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'TRXUSDT', 'LINKUSDT',
        
        # Top 20 by Volume
        'DOTUSDT', 'MATICUSDT', 'LTCUSDT', 'UNIUSDT', 'ATOMUSDT',
        'XLMUSDT', 'ETCUSDT', 'FILUSDT', 'NEARUSDT', 'APTUSDT',
        
        # High Volatility / Meme Coins
        'PEPEUSDT', 'FLOKIUSDT', 'SHIBUSDT', 'BONKUSDT', 'WIFUSDT',
        
        # DeFi Blue Chips
        'AAVEUSDT', 'COMPUSDT', 'MKRUSDT', 'SUSHIUSDT', 'CRVUSDT',
        
        # Layer 1/2 Protocols
        'OPUSDT', 'ARBUSDT', 'INJUSDT', 'SUIUSDT', 'APTUSDT',
        'RNDRUSDT', 'STXUSDT', 'ICPUSDT', 'ALGOUSDT', 'EGLDUSDT',
        
        # Gaming / Metaverse
        'AXSUSDT', 'SANDUSDT', 'MANAUSDT', 'GALAUSDT', 'ENJUSDT',
        
        # Infrastructure
        'GRTUSDT', 'FTMUSDT', 'ZILUSDT', 'CHZUSDT', 'VETUSDT'
    ]
    
    all_data = {}
    failed = []
    
    print(f"Fetching data for {len(symbols)} symbols...")
    
    for sym in symbols:
        data = fetch_candles(sym, limit=1000)
        if data:
            print(f"✓ Fetched {len(data)} candles for {sym}")
            all_data[sym] = data
        else:
            print(f"✗ Skipping {sym} due to fetch error")
            failed.append(sym)
            
    with open('backtest_data_all50.json', 'w') as f:
        json.dump(all_data, f, indent=2)
        
    print(f"\n{'='*50}")
    print(f"✅ Saved {len(all_data)}/{len(symbols)} symbols to backtest_data_all50.json")
    if failed:
        print(f"⚠️  Failed symbols: {', '.join(failed)}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
