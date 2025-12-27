import requests
import pandas as pd
import time
from datetime import datetime
import os

def fetch_bybit_15m_data(symbol="BTCUSDT", years=3):
    print(f"Fetching {years} years of {symbol} 15m data...")
    url = "https://api.bybit.com/v5/market/kline"
    end_time = int(time.time() * 1000)
    start_time = end_time - (years * 365 * 24 * 60 * 60 * 1000)
    
    all_data = []
    current_end = end_time
    
    while current_end > start_time:
        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": "15",
            "limit": 1000,
            "end": current_end
        }
        try:
            r = requests.get(url, params=params).json()
            if r['retCode'] != 0 or not r['result']['list']:
                break
            
            klines = r['result']['list']
            all_data.extend(klines)
            
            # Print progress every 10k bars
            if len(all_data) % 10000 == 0:
                print(f"  Collected {len(all_data)} bars...")
                
            current_end = int(klines[-1][0]) - 1
            if len(klines) < 1000:
                break
            time.sleep(0.02) # Respect rate limits
        except Exception as e:
            print(f"Error: {e}")
            break
            
    if not all_data:
        return None
        
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
    df = df.sort_values('timestamp').drop_duplicates().reset_index(drop=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    filename = f"quantedge/{symbol}_15m_{years}y.parquet"
    df.to_parquet(filename)
    print(f"Saved {len(df)} bars to {filename}")
    return df

if __name__ == "__main__":
    fetch_bybit_15m_data(symbol="BTCUSDT", years=3)
