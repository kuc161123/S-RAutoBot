import requests
import pandas as pd
import time
import sys

# Top 20 "Unicorns" from previous step
SYMBOLS = [
    'XAUTUSDT', 'KNCUSDT', 'WIFUSDT', 'USELESSUSDT', '1000000MOGUSDT',
    'WAXPUSDT', 'MNTUSDT', 'SCUSDT', 'SNTUSDT', 'SNXUSDT',
    'DRIFTUSDT', 'ZKUSDT', 'PUMPFUNUSDT', 'POLYXUSDT', 'RVNUSDT',
    'XVSUSDT', 'TONUSDT', 'SUSDT', 'WCTUSDT', 'XPLUSDT'
]

BASE_URL = "https://api.bybit.com"

print(f"{'SYMBOL':<15} {'24H_VOL($)':<15} {'SPREAD(%)':<10} {'STATUS':<10}")
print("-" * 55)

params = {'category': 'linear'}
for sym in SYMBOLS:
    try:
        # Get Ticker Info
        resp = requests.get(f"{BASE_URL}/v5/market/tickers", params={'category': 'linear', 'symbol': sym})
        data = resp.json().get('result', {}).get('list', [])
        
        if not data:
            print(f"{sym:<15} {'N/A':<15} {'N/A':<10} {'❌ ERROR'}")
            continue
            
        ticker = data[0]
        turnover = float(ticker.get('turnover24h', 0))
        bid = float(ticker.get('bid1Price', 0))
        ask = float(ticker.get('ask1Price', 0))
        
        if bid == 0:
            spread_pct = 0
        else:
            spread_pct = ((ask - bid) / bid) * 100
            
        # Evaluation Logic
        status = "✅ OK"
        if turnover < 1_000_000: # Less than $1M volume is risky
            status = "⚠️ LOW VOL"
        if spread_pct > 0.06: # Spread > 0.06% (3x our fee assumption)
            status = "⚠️ HI SPREAD"
        if turnover < 500_000:
            status = "❌ ILLIQUID"
            
        print(f"{sym:<15} {turnover:<15,.0f} {spread_pct:<10.3f} {status}")
        
    except Exception as e:
        print(f"{sym:<15} error: {e}")
        
    time.sleep(0.1)
