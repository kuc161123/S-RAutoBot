import requests
import pandas as pd
import time
import sys

BASE_URL = "https://api.bybit.com"
INPUT_FILE = 'market_wide_results.csv'
MIN_VOLUME = 1_000_000

def get_tickers(symbols):
    """Batch fetch tickers (limit 50 per request usually, but Bybit linear allows more sometimes)"""
    # Bybit v5 supports category=linear and no symbol arg to get ALL tickers
    # This is more efficient than batching 500 symbols manually
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/tickers", params={'category': 'linear'})
        data = resp.json().get('result', {}).get('list', [])
        return {item['symbol']: float(item.get('turnover24h', 0)) for item in data}
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return {}

def main():
    print("⏳ Loading candidates...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except:
        print("No results file found.")
        return

    # Get unique symbols that passed robustness check
    candidates = df[df['trades'] >= 10]['symbol'].unique()
    print(f"Total Candidates (Robust Stats): {len(candidates)}")
    
    print("📡 Fetching real-time market volume...")
    ticker_map = get_tickers(candidates)
    
    passed = []
    rejected = []
    
    for sym in candidates:
        vol = ticker_map.get(sym, 0)
        if vol >= MIN_VOLUME:
            passed.append(sym)
        else:
            rejected.append((sym, vol))
            
    print("-" * 30)
    print(f"✅ PASSED LIQUIDITY ($1M+): {len(passed)}")
    print(f"❌ REJECTED (Illiquid):     {len(rejected)}")
    print("-" * 30)
    
    # Save safe list for next step
    with open('safe_symbols.txt', 'w') as f:
        f.write('\n'.join(passed))

if __name__ == "__main__":
    main()
