import yaml
import pandas as pd
import numpy as np
import requests
import time
import random

# CONFIG
TEST_SIZE = 10  # Top 10 symbols
DAYS = 90
BASE_URL = "https://api.bybit.com"

def fetch_klines(symbol, interval, days):
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    all_candles = []
    current_end = end_ts
    max_req = 50 
    while current_end > start_ts and max_req > 0:
        max_req -= 1
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=5)
            data = resp.json().get('result', {}).get('list', [])
            if not data: break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            time.sleep(0.05)
        except: break
    if not all_candles: return None
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume']: df[c] = df[c].astype(float)
    df = df.sort_values('start').reset_index(drop=True)
    return df

def generate_random_data(df):
    """
    Create 'Surrogate Data' by shuffling returns.
    Preserves: Volatility, Distribution.
    Destroys: Trends, Patterns, Serial Correlation.
    """
    df_rand = df.copy()
    
    # Calculate returns
    df_rand['ret'] = df_rand['close'].pct_change()
    df_rand['range_pct'] = (df_rand['high'] - df_rand['low']) / df_rand['close']
    
    # Shuffle returns
    returns = df_rand['ret'].dropna().values
    np.random.shuffle(returns)
    
    # Reconstruct price path
    start_price = df['close'].iloc[0]
    new_closes = [start_price]
    for r in returns:
        new_closes.append(new_closes[-1] * (1 + r))
        
    # Pad or trim
    if len(new_closes) < len(df):
        new_closes.extend([new_closes[-1]] * (len(df) - len(new_closes)))
    else:
        new_closes = new_closes[:len(df)]
        
    df_rand['close'] = new_closes
    # Approximate High/Low using preserved range volatility
    df_rand['open'] = df_rand['close'].shift(1).fillna(start_price)
    ranges = df_rand['range_pct'].values
    # Randomly shuffle ranges too so volatility isn't localized
    np.random.shuffle(ranges)
    
    # Simple HL construction (not perfect candles but good enough for trend logic)
    df_rand['high'] = df_rand['close'] * (1 + ranges/2)
    df_rand['low'] = df_rand['close'] * (1 - ranges/2)
    
    return df_rand

def prepare_indicators(df):
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    # EMA
    df['ema'] = df['close'].ewm(span=200, adjust=False).mean()
    # ATR
    h, l, c_prev = df['high'], df['low'], df['close'].shift()
    tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    # Pivots
    def get_pivots(src, L=3, R=3):
        pivots = np.full(len(src), np.nan)
        for i in range(L, len(src)-R):
            window = src[i-L:i+R+1]
            if src[i] == max(window) and list(window).count(src[i])==1: pivots[i] = src[i]
            if src[i] == min(window) and list(window).count(src[i])==1: pivots[i] = src[i]
        return pivots
    df['pivot_high'] = get_pivots(df['close'].values)
    df['pivot_low'] = get_pivots(df['close'].values)
    return df

def run_strategy(df, config):
    # Simple logic injection
    c = df['close'].values; ema = df['ema'].values; rsi = df['rsi'].values
    ph = df['pivot_high'].values; pl = df['pivot_low'].values
    atr = df['atr'].values
    
    trades = []
    
    # Very Simplified Detection Loop for Speed
    # (Matches bot logic roughly)
    for i in range(200, len(df)-10):
        # BULL
        if c[i] > ema[i]: 
            # Check last 50 for 2 pivot lows
            found_p = []
            for j in range(i-3, i-50, -1):
                if not np.isnan(pl[j]): found_p.append(j)
                if len(found_p) >= 2: break
            
            if len(found_p) == 2:
                curr, prev = found_p[0], found_p[1]
                if (i - curr) <= 10:
                    # Regular Bull
                    if pl[curr] < pl[prev] and rsi[curr] > rsi[prev]:
                        # EXECUTE
                        entry = df['close'].iloc[i+1] # Breakout/Close approx
                        sl_dist = atr[i] * config['atr_mult']
                        sl = entry - sl_dist
                        tp = entry + (sl_dist * config['rr'])
                        
                        # Check outcome (on 1H for speed, less precise but valid for comparison)
                        outcome = 0 # 0=running
                        for k in range(i+2, len(df)):
                            if df['low'].iloc[k] <= sl: outcome = -1.0; break
                            if df['high'].iloc[k] >= tp: outcome = config['rr']; break
                        trades.append(outcome)

        # BEAR logic omitted for brevity, symmetry applies
        
    return sum(trades)

def main():
    print("🔥 VALIDATING NULL HYPOTHESIS (RANDOM DATA TEST)...")
    print("Goal: Verify that Strategy FAILS on random data (proving it detects REAL patterns).")
    
    with open('config_combined.yaml', 'r') as f:
        conf = yaml.safe_load(f)
        
    top_10 = sorted(conf['symbols'].items(), key=lambda x: x[1]['expected_avg_r'], reverse=True)[:TEST_SIZE]
    
    results = []
    
    for sym, cfg in top_10:
        print(f"Testing {sym}...")
        df_real = fetch_klines(sym, '60', DAYS)
        if df_real is None: continue
        
        # 1. Test REAL Data
        df_real = prepare_indicators(df_real)
        r_real = run_strategy(df_real, cfg)
        
        # 2. Test RANDOM Data (Surrogate)
        # We run 3 random iterations to average luck
        rand_rs = []
        for _ in range(3):
            df_rand = generate_random_data(df_real)
            df_rand = prepare_indicators(df_rand)
            rand_rs.append(run_strategy(df_rand, cfg))
            
        avg_rand_r = sum(rand_rs) / len(rand_rs)
        
        diff = r_real - avg_rand_r
        verdict = "✅ VALID" if r_real > avg_rand_r else "❌ SUSPICIOUS"
        
        print(f"  REAL: {r_real:+.1f}R | RANDOM (Avg): {avg_rand_r:+.1f}R | Difference: {diff:+.1f}R | {verdict}")
        results.append(diff)
        
    avg_diff = sum(results) / len(results)
    print("\n=== CONCLUSION ===")
    print(f"Average 'Alpha' (Real - Random): {avg_diff:+.2f}R")
    if avg_diff > 5:
        print("✅ PASSED: Strategy systematically exploits market structure.")
    else:
        print("❌ FAILED: Strategy cannot distinguish real trends from noise.")

if __name__ == "__main__":
    main()
