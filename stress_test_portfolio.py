import yaml
import pandas as pd
import requests
import numpy as np
import time
import concurrent.futures
import random

# CONFIG
TEST_SIZE = 50  # Test top 50 symbols (Most influential)
DAYS = 90

STRESS_LEVELS = {
    'Level 1 (Normal)':  {'slip': 0.0003, 'fee': 0.0006, 'fail_rate': 0.0},
    'Level 2 (Hard)':    {'slip': 0.0005, 'fee': 0.00075, 'fail_rate': 0.05}, # 2.5x Slip, Higher Fee, 5% Missed
    'Level 3 (Torture)': {'slip': 0.0010, 'fee': 0.0010, 'fail_rate': 0.15}   # 5x Slip, High Fee, 15% Missed
}

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

def prepare_data(df):
    df['delta'] = df['close'].diff()
    gain = df['delta'].where(df['delta'] > 0, 0).rolling(14).mean()
    loss = -df['delta'].where(df['delta'] < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
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

def run_simulation(symbol, df_1h, df_5m, config, level_params):
    slip = level_params['slip']
    fee = level_params['fee']
    fail_rate = level_params['fail_rate']
    
    rr = config['rr']
    atr_mult = config['atr_mult']
    div_type = config['divergence_type']
    
    trades = []
    
    # SCAN loop (Simplified)
    for i in range(200, len(df_1h)-10):
        # ... (Same logic as main bot) ...
        # For brevity, I'll simulate based on known entries from prior research logic
        # But to be robust, we must re-detect.
        pass
        
    # Since re-writing the full engine in a script is verbose,
    # I will assume we can reuse the logic or simplify.
    # To keep it "concise" but "robust", I'll implement the critical check:
    # 1. Detect Signal
    # 2. Apply Stress
    
    signals = []
    # ... Detection Logic ...
    # (Copying lightweight detection logic)
    c = df_1h['close'].values; rsi = df_1h['rsi'].values; ema = df_1h['ema'].values
    ph = df_1h['pivot_high'].values; pl = df_1h['pivot_low'].values
    
    for i in range(205, len(df_1h)-10):
        # REG_BULL Only for this example (if configured)
        if 'BULL' in div_type and c[i] > ema[i]:
            # ... pivot check ...
            pass
            
    # NOTE: Re-implementing the full detector is complex.
    # ALTERNATIVE: Use the debug data or just run strict re-eval.
    return 0 # Placeholder

# Let's write the FULL simplified verification logic correctly
def full_verify(sym, conf):
    # Fetch
    df_1h = fetch_klines(sym, '60', DAYS)
    df_5m = fetch_klines(sym, '5', DAYS)
    if df_1h is None or df_5m is None: return {'sym': sym, 'L1': 0, 'L2': 0, 'L3': 0}
    
    df_1h = prepare_data(df_1h)
    
    # DETECT
    signals = []
    c = df_1h['close'].values
    ema = df_1h['ema'].values
    rsi = df_1h['rsi'].values
    start_times = df_1h['start'].values
    
    # Very basic pivot
    for i in range(50, len(c)-5):
        # We only care about matching the configured div type roughly
        # If config is REG_BULL, scan for it.
        pass
        # To save code space, I will trust the "market_wide_results" row and
        # just apply the slippage penalty to the expected outcome.
        # Wait, that's not a simulation.
        # I MUST re-run the logic.
        
    # ...
    
    return {'sym': sym, 'L1': 100, 'L2': 50, 'L3': 10} # Mock return for now

def main():
    print("🔥 LOADING CONFIG...")
    with open('config_combined.yaml', 'r') as f:
        conf = yaml.safe_load(f)
        
    top_50 = sorted(conf['symbols'].items(), key=lambda x: x[1]['expected_avg_r'], reverse=True)[:TEST_SIZE]
    
    print(f"🔥 STRESS TESTING TOP {TEST_SIZE} SYMBOLS...")
    
    # We will use the 'research_market_wide.py' logic but import it to avoid rewriting
    # Or just modify the costs in a new run.
    # Actually, importing 'research_market_wide' is smart.
    
    from research_market_wide import optimize_symbol, detect_signals, prepare_1h_data, fetch_klines, execute_trade
    
    results = []
    
    for sym, cfg in top_50:
        print(f"Testing {sym}...")
        # Get Data
        df_1h = fetch_klines(sym, '60', DAYS)
        df_5m = fetch_klines(sym, '5', DAYS)
        if df_1h is None: continue
        
        df_1h = prepare_1h_data(df_1h)
        signals = detect_signals(df_1h)
        target_signals = [s for s in signals if s['type'] == cfg['divergence_type']]
        
        row = {'symbol': sym}
        
        for level, params in STRESS_LEVELS.items():
            slip = params['slip']
            fee = params['fee']
            fail = params['fail_rate']
            
            # Custom Execute with Stress
            total_r = 0
            for sig in target_signals:
                # Fail chance
                if random.random() < fail: continue
                
                # We need to hack execute_trade to accept custom slippage
                # Since execute_trade uses global constants, we can't easily change them without restart.
                # SO: We will reimplement 'execute_trade_stress'
                
                # (Defining execute_trade_stress inline for brevity)
                res = execute_trade_stress(sig, df_1h, df_5m, cfg['rr'], cfg['atr_mult'], slip, fee)
                if res: total_r += res
                
            row[level] = total_r
            
        results.append(row)
        print(f"  {sym}: Norm={row['Level 1 (Normal)']:.1f}R | Torture={row['Level 3 (Torture)']:.1f}R")
        
    # Aggregate
    df_res = pd.DataFrame(results)
    print("\n=== FINAL STRESS TEST RESULTS ===")
    print(df_res.sum(numeric_only=True))

def execute_trade_stress(signal, df_1h, df_5m, rr, atr_mult, slip, fee):
    # Simplified version of the bot's logic with custom costs
    # ...
    # (Copy logic but use 'slip' and 'fee')
    conf_idx = signal['conf_idx']
    side = signal['side']
    swing = signal['swing']
    
    # Wait for breakout
    entry_price = None
    outcome = None
    
    # 1. Find Entry
    entry_time = None
    sl_dist = 0
    
    for i in range(1, 7):
        if conf_idx + i + 1 >= len(df_1h): break
        c = df_1h.iloc[conf_idx + i]
        triggered = (c['close'] > swing) if side == 'long' else (c['close'] < swing)
        
        if triggered:
            entry_candle = df_1h.iloc[conf_idx + i + 1]
            entry_time = entry_candle['start']
            sl_dist = c['atr'] * atr_mult
            raw_entry = entry_candle['open']
            
            if side == 'long':
                entry_price = raw_entry * (1 + slip)
                sl = entry_price - sl_dist
                tp = entry_price + (sl_dist * rr)
            else:
                entry_price = raw_entry * (1 - slip)
                sl = entry_price + sl_dist
                tp = entry_price - (sl_dist * rr)
            break
            
    if not entry_price: return None
    
    # 2. Check Outcome (5m)
    sub = df_5m[df_5m['start'] >= entry_time]
    for r in sub.itertuples():
        if side == 'long':
            if r.low <= sl: outcome='loss'; break
            if r.high >= tp: outcome='win'; break
        else:
            if r.high >= sl: outcome='loss'; break
            if r.low <= tp: outcome='win'; break
            
    if not outcome: return None
    
    r_val = rr if outcome == 'win' else -1.0
    
    # Fee Drag
    risk = abs(entry_price - sl)
    if risk == 0: return -1.0
    cost = (fee * 2 * entry_price) / risk
    
    return r_val - cost

if __name__ == "__main__":
    main()
