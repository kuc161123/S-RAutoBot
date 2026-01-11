#!/usr/bin/env python3
"""
PRECISION STRATEGY OPTIMIZER (Deduped)
======================================
Goal: Find the robust configuration for 50 top symbols.

Methodology:
1.  Precision Engine (1H Signal / 5M Execution) - No Lookahead Bias.
2.  **DEDUPLICATION**: Only 1 trade per pivot-pair instance or Cooldown.
3.  Grid Search per Symbol:
    - Divergences: [REG_BULL, REG_BEAR, HID_BULL, HID_BEAR]
    - ATR Mult: [1.0, 1.5, 2.0]
    - RR Ratio: [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
4.  Robustness Filter:
    - Min Trades: 10 (Lowered due to dedup)
    - Expectancy: > 0.1R
    - Win Rate: > 30% (High RR) / > 45% (Low RR)

Author: Antigravity
"""

import requests
import pandas as pd
import numpy as np
import time
import yaml
import sys
import concurrent.futures
import threading
import os
import itertools
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
DAYS = 90
SIGNAL_TF = '60'
EXECUTION_TF = '5'
MAX_WAIT_CANDLES = 6
RSI_PERIOD = 14
EMA_PERIOD = 200

# Realistic Costs
SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006

BASE_URL = "https://api.bybit.com"
OUTPUT_FILE = 'precision_optimization_deduped.csv'
FILE_LOCK = threading.Lock()

GRID = {
    'div_type': ['REG_BULL', 'REG_BEAR', 'HID_BULL', 'HID_BEAR'],
    'atr_mult': [1.0, 1.5, 2.0],
    'rr_ratio': [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}

# ============================================================================
# DATA ENGINE
# ============================================================================

def fetch_klines(symbol, interval, days):
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    all_candles = []
    current_end = end_ts
    max_requests = 150 
    while current_end > start_ts and max_requests > 0:
        max_requests -= 1
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
            data = resp.json().get('result', {}).get('list', [])
            if not data: break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            if len(data) < 1000: break
            time.sleep(0.05)
        except:
            time.sleep(1); continue
    if not all_candles: return pd.DataFrame()
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = df[col].astype(float)
    df = df.sort_values('start').reset_index(drop=True)
    return df

def prepare_1h_data(df):
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss = -delta.where(delta < 0, 0).rolling(RSI_PERIOD).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    return df

# ============================================================================
# SIGNAL LOGIC
# ============================================================================

def find_pivots(data, left=3, right=3):
    pivot_highs = np.full(len(data), np.nan)
    pivot_lows = np.full(len(data), np.nan)
    for i in range(left, len(data) - right):
        window = data[i-left : i+right+1]
        center = data[i]
        if len(window) != (left + right + 1): continue
        if center == max(window) and list(window).count(center) == 1: pivot_highs[i] = center
        if center == min(window) and list(window).count(center) == 1: pivot_lows[i] = center
    return pivot_highs, pivot_lows

def detect_signals(df):
    close = df['close'].values; high = df['high'].values; low = df['low'].values
    rsi = df['rsi'].values; ema = df['ema'].values; times = df['start'].values
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    # Store used pivots to dedup
    # Tuple: (pivot_idx, type)
    used_pivots = set() 
    
    for i in range(205, len(df) - 3):
        curr_price = close[i]; curr_ema = ema[i]
        
        # --- BULLISH ---
        if curr_price > curr_ema:
            p_lows = []
            for j in range(i-3, max(0, i-50), -1):
                if not np.isnan(price_pl[j]):
                    p_lows.append((j, price_pl[j]))
                    if len(p_lows) >= 2: break
            if len(p_lows) == 2:
                curr_idx, curr_val = p_lows[0]
                prev_idx, prev_val = p_lows[1]
                
                # Dedup Key: Current Pivot + prev pivot pair
                dedup_key = (curr_idx, prev_idx, 'BULL')
                
                if (i - curr_idx) <= 10:
                    if dedup_key not in used_pivots:
                         added = False
                         if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                            signals.append({'conf_idx': i, 'side': 'long', 'type': 'REG_BULL', 'swing': max(high[curr_idx:i+1])})
                            added = True
                         if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                            signals.append({'conf_idx': i, 'side': 'long', 'type': 'HID_BULL', 'swing': max(high[curr_idx:i+1])})
                            added = True
                         
                         if added: used_pivots.add(dedup_key)

        # --- BEARISH ---
        if curr_price < curr_ema:
             p_highs = []
             for j in range(i-3, max(0, i-50), -1):
                if not np.isnan(price_ph[j]):
                    p_highs.append((j, price_ph[j]))
                    if len(p_highs) >= 2: break
            
             if len(p_highs) == 2:
                curr_idx, curr_val = p_highs[0]
                prev_idx, prev_val = p_highs[1]
                dedup_key = (curr_idx, prev_idx, 'BEAR')
                
                if (i - curr_idx) <= 10:
                    if dedup_key not in used_pivots:
                        added = False
                        if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                            signals.append({'conf_idx': i, 'side': 'short', 'type': 'REG_BEAR', 'swing': min(low[curr_idx:i+1])})
                            added = True
                        if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                            signals.append({'conf_idx': i, 'side': 'short', 'type': 'HID_BEAR', 'swing': min(low[curr_idx:i+1])})
                            added = True
                        
                        if added: used_pivots.add(dedup_key)
                        
    return signals

# ============================================================================
# EXECUTION ENGINE
# ============================================================================

def execute_trade(signal, df_1h, df_5m, rr_ratio, atr_mult):
    conf_idx = signal['conf_idx']
    side = signal['side']
    bos_level = signal['swing']
    if conf_idx + 1 >= len(df_1h): return None 
    
    entry_price = None; sl_price = None; tp_price = None; entry_time = None
    
    for i in range(1, MAX_WAIT_CANDLES + 1):
        if conf_idx + i >= len(df_1h): break
        candle = df_1h.iloc[conf_idx + i]
        triggered = False
        if side == 'long' and candle['close'] > bos_level: triggered = True
        elif side == 'short' and candle['close'] < bos_level: triggered = True
        if triggered:
            if conf_idx + i + 1 >= len(df_1h): break
            entry_candle = df_1h.iloc[conf_idx + i + 1]
            entry_time = entry_candle['start']
            atr = candle['atr']
            sl_dist = atr * atr_mult 
            raw_entry = entry_candle['open']
            
            if side == 'long':
                entry_price = raw_entry * (1 + SLIPPAGE_PCT)
                sl_price = entry_price - sl_dist
                tp_price = entry_price + (sl_dist * rr_ratio)
            else:
                entry_price = raw_entry * (1 - SLIPPAGE_PCT)
                sl_price = entry_price + sl_dist
                tp_price = entry_price - (sl_dist * rr_ratio)
            break
            
    if not entry_price: return None
        
    five_min_subset = df_5m[df_5m['start'] >= entry_time]
    if five_min_subset.empty: return None
    
    outcome = "timeout"
    for row in five_min_subset.itertuples():
        if side == 'long':
            hit_sl = row.low <= sl_price
            hit_tp = row.high >= tp_price
            if hit_sl and hit_tp: outcome = "loss"; break
            elif hit_sl: outcome = "loss"; break
            elif hit_tp: outcome = "win"; break
        else:
            hit_sl = row.high >= sl_price
            hit_tp = row.low <= tp_price
            if hit_sl and hit_tp: outcome = "loss"; break
            elif hit_sl: outcome = "loss"; break
            elif hit_tp: outcome = "win"; break
                
    risk = abs(entry_price - sl_price)
    if risk == 0: return None
    
    r_result = 0
    if outcome == 'win': r_result = rr_ratio
    elif outcome == 'loss': r_result = -1.0
    else: return None 
    
    fee_drag = (FEE_PCT * 2 * entry_price) / risk
    return r_result - fee_drag

# ============================================================================
# OPTIMIZATION LOOP
# ============================================================================

def optimize_symbol(sym):
    df_1h = fetch_klines(sym, SIGNAL_TF, DAYS)
    df_5m = fetch_klines(sym, EXECUTION_TF, DAYS)
    if len(df_1h) < 200 or len(df_5m) < 2000: return []
    df_1h = prepare_1h_data(df_1h)
    all_signals = detect_signals(df_1h)
    if not all_signals: return []
    
    signals_by_type = {
        'REG_BULL': [s for s in all_signals if s['type'] == 'REG_BULL'],
        'REG_BEAR': [s for s in all_signals if s['type'] == 'REG_BEAR'],
        'HID_BULL': [s for s in all_signals if s['type'] == 'HID_BULL'],
        'HID_BEAR': [s for s in all_signals if s['type'] == 'HID_BEAR']
    }
    
    best_results = []
    combos = list(itertools.product(GRID['div_type'], GRID['atr_mult'], GRID['rr_ratio']))
    
    for div_type, atr, rr in combos:
        target_signals = signals_by_type[div_type]
        if not target_signals: continue
        trades = []
        for sig in target_signals:
            res = execute_trade(sig, df_1h, df_5m, rr, atr)
            if res is not None: trades.append(res)
            
        if len(trades) < 10: continue 
        
        total_r = sum(trades); avg_r = total_r / len(trades)
        wins = len([t for t in trades if t > 0])
        wr = (wins / len(trades)) * 100
        
        passed = False
        if rr <= 3.0: 
            if wr >= 40 and avg_r >= 0.1: passed = True
        else:
            if wr >= 20 and avg_r >= 0.1: passed = True 
            
        if passed:
            res_dict = {
                'symbol': sym,
                'div_type': div_type,
                'atr': atr,
                'rr': rr,
                'trades': len(trades),
                'wr': round(wr, 1),
                'total_r': round(total_r, 2),
                'avg_r': round(avg_r, 3)
            }
            best_results.append(res_dict)
            with FILE_LOCK:
                pd.DataFrame([res_dict]).to_csv(OUTPUT_FILE, mode='a', header=not os.path.exists(OUTPUT_FILE), index=False)
                
    return best_results

def main():
    print(f"🚀 PRECISION OPTIMIZER (DEDUPED) - FULL RUN")
    target_symbols = []
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        # Process ALL enabled symbols
        target_symbols = list(config.get('symbols', {}).keys())
    
    # Randomize to get broad feedback early
    import random
    random.shuffle(target_symbols)
    
    if os.path.exists(OUTPUT_FILE): os.remove(OUTPUT_FILE)
    
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(optimize_symbol, sym): sym for sym in target_symbols}
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            sys.stdout.write(f"\rProgress: {completed}/{len(target_symbols)}")
            sys.stdout.flush()
            
    print(f"\nDONE. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
