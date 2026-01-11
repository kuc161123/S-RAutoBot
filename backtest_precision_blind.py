#!/usr/bin/env python3
"""
PRECISION BLIND TEST (1H Signal / 5M Execution)
===============================================
Goal: Validate 1H Strategy Profitability without Intra-Candle Bias.

Methodology:
1. Fetch 1H data (Signals) & 5M data (Execution) for the last 60 days.
2. Detect Divergences on 1H (Identical to Live Bot).
3. Execute Trades on 5M:
   - Walk through 5M candles starting from the 1H Entry Candle.
   - Accurately determine if Stop Loss or Take Profit was hit first.
   - Applies realistic slippage and fees.

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
import random
import os
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION
# ============================================================================
DAYS = 60
SIGNAL_TF = '60'       # 1 Hour
EXECUTION_TF = '5'     # 5 Minute
MAX_WAIT_CANDLES = 6   # Max 1H candles to wait for entry
RSI_PERIOD = 14
EMA_PERIOD = 200

# Realistic Costs
SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006

BASE_URL = "https://api.bybit.com"
OUTPUT_FILE = 'precision_blind_test_results.csv'
FILE_LOCK = threading.Lock()

# ============================================================================
# DATA ENGINE
# ============================================================================

def fetch_klines(symbol, interval, days):
    """Fetch klines with pagination"""
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_end = end_ts
    
    # Safety limit to prevent infinite loops
    max_requests = 100 
    
    while current_end > start_ts and max_requests > 0:
        max_requests -= 1
        params = {
            'category': 'linear', 
            'symbol': symbol, 
            'interval': interval, 
            'limit': 1000, 
            'end': current_end
        }
        
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
            data = resp.json().get('result', {}).get('list', [])
            
            if not data:
                break
                
            all_candles.extend(data)
            oldest = int(data[-1][0])
            current_end = oldest - 1 # Prevent overlap
            
            if len(data) < 1000:
                break
                
            time.sleep(0.05) 
            
        except Exception as e:
            # print(f"Error fetching {symbol} {interval}: {e}")
            time.sleep(1)
            continue
    
    if not all_candles:
        return pd.DataFrame()
    
    # Process Data
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df = df.sort_values('start').reset_index(drop=True)
    return df

def prepare_1h_data(df):
    """Calculate 1H Indicators for Signal Detection"""
    df = df.copy()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    # EMA
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    return df

# ============================================================================
# SIGNAL LOGIC (1H)
# ============================================================================

def find_pivots(data, left=3, right=3):
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    for i in range(left, n - right):
        window = data[i-left : i+right+1]
        center = data[i]
        
        if len(window) != (left + right + 1): continue
        
        if center == max(window) and list(window).count(center) == 1:
            pivot_highs[i] = center
        if center == min(window) and list(window).count(center) == 1:
            pivot_lows[i] = center
            
    return pivot_highs, pivot_lows

def detect_signals(df):
    """
    Detect Regular and Hidden Divergences.
    Returns list of dicts: {'conf_idx': int, 'side': 'long'/'short', 'swing': float, 'start_time': datetime}
    """
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    ema = df['ema'].values
    times = df['start'].values
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    # Start after warm-up
    for i in range(205, len(df) - 3):
        curr_price = close[i]
        curr_ema = ema[i]
        
        # --- BULLISH ---
        if curr_price > curr_ema:
            # Find recent pivots
            p_lows = []
            for j in range(i-3, max(0, i-50), -1):
                if not np.isnan(price_pl[j]):
                    p_lows.append((j, price_pl[j]))
                    if len(p_lows) >= 2: break
            
            if len(p_lows) == 2:
                curr_idx, curr_val = p_lows[0]
                prev_idx, prev_val = p_lows[1]
                
                # Check Divergence Condition
                if (i - curr_idx) <= 10:
                    if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                        # Valid Bullish Div
                        signals.append({
                            'conf_idx': i,
                            'side': 'long',
                            'swing': max(high[curr_idx:i+1]), # BOS Level
                            'start_time': times[i]
                        })

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
                
                if (i - curr_idx) <= 10:
                    if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                        # Valid Bearish Div
                        signals.append({
                            'conf_idx': i,
                            'side': 'short',
                            'swing': min(low[curr_idx:i+1]), # BOS Level
                            'start_time': times[i]
                        })
                        
    return signals

# ============================================================================
# EXECUTION LOGIC (5M)
# ============================================================================

def execute_trade(signal, df_1h, df_5m, rr_ratio):
    """
    Simulate trade execution using 5M candles.
    """
    conf_idx = signal['conf_idx']
    side = signal['side']
    bos_level = signal['swing']
    
    # Get 1H Candles for context
    if conf_idx + 1 >= len(df_1h): return None # End of data
    
    # Find Entry on 1H (Waiting for BOS Break)
    entry_price = None
    sl_price = None
    tp_price = None
    entry_time = None
    
    # Scan next MAX_WAIT_CANDLES in 1H for BOS trigger
    for i in range(1, MAX_WAIT_CANDLES + 1):
        if conf_idx + i >= len(df_1h): break
        candle = df_1h.iloc[conf_idx + i]
        
        triggered = False
        if side == 'long' and candle['close'] > bos_level:
            triggered = True
        elif side == 'short' and candle['close'] < bos_level:
            triggered = True
            
        if triggered:
            # Trade Enters on OPEN of NEXT candle
            if conf_idx + i + 1 >= len(df_1h): break
            entry_candle = df_1h.iloc[conf_idx + i + 1]
            entry_time = entry_candle['start']
            
            # Calculate TP/SL based on Entry Candle ATR
            atr = candle['atr'] # Use ATR from trigger candle
            sl_dist = atr * 1.0 # 1.0 ATR SL
            
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
            
    if not entry_price:
        return None # No entry triggered
        
    # --- 5M PLAYBACK ---
    five_min_subset = df_5m[df_5m['start'] >= entry_time]
    
    if five_min_subset.empty: return None
    
    outcome = "timeout"
    
    for row in five_min_subset.itertuples():
        # Check High/Low interaction
        if side == 'long':
            hit_sl = row.low <= sl_price
            hit_tp = row.high >= tp_price
            
            if hit_sl and hit_tp:
                outcome = "loss" # Worst case
                break
            elif hit_sl:
                outcome = "loss"
                break
            elif hit_tp:
                outcome = "win"
                break
                
        else: # Short
            hit_sl = row.high >= sl_price
            hit_tp = row.low <= tp_price
            
            if hit_sl and hit_tp:
                outcome = "loss"
                break
            elif hit_sl:
                outcome = "loss"
                break
            elif hit_tp:
                outcome = "win"
                break
                
    # Calculate R
    risk = abs(entry_price - sl_price)
    if risk == 0: return None
    
    r_result = 0
    if outcome == 'win':
        r_result = rr_ratio
    elif outcome == 'loss':
        r_result = -1.0
    else:
        return None 
        
    # Apply Fees
    fee_drag = (FEE_PCT * 2 * entry_price) / risk
    final_r = r_result - fee_drag
    
    return final_r

# ============================================================================
# MAIN
# ============================================================================

def process_symbol(sym, rr_config):
    # 1. Fetch Data
    df_1h = fetch_klines(sym, SIGNAL_TF, DAYS)
    df_5m = fetch_klines(sym, EXECUTION_TF, DAYS)
    
    if len(df_1h) < 100 or len(df_5m) < 1000:
        return None
        
    # 2. Prep & Detect
    df_1h = prepare_1h_data(df_1h)
    signals = detect_signals(df_1h)
    
    if not signals: return None
    
    # 3. Simulate
    rr = rr_config.get(sym, {}).get('rr', 3.0)
    
    sym_r = 0
    sym_trades = 0
    
    for sig in signals:
        r = execute_trade(sig, df_1h, df_5m, rr)
        if r is not None:
            sym_r += r
            sym_trades += 1
    
    result = None
    if sym_trades >= 0: # Save even if 0 trades? No, maybe only valid.
        result = {
            'symbol': sym,
            'trades': sym_trades,
            'total_r': sym_r,
            'avg_r': sym_r / sym_trades if sym_trades > 0 else 0,
            'rr': rr
        }
        
    # INCREMENTAL SAVE
    if result:
        with FILE_LOCK:
            df_row = pd.DataFrame([result])
            # If file does not exist, write header. Else append.
            header = not os.path.exists(OUTPUT_FILE)
            df_row.to_csv(OUTPUT_FILE, mode='a', header=header, index=False)
            
    return result

def main():
    print(f"🔬 STARTING PRECISION BLIND TEST (Parallel Execution + Incremental Save)")
    print(f"   Period: Last {DAYS} Days")
    
    # Load Configs
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('symbol_rr_config.yaml', 'r') as f:
        rr_config = yaml.safe_load(f)
        
    symbols = [s for s, p in config.get('symbols', {}).items() if p.get('enabled')]
    random.shuffle(symbols) # Randomize check order
    
    print(f"   Targets: {len(symbols)} Symbols")
    print("-" * 60)
    
    # Clean old file
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    
    completed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_sym = {executor.submit(process_symbol, sym, rr_config): sym for sym in symbols}
        
        for future in concurrent.futures.as_completed(future_to_sym):
            completed += 1
            sys.stdout.write(f"\rProgress: {completed}/{len(symbols)}")
            sys.stdout.flush()

    print(f"\n\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    
    if os.path.exists(OUTPUT_FILE):
        df_res = pd.read_csv(OUTPUT_FILE)
        
        total_r = df_res['total_r'].sum()
        total_trades = df_res['trades'].sum()
        avg_r = total_r / total_trades if total_trades else 0
        
        print(f"Total Trades: {total_trades}")
        print(f"Total R: {total_r:+.2f}R")
        print(f"Avg R/Trade: {avg_r:+.3f}R")
        
        print("\nSaved detailed results to 'precision_blind_test_results.csv'")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
