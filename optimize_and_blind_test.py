#!/usr/bin/env python3
"""
OPTIMIZATION FACTORY PIPELINE (FULL RUN)
========================================
1. Splits 1-Year Data into TRAIN (first 60%) and TEST (last 40%).
2. Optimizes parameters on TRAIN.
3. Validates the BEST config on TEST (Blind).
4. Outputs only PASSING symbols.
5. Saves incrementally to 'factory_results.csv'
"""

import requests
import pandas as pd
import numpy as np
import time
import warnings
import csv
import os
from itertools import product

warnings.filterwarnings('ignore')

# CONFIG
SYMBOLS_LIMIT = 400  # Full run
TRAIN_SPLIT = 0.60
OUTPUT_FILE = 'factory_results.csv'

# PARAM GRID (Deep Search)
DIV_TYPES = ['REG_BULL', 'REG_BEAR', 'HID_BULL', 'HID_BEAR']
ATR_MULTS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
RR_RATIOS = [2, 3, 4, 5, 6, 7, 8, 9, 10]

BASE_URL = "https://api.bybit.com"

# ============================================================================
# UTILS & DATA
# ============================================================================

def fetch_1yr_klines(symbol, days=365):
    all_klines = []
    end_ts = int(time.time() * 1000)
    start_ts = int((time.time() - days * 24 * 3600) * 1000)
    
    while end_ts > start_ts:
        params = {'category': 'linear', 'symbol': symbol, 'interval': '60', 'limit': 1000, 'end': end_ts}
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
            data = resp.json()
            if data.get('retCode') != 0: break
            klines = data.get('result', {}).get('list', [])
            if not klines: break
            all_klines.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.01)
        except: break
    
    if not all_klines: return pd.DataFrame()
    df = pd.DataFrame(all_klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'turnover'])
    df = df.iloc[::-1].reset_index(drop=True)
    for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
    df['ts'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
    df.set_index('ts', inplace=True)
    return df

def calc_indicators(df):
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['ema'] = df['close'].ewm(span=200, adjust=False).mean()
    return df.dropna()

def find_pivots(arr):
    n = len(arr)
    ph, pl = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(3, n - 3):
        w = arr[i-3:i+4]
        if arr[i] == max(w) and list(w).count(arr[i]) == 1: ph[i] = arr[i]
        if arr[i] == min(w) and list(w).count(arr[i]) == 1: pl[i] = arr[i]
    return ph, pl

def detect_divergence(df, idx, div_type, close, rsi, ema, ph, pl):
    if idx < 50: return False, None
    cp, ce = close[idx], ema[idx]
    side = 'long' if div_type in ['REG_BULL', 'HID_BULL'] else 'short'
    tf = 'above_ema' if side == 'long' else 'below_ema'
    if (tf == 'above_ema' and cp <= ce) or (tf == 'below_ema' and cp >= ce): return False, None
    
    pivs = pl if side == 'long' else ph
    found_pivots = []
    for j in range(idx-4, max(0, idx-50), -1):
        if not np.isnan(pivs[j]):
            found_pivots.append((j, pivs[j], rsi[j]))
            if len(found_pivots) >= 2: break
    
    if len(found_pivots) < 2: return False, None
    ci, cpv, cr = found_pivots[0]
    pi, ppv, pr = found_pivots[1]
    if (idx - ci) > 10: return False, None
    
    is_div = False
    swing = 0
    if div_type == 'REG_BULL' and cpv < ppv and cr > pr: 
        is_div = True
        swing = max(df['high'].iloc[ci:idx+1])
    elif div_type == 'HID_BULL' and cpv > ppv and cr < pr: 
        is_div = True
        swing = max(df['high'].iloc[ci:idx+1])
    elif div_type == 'REG_BEAR' and cpv > ppv and cr < pr: 
        is_div = True
        swing = min(df['low'].iloc[ci:idx+1])
    elif div_type == 'HID_BEAR' and cpv < ppv and cr > pr: 
        is_div = True
        swing = min(df['low'].iloc[ci:idx+1])
        
    if is_div:
        return True, swing
    return False, None

def simulate(df, div_type, rr, atr_mult):
    trades = []
    if len(df) < 100: return []
    close, high, low = df['close'].values, df['high'].values, df['low'].values
    rsi, ema, atr = df['rsi'].values, df['ema'].values, df['atr'].values
    ph, pl = find_pivots(close)
    
    pending_sig, pending_swing, wait = None, None, 0
    in_trade = False
    entry_p, sl, tp, side = 0, 0, 0, None
    
    for i in range(50, len(df)):
        c, h, l = close[i], high[i], low[i]
        
        if in_trade:
            win = (h >= tp) if side == 'long' else (l <= tp)
            loss = (l <= sl) if side == 'long' else (h >= sl)
            if win: trades.append(rr); in_trade = False
            elif loss: trades.append(-1.0); in_trade = False
            continue
            
        if pending_sig:
            bos = (c > pending_swing if side == 'long' else c < pending_swing)
            if bos:
                if i+1 < len(df):
                    ep = df['open'].iloc[i+1]
                    sl_dist = atr[i] * atr_mult
                    if side == 'long': tp, sl = ep + sl_dist*rr, ep - sl_dist
                    else: tp, sl = ep - sl_dist*rr, ep + sl_dist
                    in_trade = True
                    entry_p = ep
                pending_sig = None
            else:
                wait += 1
                if wait >= 12: pending_sig = None
        
        if not in_trade and not pending_sig:
            found, swing = detect_divergence(df, i, div_type, close, rsi, ema, ph, pl)
            if found:
                pending_sig = True; pending_swing = swing; wait = 0
                side = 'long' if div_type in ['REG_BULL', 'HID_BULL'] else 'short'
                
    return trades

# ============================================================================
# MAIN
# ============================================================================

def get_all_symbols(limit=400):
    try:
        r = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear")
        data = r.json()['result']['list']
        usdt = [x for x in data if x['symbol'].endswith('USDT') and 'USDC' not in x['symbol']]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        return [x['symbol'] for x in usdt[:limit]]
    except:
        return []

def main():
    print(f"🏭 OPTIMIZATION FACTORY: Starting Full Run ({SYMBOLS_LIMIT} Symbols)")
    symbols = get_all_symbols(SYMBOLS_LIMIT)
    print(f"Found {len(symbols)} symbols")
    
    # Setup CSV
    headers = ['symbol', 'divergence', 'atr_mult', 'rr', 'train_r', 'blind_r', 'total_r', 'blind_trades', 'status']
    existing_symbols = set()
    
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_symbols.add(row['symbol'])
        print(f"Resuming... {len(existing_symbols)} symbols already processed.")
    else:
        with open(OUTPUT_FILE, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    for i, symbol in enumerate(symbols):
        if symbol in existing_symbols:
            continue
            
        print(f"[{i+1}/{len(symbols)}] Processing {symbol}...", end="", flush=True)
        
        try:
            df = fetch_1yr_klines(symbol)
            if len(df) < 500:
                print(" No Data")
                # Save as skipped to avoid retry loop? No, might succeed later.
                continue
            
            df = calc_indicators(df)
            split_idx = int(len(df) * TRAIN_SPLIT)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
            
            # OPTIMIZE
            best_config = None
            best_train_r = -999
            
            for div, atr, rr in product(DIV_TYPES, ATR_MULTS, RR_RATIOS):
                trades = simulate(train_df, div, rr, atr)
                if not trades: continue
                total_r = sum(trades)
                if total_r > best_train_r and len(trades) >= 5:
                    best_train_r = total_r
                    best_config = (div, atr, rr)
            
            if not best_config:
                print(" No passing strat")
                # Record as failed
                with open(OUTPUT_FILE, 'a') as f:
                    csv.writer(f).writerow([symbol, 'NONE', 0, 0, 0, 0, 0, 0, 'FAIL_TRAIN'])
                continue
                
            # BLIND TEST
            b_div, b_atr, b_rr = best_config
            blind_trades = simulate(test_df, b_div, b_rr, b_atr)
            blind_r = sum(blind_trades)
            blind_count = len(blind_trades)
            
            status = "PASS" if blind_r > 0 and blind_count >= 3 else "FAIL_BLIND"
            icon = "✅" if status == "PASS" else "❌"
            
            print(f" Best: {b_div}/{b_atr}x/{b_rr}R (Train: {best_train_r:.1f}R) -> Blind: {blind_r:.1f}R {icon}")
            
            with open(OUTPUT_FILE, 'a') as f:
                csv.writer(f).writerow([
                    symbol, b_div, b_atr, b_rr, 
                    best_train_r, blind_r, best_train_r + blind_r, 
                    blind_count, status
                ])
                
        except Exception as e:
            print(f" Error: {e}")
            continue

    print("\noptimization run complete.")

if __name__ == "__main__":
    main()
