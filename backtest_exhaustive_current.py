#!/usr/bin/env python3
"""
EXHAUSTIVE CURRENT BOT OPTIMIZATION (5M)
========================================

Goal: Find profitable config for the EXISTING bot strategy on 5M.
Strategy: RSI Divergence (from divergence_detector.py)
Symbols: Top 20 from config.yaml
Method: Grid Search over ALL configurable parameters

Parameters to Optimize:
1. RR_RATIO: [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
2. SL_ATR_MULT: [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]
3. DIV_FILTER: ['all', 'regular_only', 'hidden_only']
4. TIME_FILTER: ['all_day', 'ny_session', 'london_ny', 'high_vol']
5. TRAILING_SL: [False, True]
6. MIN_RSI: [25, 30, 35, 40] (for Bullish Div oversold confirmation)
7. MAX_RSI: [60, 65, 70, 75] (for Bearish Div overbought confirmation)

Total Combinations: 6 × 7 × 3 × 4 × 2 × 4 × 4 = 16,128 configs
We'll test smart subsets and iterate.

Author: AutoBot Architect
"""

import pandas as pd
import numpy as np
import requests
import time
import itertools
import sys
from datetime import datetime

# ============================================================================
# SYMBOLS (From config.yaml)
# ============================================================================

SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT',
    'BNBUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT',
    'LTCUSDT', 'NEARUSDT', 'APTUSDT', 'SUIUSDT', 'ARBUSDT',
    'OPUSDT', 'ATOMUSDT', 'UNIUSDT', 'INJUSDT', 'TONUSDT'
]

DAYS = 60
TIMEFRAME = 5

# Realistic Bybit Fees
WIN_COST = 0.0006  # Maker entry + Maker TP
LOSS_COST = 0.00125  # Maker entry + Taker SL

# ============================================================================
# DATA ENGINE
# ============================================================================

def fetch_data(symbol):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': str(TIMEFRAME), 'limit': 1000, 'end': end_ts}
            r = requests.get(url, params=params).json()
            if r['retCode'] != 0 or not r['result']['list']: break
            klines = r['result']['list']
            all_kline.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.04)
        
        if not all_kline: return pd.DataFrame()
        
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
        df['datetime'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
        df['hour'] = df['datetime'].dt.hour
        return df
    except: return pd.DataFrame()

def calc_indicators(df):
    df = df.copy()
    close = df['close']
    
    # RSI 14 (Matching divergence_detector.py)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR 14
    h, l, c_prev = df['high'], df['low'], close.shift(1)
    tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Rolling Window for Divergence Detection (Matching backtest logic)
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    
    return df

# ============================================================================
# DIVERGENCE DETECTION (Matching divergence_detector.py logic)
# ============================================================================

def detect_divergences(df, div_filter, min_rsi, max_rsi):
    """
    Detect divergences using rolling window (matches backtest logic exactly).
    """
    df = df.copy()
    
    # Regular Bullish: Price LL, RSI HL, RSI < min_rsi
    df['reg_bull'] = (
        (df['low'] <= df['price_low_14']) &
        (df['rsi'] > df['rsi_low_14'].shift(14)) &
        (df['rsi'] < min_rsi)
    )
    
    # Regular Bearish: Price HH, RSI LH, RSI > max_rsi
    df['reg_bear'] = (
        (df['high'] >= df['price_high_14']) &
        (df['rsi'] < df['rsi_high_14'].shift(14)) &
        (df['rsi'] > max_rsi)
    )
    
    # Hidden Bullish: Price HL, RSI LL
    df['hid_bull'] = (
        (df['low'] > df['low'].shift(14)) &
        (df['rsi'] < df['rsi'].shift(14)) &
        (df['rsi'] < 60)
    )
    
    # Hidden Bearish: Price LH, RSI HH
    df['hid_bear'] = (
        (df['high'] < df['high'].shift(14)) &
        (df['rsi'] > df['rsi'].shift(14)) &
        (df['rsi'] > 40)
    )
    
    # Apply Filter
    if div_filter == 'regular_only':
        df['hid_bull'] = False
        df['hid_bear'] = False
    elif div_filter == 'hidden_only':
        df['reg_bull'] = False
        df['reg_bear'] = False
    
    return df

# ============================================================================
# STRATEGY BACKTEST
# ============================================================================

def backtest_config(datasets, config):
    total_r = 0
    total_trades = 0
    
    rr = config['rr_ratio']
    sl_mult = config['sl_atr']
    div_filter = config['div_filter']
    time_filter = config['time_filter']
    trailing = config['trailing']
    min_rsi = config['min_rsi']
    max_rsi = config['max_rsi']
    
    for sym, df in datasets.items():
        df = detect_divergences(df, div_filter, min_rsi, max_rsi)
        
        cooldown = 0
        for i in range(50, len(df)-1):
            if cooldown > 0: cooldown -= 1; continue
            
            row = df.iloc[i]
            
            # Time Filter
            if time_filter == 'ny_session' and not (13 <= row['hour'] < 21): continue
            elif time_filter == 'london_ny' and not (8 <= row['hour'] < 20): continue
            elif time_filter == 'high_vol' and not (12 <= row['hour'] < 22): continue
            
            # Detect Signal
            side = None
            if row['reg_bull'] or row['hid_bull']: side = 'long'
            elif row['reg_bear'] or row['hid_bear']: side = 'short'
            
            if not side: continue
            
            # Entry (Next candle open, matching bot logic)
            entry = df.iloc[i+1]['open']
            atr = row['atr']
            if pd.isna(atr) or atr == 0: continue
            
            sl_dist = atr * sl_mult
            if sl_dist/entry > 0.05: continue  # Skip if SL > 5%
            
            tp_dist = sl_dist * rr
            
            if side == 'long':
                sl = entry - sl_dist
                tp = entry + tp_dist
            else:
                sl = entry + sl_dist
                tp = entry - tp_dist
            
            # Simulate Trade
            outcome = 'timeout'
            curr_sl = sl
            be_hit = False
            
            for j in range(i+1, min(i+500, len(df))):
                c = df.iloc[j]
                
                if side == 'long':
                    if c['low'] <= curr_sl: outcome = 'loss' if not be_hit else 'be'; break
                    if c['high'] >= tp and not trailing: outcome = 'win'; break
                    
                    if trailing:
                        if c['high'] >= entry + sl_dist: curr_sl = entry; be_hit = True
                        if c['high'] >= tp: outcome = 'win'; break
                else:
                    if c['high'] >= curr_sl: outcome = 'loss' if not be_hit else 'be'; break
                    if c['low'] <= tp and not trailing: outcome = 'win'; break
                    
                    if trailing:
                        if c['low'] <= entry - sl_dist: curr_sl = entry; be_hit = True
                        if c['low'] <= tp: outcome = 'win'; break
            
            # Calculate R
            risk_pct = sl_dist / entry
            res_r = 0
            if outcome == 'win': res_r = rr - (WIN_COST / risk_pct)
            elif outcome == 'loss': res_r = -1.0 - (LOSS_COST / risk_pct)
            elif outcome == 'be': res_r = 0.0 - (WIN_COST / risk_pct)
            elif outcome == 'timeout': res_r = -0.1
            
            total_r += res_r
            total_trades += 1
            cooldown = 3
    
    return {'net_r': total_r, 'trades': total_trades, 'avg_r': total_r/total_trades if total_trades else -999}

# ============================================================================
# MAIN GRID SEARCH
# ============================================================================

def main():
    print("🚀 EXHAUSTIVE BOT OPTIMIZATION (5M)")
    print("Loading data...")
    
    datasets = {}
    for sym in SYMBOLS:
        df = fetch_data(sym)
        if not df.empty: datasets[sym] = calc_indicators(df)
    
    print(f"Loaded {len(datasets)} symbols.")
    
    # Define Grid (Start with most promising subset)
    GRID = {
        'rr_ratio': [2.0, 2.5, 3.0],
        'sl_atr': [0.6, 0.8, 1.0, 1.2],
        'div_filter': ['all', 'regular_only'],
        'time_filter': ['all_day', 'ny_session'],
        'trailing': [False, True],
        'min_rsi': [30, 35, 40],
        'max_rsi': [60, 65, 70]
    }
    
    combos = [dict(zip(GRID.keys(), v)) for v in itertools.product(*GRID.values())]
    print(f"Testing {len(combos)} configurations...\n")
    
    best_config = None
    best_r = -9999
    
    print(f"{'#':<5} {'RR':<4} {'SL':<4} {'Div':<12} {'Time':<12} {'Trail':<5} {'MRSI':<4} {'XRSI':<4} | {'Trades':<6} {'Net R':<8} {'Avg R':<8}")
    print("-" * 95)
    
    for i, cfg in enumerate(combos):
        res = backtest_config(datasets, cfg)
        
        if res['avg_r'] > best_r and res['trades'] > 50:
            best_r = res['avg_r']
            best_config = cfg
            print(f"{i+1:<5} {cfg['rr_ratio']:<4} {cfg['sl_atr']:<4} {cfg['div_filter']:<12} {cfg['time_filter']:<12} {str(cfg['trailing']):<5} {cfg['min_rsi']:<4} {cfg['max_rsi']:<4} | {res['trades']:<6} {res['net_r']:<8.1f} {res['avg_r']:<8.3f} ✅ NEW BEST")
        
        if (i+1) % 50 == 0:
            sys.stdout.write(f"\rProgress: {i+1}/{len(combos)}...")
            sys.stdout.flush()
    
    print("\n" + "="*95)
    print("🏆 CHAMPION CONFIG")
    print(best_config)
    print(f"Avg R: {best_r:+.3f}")

if __name__ == "__main__":
    main()
