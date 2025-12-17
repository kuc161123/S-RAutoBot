#!/usr/bin/env python3
"""
TRAILING STOP - DETAILED BREAKDOWN
Shows exactly where exits happen (0R, 1R, 2R, etc.)
"""

import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

TIMEFRAME = '60'
DATA_DAYS = 60
NUM_SYMBOLS = 150

SLIPPAGE_PCT = 0.0005
FEE_PCT = 0.0004
TOTAL_COST = (SLIPPAGE_PCT + FEE_PCT) * 2

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5
COOLDOWN_BARS = 10

RISK_REWARD = 3.0
MIN_SL_ATR = 0.3
MAX_SL_ATR = 2.0

BASE_URL = "https://api.bybit.com"

def get_symbols(limit):
    resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear", timeout=10)
    tickers = resp.json().get('result', {}).get('list', [])
    usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in usdt[:limit]]

def fetch_klines(symbol, interval, days):
    end_ts = int(datetime.now().timestamp() * 1000)
    candles_needed = days * 24 * 60 // int(interval)
    all_candles = []
    current_end = end_ts
    
    while len(all_candles) < candles_needed:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 
                  'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
            data = resp.json().get('result', {}).get('list', [])
            if not data: break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            if len(data) < 1000: break
        except: break
    
    if not all_candles: return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    df['date'] = df.index.date
    return df

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def find_pivots(data, left=3, right=3):
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    for i in range(left, n - right):
        is_high = all(data[j] < data[i] for j in range(i - left, i + right + 1) if j != i)
        is_low = all(data[j] > data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_high: pivot_highs[i] = data[i]
        if is_low: pivot_lows[i] = data[i]
    return pivot_highs, pivot_lows

def detect_all_divergences(df):
    if len(df) < 100: return []
    
    close = df['close'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    for i in range(30, n - 5):
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        if curr_pl and prev_pl:
            if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                if rsi[i] < RSI_OVERSOLD + 15:
                    signals.append({'idx': i, 'type': 'regular_bullish', 'side': 'long', 'swing': curr_pl})
                    continue
        
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                if rsi[i] > RSI_OVERBOUGHT - 15:
                    signals.append({'idx': i, 'type': 'regular_bearish', 'side': 'short', 'swing': curr_ph})
                    continue
        
        if curr_pl and prev_pl:
            if curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli]:
                if rsi[i] < RSI_OVERBOUGHT - 10:
                    signals.append({'idx': i, 'type': 'hidden_bullish', 'side': 'long', 'swing': curr_pl})
                    continue
        
        if curr_ph and prev_ph:
            if curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi]:
                if rsi[i] > RSI_OVERSOLD + 10:
                    signals.append({'idx': i, 'type': 'hidden_bearish', 'side': 'short', 'swing': curr_ph})
    
    return signals

def calc_pivot_sltp(rows, idx, side, atr, swing_price):
    if idx + 1 >= len(rows):
        return None, None, None, None
    
    entry = rows[idx + 1].open
    sl_dist = abs(swing_price - entry)
    
    min_dist = MIN_SL_ATR * atr
    max_dist = MAX_SL_ATR * atr
    if sl_dist < min_dist: sl_dist = min_dist
    if sl_dist > max_dist: sl_dist = max_dist
    
    if side == 'long':
        sl = entry - sl_dist
        tp = entry + (sl_dist * RISK_REWARD)
    else:
        sl = entry + sl_dist
        tp = entry - (sl_dist * RISK_REWARD)
    
    return entry, sl, sl_dist, tp

def simulate_trade_trailing(rows, signal_idx, side, sl, sl_dist, tp, entry):
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 1: return 'timeout', 0, 0
    
    if side == 'long':
        entry = entry * (1 + SLIPPAGE_PCT)
        tp = tp * (1 - TOTAL_COST)
    else:
        entry = entry * (1 - SLIPPAGE_PCT)
        tp = tp * (1 + TOTAL_COST)
    
    current_sl = sl
    max_favorable_r = 0
    
    for bar_idx, row in enumerate(rows[entry_idx:entry_idx + 100]):
        if side == 'long':
            unrealized_r = (row.high - entry) / sl_dist
            if unrealized_r > max_favorable_r:
                max_favorable_r = unrealized_r
                
                if max_favorable_r >= 2.0:
                    new_sl = entry + (max_favorable_r - 1.0) * sl_dist
                    if new_sl > current_sl:
                        current_sl = new_sl
                elif max_favorable_r >= 1.0:
                    if entry > current_sl:
                        current_sl = entry
            
            if row.low <= current_sl:
                exit_r = (current_sl - entry) / sl_dist
                return 'sl_hit', exit_r, max_favorable_r
            
            if row.high >= tp:
                return 'tp_hit', RISK_REWARD, max_favorable_r
                
        else:
            unrealized_r = (entry - row.low) / sl_dist
            if unrealized_r > max_favorable_r:
                max_favorable_r = unrealized_r
                
                if max_favorable_r >= 2.0:
                    new_sl = entry - (max_favorable_r - 1.0) * sl_dist
                    if new_sl < current_sl:
                        current_sl = new_sl
                elif max_favorable_r >= 1.0:
                    if entry < current_sl:
                        current_sl = entry
            
            if row.high >= current_sl:
                exit_r = (entry - current_sl) / sl_dist
                return 'sl_hit', exit_r, max_favorable_r
            
            if row.low <= tp:
                return 'tp_hit', RISK_REWARD, max_favorable_r
    
    exit_r = (current_sl - entry) / sl_dist if side == 'long' else (entry - current_sl) / sl_dist
    return 'timeout', exit_r, max_favorable_r

def run():
    print("=" * 70)
    print("📊 TRAILING STOP - DETAILED BREAKDOWN")
    print("=" * 70)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📋 Testing {len(symbols)} symbols...\n")
    
    # Track exits by R level
    exit_buckets = {
        'full_loss': {'count': 0, 'r': 0},      # -1R (original SL)
        'break_even': {'count': 0, 'r': 0},     # 0R exactly
        'small_profit': {'count': 0, 'r': 0},   # 0.1R to 0.9R
        'one_r': {'count': 0, 'r': 0},          # 1R to 1.9R (trailed from 2R+)
        'two_r': {'count': 0, 'r': 0},          # 2R to 2.9R (trailed from 3R+)
        'full_tp': {'count': 0, 'r': 0},        # 3R (full TP)
    }
    
    all_exits = []  # Track individual R values
    
    for idx, sym in enumerate(symbols):
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 200: continue
            
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            hl = df['high'] - df['low']
            hc = abs(df['high'] - df['close'].shift())
            lc = abs(df['low'] - df['close'].shift())
            df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            df = df.dropna()
            
            if len(df) > 0:
                df = df.iloc[:-1]
            
            if len(df) < 100: continue
            
            signals = detect_all_divergences(df)
            rows = list(df.itertuples())
            
            last_trade_idx = -COOLDOWN_BARS
            
            for sig in signals:
                i = sig['idx']
                if i - last_trade_idx < COOLDOWN_BARS: continue
                if i >= len(rows) - 50: continue
                
                row = rows[i]
                if not row.vol_ok or row.atr <= 0: continue
                
                entry, sl, sl_dist, tp = calc_pivot_sltp(rows, i, sig['side'], row.atr, sig['swing'])
                if entry is None: continue
                
                result, exit_r, max_r = simulate_trade_trailing(rows, i, sig['side'], sl, sl_dist, tp, entry)
                
                if result == 'timeout' and exit_r == 0: continue
                last_trade_idx = i
                
                all_exits.append(exit_r)
                
                # Categorize
                if exit_r <= -0.99:
                    exit_buckets['full_loss']['count'] += 1
                    exit_buckets['full_loss']['r'] += exit_r
                elif abs(exit_r) < 0.1:
                    exit_buckets['break_even']['count'] += 1
                    exit_buckets['break_even']['r'] += exit_r
                elif exit_r < 1.0:
                    exit_buckets['small_profit']['count'] += 1
                    exit_buckets['small_profit']['r'] += exit_r
                elif exit_r < 2.0:
                    exit_buckets['one_r']['count'] += 1
                    exit_buckets['one_r']['r'] += exit_r
                elif exit_r < 3.0:
                    exit_buckets['two_r']['count'] += 1
                    exit_buckets['two_r']['r'] += exit_r
                else:
                    exit_buckets['full_tp']['count'] += 1
                    exit_buckets['full_tp']['r'] += exit_r
                    
        except: continue
        
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{NUM_SYMBOLS}] Trades: {len(all_exits)}")
    
    total_trades = len(all_exits)
    total_r = sum(all_exits)
    
    print("\n" + "=" * 70)
    print("📊 EXIT BREAKDOWN")
    print("=" * 70)
    
    print(f"\n{'Exit Type':<20} {'Count':>8} {'%':>8} {'Total R':>12} {'Avg R':>10}")
    print("-" * 60)
    
    labels = {
        'full_loss': '❌ Full Loss (-1R)',
        'break_even': '⚖️ Break-Even (0R)',
        'small_profit': '📈 Small (0.1-0.9R)',
        'one_r': '✅ Trailed (+1-1.9R)',
        'two_r': '✅✅ Trailed (+2-2.9R)',
        'full_tp': '🎯 Full TP (+3R)',
    }
    
    for key, label in labels.items():
        data = exit_buckets[key]
        count = data['count']
        r_total = data['r']
        pct = (count / total_trades * 100) if total_trades > 0 else 0
        avg = (r_total / count) if count > 0 else 0
        print(f"{label:<25} {count:>6} {pct:>7.1f}% {r_total:>+10.0f}R {avg:>+9.2f}")
    
    print("-" * 60)
    print(f"{'TOTAL':<25} {total_trades:>6} {'100.0':>7}% {total_r:>+10.0f}R {total_r/total_trades:>+9.2f}")
    
    print("\n" + "=" * 70)
    print("📊 COMPARISON")
    print("=" * 70)
    print(f"Baseline (Fixed SL):  6691 trades, +7553R total, +1.13 avg")
    print(f"Trailing Stop:        {total_trades} trades, {total_r:+.0f}R total, {total_r/total_trades:+.2f} avg")
    print("=" * 70)

if __name__ == "__main__":
    run()
