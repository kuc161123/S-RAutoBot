#!/usr/bin/env python3
"""
ALL DIVERGENCES + CANDLE COLOR + VOLUME SURGE
==============================================
Dual Entry Confirmation:
1. Candle Color: Bullish for LONG, Bearish for SHORT
2. Volume Surge: Entry volume > 1.3x average volume

This combines momentum and institutional participation.
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
NUM_PERIODS = 6

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

VOLUME_MULTIPLIER = 1.3  # Entry volume must be > 1.3x average

BASE_URL = "https://api.bybit.com"

def calc_ev(wr, rr): return (wr * rr) - (1 - wr)

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
    entry = rows[idx + 1].open if idx + 1 < len(rows) else rows[idx].close
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
    
    return entry, sl, tp

def simulate_trade(rows, signal_idx, side, sl, tp, entry):
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 1: return 'timeout', 0
    
    if side == 'long':
        entry = entry * (1 + SLIPPAGE_PCT)
        tp = tp * (1 - TOTAL_COST)
    else:
        entry = entry * (1 - SLIPPAGE_PCT)
        tp = tp * (1 + TOTAL_COST)
    
    for bar_idx, row in enumerate(rows[entry_idx:entry_idx + 100]):
        if side == 'long':
            if row.low <= sl: return 'loss', bar_idx
            if row.high >= tp: return 'win', bar_idx
        else:
            if row.high >= sl: return 'loss', bar_idx
            if row.low <= tp: return 'win', bar_idx
    
    return 'timeout', 100

def run():
    print("=" * 70)
    print("📊 CANDLE COLOR + VOLUME SURGE (DUAL CONFIRMATION)")
    print("=" * 70)
    print(f"Filter 1: Candle Color (bullish/bearish)")
    print(f"Filter 2: Volume > {VOLUME_MULTIPLIER}x average")
    print(f"Timeframe: {TIMEFRAME}m | R:R: {RISK_REWARD}:1")
    print("=" * 70)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📋 Testing {len(symbols)} symbols...")
    
    type_results = defaultdict(lambda: {'w': 0, 'l': 0, 'r': 0.0})
    period_results = defaultdict(lambda: defaultdict(lambda: {'w': 0, 'l': 0}))
    filtered_candle = 0
    filtered_volume = 0
    filtered_both = 0
    
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
            
            all_dates = sorted(df['date'].unique())
            days_per_period = max(1, len(all_dates) // NUM_PERIODS)
            
            last_trade_idx = -COOLDOWN_BARS
            
            for sig in signals:
                i = sig['idx']
                if i - last_trade_idx < COOLDOWN_BARS: continue
                if i >= len(rows) - 50: continue
                
                row = rows[i]
                if not row.vol_ok or row.atr <= 0: continue
                
                entry_row = rows[i + 1] if i + 1 < len(rows) else None
                if not entry_row: continue
                
                # === DUAL CONFIRMATION ===
                # Filter 1: Candle Color
                candle_ok = False
                if sig['side'] == 'long':
                    candle_ok = entry_row.close > entry_row.open
                else:
                    candle_ok = entry_row.close < entry_row.open
                
                # Filter 2: Volume Surge
                avg_vol = row.vol_ma
                volume_ok = entry_row.volume > (avg_vol * VOLUME_MULTIPLIER)
                
                # Track filtering
                if not candle_ok and not volume_ok:
                    filtered_both += 1
                    continue
                elif not candle_ok:
                    filtered_candle += 1
                    continue
                elif not volume_ok:
                    filtered_volume += 1
                    continue
                
                entry, sl, tp = calc_pivot_sltp(rows, i, sig['side'], row.atr, sig['swing'])
                result, bars = simulate_trade(rows, i, sig['side'], sl, tp, entry)
                
                if result == 'timeout': continue
                last_trade_idx = i
                
                sig_type = sig['type']
                
                try:
                    trade_date = row.Index.date() if hasattr(row.Index, 'date') else row.date
                    day_idx = all_dates.index(trade_date)
                    period = min(day_idx // days_per_period, NUM_PERIODS - 1)
                except:
                    period = 0
                
                if result == 'win':
                    type_results[sig_type]['w'] += 1
                    type_results[sig_type]['r'] += RISK_REWARD
                    period_results[sig_type][period]['w'] += 1
                else:
                    type_results[sig_type]['l'] += 1
                    type_results[sig_type]['r'] -= 1.0
                    period_results[sig_type][period]['l'] += 1
                    
        except: continue
        
        if (idx + 1) % 30 == 0:
            total = sum(t['w'] + t['l'] for t in type_results.values())
            total_filtered = filtered_candle + filtered_volume + filtered_both
            print(f"  [{idx+1}/{NUM_SYMBOLS}] Trades: {total} | Filtered: {total_filtered}")
    
    total_filtered = filtered_candle + filtered_volume + filtered_both
    print(f"\n🔍 Filtering Breakdown:")
    print(f"   Candle only: {filtered_candle}")
    print(f"   Volume only: {filtered_volume}")
    print(f"   Both failed: {filtered_both}")
    print(f"   Total filtered: {total_filtered}")
    
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    
    for sig_type in ['regular_bullish', 'regular_bearish', 'hidden_bullish', 'hidden_bearish']:
        data = type_results[sig_type]
        n = data['w'] + data['l']
        
        if n > 0:
            wr = data['w'] / n
            ev = calc_ev(wr, RISK_REWARD)
            
            profitable_periods = 0
            for p in range(NUM_PERIODS):
                pd_data = period_results[sig_type][p]
                pd_n = pd_data['w'] + pd_data['l']
                if pd_n > 0:
                    pd_wr = pd_data['w'] / pd_n
                    if calc_ev(pd_wr, RISK_REWARD) > 0:
                        profitable_periods += 1
            
            print(f"{sig_type:<20} {n:>6} trades | {wr*100:>5.1f}% WR | {ev:>+5.2f} EV | {data['r']:>+7.0f}R | {profitable_periods}/6")
    
    total_w = sum(t['w'] for t in type_results.values())
    total_l = sum(t['l'] for t in type_results.values())
    total_n = total_w + total_l
    total_r = sum(t['r'] for t in type_results.values())
    
    if total_n > 0:
        combined_wr = total_w / total_n
        combined_ev = calc_ev(combined_wr, RISK_REWARD)
        
        print(f"\n📊 COMBINED: {total_n} trades, {combined_wr*100:.1f}% WR, {combined_ev:+.2f} EV, {total_r:+.0f}R")
        print(f"   Filtered: {total_filtered} ({total_filtered/(total_n+total_filtered)*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("📊 COMPARISON")
    print("=" * 70)
    print("No Filter:           6694 trades, 53.0% WR, +1.12 EV, +7498R")
    print("Candle Color:        6302 trades, 58.2% WR, +1.33 EV, +8370R")
    if total_n > 0:
        print(f"Candle + Volume:     {total_n} trades, {combined_wr*100:.1f}% WR, {combined_ev:+.2f} EV, {total_r:+.0f}R")
        
        if combined_wr > 0.582:
            print("\n🏆 WINNER: Candle + Volume IMPROVES on Candle Color!")
        elif total_r > 8370:
            print("\n🏆 WINNER: Candle + Volume (More total R despite similar WR)")
        else:
            print("\n⚠️ Candle Color alone is better (volume filter too restrictive)")
    print("=" * 70)

if __name__ == "__main__":
    run()
