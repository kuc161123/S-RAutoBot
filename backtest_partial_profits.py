#!/usr/bin/env python3
"""
BASELINE + PARTIAL PROFIT TAKING (50% at 1.5R)
==============================================
Exit Strategy:
- Close 50% of position at 1.5R
- Let remaining 50% run to 3R or SL

NO lookahead bias - all best practices applied.
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

FULL_RR = 3.0  # Full target
PARTIAL_RR = 1.5  # Partial target (50% of full)
MIN_SL_ATR = 0.3
MAX_SL_ATR = 2.0

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
        tp_partial = entry + (sl_dist * PARTIAL_RR)  # 1.5R
        tp_full = entry + (sl_dist * FULL_RR)  # 3R
    else:
        sl = entry + sl_dist
        tp_partial = entry - (sl_dist * PARTIAL_RR)
        tp_full = entry - (sl_dist * FULL_RR)
    
    return entry, sl, tp_partial, tp_full

def simulate_trade_partial(rows, signal_idx, side, sl, tp_partial, tp_full, entry):
    """Simulate with partial profit taking"""
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 1: return 'timeout', 0
    
    # Apply slippage/fees
    if side == 'long':
        entry = entry * (1 + SLIPPAGE_PCT)
        tp_partial = tp_partial * (1 - TOTAL_COST)
        tp_full = tp_full * (1 - TOTAL_COST)
    else:
        entry = entry * (1 - SLIPPAGE_PCT)
        tp_partial = tp_partial * (1 + TOTAL_COST)
        tp_full = tp_full * (1 + TOTAL_COST)
    
    partial_closed = False
    total_r = 0.0
    
    for bar_idx, row in enumerate(rows[entry_idx:entry_idx + 100]):
        if side == 'long':
            # Check SL first (full position or remaining 50%)
            if row.low <= sl:
                if partial_closed:
                    # 50% already closed at +1.5R, 50% hits SL at -1R
                    total_r = (PARTIAL_RR * 0.5) + (-1.0 * 0.5)  # = +0.25R
                    return 'partial_win', total_r
                else:
                    # Full position hits SL
                    return 'full_loss', -1.0
            
            # Check partial TP (50% of position)
            if not partial_closed and row.high >= tp_partial:
                partial_closed = True
                # Continue to see if remaining 50% hits full TP or SL
            
            # Check full TP (remaining 50%)
            if partial_closed and row.high >= tp_full:
                # 50% closed at 1.5R + 50% at 3R
                total_r = (PARTIAL_RR * 0.5) + (FULL_RR * 0.5)  # = +2.25R
                return 'full_win', total_r
                
        else:  # short
            if row.high >= sl:
                if partial_closed:
                    total_r = (PARTIAL_RR * 0.5) + (-1.0 * 0.5)  # = +0.25R
                    return 'partial_win', total_r
                else:
                    return 'full_loss', -1.0
            
            if not partial_closed and row.low <= tp_partial:
                partial_closed = True
            
            if partial_closed and row.low <= tp_full:
                total_r = (PARTIAL_RR * 0.5) + (FULL_RR * 0.5)  # = +2.25R
                return 'full_win', total_r
    
    # Timeout
    if partial_closed:
        # 50% closed, 50% timeout (treat as SL)
        total_r = (PARTIAL_RR * 0.5) + (-1.0 * 0.5)
        return 'partial_win', total_r
    return 'timeout', 0

def run():
    print("=" * 70)
    print("📊 PARTIAL PROFIT TAKING (50% at 1.5R)")
    print("=" * 70)
    print(f"Strategy: Close 50% at 1.5R, let 50% run to 3R")
    print(f"Full Win: +2.25R | Partial Win: +0.25R | Loss: -1.0R")
    print("=" * 70)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📋 Testing {len(symbols)} symbols...")
    
    type_results = defaultdict(lambda: {'full_wins': 0, 'partial_wins': 0, 'losses': 0, 'r': 0.0})
    
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
                
                entry, sl, tp_partial, tp_full = calc_pivot_sltp(rows, i, sig['side'], row.atr, sig['swing'])
                if entry is None: continue
                
                result, r_value = simulate_trade_partial(rows, i, sig['side'], sl, tp_partial, tp_full, entry)
                
                if result == 'timeout': continue
                last_trade_idx = i
                
                sig_type = sig['type']
                
                if result == 'full_win':
                    type_results[sig_type]['full_wins'] += 1
                    type_results[sig_type]['r'] += r_value
                elif result == 'partial_win':
                    type_results[sig_type]['partial_wins'] += 1
                    type_results[sig_type]['r'] += r_value
                else:  # full_loss
                    type_results[sig_type]['losses'] += 1
                    type_results[sig_type]['r'] += r_value
                    
        except: continue
        
        if (idx + 1) % 30 == 0:
            total = sum(t['full_wins'] + t['partial_wins'] + t['losses'] for t in type_results.values())
            print(f"  [{idx+1}/{NUM_SYMBOLS}] Trades: {total}")
    
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    
    overall_full = 0
    overall_partial = 0
    overall_losses = 0
    overall_r = 0.0
    
    for sig_type in ['regular_bullish', 'regular_bearish', 'hidden_bullish', 'hidden_bearish']:
        data = type_results[sig_type]
        full_w = data['full_wins']
        partial_w = data['partial_wins']
        losses = data['losses']
        n = full_w + partial_w + losses
        
        if n > 0:
            total_wins = full_w + partial_w
            wr = total_wins / n
            avg_r = data['r'] / n
            
            print(f"{sig_type:<20} {n:>6} trades | Full: {full_w:>4} | Partial: {partial_w:>4} | Loss: {losses:>4} | WR: {wr*100:>5.1f}% | Avg: {avg_r:>+5.2f}R | Total: {data['r']:>+7.0f}R")
            
            overall_full += full_w
            overall_partial += partial_w
            overall_losses += losses
            overall_r += data['r']
    
    total_n = overall_full + overall_partial + overall_losses
    
    if total_n > 0:
        total_wins = overall_full + overall_partial
        combined_wr = total_wins / total_n
        avg_r_per_trade = overall_r / total_n
        
        print(f"\n📊 COMBINED: {total_n} trades | {combined_wr*100:.1f}% WR | {avg_r_per_trade:+.2f}R avg | {overall_r:+.0f}R total")
        print(f"   Full Wins: {overall_full} ({overall_full/total_n*100:.1f}%)")
        print(f"   Partial Wins: {overall_partial} ({overall_partial/total_n*100:.1f}%)")
        print(f"   Losses: {overall_losses} ({overall_losses/total_n*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("📊 COMPARISON TO BASELINE")
    print("=" * 70)
    print("Baseline:        6691 trades, 53.2% WR, +1.13 avg, +7553R total")
    if total_n > 0:
        print(f"Partial Profits: {total_n} trades, {combined_wr*100:.1f}% WR, {avg_r_per_trade:+.2f} avg, {overall_r:+.0f}R total")
        
        wr_diff = (combined_wr - 0.532) * 100
        r_diff = overall_r - 7553
        
        print(f"\nChanges:")
        print(f"  Win Rate: {wr_diff:+.1f}%")
        print(f"  Total R: {r_diff:+.0f}R ({r_diff/7553*100:+.1f}%)")
        
        if combined_wr > 0.532 and overall_r > 7000:
            print("\n✅ VERDICT: Partial profits IMPROVE performance!")
        elif combined_wr > 0.532:
            print("\n⚠️ VERDICT: Higher WR but lower total R")
        else:
            print("\n❌ VERDICT: Baseline is better")
    
    print("=" * 70)

if __name__ == "__main__":
    run()
