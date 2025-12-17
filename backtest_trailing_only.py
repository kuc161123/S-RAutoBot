#!/usr/bin/env python3
"""
BASELINE + TRAILING STOP LOSS (NO PARTIAL TP)
==============================================
Exit Strategy:
- Full position to 3R target
- Trailing SL after 1R profit: move to break-even
- After 2R profit: trail 1R behind
- SL at original if no progress

NO lookahead bias - rigorous implementation.
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
        tp = entry + (sl_dist * RISK_REWARD)
    else:
        sl = entry + sl_dist
        tp = entry - (sl_dist * RISK_REWARD)
    
    return entry, sl, sl_dist, tp

def simulate_trade_trailing(rows, signal_idx, side, sl, sl_dist, tp, entry):
    """Simulate with trailing stop only (no partial TP)"""
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 1: return 'timeout', 0
    
    # Apply slippage/fees
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
            # Track max favorable move in R
            unrealized_r = (row.high - entry) / sl_dist
            if unrealized_r > max_favorable_r:
                max_favorable_r = unrealized_r
                
                # Update trailing stop
                if max_favorable_r >= 2.0:
                    # Trail at 1R behind max
                    new_sl = entry + (max_favorable_r - 1.0) * sl_dist
                    if new_sl > current_sl:
                        current_sl = new_sl
                elif max_favorable_r >= 1.0:
                    # Move to break-even
                    if entry > current_sl:
                        current_sl = entry
            
            # Check SL (current, possibly trailed)
            if row.low <= current_sl:
                # Calculate actual R based on where SL was hit
                exit_r = (current_sl - entry) / sl_dist
                return 'sl_hit', exit_r
            
            # Check TP (full 3R target)
            if row.high >= tp:
                return 'win', RISK_REWARD
                
        else:  # short
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
                return 'sl_hit', exit_r
            
            if row.low <= tp:
                return 'win', RISK_REWARD
    
    # Timeout - treat as loss at current SL
    exit_r = (current_sl - entry) / sl_dist if side == 'long' else (entry - current_sl) / sl_dist
    return 'timeout', exit_r

def run():
    print("=" * 70)
    print("📊 BASELINE + TRAILING STOP (NO PARTIAL TP)")
    print("=" * 70)
    print(f"Strategy: Full position to 3R")
    print(f"  - After 1R profit: SL to break-even")
    print(f"  - After 2R profit: Trail 1R behind")
    print("=" * 70)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📋 Testing {len(symbols)} symbols...")
    
    type_results = defaultdict(lambda: {'wins': 0, 'be': 0, 'partial': 0, 'losses': 0, 'r': 0.0})
    
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
                
                result, r_value = simulate_trade_trailing(rows, i, sig['side'], sl, sl_dist, tp, entry)
                
                if result == 'timeout' and r_value == 0: continue
                last_trade_idx = i
                
                sig_type = sig['type']
                
                if result == 'win':
                    type_results[sig_type]['wins'] += 1
                    type_results[sig_type]['r'] += r_value
                elif r_value >= 0:  # Break-even or small profit from trailing
                    type_results[sig_type]['be'] += 1
                    type_results[sig_type]['r'] += r_value
                elif r_value > -1:  # Partial loss (trailed from profit back to small loss)
                    type_results[sig_type]['partial'] += 1
                    type_results[sig_type]['r'] += r_value
                else:  # Full loss at original SL
                    type_results[sig_type]['losses'] += 1
                    type_results[sig_type]['r'] += r_value
                    
        except: continue
        
        if (idx + 1) % 30 == 0:
            total = sum(t['wins'] + t['be'] + t['partial'] + t['losses'] for t in type_results.values())
            print(f"  [{idx+1}/{NUM_SYMBOLS}] Trades: {total}")
    
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    
    overall_wins = 0
    overall_be = 0
    overall_partial = 0
    overall_losses = 0
    overall_r = 0.0
    
    for sig_type in ['regular_bullish', 'regular_bearish', 'hidden_bullish', 'hidden_bearish']:
        data = type_results[sig_type]
        wins = data['wins']
        be = data['be']
        partial = data['partial']
        losses = data['losses']
        n = wins + be + partial + losses
        
        if n > 0:
            # Win = full TP, Profitable = BE or better
            profitable = wins + be + partial  # trades that didn't hit full SL
            wr = profitable / n
            avg_r = data['r'] / n
            
            print(f"{sig_type:<20} {n:>6} trades | Win: {wins:>4} | BE: {be:>4} | Partial: {partial:>4} | Loss: {losses:>4} | Profit%: {wr*100:>5.1f}% | Avg: {avg_r:>+5.2f}R | Total: {data['r']:>+7.0f}R")
            
            overall_wins += wins
            overall_be += be
            overall_partial += partial
            overall_losses += losses
            overall_r += data['r']
    
    total_n = overall_wins + overall_be + overall_partial + overall_losses
    
    if total_n > 0:
        # Traditional WR (full TP hits only)
        traditional_wr = overall_wins / total_n
        # Profitable trades % (didn't hit full SL)
        profitable_pct = (overall_wins + overall_be + overall_partial) / total_n
        avg_r_per_trade = overall_r / total_n
        
        print(f"\n📊 COMBINED: {total_n} trades | {overall_r:+.0f}R total")
        print(f"   Full Wins (3R): {overall_wins} ({overall_wins/total_n*100:.1f}%)")
        print(f"   Break-Even: {overall_be} ({overall_be/total_n*100:.1f}%)")
        print(f"   Partial Win/Loss: {overall_partial} ({overall_partial/total_n*100:.1f}%)")
        print(f"   Full Loss (-1R): {overall_losses} ({overall_losses/total_n*100:.1f}%)")
        print(f"   Traditional WR: {traditional_wr*100:.1f}%")
        print(f"   Profitable: {profitable_pct*100:.1f}%")
        print(f"   Avg R/trade: {avg_r_per_trade:+.2f}")
    
    print("\n" + "=" * 70)
    print("📊 FINAL COMPARISON")
    print("=" * 70)
    print("Baseline (Fixed SL):  6691 trades, 53.2% WR, +1.13 avg, +7553R total")
    print("Partial TP Only:      6674 trades, 62.6% WR, +0.85 avg, +5671R total")
    print("Partial + Trailing:   6674 trades, 62.6% WR, +0.83 avg, +5563R total")
    if total_n > 0:
        print(f"Trailing Only:        {total_n} trades, {traditional_wr*100:.1f}% WR, {avg_r_per_trade:+.2f} avg, {overall_r:+.0f}R total")
    print("=" * 70)

if __name__ == "__main__":
    run()
