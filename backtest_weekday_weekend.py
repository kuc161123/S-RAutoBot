#!/usr/bin/env python3
"""
RSI DIVERGENCE WEEKDAY vs WEEKEND BACKTEST
============================================
Compares performance on:
1. Weekdays only (Mon-Fri)
2. Weekends only (Sat-Sun)
3. All days combined

Uses pivot-based SL with 3:1 R:R (current strategy settings).
"""

import requests
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TIMEFRAME = '15'
DATA_DAYS = 60
NUM_SYMBOLS = 150

SLIPPAGE_PCT = 0.0005
FEE_PCT = 0.0004
TOTAL_COST = (SLIPPAGE_PCT + FEE_PCT) * 2

RR_RATIO = 3.0  # Current strategy: 3:1 R:R

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5

BASE_URL = "https://api.bybit.com"

# =============================================================================
# HELPERS
# =============================================================================

def calc_ev(wr, rr):
    return (wr * rr) - (1 - wr)

def wilson_lb(wins, n, z=1.96):
    if n == 0: return 0.0
    p = wins / n
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    return max(0, (centre - spread) / denom)

def get_symbols(limit=150):
    resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear")
    tickers = resp.json().get('result', {}).get('list', [])
    usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in usdt[:limit]]

def fetch_klines(symbol, interval, days):
    end_ts = int(datetime.now().timestamp() * 1000)
    all_candles = []
    current_end = end_ts
    candles_needed = days * 24 * 60 // int(interval)
    
    while len(all_candles) < candles_needed:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
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
    for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = df[col].astype(float)
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
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

# =============================================================================
# DIVERGENCE DETECTION
# =============================================================================

def detect_divergence_signals(df):
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
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE: prev_pl, prev_pli = price_pl[j], j; break
        
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE: prev_ph, prev_phi = price_ph[j], j; break
        
        if curr_pl and prev_pl and curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli] and rsi[i] < RSI_OVERSOLD + 15:
            signals.append({'idx': i, 'side': 'long'}); continue
        if curr_ph and prev_ph and curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi] and rsi[i] > RSI_OVERBOUGHT - 15:
            signals.append({'idx': i, 'side': 'short'}); continue
        if curr_pl and prev_pl and curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli] and rsi[i] < RSI_OVERBOUGHT - 10:
            signals.append({'idx': i, 'side': 'long'}); continue
        if curr_ph and prev_ph and curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi] and rsi[i] > RSI_OVERSOLD + 10:
            signals.append({'idx': i, 'side': 'short'})
    
    return signals

# =============================================================================
# PIVOT-BASED SL/TP (Matches live bot)
# =============================================================================

def calc_pivot_sltp(rows, idx, side, atr, lookback=15, rr=3.0):
    """Calculate pivot-based SL/TP matching live bot logic"""
    entry = rows[idx + 1].open if idx + 1 < len(rows) else rows[idx].close
    
    # Get recent swing
    recent_lows = [rows[j].low for j in range(max(0, idx - lookback + 1), idx + 1)]
    recent_highs = [rows[j].high for j in range(max(0, idx - lookback + 1), idx + 1)]
    
    if side == 'long':
        swing_low = min(recent_lows)
        sl = swing_low
        sl_distance = abs(entry - sl)
    else:
        swing_high = max(recent_highs)
        sl = swing_high
        sl_distance = abs(sl - entry)
    
    # Apply min/max constraints
    min_sl_dist = 0.3 * atr
    max_sl_dist = 2.0 * atr
    
    if sl_distance < min_sl_dist:
        sl_distance = min_sl_dist
        sl = entry - sl_distance if side == 'long' else entry + sl_distance
    elif sl_distance > max_sl_dist:
        sl_distance = max_sl_dist
        sl = entry - sl_distance if side == 'long' else entry + sl_distance
    
    # Calculate TP
    if side == 'long':
        tp = entry + (rr * sl_distance)
    else:
        tp = entry - (rr * sl_distance)
    
    return entry, sl, tp, rr

def simulate_trade(rows, signal_idx, side, sl, tp, entry):
    start_idx = signal_idx + 1
    if start_idx >= len(rows) - 50:
        return 'timeout', 0
    
    # Apply slippage
    if side == 'long':
        entry = entry * (1 + SLIPPAGE_PCT)
        tp = tp * (1 - TOTAL_COST)
    else:
        entry = entry * (1 - SLIPPAGE_PCT)
        tp = tp * (1 + TOTAL_COST)
    
    for bar_idx, row in enumerate(rows[start_idx + 1:start_idx + 100]):
        if side == 'long':
            if row.low <= sl: return 'loss', bar_idx + 1
            if row.high >= tp: return 'win', bar_idx + 1
        else:
            if row.high >= sl: return 'loss', bar_idx + 1
            if row.low <= tp: return 'win', bar_idx + 1
    
    return 'timeout', 100

# =============================================================================
# MAIN
# =============================================================================

def run_weekend_comparison():
    print("=" * 80)
    print("🔬 RSI DIVERGENCE: WEEKDAY vs WEEKEND BACKTEST")
    print("=" * 80)
    print("\nComparing performance:")
    print("  1. Weekdays only (Mon-Fri)")
    print("  2. Weekends only (Sat-Sun)")
    print("  3. All days combined")
    print(f"\nStrategy: Pivot SL + {RR_RATIO}:1 R:R")
    print("=" * 80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📦 Fetching data for {len(symbols)} symbols...\n")
    
    # Results by day type
    results = {
        'weekday': {'wins': 0, 'losses': 0, 'by_day': defaultdict(lambda: {'w': 0, 'l': 0})},
        'weekend': {'wins': 0, 'losses': 0, 'by_day': defaultdict(lambda: {'w': 0, 'l': 0})},
        'all': {'wins': 0, 'losses': 0}
    }
    
    start_time = time.time()
    
    for idx, sym in enumerate(symbols):
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 400: continue
            
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            
            hl = df['high'] - df['low']
            hc = abs(df['high'] - df['close'].shift())
            lc = abs(df['low'] - df['close'].shift())
            df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            df = df.dropna()
            
            if len(df) < 150: continue
            
            signals = detect_divergence_signals(df)
            rows = list(df.itertuples())
            last_idx = -20
            
            for sig in signals:
                i = sig['idx']
                if i - last_idx < 10 or i >= len(rows) - 100: continue
                row = rows[i]
                if pd.isna(row.atr) or row.atr <= 0 or not row.vol_ok: continue
                
                # Determine day of week
                signal_time = row.Index
                day_of_week = signal_time.weekday()  # 0=Mon, 6=Sun
                day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day_of_week]
                is_weekend = day_of_week >= 5  # Sat=5, Sun=6
                
                # Calculate pivot-based SL/TP
                entry, sl, tp, rr = calc_pivot_sltp(rows, i, sig['side'], row.atr)
                
                # Simulate trade
                result, bars = simulate_trade(rows, i, sig['side'], sl, tp, entry)
                
                if result == 'timeout':
                    continue
                
                last_idx = i
                
                # Record results
                day_type = 'weekend' if is_weekend else 'weekday'
                
                if result == 'win':
                    results[day_type]['wins'] += 1
                    results[day_type]['by_day'][day_name]['w'] += 1
                    results['all']['wins'] += 1
                else:
                    results[day_type]['losses'] += 1
                    results[day_type]['by_day'][day_name]['l'] += 1
                    results['all']['losses'] += 1
            
            if (idx + 1) % 25 == 0:
                total = results['all']['wins'] + results['all']['losses']
                print(f"  [{idx+1}/{NUM_SYMBOLS}] Trades: {total}")
            
            time.sleep(0.02)
        except: continue
    
    elapsed = time.time() - start_time
    
    # ==========================================================================
    # RESULTS
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("📊 WEEKDAY vs WEEKEND RESULTS")
    print("=" * 80)
    print(f"\n⏱️ Completed in {elapsed/60:.1f} minutes")
    
    # Summary table
    print("\n" + "-" * 60)
    print("SUMMARY COMPARISON")
    print("-" * 60)
    print(f"\n{'Period':<15} {'Trades':<10} {'Wins':<8} {'Losses':<8} {'WR%':<10} {'EV':<10} {'Total R':<12}")
    print("-" * 75)
    
    for period in ['weekday', 'weekend', 'all']:
        r = results[period]
        total = r['wins'] + r['losses']
        if total > 0:
            wr = r['wins'] / total
            ev = calc_ev(wr, RR_RATIO)
            lb = wilson_lb(r['wins'], total)
            total_r = (r['wins'] * RR_RATIO) - r['losses']
        else:
            wr = ev = lb = total_r = 0
        
        label = period.upper()
        emoji = "📅" if period == 'weekday' else ("🏖️" if period == 'weekend' else "📊")
        print(f"{emoji} {label:<12} {total:<10} {r['wins']:<8} {r['losses']:<8} {wr*100:.1f}%{'':<4} {ev:+.2f}{'':<4} {total_r:+,.0f}R")
    
    # By day of week
    print("\n" + "-" * 60)
    print("BY DAY OF WEEK")
    print("-" * 60)
    
    day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"\n{'Day':<8} {'Trades':<10} {'WR%':<10} {'EV':<10} {'Type':<10}")
    print("-" * 50)
    
    for day in day_order:
        # Check weekday or weekend
        if day in ['Sat', 'Sun']:
            day_data = results['weekend']['by_day'][day]
            day_type = "Weekend"
        else:
            day_data = results['weekday']['by_day'][day]
            day_type = "Weekday"
        
        total = day_data['w'] + day_data['l']
        if total > 0:
            wr = day_data['w'] / total
            ev = calc_ev(wr, RR_RATIO)
        else:
            wr = ev = 0
        
        print(f"{day:<8} {total:<10} {wr*100:.1f}%{'':<4} {ev:+.2f}{'':<4} {day_type}")
    
    # Analysis and recommendation
    print("\n" + "=" * 80)
    print("🎯 ANALYSIS & RECOMMENDATION")
    print("=" * 80)
    
    wd = results['weekday']
    we = results['weekend']
    
    wd_total = wd['wins'] + wd['losses']
    we_total = we['wins'] + we['losses']
    
    if wd_total > 0 and we_total > 0:
        wd_wr = wd['wins'] / wd_total
        we_wr = we['wins'] / we_total
        wd_ev = calc_ev(wd_wr, RR_RATIO)
        we_ev = calc_ev(we_wr, RR_RATIO)
        wd_total_r = (wd['wins'] * RR_RATIO) - wd['losses']
        we_total_r = (we['wins'] * RR_RATIO) - we['losses']
        
        print(f"\n📅 **WEEKDAYS**")
        print(f"   Trades: {wd_total} | WR: {wd_wr*100:.1f}% | EV: {wd_ev:+.2f}R | P&L: {wd_total_r:+,.0f}R")
        
        print(f"\n🏖️ **WEEKENDS**")
        print(f"   Trades: {we_total} | WR: {we_wr*100:.1f}% | EV: {we_ev:+.2f}R | P&L: {we_total_r:+,.0f}R")
        
        # Comparison
        wr_diff = we_wr - wd_wr
        ev_diff = we_ev - wd_ev
        
        print(f"\n📊 **COMPARISON**")
        print(f"   Weekend vs Weekday:")
        print(f"   - Win Rate: {wr_diff*100:+.1f}%")
        print(f"   - EV: {ev_diff:+.2f}R per trade")
        
        # Recommendation
        print(f"\n💡 **RECOMMENDATION**")
        
        if we_ev > 0 and we_wr >= 0.5:
            if we_ev >= wd_ev * 0.9:
                print("   ✅ TRADE WEEKENDS - Performance is similar to weekdays")
                print(f"      Both periods are profitable with positive EV")
            else:
                print("   ⚠️ WEEKENDS OK but WEAKER than weekdays")
                print(f"      Consider reducing position size on weekends")
        elif we_ev > 0:
            print("   ⚠️ WEEKENDS MARGINAL - Lower win rate affects EV")
            print(f"      Trade with caution or reduced size")
        else:
            print("   ❌ AVOID WEEKENDS - Negative expected value")
            print(f"      Consider pausing the bot on Sat-Sun")
    else:
        print("\n⚠️ Insufficient data for comparison")

if __name__ == "__main__":
    run_weekend_comparison()
