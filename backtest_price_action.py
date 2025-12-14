#!/usr/bin/env python3
"""
RSI DIVERGENCE + PRICE ACTION CONFIRMATION BACKTEST
====================================================
Tests different price action confirmations with RSI divergence:

1. No Confirmation (current strategy)
2. Engulfing Pattern (bullish/bearish engulfing)
3. Pin Bar / Hammer (reversal candlestick)
4. Confirmation Candle (next bar closes in direction)
5. Strong Close (close in top/bottom 30% of range)

All methods avoid common backtesting pitfalls.
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

TP_ATR_MULT = 2.05
SL_ATR_MULT = 1.0

RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5

BASE_URL = "https://api.bybit.com"

# =============================================================================
# HELPERS
# =============================================================================

def calc_ev(wr, rr=2.05):
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
# PRICE ACTION PATTERNS
# =============================================================================

def is_bullish_engulfing(rows, i):
    """Check if candle at i is a bullish engulfing pattern"""
    if i < 1: return False
    curr = rows[i]
    prev = rows[i-1]
    
    # Current candle must be bullish (close > open)
    if curr.close <= curr.open: return False
    # Previous candle must be bearish (close < open)
    if prev.close >= prev.open: return False
    # Current body must engulf previous body
    if curr.open >= prev.close or curr.close <= prev.open: return False
    
    return True

def is_bearish_engulfing(rows, i):
    """Check if candle at i is a bearish engulfing pattern"""
    if i < 1: return False
    curr = rows[i]
    prev = rows[i-1]
    
    # Current candle must be bearish (close < open)
    if curr.close >= curr.open: return False
    # Previous candle must be bullish (close > open)
    if prev.close <= prev.open: return False
    # Current body must engulf previous body
    if curr.open <= prev.close or curr.close >= prev.open: return False
    
    return True

def is_hammer(rows, i):
    """Check if candle at i is a hammer/pin bar (bullish reversal)"""
    curr = rows[i]
    body = abs(curr.close - curr.open)
    candle_range = curr.high - curr.low
    
    if candle_range == 0: return False
    
    lower_wick = min(curr.open, curr.close) - curr.low
    upper_wick = curr.high - max(curr.open, curr.close)
    
    # Lower wick should be at least 2x the body
    # Upper wick should be small
    if lower_wick >= 2 * body and upper_wick <= body * 0.5:
        return True
    
    return False

def is_shooting_star(rows, i):
    """Check if candle at i is a shooting star (bearish reversal)"""
    curr = rows[i]
    body = abs(curr.close - curr.open)
    candle_range = curr.high - curr.low
    
    if candle_range == 0: return False
    
    lower_wick = min(curr.open, curr.close) - curr.low
    upper_wick = curr.high - max(curr.open, curr.close)
    
    # Upper wick should be at least 2x the body
    # Lower wick should be small
    if upper_wick >= 2 * body and lower_wick <= body * 0.5:
        return True
    
    return False

def is_confirmation_candle(rows, i, side):
    """Check if candle at i confirms the direction"""
    if i < 1: return False
    curr = rows[i]
    prev = rows[i-1]
    
    if side == 'long':
        # Current candle should close higher than it opened (bullish)
        # And close above previous close
        return curr.close > curr.open and curr.close > prev.close
    else:
        # Current candle should close lower than it opened (bearish)
        # And close below previous close
        return curr.close < curr.open and curr.close < prev.close

def is_strong_close(rows, i, side):
    """Check if the close is in favorable part of the range"""
    curr = rows[i]
    candle_range = curr.high - curr.low
    
    if candle_range == 0: return False
    
    close_position = (curr.close - curr.low) / candle_range  # 0 = low, 1 = high
    
    if side == 'long':
        # For longs, close should be in top 30% of range
        return close_position >= 0.7
    else:
        # For shorts, close should be in bottom 30% of range
        return close_position <= 0.3

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
        
        # Regular Bullish
        if curr_pl and prev_pl and curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli] and rsi[i] < RSI_OVERSOLD + 15:
            signals.append({'idx': i, 'side': 'long', 'type': 'regular'}); continue
        # Regular Bearish
        if curr_ph and prev_ph and curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi] and rsi[i] > RSI_OVERBOUGHT - 15:
            signals.append({'idx': i, 'side': 'short', 'type': 'regular'}); continue
        # Hidden Bullish
        if curr_pl and prev_pl and curr_pl > prev_pl and rsi[curr_pli] < rsi[prev_pli] and rsi[i] < RSI_OVERBOUGHT - 10:
            signals.append({'idx': i, 'side': 'long', 'type': 'hidden'}); continue
        # Hidden Bearish
        if curr_ph and prev_ph and curr_ph < prev_ph and rsi[curr_phi] > rsi[prev_phi] and rsi[i] > RSI_OVERSOLD + 10:
            signals.append({'idx': i, 'side': 'short', 'type': 'hidden'})
    
    return signals

# =============================================================================
# TRADE SIMULATION
# =============================================================================

def simulate_trade(rows, signal_idx, side, atr):
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 50:
        return 'timeout', 0
    
    base_entry = rows[entry_idx].open
    
    if side == 'long':
        entry = base_entry * (1 + SLIPPAGE_PCT)
        tp = entry + (TP_ATR_MULT * atr)
        sl = entry - (SL_ATR_MULT * atr)
        tp = tp * (1 - TOTAL_COST)
    else:
        entry = base_entry * (1 - SLIPPAGE_PCT)
        tp = entry - (TP_ATR_MULT * atr)
        sl = entry + (SL_ATR_MULT * atr)
        tp = tp * (1 + TOTAL_COST)
    
    for bar_idx, future_row in enumerate(rows[entry_idx+1:entry_idx+100]):
        if side == 'long':
            if future_row.low <= sl: return 'loss', bar_idx + 1
            if future_row.high >= tp: return 'win', bar_idx + 1
        else:
            if future_row.high >= sl: return 'loss', bar_idx + 1
            if future_row.low <= tp: return 'win', bar_idx + 1
    
    return 'timeout', 100

# =============================================================================
# MAIN
# =============================================================================

def run_comparison():
    print("=" * 80)
    print("🔬 RSI DIVERGENCE + PRICE ACTION CONFIRMATION BACKTEST")
    print("=" * 80)
    print("\nTesting Confirmation Methods:")
    print("  1. None (current strategy)")
    print("  2. Engulfing Pattern")
    print("  3. Pin Bar / Hammer")
    print("  4. Confirmation Candle (next bar direction)")
    print("  5. Strong Close (close in favorable 30%)")
    print("=" * 80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📦 Fetching data for {len(symbols)} symbols...\n")
    
    # Load all data and signals
    all_signals = []
    start = time.time()
    
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
                
                all_signals.append({
                    'rows': rows,
                    'idx': i,
                    'side': sig['side'],
                    'atr': row.atr,
                    'type': sig['type']
                })
                last_idx = i
            
            if (idx + 1) % 25 == 0:
                print(f"  [{idx+1}/{NUM_SYMBOLS}] Signals: {len(all_signals)}")
            
            time.sleep(0.02)
        except: continue
    
    print(f"\n⏱️ Data loaded in {(time.time()-start)/60:.1f}m | {len(all_signals)} signals")
    
    # Test each confirmation method
    methods = {
        'None (Current)': lambda rows, i, side: True,
        'Engulfing': lambda rows, i, side: is_bullish_engulfing(rows, i) if side == 'long' else is_bearish_engulfing(rows, i),
        'Pin Bar': lambda rows, i, side: is_hammer(rows, i) if side == 'long' else is_shooting_star(rows, i),
        'Confirm Candle': lambda rows, i, side: is_confirmation_candle(rows, i, side),
        'Strong Close': lambda rows, i, side: is_strong_close(rows, i, side)
    }
    
    results = {}
    
    for method_name, check_fn in methods.items():
        print(f"\n  Testing: {method_name}...")
        wins = losses = filtered = 0
        
        for sig in all_signals:
            rows = sig['rows']
            i = sig['idx']
            side = sig['side']
            atr = sig['atr']
            
            # Check confirmation
            if not check_fn(rows, i, side):
                filtered += 1
                continue
            
            # Simulate trade
            result, bars = simulate_trade(rows, i, side, atr)
            
            if result == 'win': wins += 1
            elif result == 'loss': losses += 1
        
        total = wins + losses
        wr = wins / total if total > 0 else 0
        ev = calc_ev(wr)
        lb = wilson_lb(wins, total)
        total_r = (wins * TP_ATR_MULT) - losses
        filter_rate = filtered / len(all_signals) * 100 if all_signals else 0
        
        results[method_name] = {
            'wins': wins, 'losses': losses, 'total': total,
            'wr': wr, 'ev': ev, 'lb': lb, 'total_r': total_r,
            'filtered': filtered, 'filter_rate': filter_rate
        }
    
    # Print comparison
    print("\n" + "=" * 80)
    print("📊 RESULTS COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Method':<18} {'Trades':<8} {'Filtered':<10} {'WR%':<8} {'LB%':<8} {'EV':<8} {'Total R':<12}")
    print("-" * 80)
    
    best_ev = max(r['ev'] for r in results.values())
    best_total = max(r['total_r'] for r in results.values())
    
    for name, r in sorted(results.items(), key=lambda x: x[1]['ev'], reverse=True):
        ev_flag = "🏆" if r['ev'] == best_ev else "  "
        total_flag = "💰" if r['total_r'] == best_total else "  "
        print(f"{name:<18} {r['total']:<8} {r['filter_rate']:.0f}%{'':<5} {r['wr']*100:.1f}%{'':<2} {r['lb']*100:.1f}%{'':<2} {r['ev']:+.2f}{'':<2} {r['total_r']:+,.0f}R {ev_flag}{total_flag}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("🏆 ANALYSIS")
    print("=" * 80)
    
    base = results.get('None (Current)', {})
    best = max(results.items(), key=lambda x: x[1]['ev'])
    
    print(f"\n📊 Current Strategy (No Confirmation):")
    print(f"   Trades: {base['total']} | WR: {base['wr']*100:.1f}% | EV: {base['ev']:+.2f} | Total: {base['total_r']:+,.0f}R")
    
    print(f"\n📊 Best by EV: {best[0]}")
    print(f"   Trades: {best[1]['total']} | WR: {best[1]['wr']*100:.1f}% | EV: {best[1]['ev']:+.2f} | Total: {best[1]['total_r']:+,.0f}R")
    
    if best[0] != 'None (Current)' and best[1]['ev'] > base['ev']:
        ev_diff = best[1]['ev'] - base['ev']
        print(f"\n✅ {best[0]} improves EV by +{ev_diff:.2f}")
        print(f"   But reduces trades by {base['total'] - best[1]['total']} ({(base['total'] - best[1]['total'])/base['total']*100:.0f}%)")
        
        # Check if total profit is better
        if best[1]['total_r'] > base['total_r']:
            print(f"   ✅ AND increases total profit by {best[1]['total_r'] - base['total_r']:+,.0f}R")
        else:
            print(f"   ⚠️ BUT reduces total profit by {base['total_r'] - best[1]['total_r']:+,.0f}R")
    else:
        print(f"\n✅ Current strategy (no confirmation) remains optimal!")

if __name__ == "__main__":
    run_comparison()
