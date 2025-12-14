#!/usr/bin/env python3
"""
PIVOT POINT VS ATR STOP LOSS COMPARISON
========================================
Compares different stop loss methods:

1. ATR Current: SL = 1×ATR from entry
2. Pivot 2:1: SL at recent swing, TP = 2× distance
3. Pivot 2.5:1: SL at recent swing, TP = 2.5× distance
4. Pivot 3:1: SL at recent swing, TP = 3× distance
5. Tight Pivot: SL at swing with small buffer
6. Wide Pivot: SL at swing with larger buffer

Using pivot points makes SL placement more dynamic based on market structure.
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
# SL/TP CALCULATION METHODS
# =============================================================================

def calc_atr_sltp(rows, idx, side, atr, rr=2.05):
    """Current method: ATR-based SL/TP"""
    entry = rows[idx + 1].open if idx + 1 < len(rows) else rows[idx].close
    
    if side == 'long':
        sl = entry - atr
        tp = entry + (rr * atr)
    else:
        sl = entry + atr
        tp = entry - (rr * atr)
    
    return entry, sl, tp, rr

def find_recent_swing(rows, idx, side, lookback=15):
    """Find the most recent swing high/low for SL placement"""
    if side == 'long':
        # Find recent swing low
        lows = [rows[j].low for j in range(max(0, idx - lookback), idx + 1)]
        if lows:
            return min(lows)
    else:
        # Find recent swing high
        highs = [rows[j].high for j in range(max(0, idx - lookback), idx + 1)]
        if highs:
            return max(highs)
    return None

def calc_pivot_sltp(rows, idx, side, atr, rr=2.0, buffer_mult=0.0):
    """Pivot-based SL/TP with configurable R:R and buffer"""
    entry = rows[idx + 1].open if idx + 1 < len(rows) else rows[idx].close
    swing = find_recent_swing(rows, idx, side)
    
    if swing is None:
        # Fallback to ATR if no swing found
        if side == 'long':
            sl = entry - atr
        else:
            sl = entry + atr
    else:
        if side == 'long':
            # SL below swing low with optional buffer
            sl = swing - (buffer_mult * atr)
        else:
            # SL above swing high with optional buffer
            sl = swing + (buffer_mult * atr)
    
    # Calculate SL distance
    sl_distance = abs(entry - sl)
    
    # Minimum SL distance (avoid too tight stops)
    min_sl = 0.3 * atr
    if sl_distance < min_sl:
        sl_distance = min_sl
        if side == 'long':
            sl = entry - sl_distance
        else:
            sl = entry + sl_distance
    
    # Maximum SL distance (avoid too wide stops)
    max_sl = 2.0 * atr
    if sl_distance > max_sl:
        sl_distance = max_sl
        if side == 'long':
            sl = entry - sl_distance
        else:
            sl = entry + sl_distance
    
    # Calculate TP based on R:R
    if side == 'long':
        tp = entry + (rr * sl_distance)
    else:
        tp = entry - (rr * sl_distance)
    
    # Calculate actual R:R achieved
    actual_rr = rr
    
    return entry, sl, tp, actual_rr

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
# TRADE SIMULATION
# =============================================================================

def simulate_trade(rows, signal_idx, side, sl, tp, entry):
    """Simulate trade with given SL and TP"""
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
            if row.low <= sl:
                return 'loss', bar_idx + 1
            if row.high >= tp:
                return 'win', bar_idx + 1
        else:
            if row.high >= sl:
                return 'loss', bar_idx + 1
            if row.low <= tp:
                return 'win', bar_idx + 1
    
    return 'timeout', 100

# =============================================================================
# MAIN
# =============================================================================

def run_comparison():
    print("=" * 80)
    print("🔬 PIVOT POINT VS ATR STOP LOSS COMPARISON")
    print("=" * 80)
    print("\nTesting SL Methods:")
    print("  1. ATR Current: SL = 1×ATR, TP = 2.05×ATR")
    print("  2. Pivot 2:1: SL at swing, TP = 2× distance")
    print("  3. Pivot 2.5:1: SL at swing, TP = 2.5× distance")
    print("  4. Pivot 3:1: SL at swing, TP = 3× distance")
    print("  5. Pivot Buffered: SL at swing + 0.2×ATR buffer")
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
                    'atr': row.atr
                })
                last_idx = i
            
            if (idx + 1) % 25 == 0:
                print(f"  [{idx+1}/{NUM_SYMBOLS}] Signals: {len(all_signals)}")
            
            time.sleep(0.02)
        except: continue
    
    print(f"\n⏱️ Data loaded in {(time.time()-start)/60:.1f}m | {len(all_signals)} signals")
    
    # Define methods
    methods = {
        'ATR (1:2.05)': lambda rows, idx, side, atr: calc_atr_sltp(rows, idx, side, atr, 2.05),
        'Pivot (1:2)': lambda rows, idx, side, atr: calc_pivot_sltp(rows, idx, side, atr, 2.0, 0.0),
        'Pivot (1:2.5)': lambda rows, idx, side, atr: calc_pivot_sltp(rows, idx, side, atr, 2.5, 0.0),
        'Pivot (1:3)': lambda rows, idx, side, atr: calc_pivot_sltp(rows, idx, side, atr, 3.0, 0.0),
        'Pivot + Buffer': lambda rows, idx, side, atr: calc_pivot_sltp(rows, idx, side, atr, 2.0, 0.2),
    }
    
    results = {}
    
    for method_name, calc_fn in methods.items():
        print(f"\n  Testing: {method_name}...")
        wins = losses = 0
        rr_sum = 0
        sl_distances = []
        bars_to_win = []
        bars_to_loss = []
        
        for sig in all_signals:
            rows = sig['rows']
            idx = sig['idx']
            side = sig['side']
            atr = sig['atr']
            
            entry, sl, tp, rr = calc_fn(rows, idx, side, atr)
            rr_sum += rr
            
            sl_dist = abs(entry - sl)
            sl_distances.append(sl_dist / atr)  # Store as ATR multiple
            
            result, bars = simulate_trade(rows, idx, side, sl, tp, entry)
            
            if result == 'win':
                wins += 1
                bars_to_win.append(bars)
            elif result == 'loss':
                losses += 1
                bars_to_loss.append(bars)
        
        total = wins + losses
        wr = wins / total if total > 0 else 0
        avg_rr = rr_sum / len(all_signals) if all_signals else 2.0
        ev = calc_ev(wr, avg_rr)
        lb = wilson_lb(wins, total)
        total_r = (wins * avg_rr) - losses
        avg_sl = np.mean(sl_distances) if sl_distances else 1.0
        
        results[method_name] = {
            'wins': wins, 'losses': losses, 'total': total,
            'wr': wr, 'ev': ev, 'lb': lb, 'total_r': total_r,
            'avg_rr': avg_rr, 'avg_sl_atr': avg_sl,
            'avg_bars_win': np.mean(bars_to_win) if bars_to_win else 0,
            'avg_bars_loss': np.mean(bars_to_loss) if bars_to_loss else 0
        }
    
    # Print comparison
    print("\n" + "=" * 80)
    print("📊 RESULTS COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Method':<18} {'N':<8} {'WR%':<8} {'Avg SL':<10} {'Avg R:R':<8} {'EV':<8} {'Total R':<12} {'Bars/Win':<10}")
    print("-" * 95)
    
    best_ev = max(r['ev'] for r in results.values())
    best_total = max(r['total_r'] for r in results.values())
    
    for name, r in sorted(results.items(), key=lambda x: x[1]['ev'], reverse=True):
        ev_flag = "🏆" if r['ev'] == best_ev else "  "
        total_flag = "💰" if r['total_r'] == best_total else "  "
        print(f"{name:<18} {r['total']:<8} {r['wr']*100:.1f}%{'':<2} {r['avg_sl_atr']:.2f}×ATR{'':<2} {r['avg_rr']:.1f}:1{'':<2} {r['ev']:+.2f}{'':<2} {r['total_r']:+,.0f}R{'':<4} {r['avg_bars_win']:.1f} {ev_flag}{total_flag}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("🏆 ANALYSIS")
    print("=" * 80)
    
    atr_current = results.get('ATR (1:2.05)', {})
    best = max(results.items(), key=lambda x: x[1]['ev'])
    best_total_method = max(results.items(), key=lambda x: x[1]['total_r'])
    
    print(f"\n📊 Current ATR Method (1:2.05):")
    print(f"   Trades: {atr_current['total']} | WR: {atr_current['wr']*100:.1f}%")
    print(f"   EV: {atr_current['ev']:+.2f} | Total: {atr_current['total_r']:+,.0f}R")
    print(f"   Avg SL: {atr_current['avg_sl_atr']:.2f}×ATR")
    
    print(f"\n📊 Best by EV: {best[0]}")
    print(f"   Trades: {best[1]['total']} | WR: {best[1]['wr']*100:.1f}%")
    print(f"   EV: {best[1]['ev']:+.2f} | Total: {best[1]['total_r']:+,.0f}R")
    
    if best[0] != 'ATR (1:2.05)' and best[1]['ev'] > atr_current['ev']:
        ev_diff = best[1]['ev'] - atr_current['ev']
        print(f"\n✅ {best[0]} beats ATR by +{ev_diff:.2f} EV per trade!")
        
        if best[1]['total_r'] > atr_current['total_r']:
            profit_diff = best[1]['total_r'] - atr_current['total_r']
            print(f"   AND increases total profit by {profit_diff:+,.0f}R!")
    else:
        print(f"\n📊 ATR method remains competitive!")

if __name__ == "__main__":
    run_comparison()
