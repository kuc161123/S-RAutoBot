#!/usr/bin/env python3
"""
RSI + OBV DIVERGENCE COMPARISON BACKTEST
=========================================
Compares:
- RSI Divergence only (current strategy)
- RSI + OBV Divergence (both must confirm)

OBV (On-Balance Volume) Divergence:
- Bullish: Price makes lower low, but OBV makes higher low
- Bearish: Price makes higher high, but OBV makes lower high
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
NUM_SYMBOLS = 100

TP_ATR_MULT = 2.05
SL_ATR_MULT = 1.0

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

def calc_ev(wr, rr=2.05):
    return (wr * rr) - (1 - wr)

def wilson_lb(wins, n, z=1.96):
    if n == 0: return 0.0
    p = wins / n
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    return max(0, (centre - spread) / denom)

def get_symbols(limit=100):
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

def calculate_obv(close, volume):
    """Calculate On-Balance Volume"""
    obv = [0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

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

def detect_signals(df, require_obv=False):
    """
    Detect divergence signals.
    If require_obv=True, both RSI and OBV must show divergence.
    """
    if len(df) < 100: return []
    
    close = df['close'].values
    rsi = df['rsi'].values
    obv = df['obv'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    obv_ph, obv_pl = find_pivots(obv, 3, 3)
    
    signals = []
    
    for i in range(30, n - 5):
        # Find price pivot lows
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE: prev_pl, prev_pli = price_pl[j], j; break
        
        # Find price pivot highs
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE: prev_ph, prev_phi = price_ph[j], j; break
        
        # Regular Bullish: Price LL, RSI HL
        if curr_pl and prev_pl and curr_pl < prev_pl:
            rsi_diverges = rsi[curr_pli] > rsi[prev_pli] and rsi[i] < RSI_OVERSOLD + 15
            
            if rsi_diverges:
                if require_obv:
                    # OBV must also diverge (HL while price makes LL)
                    obv_diverges = obv[curr_pli] > obv[prev_pli]
                    if obv_diverges:
                        signals.append({'idx': i, 'side': 'long', 'type': 'regular_bullish'})
                else:
                    signals.append({'idx': i, 'side': 'long', 'type': 'regular_bullish'})
                continue
        
        # Regular Bearish: Price HH, RSI LH
        if curr_ph and prev_ph and curr_ph > prev_ph:
            rsi_diverges = rsi[curr_phi] < rsi[prev_phi] and rsi[i] > RSI_OVERBOUGHT - 15
            
            if rsi_diverges:
                if require_obv:
                    # OBV must also diverge (LH while price makes HH)
                    obv_diverges = obv[curr_phi] < obv[prev_phi]
                    if obv_diverges:
                        signals.append({'idx': i, 'side': 'short', 'type': 'regular_bearish'})
                else:
                    signals.append({'idx': i, 'side': 'short', 'type': 'regular_bearish'})
                continue
        
        # Hidden Bullish: Price HL, RSI LL
        if curr_pl and prev_pl and curr_pl > prev_pl:
            rsi_diverges = rsi[curr_pli] < rsi[prev_pli] and rsi[i] < RSI_OVERBOUGHT - 10
            
            if rsi_diverges:
                if require_obv:
                    obv_diverges = obv[curr_pli] < obv[prev_pli]
                    if obv_diverges:
                        signals.append({'idx': i, 'side': 'long', 'type': 'hidden_bullish'})
                else:
                    signals.append({'idx': i, 'side': 'long', 'type': 'hidden_bullish'})
                continue
        
        # Hidden Bearish: Price LH, RSI HH
        if curr_ph and prev_ph and curr_ph < prev_ph:
            rsi_diverges = rsi[curr_phi] > rsi[prev_phi] and rsi[i] > RSI_OVERSOLD + 10
            
            if rsi_diverges:
                if require_obv:
                    obv_diverges = obv[curr_phi] > obv[prev_phi]
                    if obv_diverges:
                        signals.append({'idx': i, 'side': 'short', 'type': 'hidden_bearish'})
                else:
                    signals.append({'idx': i, 'side': 'short', 'type': 'hidden_bearish'})
    
    return signals

def simulate_trade(rows, signal_idx, side, atr):
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 50: return 'timeout'
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
    
    for future_row in rows[entry_idx+1:entry_idx+100]:
        if side == 'long':
            if future_row.low <= sl: return 'loss'
            if future_row.high >= tp: return 'win'
        else:
            if future_row.high >= sl: return 'loss'
            if future_row.low <= tp: return 'win'
    
    return 'timeout'

# =============================================================================
# MAIN
# =============================================================================

def run_comparison():
    print("=" * 70)
    print("🔬 RSI vs RSI+OBV DIVERGENCE COMPARISON")
    print("=" * 70)
    print("Comparing:")
    print("  A) RSI Divergence Only (current strategy)")
    print("  B) RSI + OBV Divergence (both must confirm)")
    print("=" * 70)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📦 Fetching data for {len(symbols)} symbols...\n")
    
    # Store processed data
    all_symbols_data = []
    start = time.time()
    
    for idx, sym in enumerate(symbols):
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 400: continue
            
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            df['obv'] = calculate_obv(df['close'], df['volume'])
            
            hl = df['high'] - df['low']
            hc = abs(df['high'] - df['close'].shift())
            lc = abs(df['low'] - df['close'].shift())
            df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            df = df.dropna()
            
            if len(df) < 150: continue
            
            all_symbols_data.append({'df': df, 'symbol': sym})
            
            if (idx + 1) % 20 == 0:
                print(f"  [{idx+1}/{NUM_SYMBOLS}] Loaded: {len(all_symbols_data)} symbols")
            
            time.sleep(0.02)
        except: continue
    
    print(f"\n⏱️ Data loaded in {(time.time()-start)/60:.1f}m | {len(all_symbols_data)} symbols")
    
    # Test both strategies
    results = {}
    
    for mode in ['RSI Only', 'RSI + OBV']:
        require_obv = (mode == 'RSI + OBV')
        wins = losses = 0
        by_type = defaultdict(lambda: {'w': 0, 'l': 0})
        
        for sym_data in all_symbols_data:
            df = sym_data['df']
            signals = detect_signals(df, require_obv=require_obv)
            rows = list(df.itertuples())
            last_idx = -20
            
            for sig in signals:
                i = sig['idx']
                if i - last_idx < 10 or i >= len(rows) - 100: continue
                row = rows[i]
                if pd.isna(row.atr) or row.atr <= 0 or not row.vol_ok: continue
                
                result = simulate_trade(rows, i, sig['side'], row.atr)
                if result == 'win':
                    wins += 1
                    by_type[sig['type']]['w'] += 1
                elif result == 'loss':
                    losses += 1
                    by_type[sig['type']]['l'] += 1
                
                last_idx = i
        
        total = wins + losses
        wr = wins / total if total > 0 else 0
        ev = calc_ev(wr)
        lb = wilson_lb(wins, total)
        
        results[mode] = {'wins': wins, 'losses': losses, 'total': total, 'wr': wr, 'ev': ev, 'lb': lb, 'by_type': by_type}
    
    # Print comparison
    print("\n" + "=" * 70)
    print("📊 RESULTS COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Strategy':<15} {'Trades':<10} {'Wins':<8} {'WR%':<10} {'LB%':<10} {'EV':<10}")
    print("-" * 65)
    
    for mode, r in results.items():
        print(f"{mode:<15} {r['total']:<10} {r['wins']:<8} {r['wr']*100:.1f}%{'':<4} {r['lb']*100:.1f}%{'':<4} {r['ev']:+.2f}")
    
    # By divergence type
    print("\n" + "-" * 70)
    print("BY DIVERGENCE TYPE")
    print("-" * 70)
    
    for mode in ['RSI Only', 'RSI + OBV']:
        print(f"\n📊 {mode}:")
        by_type = results[mode]['by_type']
        for sig_type, data in sorted(by_type.items()):
            total = data['w'] + data['l']
            if total > 0:
                wr = data['w'] / total
                ev = calc_ev(wr)
                print(f"   {sig_type:<20} N={total:<6} WR={wr*100:.1f}% EV={ev:+.2f}")
    
    # Winner
    print("\n" + "=" * 70)
    print("🏆 ANALYSIS")
    print("=" * 70)
    
    rsi_only = results['RSI Only']
    rsi_obv = results['RSI + OBV']
    
    print(f"\n📊 RSI Only: {rsi_only['total']} trades | {rsi_only['wr']*100:.1f}% WR | {rsi_only['ev']:+.2f} EV")
    print(f"📊 RSI + OBV: {rsi_obv['total']} trades | {rsi_obv['wr']*100:.1f}% WR | {rsi_obv['ev']:+.2f} EV")
    
    trade_diff = rsi_only['total'] - rsi_obv['total']
    ev_diff = rsi_obv['ev'] - rsi_only['ev']
    
    print(f"\n📈 Adding OBV filter:")
    print(f"   - Reduces trades by {trade_diff} ({trade_diff/rsi_only['total']*100:.1f}%)")
    print(f"   - Changes EV by {ev_diff:+.2f}")
    
    if rsi_obv['ev'] > rsi_only['ev']:
        print(f"\n✅ RSI + OBV is MORE PROFITABLE per trade")
        total_r_rsi = rsi_only['ev'] * rsi_only['total']
        total_r_obv = rsi_obv['ev'] * rsi_obv['total']
        print(f"   But total R: RSI={total_r_rsi:+,.0f}R vs RSI+OBV={total_r_obv:+,.0f}R")
    else:
        print(f"\n❌ RSI + OBV does NOT improve performance")
        print(f"   Stick with RSI Divergence only")

if __name__ == "__main__":
    run_comparison()
