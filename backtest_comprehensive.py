#!/usr/bin/env python3
"""
COMPREHENSIVE R:R AND TIMEFRAME BACKTEST
=========================================
Tests multiple R:R ratios across different timeframes.
Optimized for speed with progress tracking.
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

TIMEFRAMES = ['15', '60']  # 15m and 1h
RR_RATIOS = [1.0, 1.5, 2.0, 2.5, 3.0]
DATA_DAYS = 60
NUM_SYMBOLS = 100  # Reduced for speed

# Costs
SLIPPAGE_PCT = 0.0005
FEE_PCT = 0.0004
TOTAL_COST = (SLIPPAGE_PCT + FEE_PCT) * 2

# Divergence settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5

BASE_URL = "https://api.bybit.com"

# =============================================================================
# HELPERS
# =============================================================================

def wilson_lb(wins, n, z=1.96):
    if n == 0: return 0.0
    p = wins / n
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    return max(0, (centre - spread) / denom)

def calc_ev(wr, rr):
    return (wr * rr) - (1 - wr)

def get_symbols(limit=100):
    url = f"{BASE_URL}/v5/market/tickers?category=linear"
    resp = requests.get(url)
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
        url = f"{BASE_URL}/v5/market/kline"
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(url, params=params, timeout=10)
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

def detect_signals(df):
    if len(df) < 100:
        return []
    
    close = df['close'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_pivot_highs, price_pivot_lows = find_pivots(close, left=3, right=3)
    signals = []
    
    for i in range(30, n - 5):
        curr_pl = curr_pl_idx = prev_pl = prev_pl_idx = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pivot_lows[j]):
                if curr_pl is None:
                    curr_pl, curr_pl_idx = price_pivot_lows[j], j
                elif prev_pl is None and j < curr_pl_idx - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pl_idx = price_pivot_lows[j], j
                    break
        
        curr_ph = curr_ph_idx = prev_ph = prev_ph_idx = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pivot_highs[j]):
                if curr_ph is None:
                    curr_ph, curr_ph_idx = price_pivot_highs[j], j
                elif prev_ph is None and j < curr_ph_idx - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_ph_idx = price_pivot_highs[j], j
                    break
        
        # Regular Bullish
        if curr_pl and prev_pl and curr_pl < prev_pl:
            if rsi[curr_pl_idx] > rsi[prev_pl_idx] and rsi[i] < RSI_OVERSOLD + 15:
                signals.append({'idx': i, 'side': 'long'})
                continue
        
        # Regular Bearish
        if curr_ph and prev_ph and curr_ph > prev_ph:
            if rsi[curr_ph_idx] < rsi[prev_ph_idx] and rsi[i] > RSI_OVERBOUGHT - 15:
                signals.append({'idx': i, 'side': 'short'})
                continue
        
        # Hidden Bullish
        if curr_pl and prev_pl and curr_pl > prev_pl:
            if rsi[curr_pl_idx] < rsi[prev_pl_idx] and rsi[i] < RSI_OVERBOUGHT - 10:
                signals.append({'idx': i, 'side': 'long'})
                continue
        
        # Hidden Bearish
        if curr_ph and prev_ph and curr_ph < prev_ph:
            if rsi[curr_ph_idx] > rsi[prev_ph_idx] and rsi[i] > RSI_OVERSOLD + 10:
                signals.append({'idx': i, 'side': 'short'})
    
    return signals

def simulate_trade(rows, signal_idx, side, atr, tp_mult):
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 50:
        return 'timeout'
    
    entry_row = rows[entry_idx]
    base_entry = entry_row.open
    
    if side == 'long':
        entry = base_entry * (1 + SLIPPAGE_PCT)
        tp = entry + (tp_mult * atr)
        sl = entry - atr
        tp = tp * (1 - TOTAL_COST)
    else:
        entry = base_entry * (1 - SLIPPAGE_PCT)
        tp = entry - (tp_mult * atr)
        sl = entry + atr
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

def run_comprehensive_backtest():
    print("=" * 80)
    print("🔬 COMPREHENSIVE R:R & TIMEFRAME BACKTEST")
    print("=" * 80)
    print(f"Timeframes: {TIMEFRAMES} | R:R ratios: {RR_RATIOS}")
    print(f"Data: {DATA_DAYS} days | Symbols: {NUM_SYMBOLS}")
    print("=" * 80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    all_results = {}
    
    for tf in TIMEFRAMES:
        tf_name = f"{tf}m"
        print(f"\n{'='*80}")
        print(f"📊 TESTING TIMEFRAME: {tf_name}")
        print("="*80)
        
        # Collect signals for this timeframe
        all_signals = []
        start = time.time()
        
        for idx, sym in enumerate(symbols):
            try:
                df = fetch_klines(sym, tf, DATA_DAYS)
                if df.empty or len(df) < 300:
                    continue
                
                df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift())
                low_close = abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df['atr'] = tr.rolling(14).mean()
                df['vol_ma'] = df['volume'].rolling(20).mean()
                df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
                df = df.dropna()
                
                if len(df) < 150:
                    continue
                
                signals = detect_signals(df)
                rows = list(df.itertuples())
                last_idx = -20
                
                for sig in signals:
                    i = sig['idx']
                    if i - last_idx < 10 or i >= len(rows) - 100:
                        continue
                    row = rows[i]
                    if pd.isna(row.atr) or row.atr <= 0 or not row.vol_ok:
                        continue
                    
                    all_signals.append({
                        'rows': rows,
                        'idx': i,
                        'side': sig['side'],
                        'atr': row.atr
                    })
                    last_idx = i
                
                if (idx + 1) % 20 == 0:
                    print(f"  [{idx+1}/{NUM_SYMBOLS}] Signals: {len(all_signals)}")
                
                time.sleep(0.02)
            except:
                continue
        
        print(f"  ⏱️ Data collected in {(time.time()-start)/60:.1f}m | {len(all_signals)} signals")
        
        # Test each R:R
        tf_results = {}
        for rr in RR_RATIOS:
            wins = losses = 0
            for sig in all_signals:
                result = simulate_trade(sig['rows'], sig['idx'], sig['side'], sig['atr'], rr)
                if result == 'win': wins += 1
                elif result == 'loss': losses += 1
            
            total = wins + losses
            wr = wins / total if total > 0 else 0
            ev = calc_ev(wr, rr)
            lb = wilson_lb(wins, total)
            
            tf_results[rr] = {'wins': wins, 'losses': losses, 'total': total, 'wr': wr, 'lb': lb, 'ev': ev}
        
        all_results[tf_name] = tf_results
        
        # Print results for this timeframe
        print(f"\n  {'R:R':<6} {'N':<8} {'Wins':<8} {'WR%':<10} {'LB%':<10} {'EV':<10}")
        print("  " + "-"*55)
        for rr in RR_RATIOS:
            r = tf_results[rr]
            emoji = "✅" if r['ev'] > 0.5 else "⚠️" if r['ev'] > 0 else "❌"
            print(f"  {rr}:1{'':<3} {r['total']:<8} {r['wins']:<8} {r['wr']*100:.1f}%{'':<4} {r['lb']*100:.1f}%{'':<4} {r['ev']:+.2f} {emoji}")
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON: ALL TIMEFRAMES & R:R")
    print("="*80)
    
    print(f"\n{'Config':<15} {'Trades':<10} {'WR%':<10} {'EV':<10} {'EV×N':<12}")
    print("-"*60)
    
    best_config = None
    best_metric = -999
    
    for tf_name in all_results:
        for rr in RR_RATIOS:
            r = all_results[tf_name][rr]
            config = f"{tf_name} {rr}:1"
            total_ev = r['ev'] * r['total']
            emoji = "✅" if r['ev'] > 0.5 else "⚠️" if r['ev'] > 0 else "❌"
            
            print(f"{config:<15} {r['total']:<10} {r['wr']*100:.1f}%{'':<4} {r['ev']:+.2f}{'':<4} {total_ev:+,.0f}R {emoji}")
            
            # Use EV × sqrt(N) as optimization metric (balances edge and sample size)
            metric = r['ev'] * math.sqrt(r['total']) if r['ev'] > 0 else -999
            if metric > best_metric:
                best_metric = metric
                best_config = (tf_name, rr, r)
    
    print("\n" + "="*80)
    print("🏆 BEST CONFIGURATION")
    print("="*80)
    
    if best_config:
        tf, rr, r = best_config
        print(f"\n  📊 Timeframe: {tf}")
        print(f"  🎯 R:R Ratio: {rr}:1")
        print(f"  📈 Win Rate: {r['wr']*100:.1f}% (LB: {r['lb']*100:.1f}%)")
        print(f"  💰 EV: {r['ev']:+.2f}R per trade")
        print(f"  📊 Total Trades: {r['total']}")
        print(f"  💵 Total Expected: {r['ev'] * r['total']:+,.0f}R")

if __name__ == "__main__":
    run_comprehensive_backtest()
