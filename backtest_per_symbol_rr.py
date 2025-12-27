#!/usr/bin/env python3
"""
PER-SYMBOL R:R OPTIMIZATION
===========================
Finding the optimal R:R ratio for each of the Top 40 symbols individually.
Comparing per-symbol optimization vs global R:R strategy.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Config
TIMEFRAME = '240'  # 4H
DATA_DAYS = 1000   # ~3 Years
NUM_SYMBOLS = 40
MAX_WAIT_CANDLES = 6
SL_MULT = 1.0

# Costs
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

# Indicators
RSI_PERIOD = 14
LOOKBACK_BARS = 10
MIN_PIVOT_DISTANCE = 3
PIVOT_RIGHT = 3

BASE_URL = "https://api.bybit.com"

def get_symbols(limit):
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear", timeout=10)
        tickers = resp.json().get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        return [t['symbol'] for t in usdt[:limit]]
    except:
        return []

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
    return df

def add_indicators(df):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    df['ema_daily'] = df['close'].ewm(span=1200, adjust=False).mean()
    
    return df.dropna()

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

def detect_divergences(df):
    if len(df) < 100: return []
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    n = len(df)
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    for i in range(30, n - 15):
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i - PIVOT_RIGHT, max(i - LOOKBACK_BARS - PIVOT_RIGHT, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        if curr_pl and prev_pl:
            if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                swing_high = max(high[max(0, i-LOOKBACK_BARS):i+1])
                signals.append({'idx': i, 'side': 'long', 'swing': swing_high})
                
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i - PIVOT_RIGHT, max(i - LOOKBACK_BARS - PIVOT_RIGHT, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_ph[j], j
                    break
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                swing_low = min(low[max(0, i-LOOKBACK_BARS):i+1])
                signals.append({'idx': i, 'side': 'short', 'swing': swing_low})
                
    return signals

def run_backtest(df, signals, rr):
    rows = list(df.itertuples())
    trades = []
    
    for sig in signals:
        div_idx = sig['idx']
        side = sig['side']
        current_row = rows[div_idx]
        
        # Trend Filter
        if side == 'long' and current_row.close < current_row.ema_daily: continue
        if side == 'short' and current_row.close > current_row.ema_daily: continue
            
        bos_idx = None
        for j in range(div_idx + 1, min(div_idx + 1 + MAX_WAIT_CANDLES, len(rows) - 10)):
            curr_row = rows[j]
            if side == 'long':
                if curr_row.close > sig['swing']:
                    bos_idx = j
                    break
            else:
                if curr_row.close < sig['swing']:
                    bos_idx = j
                    break
        
        if bos_idx is None: continue
        entry_idx = bos_idx + 1
        if entry_idx >= len(rows) - 50: continue
        
        entry_row = rows[entry_idx]
        entry_price = entry_row.open
        if side == 'long': entry_price *= (1 + SLIPPAGE_PCT)
        else: entry_price *= (1 - SLIPPAGE_PCT)
        
        entry_atr = entry_row.atr
        sl_dist = entry_atr * SL_MULT
        
        if side == 'long':
            sl = entry_price - sl_dist
            tp = entry_price + (sl_dist * rr)
        else:
            sl = entry_price + sl_dist
            tp = entry_price - (sl_dist * rr)
        
        result = None
        for k in range(entry_idx, min(entry_idx + 200, len(rows))):
            curr_row = rows[k]
            if side == 'long':
                if curr_row.low <= sl: result = 'loss'; break
                if curr_row.high >= tp: result = 'win'; break
            else:
                if curr_row.high >= sl: result = 'loss'; break
                if curr_row.low <= tp: result = 'win'; break
        
        if result:
            risk_pct = sl_dist / entry_price
            fee_cost = (FEE_PCT * 2 + SLIPPAGE_PCT) / risk_pct
            r = (rr - fee_cost) if result == 'win' else (-1.0 - fee_cost)
            trades.append({'result': result, 'r': r})
    return trades

def main():
    print("="*80)
    print("PER-SYMBOL R:R OPTIMIZATION")
    print("="*80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"Fetching data for {len(symbols)} symbols...")
    
    symbol_data = {}
    for idx, sym in enumerate(symbols):
        df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
        if df.empty or len(df) < 500: continue
        df = add_indicators(df)
        if len(df) < 200: continue
        signals = detect_divergences(df)
        if signals:
            symbol_data[sym] = (df, signals)
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(symbols)} symbols...")
            
    print(f"\nSymbols with signals: {len(symbol_data)}")
    
    rr_ratios = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]
    
    results = []
    
    print("\nOptimizing R:R for each symbol...")
    print(f"{'Symbol':<12} | {'Best RR':<8} | {'N':<5} | {'WR':<7} | {'Avg R':<10} | {'Total R':<9}")
    print("-"*75)
    
    for sym, (df, signals) in sorted(symbol_data.items()):
        best_rr = None
        best_avg_r = -999
        best_stats = None
        
        for rr in rr_ratios:
            trades = run_backtest(df, signals, rr)
            if not trades: continue
            
            wins = sum(1 for t in trades if t['result'] == 'win')
            n = len(trades)
            wr = wins / n * 100
            total_r = sum(t['r'] for t in trades)
            avg_r = total_r / n
            
            if avg_r > best_avg_r:
                best_avg_r = avg_r
                best_rr = rr
                best_stats = {'n': n, 'wr': wr, 'total_r': total_r}
        
        if best_rr and best_stats:
            print(f"{sym:<12} | {best_rr:<8.1f} | {best_stats['n']:<5} | {best_stats['wr']:>5.1f}% | {best_avg_r:>+8.4f}R | {best_stats['total_r']:>+7.1f}R")
            results.append({
                'symbol': sym,
                'best_rr': best_rr,
                'n': best_stats['n'],
                'wr': best_stats['wr'],
                'avg_r': best_avg_r,
                'total_r': best_stats['total_r']
            })
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    rr_counts = {}
    for r in results:
        rr = r['best_rr']
        rr_counts[rr] = rr_counts.get(rr, 0) + 1
    
    print("\nR:R Distribution:")
    for rr in sorted(rr_counts.keys()):
        count = rr_counts[rr]
        pct = count / len(results) * 100
        print(f"  {rr}:1 → {count} symbols ({pct:.1f}%)")
    
    # Compare per-symbol vs global
    print("\n" + "="*80)
    print("PER-SYMBOL vs GLOBAL 5:1 COMPARISON")
    print("="*80)
    
    # Calculate performance with per-symbol optimal
    per_symbol_total_r = sum(r['total_r'] for r in results)
    per_symbol_total_n = sum(r['n'] for r in results)
    per_symbol_avg_r = per_symbol_total_r / per_symbol_total_n if per_symbol_total_n > 0 else 0
    
    # Calculate performance with global 5:1
    global_total_r = 0
    global_total_n = 0
    for sym, (df, signals) in symbol_data.items():
        trades = run_backtest(df, signals, 5.0)
        global_total_r += sum(t['r'] for t in trades)
        global_total_n += len(trades)
    
    global_avg_r = global_total_r / global_total_n if global_total_n > 0 else 0
    
    print(f"\nPer-Symbol Optimal:")
    print(f"  Total R:     {per_symbol_total_r:+.1f}R")
    print(f"  Avg R/Trade: {per_symbol_avg_r:+.4f}R")
    
    print(f"\nGlobal 5:1 R:R:")
    print(f"  Total R:     {global_total_r:+.1f}R")
    print(f"  Avg R/Trade: {global_avg_r:+.4f}R")
    
    improvement = ((per_symbol_avg_r - global_avg_r) / abs(global_avg_r)) * 100 if global_avg_r != 0 else 0
    
    print(f"\nImprovement: {improvement:+.1f}%")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    most_common_rr = max(rr_counts, key=rr_counts.get)
    
    if improvement > 10:
        print(f"\n✅ USE PER-SYMBOL R:R")
        print(f"   Improvement is significant (+{improvement:.1f}%)")
        print(f"   Save optimal R:R for each symbol in config")
    elif improvement > 3:
        print(f"\n⚠️  MARGINAL BENEFIT (+{improvement:.1f}%)")
        print(f"   Per-symbol is slightly better but adds complexity")
        print(f"   Most common optimal: {most_common_rr}:1 ({rr_counts[most_common_rr]} symbols)")
        print(f"   Recommendation: Use global {most_common_rr}:1 for simplicity")
    else:
        print(f"\n❌ USE GLOBAL 5:1 R:R")
        print(f"   Per-symbol provides minimal improvement (+{improvement:.1f}%)")
        print(f"   Not worth the added complexity")
        print(f"   Most symbols prefer {most_common_rr}:1 anyway ({rr_counts[most_common_rr]} symbols)")

if __name__ == "__main__":
    main()
