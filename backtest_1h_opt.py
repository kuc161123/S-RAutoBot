#!/usr/bin/env python3
"""
1H STRATEGY OPTIMIZATION
=========================
Focus: Optimizing the most promising configuration found so far:
TIMEFRAME: 1H
BASE: Divergence + BOS
FILTERS: Volume + Trend

Variables to Optimize:
1. R:R Ratio: [1.0, 1.3, 1.5, 1.8, 2.0, 2.5]
2. SL ATR Multiplier: [0.8, 1.0, 1.2, 1.5]
3. Volume Multiplier: [1.2, 1.5, 1.8, 2.0]
4. Trend EMA: [50, 100, 200]
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import itertools
import warnings
warnings.filterwarnings('ignore')

# Config
TIMEFRAME = '60'
DATA_DAYS = 365  # Increased to 1 year for 1H robustness
NUM_SYMBOLS = 60
MAX_WAIT_CANDLES = 12

# Costs
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

# Indicators
RSI_PERIOD = 14
LOOKBACK_BARS = 10
MIN_PIVOT_DISTANCE = 4

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
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    # EMAs
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Volume MA
    df['vol_ma'] = df['volume'].rolling(20).mean()
    
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
        # Bullish
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
        if curr_pl and prev_pl:
            if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                if rsi[i] < 45: # Slightly relaxed for 1H
                    swing_high = max(high[max(0, i-LOOKBACK_BARS):i+1])
                    signals.append({'idx': i, 'side': 'long', 'swing': swing_high})
                    continue
        
        # Bearish
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                if rsi[i] > 55: # Slightly relaxed for 1H
                    swing_low = min(low[max(0, i-LOOKBACK_BARS):i+1])
                    signals.append({'idx': i, 'side': 'short', 'swing': swing_low})
    
    return signals

def run_backtest(df, signals, rr, sl_mult, vol_mult, trend_ema_col):
    rows = list(df.itertuples())
    atr = df['atr'].values
    ema_vals = df[trend_ema_col].values
    volume = df['volume'].values
    vol_ma = df['vol_ma'].values
    close_arr = df['close'].values
    
    trades = []
    
    for sig in signals:
        div_idx = sig['idx']
        side = sig['side']
        
        # Trend Filter
        if side == 'long' and close_arr[div_idx] < ema_vals[div_idx]:
            continue
        if side == 'short' and close_arr[div_idx] > ema_vals[div_idx]:
            continue
        
        bos_idx = None
        for j in range(div_idx + 1, min(div_idx + 1 + MAX_WAIT_CANDLES, len(rows) - 10)):
            # Volume Filter
            if volume[j] < vol_ma[j] * vol_mult:
                continue
            
            if side == 'long':
                if rows[j].close > sig['swing']:
                    bos_idx = j
                    break
            else:
                if rows[j].close < sig['swing']:
                    bos_idx = j
                    break
        
        if bos_idx is None:
            continue
        
        entry_idx = bos_idx + 1
        if entry_idx >= len(rows) - 50:
            continue
        
        entry_price = rows[entry_idx].open
        if side == 'long':
            entry_price *= (1 + SLIPPAGE_PCT)
        else:
            entry_price *= (1 - SLIPPAGE_PCT)
        
        entry_atr = atr[entry_idx - 1]
        sl_dist = entry_atr * sl_mult
        
        if side == 'long':
            sl = entry_price - sl_dist
            tp = entry_price + (sl_dist * rr)
        else:
            sl = entry_price + sl_dist
            tp = entry_price - (sl_dist * rr)
        
        result = None
        for k in range(entry_idx, min(entry_idx + 500, len(rows))):
            row = rows[k]
            if side == 'long':
                if row.low <= sl: result = 'loss'; break
                if row.high >= tp: result = 'win'; break
            else:
                if row.high >= sl: result = 'loss'; break
                if row.low <= tp: result = 'win'; break
        
        if result:
            risk_pct = sl_dist / entry_price
            fee_cost = (FEE_PCT * 2 + SLIPPAGE_PCT) / risk_pct
            r = (rr - fee_cost) if result == 'win' else (-1.0 - fee_cost)
            trades.append({'result': result, 'r': r})
    
    return trades

def main():
    print("="*80)
    print("1H STRATEGY OPTIMIZATION")
    print("="*80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"Fetching 1 year of 1H data for {len(symbols)} symbols...")
    
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
    
    # Optimization Grid
    rr_ratios = [1.0, 1.2, 1.3, 1.5, 1.8, 2.0]
    sl_mults = [0.8, 1.0, 1.2, 1.5]
    vol_mults = [1.2, 1.5, 1.8]
    trend_emas = ['ema50', 'ema100', 'ema200']
    
    total_combs = len(rr_ratios) * len(sl_mults) * len(vol_mults) * len(trend_emas)
    print(f"\nTesting {total_combs} combinations...")
    
    best_results = []
    
    count = 0
    # To save time, we can pre-calculate some aspects or just run nested loops efficiently
    # Since backtest is fast, nested loops are fine for ~200 configs
    
    # We'll prioritize iterating likely best configs
    
    print(f"\n{'RR':<4} | {'SL':<4} | {'Vol':<4} | {'EMA':<6} | {'N':<6} | {'WR':<7} | {'Avg R':<10} | {'Status'}")
    print("-"*80)
    
    for trend in trend_emas:
        for vol in vol_mults:
            for sl in sl_mults:
                for rr in rr_ratios:
                    if count > 500: break # Safety limit
                    
                    all_trades = []
                    for sym, (df, signals) in symbol_data.items():
                        trades = run_backtest(df, signals, rr, sl, vol, trend)
                        all_trades.extend(trades)
                    
                    if len(all_trades) > 50: # Min sample size
                        wins = sum(1 for t in all_trades if t['result'] == 'win')
                        n = len(all_trades)
                        wr = wins / n * 100
                        total_r = sum(t['r'] for t in all_trades)
                        avg_r = total_r / n
                        
                        count += 1
                        
                        if avg_r > -0.1: # Only print interesting ones
                            status = "✅ PROFIT" if avg_r > 0 else "Approaching"
                            print(f"{rr:<4.1f} | {sl:<4.1f} | {vol:<4.1f} | {trend:<6} | {n:<6} | {wr:>5.1f}% | {avg_r:>+8.4f}R | {status}")
                            
                            if avg_r > 0:
                                best_results.append({
                                    'rr': rr, 'sl': sl, 'vol': vol, 'trend': trend,
                                    'n': n, 'wr': wr, 'avg_r': avg_r
                                })

    print("\n" + "="*80)
    
    if best_results:
        print(f"\n🏆 FOUND {len(best_results)} PROFITABLE CONFIGURATIONS!")
        best_results.sort(key=lambda x: x['avg_r'], reverse=True)
        for r in best_results[:5]:
            print(f"  ✅ RR={r['rr']} | SL={r['sl']} | Vol={r['vol']} | {r['trend']} | {r['wr']:.1f}% WR | {r['avg_r']:+.4f}R avg")
    else:
        print("\n❌ NO PROFITABLE CONFIGURATIONS FOUND (1H)")

if __name__ == "__main__":
    main()
