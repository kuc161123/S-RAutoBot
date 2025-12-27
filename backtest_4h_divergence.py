#!/usr/bin/env python3
"""
4H DIVERGENCE STRATEGY TEST
===========================
Testing the "Divergence + BOS" strategy on the 4H timeframe.
Hypothesis: 4H has less noise and more reliable structural breaks.

Configuration:
- Timeframe: 4H (240m)
- History: 3 Years (to get enough trades)
- Filters: Volume (>1.2x MA) + Trend (EMA200)
- BOS: Close Break
- R:R: Testing [1.5, 2.0, 3.0, 4.0]
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
NUM_SYMBOLS = 40   # Top 40 symbols
MAX_WAIT_CANDLES = 6 # 24 hours (6 * 4h)

# Costs
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

# Indicators
RSI_PERIOD = 14
LOOKBACK_BARS = 10
MIN_PIVOT_DISTANCE = 3

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
    
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
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
                swing_high = max(high[max(0, i-LOOKBACK_BARS):i+1])
                signals.append({'idx': i, 'side': 'long', 'swing': swing_high})
        
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                swing_low = min(low[max(0, i-LOOKBACK_BARS):i+1])
                signals.append({'idx': i, 'side': 'short', 'swing': swing_low})
    
    return signals

def run_backtest(df, signals, rr, use_filter=True):
    rows = list(df.itertuples())
    atr = df['atr'].values
    ema200 = df['ema200'].values
    volume = df['volume'].values
    vol_ma = df['vol_ma'].values
    close_arr = df['close'].values
    
    trades = []
    
    for sig in signals:
        div_idx = sig['idx']
        side = sig['side']
        
        # Filters (Trend + Volume)
        if use_filter:
            # Trend
            if side == 'long' and close_arr[div_idx] < ema200[div_idx]:
                continue
            if side == 'short' and close_arr[div_idx] > ema200[div_idx]:
                continue
            # Volume
            if volume[div_idx] < vol_ma[div_idx] * 1.2:
                continue
        
        # BOS
        bos_idx = None
        for j in range(div_idx + 1, min(div_idx + 1 + MAX_WAIT_CANDLES, len(rows) - 10)):
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
        sl_dist = entry_atr * 1.0 # 1.0 ATR SL
        
        if side == 'long':
            sl = entry_price - sl_dist
            tp = entry_price + (sl_dist * rr)
        else:
            sl = entry_price + sl_dist
            tp = entry_price - (sl_dist * rr)
        
        result = None
        for k in range(entry_idx, min(entry_idx + 200, len(rows))):
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
    print("4H DIVERGENCE STRATEGY TEST")
    print("="*80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"Fetching 3 years of 4H data for {len(symbols)} symbols...")
    
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
    
    rrs = [1.5, 2.0, 3.0, 4.0, 5.0]
    configs = [
        {'name': 'No Filter', 'filter': False},
        {'name': 'Trend + Vol', 'filter': True},
    ]
    
    results = []
    
    for rr in rrs:
        print(f"\n--- R:R = {rr}:1 ---")
        print(f"{'Config':<15} | {'N':<6} | {'WR':<7} | {'Avg R':<10} | {'Status'}")
        print("-"*60)
        
        for cfg in configs:
            all_trades = []
            for sym, (df, signals) in symbol_data.items():
                trades = run_backtest(df, signals, rr, use_filter=cfg['filter'])
                all_trades.extend(trades)
            
            if all_trades:
                wins = sum(1 for t in all_trades if t['result'] == 'win')
                n = len(all_trades)
                wr = wins / n * 100
                total_r = sum(t['r'] for t in all_trades)
                avg_r = total_r / n
                status = "✅ PROFIT" if avg_r > 0 else "❌"
                print(f"{cfg['name']:<15} | {n:<6} | {wr:>5.1f}% | {avg_r:>+8.4f}R | {status}")
                results.append({'cfg': cfg['name'], 'rr': rr, 'n': n, 'wr': wr, 'avg_r': avg_r})
    
    print("\n" + "="*80)
    
    profitable = [r for r in results if r['avg_r'] > 0]
    if profitable:
        print(f"\n🏆 FOUND {len(profitable)} PROFITABLE CONFIGURATIONS!")
        profitable.sort(key=lambda x: x['avg_r'], reverse=True)
        for r in profitable[:3]:
            print(f"  ✅ {r['cfg']} | RR={r['rr']} | {r['wr']:.1f}% WR | {r['avg_r']:+.4f}R")
    else:
        print("\n❌ NO PROFITABLE CONFIGURATIONS ON 4H")

if __name__ == "__main__":
    main()
