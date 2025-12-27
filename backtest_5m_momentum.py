#!/usr/bin/env python3
"""
5M DIVERGENCE - MOMENTUM CONFIRMATION
=====================================
Testing stricter momentum-based BOS confirmation:
1. Strong BOS: Close breaks swing by at least 0.5 ATR
2. Momentum candle: Large body (body > 60% of range)
3. Volume spike on BOS candle
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Config
TIMEFRAME = '5'
DATA_DAYS = 60
NUM_SYMBOLS = 40
MAX_WAIT_CANDLES = 10
SL_ATR_MULT = 0.8

# Test R:R
RR_RATIOS = [1.0, 1.5, 2.0]

# Costs
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

RSI_PERIOD = 14
LOOKBACK_BARS = 10
MIN_PIVOT_DISTANCE = 3

BASE_URL = "https://api.bybit.com"

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
    
    # Body size as % of range
    df['body'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    df['body_pct'] = df['body'] / (df['range'] + 1e-9)
    
    # Direction
    df['bullish'] = df['close'] > df['open']
    df['bearish'] = df['close'] < df['open']
    
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
                if rsi[i] < 40:
                    swing_high = max(high[max(0, i-LOOKBACK_BARS):i+1])
                    signals.append({'idx': i, 'side': 'long', 'swing': swing_high})
                    continue
        
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                if rsi[i] > 60:
                    swing_low = min(low[max(0, i-LOOKBACK_BARS):i+1])
                    signals.append({'idx': i, 'side': 'short', 'swing': swing_low})
    
    return signals

def run_backtest(df, signals, rr, config):
    rows = list(df.itertuples())
    atr = df['atr'].values
    bullish = df['bullish'].values
    bearish = df['bearish'].values
    body_pct = df['body_pct'].values
    volume = df['volume'].values
    vol_ma = df['vol_ma'].values
    close_arr = df['close'].values
    ema200 = df['ema200'].values
    
    trades = []
    
    for sig in signals:
        div_idx = sig['idx']
        side = sig['side']
        
        # Trend filter
        if config.get('trend'):
            if side == 'long' and close_arr[div_idx] < ema200[div_idx]:
                continue
            if side == 'short' and close_arr[div_idx] > ema200[div_idx]:
                continue
        
        bos_idx = None
        for j in range(div_idx + 1, min(div_idx + 1 + MAX_WAIT_CANDLES, len(rows) - 10)):
            row = rows[j]
            
            # Volume spike check
            if config.get('vol_spike') and volume[j] < vol_ma[j] * 1.5:
                continue
            
            # Momentum candle check (large body)
            if config.get('momentum') and body_pct[j] < 0.6:
                continue
            
            # Strong BOS check (break by at least 0.3 ATR)
            break_dist = 0
            if side == 'long':
                if row.close > sig['swing']:
                    break_dist = row.close - sig['swing']
                    # Candle must be bullish
                    if config.get('candle_dir') and not bullish[j]:
                        continue
                    # Strong break
                    if config.get('strong_bos') and break_dist < atr[j] * 0.3:
                        continue
                    bos_idx = j
                    break
            else:
                if row.close < sig['swing']:
                    break_dist = sig['swing'] - row.close
                    if config.get('candle_dir') and not bearish[j]:
                        continue
                    if config.get('strong_bos') and break_dist < atr[j] * 0.3:
                        continue
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
        sl_dist = entry_atr * SL_ATR_MULT
        
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
    print("5M DIVERGENCE - MOMENTUM CONFIRMATION")
    print("="*80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"Fetching data for {len(symbols)} symbols...")
    
    symbol_data = {}
    for idx, sym in enumerate(symbols):
        df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
        if df.empty or len(df) < 500:
            continue
        df = add_indicators(df)
        if len(df) < 100:
            continue
        signals = detect_divergences(df)
        if signals:
            symbol_data[sym] = (df, signals)
        if (idx + 1) % 10 == 0:
            print(f"  Fetched {idx+1}/{len(symbols)} symbols...")
    
    print(f"\nSymbols: {len(symbol_data)}")
    
    configs = [
        {'name': 'No Filter', 'cfg': {}},
        {'name': 'Candle Dir Only', 'cfg': {'candle_dir': True}},
        {'name': 'Strong BOS (0.3 ATR)', 'cfg': {'strong_bos': True, 'candle_dir': True}},
        {'name': 'Momentum Candle', 'cfg': {'momentum': True, 'candle_dir': True}},
        {'name': 'Vol Spike', 'cfg': {'vol_spike': True, 'candle_dir': True}},
        {'name': 'Trend Filter', 'cfg': {'trend': True, 'candle_dir': True}},
        {'name': 'Strong + Momentum', 'cfg': {'strong_bos': True, 'momentum': True, 'candle_dir': True}},
        {'name': 'Strong + Vol + Trend', 'cfg': {'strong_bos': True, 'vol_spike': True, 'trend': True, 'candle_dir': True}},
        {'name': 'ALL FILTERS', 'cfg': {'strong_bos': True, 'momentum': True, 'vol_spike': True, 'trend': True, 'candle_dir': True}},
    ]
    
    for rr in RR_RATIOS:
        print(f"\n{'='*80}")
        print(f"R:R = {rr}:1")
        print(f"{'='*80}")
        print(f"{'Config':<25} | {'N':<6} | {'WR':<8} | {'Avg R':<10} | {'Total R':<9}")
        print("-"*75)
        
        for c in configs:
            all_trades = []
            for sym, (df, signals) in symbol_data.items():
                trades = run_backtest(df, signals, rr, c['cfg'])
                all_trades.extend(trades)
            
            if all_trades:
                wins = sum(1 for t in all_trades if t['result'] == 'win')
                n = len(all_trades)
                wr = wins / n * 100
                total_r = sum(t['r'] for t in all_trades)
                avg_r = total_r / n
                status = "✅" if avg_r > 0 else "❌"
                print(f"{c['name']:<25} | {n:<6} | {wr:>6.1f}% | {avg_r:>+8.4f}R | {total_r:>+8.1f}R {status}")
            else:
                print(f"{c['name']:<25} | {'0':<6} | {'--':<8} | {'--':<10} | {'--':<9}")

if __name__ == "__main__":
    main()
