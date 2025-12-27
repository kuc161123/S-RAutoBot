#!/usr/bin/env python3
"""
CREATIVE STRATEGY DISCOVERY ENGINE V5
=====================================
Focusing on Volume-Price Analysis (VPA) and Volatility Exhaustion.

1. ABSORPTION_REV: Volume > 3x MA + (Body < 0.3x ATR) -> Sign of Absorption.
2. VOL_EXHAUSTION: BB Width at 100-candle high (Overstretched) + RSI Extreme.
3. OBV_DIVERGENCE: Price higher high, OBV lower high (Smart Money Divergence).
"""

import pandas as pd
import numpy as np
import requests
import time

# Settings
DAYS = 300
FEE = 0.0006
SLIPPAGE = 0.0003
INTERVAL = '60' # 1H

def get_symbols(n=25):
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        return [t['symbol'] for t in usdt[:n]]
    except: return []

def fetch_data(symbol):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': INTERVAL, 'limit': 1000, 'end': end_ts}
            r = requests.get(url, params=params).json()
            if r['retCode'] != 0 or not r['result']['list']: break
            klines = r['result']['list']
            all_kline.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.01)
        if not all_kline: return pd.DataFrame()
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
        return df
    except: return pd.DataFrame()

def calc_indicators(df):
    if len(df) < 50: return pd.DataFrame()
    df = df.copy()
    close, high, low, vol = df['close'], df['high'], df['low'], df['vol']
    
    # ATR & MA
    df['atr'] = (high - low).rolling(20).mean()
    df['vol_ma'] = vol.rolling(20).mean()
    df['body'] = abs(close - df['open'])
    
    # OBV
    obv = [0]
    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + vol.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - vol.iloc[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    
    # BB Width
    ma = close.rolling(20).mean()
    std = close.rolling(20).std()
    df['bb_width'] = (std * 4) / ma
    df['bb_width_max'] = df['bb_width'].rolling(100).max()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-9)))
    
    return df

def run_absorption(df):
    """
    Trigger: Volume > 3x MA but Price Body < 0.3x ATR (Absorption)
    """
    trades = []
    for i in range(20, len(df)-2):
        row = df.iloc[i]
        if row['vol'] > row['vol_ma'] * 3.0 and row['body'] < row['atr'] * 0.3:
            # Side depends on price location relative to mid-band or prev trend
            # Simplified: If close is in bottom 30% of candle -> Bullish Absorption
            candle_range = row['high'] - row['low']
            if candle_range == 0: continue
            
            side = None
            if (row['close'] - row['low']) / candle_range > 0.7: side = 'long' # Exhaustion/Reversal
            if (row['high'] - row['close']) / candle_range > 0.7: side = 'short'
            
            if not side: continue
            
            entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
            sl_dist = row['atr'] * 1.5
            tp_dist = sl_dist * 2.5
            sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
            
            outcome = None
            for j in range(i+1, min(i+100, len(df))):
                c = df.iloc[j]
                if side == 'long':
                    if c['low'] <= sl: outcome = 'loss'; break
                    if c['high'] >= tp: outcome = 'win'; break
                else:
                    if c['high'] >= sl: outcome = 'loss'; break
                    if c['low'] <= tp: outcome = 'win'; break
            if outcome:
                res_r = 2.5 - (FEE+SLIPPAGE)/(sl_dist/entry) if outcome == 'win' else -1.0 - (FEE+SLIPPAGE)/(sl_dist/entry)
                trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def run_vol_exhaustion(df):
    """
    Trigger: BB Width at 100-candle high + RSI at extremes.
    """
    trades = []
    for i in range(100, len(df)-2):
        row = df.iloc[i]
        if row['bb_width'] >= row['bb_width_max'] and (row['rsi'] < 30 or row['rsi'] > 70):
            side = 'long' if row['rsi'] < 30 else 'short'
            entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
            sl_dist = row['atr'] * 2.0
            tp_dist = sl_dist * 3.0
            sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
            
            outcome = None
            for j in range(i+1, min(i+150, len(df))):
                c = df.iloc[j]
                if side == 'long':
                    if c['low'] <= sl: outcome = 'loss'; break
                    if c['high'] >= tp: outcome = 'win'; break
                else:
                    if c['high'] >= sl: outcome = 'loss'; break
                    if c['low'] <= tp: outcome = 'win'; break
            if outcome:
                res_r = 3.0 - FEE/(sl_dist/entry) if outcome == 'win' else -1.0 - FEE/(sl_dist/entry)
                trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def main():
    symbols = get_symbols(25)
    print(f"DISCOVERY ENGINE V5: Testing {len(symbols)} symbols on 1H...")
    
    all_dfs = {}
    for sym in symbols:
        df = fetch_data(sym)
        if len(df) > 200:
            all_dfs[sym] = calc_indicators(df)
            
    strategies = [
        ("Volume Absorption", run_absorption),
        ("Vol Exhaustion", run_vol_exhaustion)
    ]
    
    print("\n" + "="*60)
    print(f"{'Strategy':<20} | {'WR':<8} | {'Avg R':<8} | {'Total R':<8} | {'N':<6}")
    print("-"*60)
    
    for name, func in strategies:
        trades = []
        for sym, df in all_dfs.items():
            trades.extend(func(df))
        if trades:
            wr = sum(1 for t in trades if t['win']) / len(trades) * 100
            total_r = sum(t['r'] for t in trades)
            avg_r = total_r / len(trades)
            print(f"{name:<20} | {wr:4.1f}% | {avg_r:+5.3f}R | {total_r:+7.1f}R | {len(trades):<6}")
        else:
            print(f"{name:<20} | NO TRADES")
    print("="*60)

if __name__ == "__main__":
    main()
