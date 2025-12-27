#!/usr/bin/env python3
"""
CREATIVE STRATEGY DISCOVERY ENGINE V6
=====================================
Focusing on Session Timing and Multi-EMA Trend Alignment.

1. SESSION_DIV: RSI Divergence ONLY during London/NY Overlap (12:00-16:00 UTC).
2. TRIPLE_EMA_PULLBACK: Entry on RSI OS when 20 > 50 > 200 EMA (Strong Trend).
3. THREE_BAR_REVERSAL: Divergence + 3-candle reversal pattern.
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# Settings
DAYS = 250
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
        # Add Hour
        df['hour'] = pd.to_datetime(df['ts'], unit='ms').dt.hour
        return df
    except: return pd.DataFrame()

def calc_indicators(df):
    if len(df) < 200: return pd.DataFrame()
    df = df.copy()
    close = df['close']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-9)))
    
    # EMAs
    df['ema20'] = close.ewm(span=20, adjust=False).mean()
    df['ema50'] = close.ewm(span=50, adjust=False).mean()
    df['ema200'] = close.ewm(span=200, adjust=False).mean()
    df['atr'] = (df['high'] - df['low']).rolling(20).mean()
    
    # Divergence
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    df['reg_bull'] = (df['low'] <= df['price_low_14']) & (df['rsi'] > df['rsi_low_14'].shift(14)) & (df['rsi'] < 40)
    df['reg_bear'] = (df['high'] >= df['price_high_14']) & (df['rsi'] < df['rsi_high_14'].shift(14)) & (df['rsi'] > 60)
    
    return df

def run_session_div(df):
    """
    Trigger: Divergence during London/NY overlap (12:00 - 16:00 UTC)
    """
    trades = []
    for i in range(50, len(df)-2):
        row = df.iloc[i]
        if not (12 <= row['hour'] <= 16): continue
        
        side = 'long' if row['reg_bull'] else 'short' if row['reg_bear'] else None
        if not side: continue
        
        entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        sl_dist = row['atr'] * 1.5
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
            res_r = 3.0 - (FEE+SLIPPAGE)/(sl_dist/entry) if outcome == 'win' else -1.0 - (FEE+SLIPPAGE)/(sl_dist/entry)
            trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def run_ema_stack_pullback(df):
    """
    Trigger: RSI Pullback to 40 (Long) or 60 (Short) while EMAs are perfectly stacked.
    EMA 20 > 50 > 200 for Long
    EMA 20 < 50 < 200 for Short
    """
    trades = []
    for i in range(200, len(df)-2):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        side = None
        # Long Pullback: Perfectly stacked EMAs + RSI crossing UP through 35
        if row['ema20'] > row['ema50'] > row['ema200'] and prev['rsi'] < 35 and row['rsi'] >= 35:
            side = 'long'
        # Short Pullback: Perfectly stacked EMAs + RSI crossing DOWN through 65
        if row['ema20'] < row['ema50'] < row['ema200'] and prev['rsi'] > 65 and row['rsi'] <= 65:
            side = 'short'
            
        if not side: continue
        
        entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        sl_dist = row['atr'] * 2.0
        tp_dist = sl_dist * 3.0
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        
        outcome = None
        for j in range(i+1, min(i+200, len(df))):
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
    print(f"DISCOVERY ENGINE V6: Testing {len(symbols)} symbols on 1H...")
    
    all_dfs = {}
    for sym in symbols:
        df = fetch_data(sym)
        if len(df) > 200:
            all_dfs[sym] = calc_indicators(df)
            
    strategies = [
        ("Session Divergence", run_session_div),
        ("EMA Stack Pullback", run_ema_stack_pullback)
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
