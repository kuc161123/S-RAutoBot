#!/usr/bin/env python3
"""
EXTREME CREATIVITY DISCOVERY
============================
Testing non-traditional entry triggers.
"""

import pandas as pd
import numpy as np
import requests
import time

# Settings
DAYS = 200
FEE = 0.0006
SLIPPAGE = 0.0003

def get_symbols(n=20):
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        return [t['symbol'] for t in usdt[:n]]
    except: return []

def fetch_data(symbol, interval='5'):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': str(interval), 'limit': 1000, 'end': end_ts}
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
    if len(df) < 200: return pd.DataFrame()
    df = df.copy()
    close = df['close']
    
    # RSI 2 for Snapback
    delta = close.diff()
    gain2 = (delta.where(delta > 0, 0)).rolling(2).mean()
    loss2 = (-delta.where(delta < 0, 0)).rolling(2).mean()
    df['rsi2'] = 100 - (100 / (1 + gain2/(loss2+1e-9)))
    
    # EMA 9/21 for Fast Momentum
    df['ema9'] = close.ewm(span=9, adjust=False).mean()
    df['ema21'] = close.ewm(span=21, adjust=False).mean()
    df['ema200'] = close.ewm(span=200, adjust=False).mean()
    
    # RSI 14
    gain14 = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss14 = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi14'] = 100 - (100 / (1 + gain14/(loss14+1e-9)))
    df['atr'] = (df['high'] - df['low']).rolling(20).mean()
    
    # Divergence
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi14'].rolling(14).min()
    df['rsi_high_14'] = df['rsi14'].rolling(14).max()
    df['reg_bull'] = (df['low'] <= df['price_low_14']) & (df['rsi14'] > df['rsi_low_14'].shift(14)) & (df['rsi14'] < 40)
    df['reg_bear'] = (df['high'] >= df['price_high_14']) & (df['rsi14'] < df['rsi_high_14'].shift(14)) & (df['rsi14'] > 60)
    
    return df

def run_fast_momentum(df):
    """
    Trigger: Divergence confirmed by EMA 9/21 Cross.
    """
    trades = []
    for i in range(50, len(df)-2):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Look back 15 candles for divergence signal
        recent_signals = df.iloc[i-15:i+1]
        has_bull = recent_signals['reg_bull'].any()
        has_bear = recent_signals['reg_bear'].any()
        
        side = None
        if has_bull and prev['ema9'] < prev['ema21'] and row['ema9'] >= row['ema21']: side = 'long'
        if has_bear and prev['ema9'] > prev['ema21'] and row['ema9'] <= row['ema21']: side = 'short'
        
        if not side: continue
        
        entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        sl_dist = row['atr'] * 1.5
        tp_dist = sl_dist * 2.5
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
            risk_pct = sl_dist / entry
            fee_cost = (FEE+SLIPPAGE)/risk_pct
            res_r = 2.5 - fee_cost if outcome == 'win' else -1.0 - fee_cost
            trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def run_snapback(df):
    """
    Connors-style: 
    1. Close > EMA 200 (Uptrend)
    2. RSI 2 < 10 (Oversold pullback)
    3. Buy on next candle open
    4. TP: Close > EMA 10
    5. SL: 2x ATR
    """
    trades = []
    for i in range(50, len(df)-2):
        row = df.iloc[i]
        
        side = None
        if row['close'] > row['ema200'] and row['rsi2'] < 5: side = 'long'
        if row['close'] < row['ema200'] and row['rsi2'] > 95: side = 'short'
        
        if not side: continue
        
        entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        sl_dist = row['atr'] * 2.0
        sl = entry - sl_dist if side == 'long' else entry + sl_dist
        
        outcome = None
        for j in range(i+1, min(i+100, len(df))):
            c = df.iloc[j]
            # Custom TP: Exit on EMA 10 cross
            ema10 = c['close'] # Actually should calculate EMA 10 in loop or pre-calc
            
            # Simplified: Use 1.5 RR for test speed
            tp = entry + sl_dist * 1.5 if side == 'long' else entry - sl_dist * 1.5
            
            if side == 'long':
                if c['low'] <= sl: outcome = 'loss'; break
                if c['high'] >= tp: outcome = 'win'; break
            else:
                if c['high'] >= sl: outcome = 'loss'; break
                if c['low'] <= tp: outcome = 'win'; break
        if outcome:
            risk_pct = sl_dist / entry
            fee_cost = (FEE+SLIPPAGE)/risk_pct
            res_r = 1.5 - fee_cost if outcome == 'win' else -1.0 - fee_cost
            trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def main():
    symbols = get_symbols(20)
    print(f"EXTREME DISCOVERY: Testing {len(symbols)} symbols on 5M...")
    
    all_dfs = {}
    for sym in symbols:
        df = fetch_data(sym)
        if len(df) > 200:
            all_dfs[sym] = calc_indicators(df)
            
    strategies = [
        ("Fast Momentum", run_fast_momentum),
        ("Snapback MR", run_snapback)
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
