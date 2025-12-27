#!/usr/bin/env python3
"""
STRATEGY DISCOVERY ENGINE V2: VOLATILITY PIERCING
================================================
Tests mean reversion strategies that use Bollinger Band pierces.
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

def fetch_data(symbol, interval='60'):
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
    if len(df) < 50: return pd.DataFrame()
    df = df.copy()
    close = df['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['atr'] = (df['high'] - df['low']).rolling(20).mean()
    
    # BB
    ma = close.rolling(20).mean()
    std = close.rolling(20).std()
    df['bb_up'] = ma + (std * 2)
    df['bb_low'] = ma - (std * 2)
    
    # Divergence
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    df['reg_bull'] = (df['low'] <= df['price_low_14']) & (df['rsi'] > df['rsi_low_14'].shift(14)) & (df['rsi'] < 40)
    df['reg_bear'] = (df['high'] >= df['price_high_14']) & (df['rsi'] < df['rsi_high_14'].shift(14)) & (df['rsi'] > 60)
    return df

def run_bb_divergence(df):
    """
    Trigger: RSI Divergence + Price pierced BB Band and closed back inside.
    """
    trades = []
    for i in range(20, len(df)-2):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        side = None
        # Bullish: Lower low pierced BB_LOW, but current close is back inside
        if row['reg_bull'] and prev['low'] < prev['bb_low'] and row['close'] > row['bb_low']:
            side = 'long'
        # Bearish: Higher high pierced BB_UP, but current close is back inside
        if row['reg_bear'] and prev['high'] > prev['bb_up'] and row['close'] < row['bb_up']:
            side = 'short'
            
        if not side: continue
        
        entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        sl_dist = row['atr'] * 1.0
        tp_dist = sl_dist * 2.0
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
            res_r = 2.0 - (FEE/(sl_dist/entry)) if outcome == 'win' else -1.0 - (FEE/(sl_dist/entry))
            trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def run_rsi_extreme_reversal(df):
    """
    Trigger: RSI < 20 or RSI > 80.
    Entry: On first candle closing back across 25/75.
    SL: Pivot High/Low
    """
    trades = []
    for i in range(20, len(df)-2):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        side = None
        if prev['rsi'] < 20 and row['rsi'] >= 25: side = 'long'
        if prev['rsi'] > 80 and row['rsi'] <= 75: side = 'short'
        
        if not side: continue
        
        entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        # SL is the extreme low/high of the last 3 candles
        sl = df.iloc[i-3:i+1]['low'].min() * 0.999 if side == 'long' else df.iloc[i-3:i+1]['high'].max() * 1.001
        risk = abs(entry - sl)
        if risk <= 0: continue
        tp = entry + (risk * 2.5) if side == 'long' else entry - (risk * 2.5)
        
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
            res_r = 2.5 - (FEE/(risk/entry)) if outcome == 'win' else -1.0 - (FEE/(risk/entry))
            trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def main():
    symbols = get_symbols(20)
    print(f"DISCOVERY ENGINE V2 (PIERCING): Testing {len(symbols)} symbols on 1H...")
    
    all_dfs = {}
    for sym in symbols:
        df = fetch_data(sym)
        if len(df) > 200:
            all_dfs[sym] = calc_indicators(df)
    
    strategies = [
        ("BB Divergence", run_bb_divergence),
        ("RSI Reversal", run_rsi_extreme_reversal)
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
