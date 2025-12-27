#!/usr/bin/env python3
"""
CREATIVE STRATEGY DISCOVERY ENGINE V3
=====================================
Focusing on High-Precision / Trend-Continuation models.

Strategies:
1. HIDDEN_DIV: Price Higher Low + RSI Lower Low (Bullish Trend Continuation)
2. LIQUIDITY_SWEEP: Wick rejection of swing H/L + Regular Divergence
3. GAP_MEAN_REVERSION: Price Far from EMA 200 + RSI Extreme Reject
"""

import pandas as pd
import numpy as np
import requests
import time

# Settings
DAYS = 300
FEE = 0.0006
SLIPPAGE = 0.0003
INTERVAL = '60' # 1H for quality

def get_symbols(n=30):
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
    if len(df) < 200: return pd.DataFrame()
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    
    # RSI 14
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-9)))
    
    df['atr'] = (high - low).rolling(20).mean()
    df['ema200'] = close.ewm(span=200, adjust=False).mean()
    
    # Lows/Highs for Divergence
    df['price_low_14'] = low.rolling(14).min()
    df['price_high_14'] = high.rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    
    # Regular Divergence
    df['reg_bull'] = (low <= df['price_low_14']) & (df['rsi'] > df['rsi_low_14'].shift(14)) & (df['rsi'] < 40)
    df['reg_bear'] = (high >= df['price_high_14']) & (df['rsi'] < df['rsi_high_14'].shift(14)) & (df['rsi'] > 60)
    
    # Hidden Divergence (Trend Continuation)
    # Hidden Bull: Price higher low, RSI lower low
    df['hidden_bull'] = (low > df['price_low_14'].shift(14)) & (df['rsi'] <= df['rsi_low_14'].shift(14)) & (df['rsi'] < 45)
    # Hidden Bear: Price lower high, RSI higher high
    df['hidden_bear'] = (high < df['price_high_14'].shift(14)) & (df['rsi'] >= df['rsi_high_14'].shift(14)) & (df['rsi'] > 55)
    
    # Swing H/L for Sweep detection
    df['swing_h_20'] = high.rolling(20).max().shift(1)
    df['swing_l_20'] = low.rolling(20).min().shift(1)
    
    return df

def run_hidden_div(df):
    trades = []
    for i in range(50, len(df)-2):
        row = df.iloc[i]
        side = None
        if row['hidden_bull'] and row['close'] > row['ema200']: side = 'long'
        if row['hidden_bear'] and row['close'] < row['ema200']: side = 'short'
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

def run_liquidity_sweep(df):
    """
    Trigger: Price swept a 20-candle high/low but closed back inside AND has divergence.
    """
    trades = []
    for i in range(50, len(df)-2):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        side = None
        # Long: Low went below 20-candle swing low, but close is above it.
        if row['low'] < row['swing_l_20'] and row['close'] > row['swing_l_20'] and row['reg_bull']:
            side = 'long'
        # Short: High went above 20-candle swing high, but close is below it.
        if row['high'] > row['swing_h_20'] and row['close'] < row['swing_h_20'] and row['reg_bear']:
            side = 'short'
            
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

def main():
    symbols = get_symbols(25)
    print(f"DISCOVERY ENGINE V3: Testing {len(symbols)} symbols on 1H...")
    
    all_dfs = {}
    for sym in symbols:
        df = fetch_data(sym)
        if len(df) > 200:
            all_dfs[sym] = calc_indicators(df)
            
    strategies = [
        ("Hidden Divergence", run_hidden_div),
        ("Liquidity Sweep", run_liquidity_sweep)
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
