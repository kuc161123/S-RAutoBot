#!/usr/bin/env python3
"""
CREATIVE STRATEGY DISCOVERY ENGINE V4
=====================================
Experimental "Out of the box" strategies.

1. SQUEEZE_DIV: BB Squeeze (Width < 2% of Price) + Regular Divergence
2. ADX_TREND_CONT: Hidden Divergence + ADX > 25 + EMA 200 Alignment
3. RSI_TL_BREAK: RSI breaking its own 10-period trendline + Divergence
"""

import pandas as pd
import numpy as np
import requests
import time

# Settings
DAYS = 365 # 1 year for robustness
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
    close, high, low = df['close'], df['high'], df['low']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-9)))
    
    # BB Squeeze
    ma = close.rolling(20).mean()
    std = close.rolling(20).std()
    df['bb_width'] = (std * 4) / ma
    df['is_squeeze'] = df['bb_width'] < df['bb_width'].rolling(100).quantile(0.2) # Bottom 20% of width
    
    # ADX
    plus_dm = high.diff().where((high.diff() > low.diff().abs()) & (high.diff() > 0), 0)
    minus_dm = low.diff().abs().where((low.diff().abs() > high.diff()) & (low.diff().abs() > 0), 0)
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    df['adx'] = 100 * (abs(plus_dm.rolling(14).mean() - minus_dm.rolling(14).mean()) / (plus_dm.rolling(14).mean() + minus_dm.rolling(14).mean() + 1e-9))
    df['adx'] = df['adx'].rolling(14).mean()
    
    df['ema200'] = close.ewm(span=200, adjust=False).mean()
    df['atr'] = tr.rolling(20).mean()
    
    # Divergence (simplified for speed)
    df['price_low_14'] = low.rolling(14).min()
    df['price_high_14'] = high.rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    
    df['reg_bull'] = (low <= df['price_low_14']) & (df['rsi'] > df['rsi_low_14'].shift(14)) & (df['rsi'] < 40)
    df['reg_bear'] = (high >= df['price_high_14']) & (df['rsi'] < df['rsi_high_14'].shift(14)) & (df['rsi'] > 60)
    
    df['hidden_bull'] = (low > df['price_low_14'].shift(14)) & (df['rsi'] <= df['rsi_low_14'].shift(14)) & (df['rsi'] < 45)
    df['hidden_bear'] = (high < df['price_high_14'].shift(14)) & (df['rsi'] >= df['rsi_high_14'].shift(14)) & (df['rsi'] > 55)
    
    return df

def run_squeeze_div(df):
    """
    Trigger: RSI Divergence during or immediately after a BB Squeeze.
    """
    trades = []
    for i in range(50, len(df)-2):
        row = df.iloc[i]
        # Check if squeeze in last 5 candles
        if not df.iloc[i-5:i+1]['is_squeeze'].any(): continue
        
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

def run_hidden_div_pro(df):
    """
    Trigger: Hidden Divergence + ADX > 25 (Trend Strength)
    """
    trades = []
    for i in range(50, len(df)-2):
        row = df.iloc[i]
        if row['adx'] < 25: continue
        
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

def main():
    symbols = get_symbols(25)
    print(f"DISCOVERY ENGINE V4: Testing {len(symbols)} symbols on 1H (365 days)...")
    
    all_dfs = {}
    for sym in symbols:
        df = fetch_data(sym)
        if len(df) > 200:
            all_dfs[sym] = calc_indicators(df)
            
    strategies = [
        ("Squeeze Divergence", run_squeeze_div),
        ("Hidden Div Pro (ADX)", run_hidden_div_pro)
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
