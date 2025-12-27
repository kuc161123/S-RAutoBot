#!/usr/bin/env python3
"""
CREATIVE STRATEGY DISCOVERY ENGINE V1
=====================================
Tests new, low-lag entry triggers to find a profitable realistic strategy.

Strategies:
1. PIVOT_REVERSAL: RSI Oversold + Bullish Engulfing (Tight Pivot SL)
2. TREND_DIVERGENCE: Divergence only IF aligned with EMA 200
3. VOL_CLIMAX: Divergence + 2x Volume spike
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# Common Settings
DAYS = 200
FEE = 0.0006
SLIPPAGE = 0.0003
SYMBOLS_COUNT = 30

def get_symbols(n=SYMBOLS_COUNT):
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        BAD = ['XAUTUSDT', 'PAXGUSDT', 'USTCUSDT', 'USDCUSDT', 'BUSDUSDT', 'DAIUSDT']
        return [t['symbol'] for t in usdt[:n] if t['symbol'] not in BAD][:n]
    except: return []

def fetch_data(symbol, interval='60'): # 1H TF usually more stable for trend
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
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['atr'] = (df['high'] - df['low']).rolling(20).mean()
    df['ema200'] = close.ewm(span=200, adjust=False).mean()
    df['vol_ma'] = df['vol'].rolling(20).mean()
    # Swing points for divergence
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    df['reg_bull'] = (df['low'] <= df['price_low_14']) & (df['rsi'] > df['rsi_low_14'].shift(14)) & (df['rsi'] < 40)
    df['reg_bear'] = (df['high'] >= df['price_high_14']) & (df['rsi'] < df['rsi_high_14'].shift(14)) & (df['rsi'] > 60)
    return df

def run_pivot_reversal(df):
    """
    Trigger: RSI < 30 + Bullish Engulfing
    SL: Signal Candle Low - 0.1% buffer
    TP: 2.0 RR
    """
    trades = []
    for i in range(20, len(df)-2):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Bullish Engulfing + RSI OS
        is_bull_engulf = (row['close'] > prev['open']) and (prev['close'] < prev['open']) and (row['close'] > row['open'])
        if is_bull_engulf and row['rsi'] < 30:
            entry = df.iloc[i+1]['open'] * (1 + SLIPPAGE)
            sl = min(row['low'], prev['low']) * 0.999
            risk = entry - sl
            if risk <= 0: continue
            tp = entry + (risk * 2.0)
            
            outcome = None
            for j in range(i+1, min(i+100, len(df))):
                c = df.iloc[j]
                if c['low'] <= sl: outcome = 'loss'; break
                if c['high'] >= tp: outcome = 'win'; break
            
            if outcome:
                res_r = 2.0 - (FEE/ (risk/entry)) if outcome == 'win' else -1.0 - (FEE/ (risk/entry))
                trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def run_trend_alignment(df):
    """
    Trigger: Divergence aligned with EMA 200
    Entry: Immediate on next candle
    SL: 1.5x ATR
    TP: 3.0 RR
    """
    trades = []
    for i in range(50, len(df)-2):
        row = df.iloc[i]
        side = None
        if row['reg_bull'] and row['close'] > row['ema200']: side = 'long'
        if row['reg_bear'] and row['close'] < row['ema200']: side = 'short'
        if not side: continue
        
        entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        atr = row['atr']
        sl_dist = atr * 1.5
        tp_dist = sl_dist * 3.0
        sl = entry - sl_dist if side == 'long' else entry + sl_dist
        tp = entry + tp_dist if side == 'long' else entry - tp_dist
        
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
            risk_pct = sl_dist / entry
            fee_cost = FEE / risk_pct
            res_r = 3.0 - fee_cost if outcome == 'win' else -1.0 - fee_cost
            trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def run_vol_climax(df):
    """
    Trigger: RSI Divergence + Volume > 2x MA
    """
    trades = []
    for i in range(50, len(df)-2):
        row = df.iloc[i]
        if (row['reg_bull'] or row['reg_bear']) and row['vol'] > row['vol_ma'] * 2.0:
            side = 'long' if row['reg_bull'] else 'short'
            entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
            sl_dist = row['atr'] * 1.2
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
                res_r = 2.5 - (FEE/(sl_dist/entry)) if outcome == 'win' else -1.0 - (FEE/(sl_dist/entry))
                trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def main():
    symbols = get_symbols(20)
    print(f"DISCOVERY ENGINE V1: Testing {len(symbols)} symbols on 1H...")
    
    all_dfs = {}
    for sym in symbols:
        df = fetch_data(sym)
        if len(df) > 200:
            all_dfs[sym] = calc_indicators(df)
    
    strategies = [
        ("Pivot Reversal", run_pivot_reversal),
        ("Trend Alignment", run_trend_alignment),
        ("Volume Climax", run_vol_climax)
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
            print(f"{name:<20} | {wr:5.1f}% | {avg_r:+5.3f}R | {total_r:+7.1f}R | {len(trades):<6}")
        else:
            print(f"{name:<20} | NO TRADES")
    print("="*60)

if __name__ == "__main__":
    main()
