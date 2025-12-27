#!/usr/bin/env python3
"""
TREND DIVERGENCE PRO
====================
Combining Trend Alignment with ADX filtering and RSI optimization.
"""

import pandas as pd
import numpy as np
import requests
import time

# Settings
DAYS = 250
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

def calc_indicators(df, rsi_period=14):
    if len(df) < 50: return pd.DataFrame()
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    df['atr'] = (high - low).rolling(20).mean()
    
    # EMA
    df['ema200'] = close.ewm(span=200, adjust=False).mean()
    
    # ADX
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
    atr_adx = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_adx)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_adx)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['adx'] = dx.rolling(14).mean()
    
    # Divergence
    df['price_low_14'] = low.rolling(14).min()
    df['price_high_14'] = high.rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    df['reg_bull'] = (low <= df['price_low_14']) & (df['rsi'] > df['rsi_low_14'].shift(14)) & (df['rsi'] < 40)
    df['reg_bear'] = (high >= df['price_high_14']) & (df['rsi'] < df['rsi_high_14'].shift(14)) & (df['rsi'] > 60)
    return df

def run_strategy(df, adx_min=20):
    trades = []
    for i in range(200, len(df)-2):
        row = df.iloc[i]
        if row['adx'] < adx_min: continue
        
        side = None
        if row['reg_bull'] and row['close'] > row['ema200']: side = 'long'
        if row['reg_bear'] and row['close'] < row['ema200']: side = 'short'
        if not side: continue
        
        entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        sl_dist = row['atr'] * 1.5
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
            risk_pct = sl_dist / entry
            fee_cost = (FEE+SLIPPAGE)/risk_pct
            res_r = 3.0 - fee_cost if outcome == 'win' else -1.0 - fee_cost
            trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def main():
    symbols = get_symbols(25)
    print(f"TREND PRO: Testing {len(symbols)} symbols on 1H with ADX...")
    
    all_trades = []
    for sym in symbols:
        df = fetch_data(sym)
        if len(df) > 200:
            df = calc_indicators(df, rsi_period=9) # Testing tighter RSI
            trades = run_strategy(df, adx_min=25)
            all_trades.extend(trades)
            print(f"  {sym}: {len(trades)} trades")
            
    if all_trades:
        wr = sum(1 for t in all_trades if t['win']) / len(all_trades) * 100
        total_r = sum(t['r'] for t in all_trades)
        avg_r = total_r / len(all_trades)
        print("\n" + "="*30)
        print(f"FINAL RESULTS (Trend Pro)")
        print(f"Win Rate: {wr:.1f}%")
        print(f"Avg R/Trade: {avg_r:+.3f}R")
        print(f"Total R: {total_r:+.1f}R")
        print(f"N: {len(all_trades)}")
        print("="*30)

if __name__ == "__main__":
    main()
