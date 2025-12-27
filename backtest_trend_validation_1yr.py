#!/usr/bin/env python3
"""
TREND DIVERGENCE VALIDATION (1 YEAR)
====================================
Validating the trend-alignment strategy across 50 symbols for 365 days.
"""

import pandas as pd
import numpy as np
import requests
import time

# Settings
DAYS = 365
TIMEFRAMES = ["60", "15"]
RR = 3.0
SL_MULT = 1.5
FEE = 0.0006
SLIPPAGE = 0.0003

def get_symbols(n=50):
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        return [t['symbol'] for t in usdt[:n]]
    except: return []

def fetch_data(symbol, interval):
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
    if len(df) < 250: return pd.DataFrame()
    df = df.copy()
    close = df['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['atr'] = (df['high'] - df['low']).rolling(20).mean()
    df['ema200'] = close.ewm(span=200, adjust=False).mean()
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    df['reg_bull'] = (df['low'] <= df['price_low_14']) & (df['rsi'] > df['rsi_low_14'].shift(14)) & (df['rsi'] < 40)
    df['reg_bear'] = (df['high'] >= df['price_high_14']) & (df['rsi'] < df['rsi_high_14'].shift(14)) & (df['rsi'] > 60)
    return df

def run_strategy(df):
    trades = []
    for i in range(200, len(df)-2):
        row = df.iloc[i]
        side = None
        if row['reg_bull'] and row['close'] > row['ema200']: side = 'long'
        if row['reg_bear'] and row['close'] < row['ema200']: side = 'short'
        if not side: continue
        
        entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        sl_dist = row['atr'] * SL_MULT
        tp_dist = sl_dist * RR
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        
        outcome = None
        for j in range(i+1, min(i+300, len(df))):
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
            res_r = RR - fee_cost if outcome == 'win' else -1.0 - fee_cost
            trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def main():
    symbols = get_symbols(30)
    print(f"VALIDATION: Testing {len(symbols)} symbols for 365 days...")
    
    for tf in TIMEFRAMES:
        print(f"\n--- TF: {tf}m ---")
        all_trades = []
        for i, sym in enumerate(symbols):
            df = fetch_data(sym, tf)
            if len(df) < 300: continue
            df = calc_indicators(df)
            trades = run_strategy(df)
            all_trades.extend(trades)
            if (i+1) % 5 == 0: print(f"  Processed {i+1} symbols...")
            
        if all_trades:
            wr = sum(1 for t in all_trades if t['win']) / len(all_trades) * 100
            total_r = sum(t['r'] for t in all_trades)
            avg_r = total_r / len(all_trades)
            print(f"  RESULT: WR={wr:.1f}%, Avg={avg_r:+.3f}R, Total={total_r:+.1f}R (N={len(all_trades)})")

if __name__ == "__main__":
    main()
