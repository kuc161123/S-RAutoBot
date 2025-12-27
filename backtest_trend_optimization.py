#!/usr/bin/env python3
"""
TREND ALIGNMENT OPTIMIZATION
============================
Refining the profitable "Trend Alignment" strategy found in discovery.
Grid Search:
- Timeframes: 15m, 30m, 1h
- SL Multiplier: 1.0, 1.2, 1.5, 2.0
- RR Ratio: 2.0, 2.5, 3.0, 4.0
"""

import pandas as pd
import numpy as np
import requests
import time
import itertools

# Common Settings
DAYS = 250
FEE = 0.0006
SLIPPAGE = 0.0003

# Grid
TIMEFRAMES = ["15", "30", "60"]
SL_MULTS = [1.0, 1.2, 1.5, 2.0]
RR_RATIOS = [2.0, 2.5, 3.0, 4.0]

def get_symbols(n=20):
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

def run_strategy(df, sl_mult, rr):
    trades = []
    for i in range(200, len(df)-2):
        row = df.iloc[i]
        side = None
        if row['reg_bull'] and row['close'] > row['ema200']: side = 'long'
        if row['reg_bear'] and row['close'] < row['ema200']: side = 'short'
        if not side: continue
        
        entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        sl_dist = row['atr'] * sl_mult
        tp_dist = sl_dist * rr
        sl = entry - sl_dist if side == 'long' else entry + sl_dist
        tp = entry + tp_dist if side == 'long' else entry - tp_dist
        
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
            fee_cost = (FEE + SLIPPAGE) / risk_pct
            res_r = rr - fee_cost if outcome == 'win' else -1.0 - fee_cost
            trades.append({'r': res_r, 'win': outcome == 'win'})
    return trades

def main():
    symbols = get_symbols(15)
    print(f"TREND OPTIMIZATION: Testing {len(symbols)} symbols...")
    
    # Pre-fetch data for all TFs
    data = {}
    for tf in TIMEFRAMES:
        print(f"Fetching {tf}m data...")
        data[tf] = {}
        for sym in symbols:
            df = fetch_data(sym, tf)
            if len(df) > 250:
                data[tf][sym] = calc_indicators(df)

    results = []
    combinations = list(itertools.product(TIMEFRAMES, SL_MULTS, RR_RATIOS))
    print(f"\nTesting {len(combinations)} combinations...")
    
    for tf, sl_mult, rr in combinations:
        all_trades = []
        for sym, df in data[tf].items():
            all_trades.extend(run_strategy(df, sl_mult, rr))
        
        if all_trades:
            total_r = sum(t['r'] for t in all_trades)
            avg_r = total_r / len(all_trades)
            wr = sum(1 for t in all_trades if t['win']) / len(all_trades) * 100
            results.append({'tf': tf, 'sl': sl_mult, 'rr': rr, 'wr': wr, 'avg_r': avg_r, 'total_r': total_r, 'n': len(all_trades)})
            
    # Sort and display top 10
    results.sort(key=lambda x: x['avg_r'], reverse=True)
    print("\n" + "="*70)
    print(f"{'TF':<4} | {'SL':<4} | {'RR':<4} | {'WR':<6} | {'Avg R':<8} | {'Total R':<8} | {'N':<5}")
    print("-" * 70)
    for r in results[:20]:
        print(f"{r['tf']:<4} | {r['sl']:<4.1f} | {r['rr']:<4.1f} | {r['wr']:4.1f}% | {r['avg_r']:+6.3f}R | {r['total_r']:+7.1f}R | {r['n']:<5}")
    print("="*70)

if __name__ == "__main__":
    main()
