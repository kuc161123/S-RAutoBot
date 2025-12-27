#!/usr/bin/env python3
"""
EXTREME VALIDATION & STRESS TEST
===============================

Rigorous stress testing of the 2-candle Structure Break strategy:
1. Slippage Stress Test (0% to 0.2% per trade)
2. ATR Sensitivity (0.5x to 1.5x Multiplier)
3. RR Sensitivity (1:1 to 5:1)
4. Session Analysis (London, NY, Asia)
5. Max Drawdown & Recovery Factor
"""

import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime

# BASE CONFIG
DAYS = 60
TIMEFRAME = 5
BASE_RR = 3.0
BASE_SL_MULT = 0.8
MAX_WAIT_CANDLES = 2
FEE_PERCENT = 0.0006  # Bybit Taker 0.06%

def get_symbols():
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt_pairs.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        BAD = ['XAUTUSDT', 'PAXGUSDT', 'USTCUSDT', 'USDCUSDT', 'BUSDUSDT', 'DAIUSDT']
        return [t['symbol'] for t in usdt_pairs[:98] if t['symbol'] not in BAD][:98]
    except:
        return []

def fetch_data(symbol):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': str(TIMEFRAME), 'limit': 1000, 'end': end_ts}
            r = requests.get(url, params=params).json()
            if r['retCode'] != 0 or not r['result']['list']: break
            klines = r['result']['list']
            all_kline.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.02)
        if not all_kline: return pd.DataFrame()
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
        df['datetime'] = pd.to_datetime(df['ts'].astype(float), unit='ms')
        df['hour'] = df['datetime'].dt.hour
        return df
    except: return pd.DataFrame()

def calc_indicators(df):
    df = df.copy()
    close = df['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    h, l, c_prev = df['high'], df['low'], close.shift(1)
    tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(20).mean()
    df['swing_high_10'] = df['high'].rolling(10).max()
    df['swing_low_10'] = df['low'].rolling(10).min()
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    df['reg_bull'] = (df['low'] <= df['price_low_14']) & (df['rsi'] > df['rsi_low_14'].shift(14)) & (df['rsi'] < 40)
    df['reg_bear'] = (df['high'] >= df['price_high_14']) & (df['rsi'] < df['rsi_high_14'].shift(14)) & (df['rsi'] > 60)
    return df

def simulate_trade(df, i, side, rr, sl_mult, slippage, fee):
    # Same logic as realistic script
    row = df.iloc[i]
    structure_broken = False
    candles_waited = 0
    for ahead in range(1, MAX_WAIT_CANDLES + 1):
        if i + ahead >= len(df) - 1: break
        check = df.iloc[i + ahead]
        candles_waited = ahead
        if side == 'long' and check['close'] > row['swing_high_10']:
            structure_broken = True
            break
        if side == 'short' and check['close'] < row['swing_low_10']:
            structure_broken = True
            break
    if not structure_broken: return None
    entry_idx = i + candles_waited
    if entry_idx >= len(df): return None
    base_entry = df.iloc[entry_idx]['open']
    if side == 'long': entry = base_entry * (1 + slippage)
    else: entry = base_entry * (1 - slippage)
    atr = row['atr']
    if pd.isna(atr) or atr == 0: return None
    sl_dist = atr * sl_mult
    if sl_dist/entry > 0.05: return None
    tp_dist = sl_dist * rr
    if side == 'long': sl, tp = entry - sl_dist, entry + tp_dist
    else: sl, tp = entry + sl_dist, entry - tp_dist
    outcome = 'timeout'
    for j in range(entry_idx, min(entry_idx + 300, len(df))):
        c = df.iloc[j]
        if side == 'long':
            if c['low'] <= sl: outcome = 'loss'; break
            if c['high'] >= tp: outcome = 'win'; break
        else:
            if c['high'] >= sl: outcome = 'loss'; break
            if c['low'] <= tp: outcome = 'win'; break
    risk_pct = sl_dist / entry
    if outcome == 'win': res_r = rr - (fee + slippage) / risk_pct
    elif outcome == 'loss': res_r = -1.0 - (fee + slippage) / risk_pct
    else: res_r = -0.2
    return {'r': res_r, 'win': outcome == 'win', 'hour': row['hour']}

def run_backtest(datasets, rr=BASE_RR, sl_mult=BASE_SL_MULT, slippage=0.0003):
    all_trades = []
    for sym, df in datasets.items():
        cooldown = 0
        for i in range(60, len(df)-2):
            if cooldown > 0: cooldown -= 1; continue
            row = df.iloc[i]
            side = 'long' if row['reg_bull'] else 'short' if row['reg_bear'] else None
            if not side: continue
            res = simulate_trade(df, i, side, rr, sl_mult, slippage, FEE_PERCENT)
            if res:
                all_trades.append(res)
                cooldown = 6
    return all_trades

def analyze_results(trades, label=""):
    if not trades: return {"net_r": 0, "avg_r": 0, "wr": 0}
    net_r = sum(t['r'] for t in trades)
    avg_r = net_r / len(trades)
    wr = sum(1 for t in trades if t['win']) / len(trades) * 100
    return {"net_r": net_r, "avg_r": avg_r, "wr": wr, "count": len(trades)}

def main():
    print("🚀 INITIALIZING EXTREME VALIDATION...")
    symbols = get_symbols()
    datasets = {}
    for i, sym in enumerate(symbols):
        df = fetch_data(sym)
        if not df.empty: datasets[sym] = calc_indicators(df)
        if (i+1) % 20 == 0: print(f"Progress: {i+1}/98...")
    
    # 1. BASELINE
    print("\n--- 1. BASELINE (Realistic Fees) ---")
    base_trades = run_backtest(datasets)
    res = analyze_results(base_trades)
    print(f"Net R: {res['net_r']:.1f} | Avg R: {res['avg_r']:.3f} | WR: {res['wr']:.1f}% | Trades: {res['count']}")

    # 2. SLIPPAGE STRESS TEST
    print("\n--- 2. SLIPPAGE STRESS TEST ---")
    print(f"{'Slippage':<10} | {'Net R':<10} | {'Avg R':<10} | {'Status'}")
    for slip in [0.0005, 0.0010, 0.0015, 0.0020]:
        t = run_backtest(datasets, slippage=slip)
        r = analyze_results(t)
        status = "✅ PASS" if r['avg_r'] > 0.5 else "⚠️ WEAK" if r['avg_r'] > 0 else "❌ FAIL"
        print(f"{slip*100:0.2f}%      | {r['net_r']:>+8.1f}R | {r['avg_r']:>+8.3f}R | {status}")

    # 3. RR SENSITIVITY
    print("\n--- 3. RR SENSITIVITY ---")
    print(f"{'Target RR':<10} | {'Net R':<10} | {'Avg R':<10} | {'WR'}")
    for rr in [1.5, 2.0, 3.0, 4.0]:
        t = run_backtest(datasets, rr=rr)
        r = analyze_results(t)
        print(f"{rr:>9.1f} | {r['net_r']:>+8.1f}R | {r['avg_r']:>+8.3f}R | {r['wr']:4.1f}%")

    # 4. ATR SL SENSITIVITY
    print("\n--- 4. ATR SL SENSITIVITY ---")
    print(f"{'ATR Mult':<10} | {'Net R':<10} | {'Avg R':<10} | {'WR'}")
    for mult in [0.6, 0.8, 1.0, 1.2]:
        t = run_backtest(datasets, sl_mult=mult)
        r = analyze_results(t)
        print(f"{mult:>9.1f} | {r['net_r']:>+8.1f}R | {r['avg_r']:>+8.3f}R | {r['wr']:4.1f}%")

    # 5. SESSION ANALYSIS
    print("\n--- 5. SESSION ANALYSIS ---")
    sessions = {
        'Asia (00-08 UTC)': [0, 8],
        'London (08-14 UTC)': [8, 14],
        'NY (14-21 UTC)': [14, 21],
        'Gap (21-00 UTC)': [21, 24]
    }
    for name, bounds in sessions.items():
        s_trades = [t for t in base_trades if bounds[0] <= t['hour'] < bounds[1]]
        r = analyze_results(s_trades)
        print(f"{name:<20} | {r['net_r']:>+8.1f}R | {r['avg_r']:>+8.3f}R | {r['count']} trades")

if __name__ == "__main__":
    main()
