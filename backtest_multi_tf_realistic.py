#!/usr/bin/env python3
"""
MULTI-TIMEFRAME REALISTIC BACKTEST
==================================
Tests the strategy across 15M, 1H, and 4H timeframes to see if 
higher timeframes reduce the impact of realistic entry lag.
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# Settings
DAYS = 200  # More days for higher timeframes
TIMEFRAMES = ["15", "60", "240"]
RR = 3.0
SL_MULT = 1.0  # Slightly looser SL for higher TFs
FEE_PERCENT = 0.0006
SLIPPAGE_PERCENT = 0.0003
MAX_WAIT_CANDLES = 10

def get_top_symbols(n=20):
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        BAD = ['XAUTUSDT', 'PAXGUSDT', 'USTCUSDT', 'USDCUSDT', 'BUSDUSDT', 'DAIUSDT']
        return [t['symbol'] for t in usdt[:n] if t['symbol'] not in BAD][:n]
    except:
        return []

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
    if len(df) < 50: return pd.DataFrame()
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

def run_strategy_realistic(df, slippage_pct=SLIPPAGE_PERCENT):
    trades = []
    cooldown = 0
    for i in range(50, len(df)-2):
        if cooldown > 0: cooldown -= 1; continue
        row = df.iloc[i]
        side = 'long' if row['reg_bull'] else 'short' if row['reg_bear'] else None
        if not side: continue
        
        structure_broken, candles_waited = False, 0
        for ahead in range(1, MAX_WAIT_CANDLES + 1):
            if i+ahead >= len(df): break
            check = df.iloc[i+ahead]
            candles_waited = ahead
            if (side == 'long' and check['close'] > row['swing_high_10']) or \
               (side == 'short' and check['close'] < row['swing_low_10']):
                structure_broken = True; break
        
        if not structure_broken: continue
        
        idx = i + candles_waited
        entry_idx = idx + 1
        if entry_idx >= len(df): continue
        
        base = df.iloc[entry_idx]['open']
        entry = base * (1 + slippage_pct) if side == 'long' else base * (1 - slippage_pct)
        
        atr = row['atr']
        if pd.isna(atr) or atr <= 0: continue
        sl_dist = atr * SL_MULT
        tp_dist = sl_dist * RR
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        
        outcome = 'timeout'
        for j in range(entry_idx, min(entry_idx+300, len(df))):
            c = df.iloc[j]
            if side == 'long':
                if c['low'] <= sl: outcome = 'loss'; break
                if c['high'] >= tp: outcome = 'win'; break
            else:
                if c['high'] >= sl: outcome = 'loss'; break
                if c['low'] <= tp: outcome = 'win'; break
        
        risk_pct = sl_dist / entry
        fee_cost = (FEE_PERCENT + slippage_pct) / risk_pct
        res_r = RR - fee_cost if outcome == 'win' else -1.0 - fee_cost if outcome == 'loss' else -0.2
        trades.append({'r': res_r, 'win': outcome == 'win'})
        cooldown = 6
    return trades

def main():
    symbols = get_top_symbols(15)
    print(f"Testing {len(symbols)} symbols across {TIMEFRAMES} timeframes...")
    
    overall_results = []
    
    for tf in TIMEFRAMES:
        print(f"\n--- TIMEFRAME: {tf}m ---")
        tf_trades = []
        for sym in symbols:
            df = fetch_data(sym, tf)
            if len(df) < 100: continue
            df = calc_indicators(df)
            trades = run_strategy_realistic(df)
            tf_trades.extend(trades)
            print(f"  {sym}: {len(trades)} trades", flush=True)
            
        if tf_trades:
            total_r = sum(t['r'] for t in tf_trades)
            wr = sum(1 for t in tf_trades if t['win']) / len(tf_trades) * 100
            avg_r = total_r / len(tf_trades)
            print(f"\nSUMMARY {tf}m:")
            print(f"  Trades: {len(tf_trades)}")
            print(f"  Win Rate: {wr:.1f}%")
            print(f"  Total R: {total_r:+.1f}R")
            print(f"  Avg R/Trade: {avg_r:+.3f}R")
            overall_results.append({'tf': tf, 'wr': wr, 'total_r': total_r, 'avg_r': avg_r, 'n': len(tf_trades)})

    print("\n" + "="*30)
    print("FINAL COMPARISON")
    print("="*30)
    for res in overall_results:
        print(f"{res['tf']}m: WR {res['wr']:.1f}% | Avg {res['avg_r']:+.3f}R | Total {res['total_r']:+.1f}R | N={res['n']}")

if __name__ == "__main__":
    main()
