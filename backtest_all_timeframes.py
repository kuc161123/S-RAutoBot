#!/usr/bin/env python3
"""
COMPREHENSIVE TIMEFRAME COMPARISON
==================================
Tests the exact bot strategy across ALL timeframes to find profitable ones.
Timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 4h
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# Bot Config (from config.yaml)
MAX_WAIT_CANDLES = 10
STRUCTURE_LOOKBACK = 10
RR = 3.0
SL_MULT = 0.8
RSI_OVERSOLD = 40
RSI_OVERBOUGHT = 60
COOLDOWN_BARS = 6
FEE_PERCENT = 0.0006
SLIPPAGE_PERCENT = 0.0003

# Timeframes to test (Bybit interval codes)
TIMEFRAMES = {
    '1': {'name': '1M', 'days': 30},
    '3': {'name': '3M', 'days': 60},
    '5': {'name': '5M', 'days': 90},
    '15': {'name': '15M', 'days': 120},
    '30': {'name': '30M', 'days': 180},
    '60': {'name': '1H', 'days': 365},
    '240': {'name': '4H', 'days': 365}
}

def get_symbols(n=20):
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

def fetch_data(symbol, interval, days):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - days * 24 * 3600) * 1000)
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
    df['swing_high'] = df['high'].rolling(STRUCTURE_LOOKBACK).max()
    df['swing_low'] = df['low'].rolling(STRUCTURE_LOOKBACK).min()
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    df['reg_bull'] = (df['low'] <= df['price_low_14']) & (df['rsi'] > df['rsi_low_14'].shift(14)) & (df['rsi'] < RSI_OVERSOLD)
    df['reg_bear'] = (df['high'] >= df['price_high_14']) & (df['rsi'] < df['rsi_high_14'].shift(14)) & (df['rsi'] > RSI_OVERBOUGHT)
    return df

def run_strategy(df):
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
            if (side == 'long' and check['close'] > row['swing_high']) or \
               (side == 'short' and check['close'] < row['swing_low']):
                structure_broken = True; break
        
        if not structure_broken: continue
        
        entry_idx = i + candles_waited + 1
        if entry_idx >= len(df): continue
        
        base = df.iloc[entry_idx]['open']
        entry = base * (1 + SLIPPAGE_PERCENT) if side == 'long' else base * (1 - SLIPPAGE_PERCENT)
        
        atr = row['atr']
        if pd.isna(atr) or atr <= 0: continue
        sl_dist = atr * SL_MULT
        if sl_dist/entry > 0.05: continue
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
        fee_cost = (FEE_PERCENT + SLIPPAGE_PERCENT) / risk_pct
        res_r = RR - fee_cost if outcome == 'win' else -1.0 - fee_cost if outcome == 'loss' else -0.2
        trades.append({'r': res_r, 'win': outcome == 'win'})
        cooldown = COOLDOWN_BARS
    return trades

def main():
    print("=" * 60)
    print("COMPREHENSIVE TIMEFRAME COMPARISON")
    print("=" * 60)
    print(f"Strategy: RSI Divergence + Structure Break (Realistic)")
    print(f"Config: MaxWait={MAX_WAIT_CANDLES}, RR={RR}:1, SL={SL_MULT}x ATR")
    print()
    
    symbols = get_symbols(20)
    print(f"Testing {len(symbols)} symbols across {len(TIMEFRAMES)} timeframes...")
    print()
    
    results = []
    
    for tf_code, tf_info in TIMEFRAMES.items():
        print(f"--- {tf_info['name']} ({tf_info['days']} days) ---")
        tf_trades = []
        
        for sym in symbols:
            df = fetch_data(sym, tf_code, tf_info['days'])
            if len(df) < 100: continue
            df = calc_indicators(df)
            trades = run_strategy(df)
            tf_trades.extend(trades)
        
        if tf_trades:
            total_r = sum(t['r'] for t in tf_trades)
            wr = sum(1 for t in tf_trades if t['win']) / len(tf_trades) * 100
            avg_r = total_r / len(tf_trades)
            profitable = avg_r > 0
            status = "✅ PROFITABLE" if profitable else "❌ LOSING"
            print(f"  {status} | WR={wr:.1f}% | Avg={avg_r:+.3f}R | Total={total_r:+.1f}R | N={len(tf_trades)}")
            results.append({'tf': tf_info['name'], 'wr': wr, 'avg_r': avg_r, 'total_r': total_r, 'n': len(tf_trades), 'profitable': profitable})
        else:
            print(f"  No trades")
        print()
    
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    results.sort(key=lambda x: x['avg_r'], reverse=True)
    for r in results:
        status = "✅" if r['profitable'] else "❌"
        print(f"{status} {r['tf']:>4}: WR={r['wr']:.1f}% | Avg={r['avg_r']:+.3f}R | Total={r['total_r']:+.1f}R | N={r['n']}")
    
    profitable = [r for r in results if r['profitable']]
    print()
    if profitable:
        print(f"🎉 FOUND {len(profitable)} PROFITABLE TIMEFRAME(S)!")
        best = profitable[0]
        print(f"🏆 BEST: {best['tf']} with {best['avg_r']:+.3f}R per trade")
    else:
        print("❌ NO PROFITABLE TIMEFRAMES FOUND")

if __name__ == "__main__":
    main()
