#!/usr/bin/env python3
"""
1-CANDLE WAIT + RR GRID SEARCH
==============================
Testing current strategy with IMMEDIATE entry (1 candle wait) and different R:R ratios.

Strategy: RSI Divergence + IMMEDIATE structure break entry
- Wait: 1 candle only (reduce entry lag)
- R:R ratios tested: 1.5, 2.0, 2.5, 3.0, 4.0, 5.0
- Timeframes: 5M, 15M, 30M
- SL: 0.8x ATR (current config)
"""

import pandas as pd
import numpy as np
import requests
import time
import concurrent.futures
import itertools

# Configuration
DAYS = 200
SYMBOL_COUNT = 30
RSI_PERIOD = 14
LOOKBACK = 10
MAX_WAIT = 1  # IMMEDIATE ENTRY
SL_MULT = 0.8
RR_RATIOS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
TIMEFRAMES = ['5', '15', '30']
FEE = 0.0006
SLIPPAGE = 0.0003

def get_symbols():
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        return [t['symbol'] for t in usdt[:SYMBOL_COUNT]]
    except: return []

def fetch_data(symbol, interval):
    try:
        url = "https://api.bybit.com/v5/market/kline"
        all_kline = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - DAYS * 24 * 3600) * 1000)
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': end_ts}
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
    if len(df) < 100: return pd.DataFrame()
    df = df.copy()
    close, high, low = df['close'], df['high'], df['low']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-9)))
    df['atr'] = (high - low).rolling(20).mean()
    df['swing_low'] = low.rolling(LOOKBACK).min()
    df['swing_high'] = high.rolling(LOOKBACK).max()
    df['rsi_low'] = df['rsi'].rolling(LOOKBACK).min()
    df['rsi_high'] = df['rsi'].rolling(LOOKBACK).max()
    return df

def run_strategy(df, rr):
    trades = []
    
    for i in range(50, len(df) - 50):
        row = df.iloc[i]
        
        # RSI Divergence
        bullish_div = (row['low'] <= row['swing_low']) and (row['rsi'] > row['rsi_low']) and (row['rsi'] < 35)
        bearish_div = (row['high'] >= row['swing_high']) and (row['rsi'] < row['rsi_high']) and (row['rsi'] > 65)
        
        if not (bullish_div or bearish_div):
            continue
        
        side = 'long' if bullish_div else 'short'
        
        # Check NEXT candle for structure break (1 candle wait only)
        if i + 1 >= len(df) - 10:
            continue
            
        next_candle = df.iloc[i + 1]
        
        # Structure break must happen on very next candle
        bos_found = False
        if side == 'long' and next_candle['close'] > row['swing_high']:
            bos_found = True
        elif side == 'short' and next_candle['close'] < row['swing_low']:
            bos_found = True
        
        if not bos_found:
            continue
        
        # Entry at candle AFTER the structure break (i+2)
        if i + 2 >= len(df):
            continue
            
        entry = df.iloc[i + 2]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        
        atr = row['atr']
        if pd.isna(atr) or atr <= 0:
            continue
        
        sl_dist = atr * SL_MULT
        tp_dist = sl_dist * rr
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        
        # Execute
        outcome = None
        for k in range(i+2, min(i+2+150, len(df))):
            c = df.iloc[k]
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
    symbols = get_symbols()
    print("="*80)
    print("1-CANDLE WAIT + RR GRID SEARCH")
    print("="*80)
    print(f"Testing {len(symbols)} symbols over {DAYS} days")
    print(f"Strategy: RSI Divergence + IMMEDIATE entry (1 candle wait)")
    print(f"Testing {len(RR_RATIOS)} R:R ratios across {len(TIMEFRAMES)} timeframes")
    print()
    
    # Pre-fetch data for all timeframes
    data = {}
    for tf in TIMEFRAMES:
        print(f"Fetching {tf}m data...")
        data[tf] = {}
        for sym in symbols:
            df = fetch_data(sym, tf)
            if len(df) > 100:
                df = calc_indicators(df)
                data[tf][sym] = df
        print(f"  Got {len(data[tf])} symbols")
    
    # Test all combinations
    results = []
    
    for tf in TIMEFRAMES:
        for rr in RR_RATIOS:
            all_trades = []
            for sym, df in data[tf].items():
                trades = run_strategy(df, rr)
                all_trades.extend(trades)
            
            if all_trades:
                wr = sum(1 for t in all_trades if t['win']) / len(all_trades) * 100
                total_r = sum(t['r'] for t in all_trades)
                avg_r = total_r / len(all_trades)
                
                results.append({
                    'tf': tf,
                    'rr': rr,
                    'wr': wr,
                    'avg_r': avg_r,
                    'total_r': total_r,
                    'n': len(all_trades)
                })
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS: 1-CANDLE WAIT STRATEGY")
    print("="*80)
    print(f"{'TF':<4} | {'RR':<4} | {'WR':<8} | {'Avg R':<10} | {'Total R':<10} | {'N':<6}")
    print("-"*80)
    
    winners = []
    for r in results:
        status = "✅" if r['avg_r'] > 0 else "  "
        print(f"{r['tf']:<4} | {r['rr']:<4.1f} | {r['wr']:6.1f}% | {r['avg_r']:+8.3f}R | {r['total_r']:+8.1f}R | {r['n']:<6}")
        if r['avg_r'] > 0:
            winners.append(r)
    
    print("="*80)
    
    if winners:
        print(f"\n🏆 FOUND {len(winners)} PROFITABLE CONFIGURATIONS!")
        winners.sort(key=lambda x: x['avg_r'], reverse=True)
        best = winners[0]
        print(f"\nBEST CONFIG:")
        print(f"  Timeframe: {best['tf']}m")
        print(f"  R:R Ratio: {best['rr']}:1")
        print(f"  Win Rate:  {best['wr']:.1f}%")
        print(f"  Avg R:     {best['avg_r']:+.3f}R")
        print(f"  Total R:   {best['total_r']:+.1f}R")
        print(f"  Trades:    {best['n']}")
    else:
        print("\n❌ NO PROFITABLE CONFIGURATIONS FOUND")
        print("1-candle wait did not improve profitability")
    
    print("="*80)

if __name__ == "__main__":
    main()
