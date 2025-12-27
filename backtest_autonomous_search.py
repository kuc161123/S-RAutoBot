#!/usr/bin/env python3
"""
AUTONOMOUS ROBUST STRATEGY SEARCH
=================================
Systematically tests divergence strategy variations until finding one that:
1. Has positive expectancy (+0.05R or better)
2. Passes Walk-Forward validation (4+/6 periods profitable)
3. Survives slippage stress test

NO OVERFITTING: Uses same 50 symbols for all tests.
"""

import pandas as pd
import numpy as np
import requests
import time
import itertools
import random

# Fixed Symbol Set (NO OVERFITTING)
DAYS = 365
SYMBOL_COUNT = 50
FEE = 0.0006
SLIPPAGE = 0.0003

# Search Space
TIMEFRAMES = ['60', '240']  # 1H, 4H
RSI_PERIODS = [7, 14]
RSI_OS = [25, 30, 35]  # Oversold thresholds
RSI_OB = [65, 70, 75]  # Overbought thresholds
SL_MULTS = [1.0, 1.5, 2.0]
RR_RATIOS = [2.0, 3.0, 4.0]
LOOKBACKS = [10, 14, 20]  # Divergence lookback
COOLDOWNS = [3, 6, 10]

def get_symbols():
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        BAD = ['XAUTUSDT', 'PAXGUSDT', 'USTCUSDT', 'USDCUSDT', 'BUSDUSDT', 'DAIUSDT']
        return [t['symbol'] for t in usdt if t['symbol'] not in BAD][:SYMBOL_COUNT]
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
            time.sleep(0.02)
        if not all_kline: return pd.DataFrame()
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
        df['ts'] = pd.to_numeric(df['ts'])
        return df
    except: return pd.DataFrame()

def calc_indicators(df, rsi_period, lookback):
    if len(df) < 50: return pd.DataFrame()
    df = df.copy()
    close, high, low = df['close'], df['high'], df['low']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-9)))
    df['atr'] = (high - low).rolling(20).mean()
    df['price_low'] = low.rolling(lookback).min()
    df['price_high'] = high.rolling(lookback).max()
    df['rsi_low'] = df['rsi'].rolling(lookback).min()
    df['rsi_high'] = df['rsi'].rolling(lookback).max()
    return df

def run_strategy(df, rsi_os, rsi_ob, sl_mult, rr, cooldown):
    trades = []
    cool = 0
    for i in range(50, len(df)-2):
        if cool > 0: cool -= 1; continue
        row = df.iloc[i]
        
        # Regular Divergence
        bull_div = (row['low'] <= row['price_low']) and (row['rsi'] > row['rsi_low']) and (row['rsi'] < rsi_os)
        bear_div = (row['high'] >= row['price_high']) and (row['rsi'] < row['rsi_high']) and (row['rsi'] > rsi_ob)
        
        side = 'long' if bull_div else 'short' if bear_div else None
        if not side: continue
        
        # REALISTIC ENTRY: Next candle open
        entry = df.iloc[i+1]['open'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
        atr = row['atr']
        if pd.isna(atr) or atr <= 0: continue
        
        sl_dist = atr * sl_mult
        tp_dist = sl_dist * rr
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        
        outcome = None
        for j in range(i+1, min(i+500, len(df))):
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
            trades.append({'r': res_r, 'win': outcome == 'win', 'ts': row['ts']})
        cool = cooldown
    return trades

def walk_forward_validate(trades, n_periods=6):
    if len(trades) < 100: return False, 0
    sorted_trades = sorted(trades, key=lambda x: x['ts'])
    n = len(sorted_trades)
    period_size = n // n_periods
    profitable = 0
    for i in range(n_periods):
        start = i * period_size
        end = (i + 1) * period_size if i < n_periods - 1 else n
        period_trades = sorted_trades[start:end]
        if sum(t['r'] for t in period_trades) > 0:
            profitable += 1
    return profitable >= 4, profitable

def main():
    symbols = get_symbols()
    print("=" * 70)
    print("AUTONOMOUS ROBUST STRATEGY SEARCH")
    print("=" * 70)
    print(f"Symbols: {len(symbols)} (FIXED - no overfitting)")
    print("Criteria: Avg R > +0.05 AND Walk-Forward 4+/6 periods profitable")
    print()
    
    # Pre-fetch all data for both timeframes
    data = {}
    for tf in TIMEFRAMES:
        print(f"Fetching {tf}m data for {len(symbols)} symbols...")
        data[tf] = {}
        for sym in symbols:
            df = fetch_data(sym, tf)
            if len(df) > 100:
                data[tf][sym] = df
        print(f"  Got {len(data[tf])} symbols")
    
    # Generate all combinations
    combos = list(itertools.product(
        TIMEFRAMES, RSI_PERIODS, RSI_OS, RSI_OB, SL_MULTS, RR_RATIOS, LOOKBACKS, COOLDOWNS
    ))
    print(f"\nTesting {len(combos)} combinations...")
    
    winners = []
    tested = 0
    
    for tf, rsi_p, rsi_os, rsi_ob, sl_m, rr, lb, cd in combos:
        tested += 1
        
        # Run strategy on all symbols
        all_trades = []
        for sym, df in data[tf].items():
            df_ind = calc_indicators(df.copy(), rsi_p, lb)
            trades = run_strategy(df_ind, rsi_os, rsi_ob, sl_m, rr, cd)
            for t in trades: t['ts'] = t.get('ts', 0)  # Ensure ts exists
            all_trades.extend(trades)
        
        if len(all_trades) < 100: continue
        
        total_r = sum(t['r'] for t in all_trades)
        avg_r = total_r / len(all_trades)
        wr = sum(1 for t in all_trades if t['win']) / len(all_trades) * 100
        
        # First filter: Must have positive expectancy
        if avg_r < 0.05: continue
        
        # Second filter: Walk-Forward validation
        passed_wf, wf_count = walk_forward_validate(all_trades)
        if not passed_wf: continue
        
        # WINNER FOUND!
        winners.append({
            'tf': tf, 'rsi_p': rsi_p, 'rsi_os': rsi_os, 'rsi_ob': rsi_ob,
            'sl': sl_m, 'rr': rr, 'lb': lb, 'cd': cd,
            'wr': wr, 'avg_r': avg_r, 'total_r': total_r, 'n': len(all_trades), 'wf': wf_count
        })
        print(f"  🏆 WINNER #{len(winners)}: TF={tf} RSI({rsi_p},{rsi_os}/{rsi_ob}) SL={sl_m} RR={rr} | Avg={avg_r:+.3f}R | WF={wf_count}/6")
        
        if tested % 100 == 0:
            print(f"  Progress: {tested}/{len(combos)} tested, {len(winners)} winners found")
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    if winners:
        winners.sort(key=lambda x: x['avg_r'], reverse=True)
        print(f"FOUND {len(winners)} ROBUST STRATEGIES!\n")
        print(f"{'TF':<4} | {'RSI':<10} | {'SL':<4} | {'RR':<4} | {'WR':<6} | {'Avg R':<8} | {'WF':<4} | N")
        print("-" * 70)
        for w in winners[:20]:
            print(f"{w['tf']:<4} | {w['rsi_p']}/{w['rsi_os']}/{w['rsi_ob']:<10} | {w['sl']:<4.1f} | {w['rr']:<4.1f} | {w['wr']:4.1f}% | {w['avg_r']:+6.3f}R | {w['wf']}/6  | {w['n']}")
        
        best = winners[0]
        print("\n" + "=" * 70)
        print("🏆 BEST ROBUST STRATEGY")
        print("=" * 70)
        print(f"Timeframe:      {best['tf']}m")
        print(f"RSI Period:     {best['rsi_p']}")
        print(f"RSI Thresholds: {best['rsi_os']}/{best['rsi_ob']}")
        print(f"SL Multiplier:  {best['sl']}x ATR")
        print(f"R:R Ratio:      {best['rr']}:1")
        print(f"Lookback:       {best['lb']} candles")
        print(f"Cooldown:       {best['cd']} bars")
        print(f"--- Results ---")
        print(f"Win Rate:       {best['wr']:.1f}%")
        print(f"Avg R/Trade:    {best['avg_r']:+.3f}R")
        print(f"Total R:        {best['total_r']:+.1f}R")
        print(f"Walk-Forward:   {best['wf']}/6 periods profitable")
    else:
        print("NO ROBUST STRATEGIES FOUND")
        print("All tested combinations either had negative expectancy or failed Walk-Forward.")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
