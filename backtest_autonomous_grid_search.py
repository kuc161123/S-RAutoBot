#!/usr/bin/env python3
"""
AUTONOMOUS CONFIGURATION GRID SEARCH
====================================
Systematically tests hundreds of strategy configurations until finding profitable ones.

Will test:
- Timeframes: 5M, 15M, 30M
- RSI Periods: 7, 14, 21
- RSI Thresholds: 25, 30, 35, 40
- Lookbacks: 7, 10, 14, 20
- SL Multipliers: 0.5, 0.8, 1.0, 1.5, 2.0
- RR Ratios: 1.5, 2.0, 2.5, 3.0, 4.0
- Pullback modes: With/Without
- Max wait for BOS: 5, 10, 15 candles
"""

import pandas as pd
import numpy as np
import requests
import time
import itertools

# Fixed symbol set to prevent overfitting
DAYS = 200
SYMBOL_COUNT = 20
FEE = 0.0006
SLIPPAGE = 0.0003

# Search Grid
TIMEFRAMES = ['5', '15', '30']
RSI_PERIODS = [7, 14]
RSI_THRESHOLDS = [25, 30, 35]
LOOKBACKS = [7, 10, 14]
SL_MULTS = [0.8, 1.0, 1.5]
RR_RATIOS = [2.0, 2.5, 3.0]
PULLBACK_MODES = [True, False]
MAX_WAITS = [10, 15]

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

def calc_indicators(df, rsi_period, lookback):
    if len(df) < 100: return pd.DataFrame()
    df = df.copy()
    close, high, low = df['close'], df['high'], df['low']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    df['rsi'] = 100 - (100 / (1 + gain/(loss+1e-9)))
    df['atr'] = (high - low).rolling(20).mean()
    df['swing_low'] = low.rolling(lookback).min()
    df['swing_high'] = high.rolling(lookback).max()
    df['rsi_low'] = df['rsi'].rolling(lookback).min()
    df['rsi_high'] = df['rsi'].rolling(lookback).max()
    return df

def run_strategy(df, rsi_thresh, sl_mult, rr, use_pullback, max_wait):
    trades = []
    for i in range(50, len(df) - 100):
        row = df.iloc[i]
        
        # Detect Divergence
        bullish_div = (row['low'] <= row['swing_low']) and (row['rsi'] > row['rsi_low']) and (row['rsi'] < rsi_thresh)
        bearish_div = (row['high'] >= row['swing_high']) and (row['rsi'] < row['rsi_high']) and (row['rsi'] > (100-rsi_thresh))
        
        if not (bullish_div or bearish_div): continue
        side = 'long' if bullish_div else 'short'
        
        # Wait for BOS
        bos_found, bos_level, bos_idx = False, None, None
        for j in range(i+1, min(i+1+max_wait, len(df)-50)):
            c = df.iloc[j]
            if side == 'long' and c['high'] > row['swing_high']:
                bos_found, bos_level, bos_idx = True, row['swing_high'], j
                break
            elif side == 'short' and c['low'] < row['swing_low']:
                bos_found, bos_level, bos_idx = True, row['swing_low'], j
                break
        if not bos_found: continue
        
        if use_pullback:
            # Wait for pullback
            pullback_found, pullback_idx = False, None
            for k in range(bos_idx+1, min(bos_idx+1+20, len(df)-10)):
                c = df.iloc[k]
                if side == 'long' and c['low'] <= bos_level * 1.002:
                    pullback_found, pullback_idx = True, k
                    break
                elif side == 'short' and c['high'] >= bos_level * 0.998:
                    pullback_found, pullback_idx = True, k
                    break
            if not pullback_found: continue
            
            # Bounce confirmation
            if pullback_idx + 1 >= len(df): continue
            confirmation_candle = df.iloc[pullback_idx + 1]
            bounce_confirmed = False
            if side == 'long' and confirmation_candle['close'] > df.iloc[pullback_idx]['close']:
                bounce_confirmed = True
            elif side == 'short' and confirmation_candle['close'] < df.iloc[pullback_idx]['close']:
                bounce_confirmed = True
            if not bounce_confirmed: continue
            entry = confirmation_candle['close'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
            entry_idx = pullback_idx + 1
        else:
            # Enter immediately on BOS candle close
            entry = df.iloc[bos_idx]['close'] * (1 + (SLIPPAGE if side == 'long' else -SLIPPAGE))
            entry_idx = bos_idx
        
        atr = row['atr']
        if pd.isna(atr) or atr <= 0: continue
        
        sl_dist = atr * sl_mult
        tp_dist = sl_dist * rr
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        
        outcome = None
        for m in range(entry_idx, min(entry_idx+150, len(df))):
            c = df.iloc[m]
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
    print("AUTONOMOUS CONFIGURATION GRID SEARCH")
    print("="*80)
    print(f"Symbols: {len(symbols)} (FIXED)")
    print("Searching for profitable configurations...")
    print()
    
    # Pre-fetch data for all timeframes
    data = {}
    for tf in TIMEFRAMES:
        print(f"Fetching {tf}m data...")
        data[tf] = {}
        for sym in symbols:
            df = fetch_data(sym, tf)
            if len(df) > 100:
                data[tf][sym] = df
        print(f"  Got {len(data[tf])} symbols")
    
    # Generate all combinations
    combos = list(itertools.product(
        TIMEFRAMES, RSI_PERIODS, RSI_THRESHOLDS, LOOKBACKS, SL_MULTS, RR_RATIOS, PULLBACK_MODES, MAX_WAITS
    ))
    
    print(f"\nTesting {len(combos)} combinations...")
    print("Will only report PROFITABLE configurations.\n")
    
    winners = []
    tested = 0
    
    for tf, rsi_p, rsi_t, lb, sl_m, rr, pb, mw in combos:
        tested += 1
        
        all_trades = []
        for sym, df_raw in data[tf].items():
            df = calc_indicators(df_raw.copy(), rsi_p, lb)
            trades = run_strategy(df, rsi_t, sl_m, rr, pb, mw)
            all_trades.extend(trades)
        
        if len(all_trades) < 50: continue
        
        total_r = sum(t['r'] for t in all_trades)
        avg_r = total_r / len(all_trades)
        wr = sum(1 for t in all_trades if t['win']) / len(all_trades) * 100
        
        # Filter: Must be profitable
        if avg_r <= 0: continue
        
        winners.append({
            'tf': tf, 'rsi_p': rsi_p, 'rsi_t': rsi_t, 'lb': lb,
            'sl': sl_m, 'rr': rr, 'pb': pb, 'mw': mw,
            'wr': wr, 'avg_r': avg_r, 'total_r': total_r, 'n': len(all_trades)
        })
        
        print(f"🏆 WINNER #{len(winners)}: TF={tf}m RSI({rsi_p},{rsi_t}) LB={lb} SL={sl_m} RR={rr} PB={pb} | WR={wr:.1f}% Avg={avg_r:+.3f}R N={len(all_trades)}")
        
        if tested % 100 == 0:
            print(f"  Progress: {tested}/{len(combos)} tested, {len(winners)} winners")
    
    print("\n" + "="*80)
    print("SEARCH COMPLETE")
    print("="*80)
    
    if winners:
        winners.sort(key=lambda x: x['avg_r'], reverse=True)
        print(f"FOUND {len(winners)} PROFITABLE CONFIGURATIONS!\n")
        print(f"{'TF':<4} | {'RSI':<8} | {'LB':<3} | {'SL':<4} | {'RR':<4} | {'PB':<5} | {'WR':<6} | {'Avg R':<8} | N")
        print("-"*80)
        for w in winners[:30]:
            pb_str = "YES" if w['pb'] else "NO"
            print(f"{w['tf']:<4} | {w['rsi_p']}/{w['rsi_t']:<8} | {w['lb']:<3} | {w['sl']:<4.1f} | {w['rr']:<4.1f} | {pb_str:<5} | {w['wr']:4.1f}% | {w['avg_r']:+6.3f}R | {w['n']}")
        
        print("\n🏆 BEST CONFIGURATION:")
        best = winners[0]
        print(f"  Timeframe: {best['tf']}m")
        print(f"  RSI Period: {best['rsi_p']}")
        print(f"  RSI Threshold: {best['rsi_t']}")
        print(f"  Lookback: {best['lb']}")
        print(f"  SL: {best['sl']}x ATR")
        print(f"  RR: {best['rr']}:1")
        print(f"  Use Pullback: {best['pb']}")
        print(f"  Win Rate: {best['wr']:.1f}%")
        print(f"  Avg R: {best['avg_r']:+.3f}R")
        print(f"  Total R: {best['total_r']:+.1f}R")
    else:
        print("❌ NO PROFITABLE CONFIGURATIONS FOUND")
        print(f"Tested {tested} combinations, all showed negative expectancy.")
    
    print("="*80)

if __name__ == "__main__":
    main()
