#!/usr/bin/env python3
"""
CANDLE WAIT OPTIMIZATION - 100 Symbols
=======================================

Test different max_wait_candles (1-5) on 100 symbols
to find optimal structure break timing.

Current: 1 candle (+2.071R/trade)
Testing: 1, 2, 3, 4, 5 candles
"""

import pandas as pd
import numpy as np
import requests
import time

DAYS = 60
TIMEFRAME = 5
RR = 3.0
SL_MULT = 0.8
WIN_COST = 0.0006
LOSS_COST = 0.00125

def get_top_100_symbols():
    try:
        url = "https://api.bybit.com/v5/market/tickers?category=linear"
        resp = requests.get(url, timeout=10).json()
        tickers = resp.get('result', {}).get('list', [])
        usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt_pairs.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        BAD = ['XAUTUSDT', 'PAXGUSDT', 'USTCUSDT', 'USDCUSDT', 'BUSDUSDT', 'DAIUSDT']
        return [t['symbol'] for t in usdt_pairs[:100] if t['symbol'] not in BAD][:100]
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
            time.sleep(0.04)
        if not all_kline: return pd.DataFrame()
        df = pd.DataFrame(all_kline, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'to'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']: df[c] = df[c].astype(float)
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
    df['atr'] = tr.rolling(14).mean()
    df['swing_high_10'] = df['high'].rolling(10).max()
    df['swing_low_10'] = df['low'].rolling(10).min()
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    df['reg_bull'] = (df['low'] <= df['price_low_14']) & (df['rsi'] > df['rsi_low_14'].shift(14)) & (df['rsi'] < 40)
    df['reg_bear'] = (df['high'] >= df['price_high_14']) & (df['rsi'] < df['rsi_high_14'].shift(14)) & (df['rsi'] > 60)
    return df

def run_strategy(df, max_wait_candles=1):
    """Run strategy with specified max wait candles"""
    trades = []
    cooldown = 0
    
    for i in range(50, len(df)-1):
        if cooldown > 0: cooldown -= 1; continue
        row = df.iloc[i]
        
        side = None
        if row['reg_bull']: side = 'long'
        elif row['reg_bear']: side = 'short'
        if not side: continue
        
        # Structure break check with variable wait
        structure_broken = False
        candles_waited = 0
        
        for ahead in range(1, max_wait_candles + 1):
            if i+ahead >= len(df): break
            check = df.iloc[i+ahead]
            candles_waited = ahead
            
            if side == 'long' and check['close'] > row['swing_high_10']:
                structure_broken = True
                break
            if side == 'short' and check['close'] < row['swing_low_10']:
                structure_broken = True
                break
        
        if not structure_broken: continue
        
        # Entry on the candle that broke structure
        entry = df.iloc[i+candles_waited]['open']
        atr = row['atr']
        if pd.isna(atr) or atr == 0: continue
        
        sl_dist = atr * SL_MULT
        if sl_dist/entry > 0.05: continue
        tp_dist = sl_dist * RR
        
        if side == 'long': sl, tp = entry - sl_dist, entry + tp_dist
        else: sl, tp = entry + sl_dist, entry - tp_dist
        
        outcome = 'timeout'
        for j in range(i+candles_waited, min(i+candles_waited+301, len(df))):
            c = df.iloc[j]
            if side == 'long':
                if c['low'] <= sl: outcome = 'loss'; break
                if c['high'] >= tp: outcome = 'win'; break
            else:
                if c['high'] >= sl: outcome = 'loss'; break
                if c['low'] <= tp: outcome = 'win'; break
        
        risk_pct = sl_dist / entry
        res_r = 0
        if outcome == 'win': res_r = RR - (WIN_COST / risk_pct)
        elif outcome == 'loss': res_r = -1.0 - (LOSS_COST / risk_pct)
        elif outcome == 'timeout': res_r = -0.1
        
        trades.append({'r': res_r, 'win': outcome == 'win', 'waited': candles_waited})
        cooldown = 5
    
    return trades

def main():
    print("=" * 70)
    print("🔬 CANDLE WAIT OPTIMIZATION (1-5 candles)")
    print("=" * 70)
    print("Testing max_wait_candles: 1, 2, 3, 4, 5")
    print("=" * 70)
    
    print("\n📥 Fetching top 100 symbols...")
    symbols = get_top_100_symbols()
    print(f"Found {len(symbols)} symbols")
    
    print(f"\n📥 Loading {DAYS}-day data...")
    datasets = {}
    for i, sym in enumerate(symbols):
        df = fetch_data(sym)
        if not df.empty: datasets[sym] = calc_indicators(df)
        if (i+1) % 20 == 0: print(f"Progress: {i+1}/{len(symbols)}...")
    print(f"✅ Loaded {len(datasets)} symbols\n")
    
    # Test each wait time
    results = []
    
    for wait in range(1, 6):
        print(f"Testing max_wait_candles = {wait}...")
        
        all_trades = []
        for sym, df in datasets.items():
            trades = run_strategy(df, max_wait_candles=wait)
            all_trades.extend(trades)
        
        if all_trades:
            total_r = sum(t['r'] for t in all_trades)
            avg_r = total_r / len(all_trades)
            wins = sum(1 for t in all_trades if t['win'])
            wr = wins / len(all_trades) * 100
            avg_wait = sum(t['waited'] for t in all_trades) / len(all_trades)
            
            results.append({
                'wait': wait,
                'trades': len(all_trades),
                'net_r': total_r,
                'avg_r': avg_r,
                'wr': wr,
                'avg_waited': avg_wait
            })
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Wait':<6} | {'Trades':<7} | {'Net R':<10} | {'Avg R':<10} | {'WR':<8} | {'Avg Wait':<8}")
    print("-" * 70)
    
    best = None
    best_r = -999
    
    for r in results:
        marker = ""
        if r['avg_r'] > best_r:
            best_r = r['avg_r']
            best = r
            marker = " ⭐ BEST"
        
        print(f"{r['wait']:<6} | {r['trades']:<7} | {r['net_r']:>+9.1f} | {r['avg_r']:>+9.3f} | {r['wr']:>6.1f}% | {r['avg_waited']:.1f} cdls{marker}")
    
    # Recommendation
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATION")
    print("=" * 70)
    
    if best:
        print(f"\n🏆 OPTIMAL: max_wait_candles = {best['wait']}")
        print(f"   - Trades: {best['trades']}")
        print(f"   - Net R: {best['net_r']:+.1f}R")
        print(f"   - Avg R: {best['avg_r']:+.3f}R")
        print(f"   - Win Rate: {best['wr']:.1f}%")
        
        if best['wait'] == 1:
            print("\n✅ CURRENT SETTING (1 candle) IS OPTIMAL!")
        else:
            print(f"\n⚠️ Consider changing to {best['wait']} candles for better results")

if __name__ == "__main__":
    main()
