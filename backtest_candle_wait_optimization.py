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
TIMEFRDAYS = 90 # Increased to 90 days for higher timeframes
# REALISTIC COSTS (Synchronized with validation)
FEE_PERCENT = 0.0006      # 0.06% (Bybit taker fee)
SLIPPAGE_PERCENT = 0.0003  # 0.03% slippage per trade
RR = 3.0
SL_MULT = 0.8

def get_top_98_symbols():
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
            time.sleep(0.02)
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

def run_strategy(df, max_wait_candles=1):
    trades = []
    cooldown = 0
    for i in range(50, len(df)-1):
        if cooldown > 0: cooldown -= 1; continue
        row = df.iloc[i]
        side = 'long' if row['reg_bull'] else 'short' if row['reg_bear'] else None
        if not side: continue
        structure_broken, candles_waited = False, 0
        for ahead in range(1, max_wait_candles + 1):
            if i+ahead >= len(df): break
            check = df.iloc[i+ahead]
            candles_waited = ahead
            if (side == 'long' and check['close'] > row['swing_high_10']) or (side == 'short' and check['close'] < row['swing_low_10']):
                structure_broken = True; break
        if not structure_broken: continue
        idx = i + candles_waited
        base = df.iloc[idx]['open']
        entry = base * (1 + SLIPPAGE_PERCENT) if side == 'long' else base * (1 - SLIPPAGE_PERCENT)
        atr = row['atr']
        if pd.isna(atr) or atr <= 0: continue
        sl_dist = atr * SL_MULT
        if sl_dist/entry > 0.05: continue
        tp_dist = sl_dist * RR
        sl, tp = (entry - sl_dist, entry + tp_dist) if side == 'long' else (entry + sl_dist, entry - tp_dist)
        outcome = 'timeout'
        for j in range(idx, min(idx+300, len(df))):
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
        trades.append({'r': res_r, 'win': outcome == 'win', 'waited': candles_waited})
        cooldown = 6
    return trades

def main():
    intervals = [15, 30, 60]
    symbols = get_top_98_symbols()
    print(f"🔬 MULTI-TIMEFRAME OPTIMIZATION ({DAYS} Days, 98 symbols)")
    
    for tf in intervals:
        print(f"\n{'-'*30}\n🕒 TIMEFRAME: {tf}m\n{'-'*30}")
        datasets = {}
        for i, sym in enumerate(symbols):
            df = fetch_data(sym, tf)
            if not df.empty:
                processed = calc_indicators(df)
                if not processed.empty: datasets[sym] = processed
            if (i+1) % 50 == 0: print(f"  Fetch Progress: {i+1}/{len(symbols)}...")
        
        print(f"📊 Results for {tf}m ({len(datasets)} coins):")
        print(f"{'Wait':<6} | {'Trades':<7} | {'Net R':<10} | {'Avg R':<10} | {'WR':<8} | {'Avg Wait'}")
        print("-" * 65)
        
        for wait in range(1, 11):
            all_trades = []
            for df in datasets.values(): all_trades.extend(run_strategy(df, wait))
            if all_trades:
                total_r = sum(t['r'] for t in all_trades)
                avg_r = total_r / len(all_trades)
                wins = sum(1 for t in all_trades if t['win'])
                wr = (wins / len(all_trades)) * 100
                avg_wait = sum(t['waited'] for t in all_trades) / len(all_trades)
                print(f"{wait:<6} | {len(all_trades):<7} | {total_r:>+9.1f} | {avg_r:>+9.3f} | {wr:>6.1f}% | {avg_wait:.1f}")

if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
