#!/usr/bin/env python3
"""
1H REALISTIC BACKTEST WITH VOLUME CONFIRMATION
===============================================
Tests if adding a volume filter to the structure break improves performance.
Filter: break_candle_vol > avg_vol * 1.5
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# Settings
DAYS = 300
TF = "60"
RR = 3.0
SL_MULT = 1.0
FEE_PERCENT = 0.0006
SLIPPAGE_PERCENT = 0.0003
MAX_WAIT_CANDLES = 12

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
    close_df = df['close']
    delta = close_df.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    h, l, c_prev = df['high'], df['low'], close_df.shift(1)
    tr = pd.concat([h-l, (h-c_prev).abs(), (l-c_prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(20).mean()
    df['vol_ma'] = df['vol'].rolling(20).mean()
    df['swing_high_10'] = df['high'].rolling(10).max()
    df['swing_low_10'] = df['low'].rolling(10).min()
    df['price_low_14'] = df['low'].rolling(14).min()
    df['price_high_14'] = df['high'].rolling(14).max()
    df['rsi_low_14'] = df['rsi'].rolling(14).min()
    df['rsi_high_14'] = df['rsi'].rolling(14).max()
    df['reg_bull'] = (df['low'] <= df['price_low_14']) & (df['rsi'] > df['rsi_low_14'].shift(14)) & (df['rsi'] < 40)
    df['reg_bear'] = (df['high'] >= df['price_high_14']) & (df['rsi'] < df['rsi_high_14'].shift(14)) & (df['rsi'] > 60)
    return df

def run_strategy(df, vol_filter=False):
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
            
            # Structure Break Logic
            is_break = False
            if side == 'long' and check['close'] > row['swing_high_10']: is_break = True
            if side == 'short' and check['close'] < row['swing_low_10']: is_break = True
            
            if is_break:
                # Volume Filter
                if vol_filter:
                    if check['vol'] < row['vol_ma'] * 1.2: continue # Need volume surge
                
                structure_broken = True
                candles_waited = ahead
                break
        
        if not structure_broken: continue
        
        idx = i + candles_waited
        entry_idx = idx + 1
        if entry_idx >= len(df): continue
        
        base = df.iloc[entry_idx]['open']
        entry = base * (1 + SLIPPAGE_PERCENT) if side == 'long' else base * (1 - SLIPPAGE_PERCENT)
        
        atr = row['atr']
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
        fee_cost = (FEE_PERCENT + SLIPPAGE_PERCENT) / risk_pct
        res_r = RR - fee_cost if outcome == 'win' else -1.0 - fee_cost if outcome == 'loss' else -0.2
        trades.append({'r': res_r, 'win': outcome == 'win'})
        cooldown = 12 # Longer cooldown for 1H
    return trades

def main():
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'LINKUSDT', 'BCHUSDT', 'AVAXUSDT', 'DOTUSDT']
    print(f"Testing 1H Divergence + Structure Break with Volume Filter...")
    
    for vf in [False, True]:
        print(f"\n--- VOLUME FILTER: {vf} ---")
        all_trades = []
        for sym in symbols:
            df = fetch_data(sym, TF)
            if len(df) < 100: continue
            df = calc_indicators(df)
            trades = run_strategy(df, vol_filter=vf)
            all_trades.extend(trades)
            print(f"  {sym}: {len(trades)} trades", flush=True)
            
        if all_trades:
            total_r = sum(t['r'] for t in all_trades)
            wr = sum(1 for t in all_trades if t['win']) / len(all_trades) * 100
            avg_r = total_r / len(all_trades)
            print(f"SUMMARY (Filter={vf}): WR={wr:.1f}%, Avg={avg_r:+.3f}R, Total={total_r:+.1f}R (N={len(all_trades)})")

if __name__ == "__main__":
    main()
