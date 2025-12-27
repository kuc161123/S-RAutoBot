#!/usr/bin/env python3
"""
REALISTIC Backtest - NO Look-Ahead Bias
========================================
The original backtest had a critical flaw: it entered at the OPEN of a candle
whose CLOSE it needed to confirm structure break. This is impossible in live trading.

This version fixes that by entering at the NEXT candle's OPEN after structure break.
"""

import pandas as pd
import numpy as np
import requests
import time
import random
from datetime import datetime

# Configuration
DAYS = 90
TIMEFRAME = 5
MAX_WAIT_CANDLES = 10
RR = 3.0
SL_MULT = 0.8
FEE_PERCENT = 0.0006
SLIPPAGE_PERCENT = 0.0003

def get_top_symbols(n=50):
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

def run_strategy_REALISTIC(df, slippage_pct=SLIPPAGE_PERCENT):
    """
    REALISTIC VERSION: Entry at NEXT candle's OPEN after structure break confirmation.
    
    The original version had look-ahead bias by entering at the OPEN of the 
    candle whose CLOSE confirmed the structure break (impossible in live).
    """
    trades = []
    cooldown = 0
    for i in range(50, len(df)-2):  # -2 because we need next candle for entry
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
                structure_broken = True
                break
        
        if not structure_broken: continue
        
        # CRITICAL FIX: Enter at NEXT candle's OPEN (realistic)
        # The original entered at df.iloc[idx]['open'] which is look-ahead bias
        confirm_idx = i + candles_waited  # Candle that confirmed structure break
        entry_idx = confirm_idx + 1  # NEXT candle for realistic entry
        
        if entry_idx >= len(df): continue
        
        # Entry at NEXT candle's OPEN (no look-ahead bias)
        base = df.iloc[entry_idx]['open']
        entry = base * (1 + slippage_pct) if side == 'long' else base * (1 - slippage_pct)
        
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
        fee_cost = (FEE_PERCENT + slippage_pct) / risk_pct
        res_r = RR - fee_cost if outcome == 'win' else -1.0 - fee_cost if outcome == 'loss' else -0.2
        trades.append({'r': res_r, 'win': outcome == 'win', 'symbol': df.iloc[i].get('symbol', 'UNK'), 'idx': i, 'side': side})
        cooldown = 6
    return trades

def run_strategy_ORIGINAL(df, slippage_pct=SLIPPAGE_PERCENT):
    """
    ORIGINAL VERSION (with look-ahead bias) for comparison.
    Entry at OPEN of structure break candle (impossible in live).
    """
    trades = []
    cooldown = 0
    for i in range(50, len(df)-1):
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
        base = df.iloc[idx]['open']  # LOOK-AHEAD BIAS: Using OPEN of candle whose CLOSE we needed
        entry = base * (1 + slippage_pct) if side == 'long' else base * (1 - slippage_pct)
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
        fee_cost = (FEE_PERCENT + slippage_pct) / risk_pct
        res_r = RR - fee_cost if outcome == 'win' else -1.0 - fee_cost if outcome == 'loss' else -0.2
        trades.append({'r': res_r, 'win': outcome == 'win', 'symbol': df.iloc[i].get('symbol', 'UNK'), 'idx': i, 'side': side})
        cooldown = 6
    return trades

def main():
    print("=" * 60)
    print("REALISTIC vs ORIGINAL BACKTEST COMPARISON")
    print("=" * 60)
    print(f"Config: {DAYS} days | {TIMEFRAME}m TF | {MAX_WAIT_CANDLES} max wait | {RR}:1 R:R | {SL_MULT}x ATR SL")
    print()
    
    symbols = get_top_symbols(50)
    print(f"Testing {len(symbols)} symbols...")
    
    all_trades_realistic = []
    all_trades_original = []
    
    for i, sym in enumerate(symbols):
        print(f"  [{i+1}/{len(symbols)}] {sym}...", end=" ", flush=True)
        df = fetch_data(sym)
        if len(df) < 100:
            print("skip (not enough data)")
            continue
        df = calc_indicators(df)
        if len(df) < 100:
            print("skip (indicators failed)")
            continue
        
        trades_r = run_strategy_REALISTIC(df)
        trades_o = run_strategy_ORIGINAL(df)
        
        for t in trades_r: t['symbol'] = sym
        for t in trades_o: t['symbol'] = sym
        
        all_trades_realistic.extend(trades_r)
        all_trades_original.extend(trades_o)
        
        print(f"R:{len(trades_r)} O:{len(trades_o)}")
    
    print()
    print("=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    
    # Original (with look-ahead bias)
    if all_trades_original:
        total_r_o = sum(t['r'] for t in all_trades_original)
        wins_o = sum(1 for t in all_trades_original if t['win'])
        wr_o = wins_o / len(all_trades_original) * 100
        avg_r_o = total_r_o / len(all_trades_original)
        print(f"\n📊 ORIGINAL (Look-Ahead Bias - UNREALISTIC):")
        print(f"   Trades: {len(all_trades_original)}")
        print(f"   Win Rate: {wr_o:.1f}%")
        print(f"   Total R: {total_r_o:+.1f}R")
        print(f"   Avg R/Trade: {avg_r_o:+.3f}R")
    
    # Realistic (no look-ahead bias)
    if all_trades_realistic:
        total_r_r = sum(t['r'] for t in all_trades_realistic)
        wins_r = sum(1 for t in all_trades_realistic if t['win'])
        wr_r = wins_r / len(all_trades_realistic) * 100
        avg_r_r = total_r_r / len(all_trades_realistic)
        print(f"\n📊 REALISTIC (No Look-Ahead - LIVE EXPECTATION):")
        print(f"   Trades: {len(all_trades_realistic)}")
        print(f"   Win Rate: {wr_r:.1f}%")
        print(f"   Total R: {total_r_r:+.1f}R")
        print(f"   Avg R/Trade: {avg_r_r:+.3f}R")
    
    # Side breakdown for realistic
    if all_trades_realistic:
        longs = [t for t in all_trades_realistic if t['side'] == 'long']
        shorts = [t for t in all_trades_realistic if t['side'] == 'short']
        
        print(f"\n📈 REALISTIC - LONG trades:")
        if longs:
            wr_l = sum(1 for t in longs if t['win']) / len(longs) * 100
            r_l = sum(t['r'] for t in longs)
            print(f"   Trades: {len(longs)} | WR: {wr_l:.1f}% | R: {r_l:+.1f}R")
        
        print(f"\n📉 REALISTIC - SHORT trades:")
        if shorts:
            wr_s = sum(1 for t in shorts if t['win']) / len(shorts) * 100
            r_s = sum(t['r'] for t in shorts)
            print(f"   Trades: {len(shorts)} | WR: {wr_s:.1f}% | R: {r_s:+.1f}R")
    
    print()
    print("=" * 60)
    if all_trades_realistic and all_trades_original:
        diff = wr_o - wr_r
        print(f"⚠️  LOOK-AHEAD BIAS IMPACT: {diff:+.1f}% WR difference")
        print(f"⚠️  The original backtest was {diff:.1f}% too optimistic!")
    print("=" * 60)

if __name__ == "__main__":
    main()
