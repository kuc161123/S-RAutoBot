#!/usr/bin/env python3
"""
EXACT BOT CONFIGURATION BACKTEST
================================
This backtest uses the EXACT parameters from config.yaml:
- Timeframe: 5M
- Structure Lookback: 10 candles
- Max Wait Candles: 10
- R:R Ratio: 3.0:1
- SL Multiplier: 0.8x ATR
- RSI Thresholds: 40/60
- Realistic Execution: Entry at NEXT candle's OPEN after structure break
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime

# ===== EXACT BOT CONFIGURATION =====
DAYS = 90
TIMEFRAME = 5
MAX_WAIT_CANDLES = 10
STRUCTURE_LOOKBACK = 10
RR = 3.0
SL_MULT = 0.8
RSI_OVERSOLD = 40
RSI_OVERBOUGHT = 60
COOLDOWN_BARS = 6
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
            time.sleep(0.015)
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
    # Divergence detection matching bot logic
    df['reg_bull'] = (df['low'] <= df['price_low_14']) & (df['rsi'] > df['rsi_low_14'].shift(14)) & (df['rsi'] < RSI_OVERSOLD)
    df['reg_bear'] = (df['high'] >= df['price_high_14']) & (df['rsi'] < df['rsi_high_14'].shift(14)) & (df['rsi'] > RSI_OVERBOUGHT)
    return df

def run_strategy_realistic(df, slippage_pct=SLIPPAGE_PERCENT):
    """
    REALISTIC execution matching live bot:
    1. Detect divergence signal
    2. Wait for structure break (close > swing_high for long, close < swing_low for short)
    3. Enter at NEXT candle's OPEN (realistic, no look-ahead)
    """
    trades = []
    cooldown = 0
    
    for i in range(50, len(df)-2):
        if cooldown > 0: 
            cooldown -= 1
            continue
            
        row = df.iloc[i]
        side = 'long' if row['reg_bull'] else 'short' if row['reg_bear'] else None
        if not side: continue
        
        # Wait for structure break (matching bot's check_pending_trio_triggers)
        structure_broken, candles_waited = False, 0
        for ahead in range(1, MAX_WAIT_CANDLES + 1):
            if i+ahead >= len(df): break
            check = df.iloc[i+ahead]
            candles_waited = ahead
            if (side == 'long' and check['close'] > row['swing_high']) or \
               (side == 'short' and check['close'] < row['swing_low']):
                structure_broken = True
                break
        
        if not structure_broken: continue
        
        # REALISTIC: Enter at NEXT candle's OPEN (matching bot's execution)
        confirm_idx = i + candles_waited
        entry_idx = confirm_idx + 1
        
        if entry_idx >= len(df): continue
        
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
        trades.append({'r': res_r, 'win': outcome == 'win', 'side': side, 'symbol': df.iloc[i].get('symbol', 'UNK')})
        cooldown = COOLDOWN_BARS
    
    return trades

def main():
    print("=" * 60)
    print("EXACT BOT CONFIGURATION BACKTEST")
    print("=" * 60)
    print(f"Parameters matching config.yaml:")
    print(f"  Timeframe: {TIMEFRAME}m")
    print(f"  Structure Lookback: {STRUCTURE_LOOKBACK} candles")
    print(f"  Max Wait Candles: {MAX_WAIT_CANDLES}")
    print(f"  R:R Ratio: {RR}:1")
    print(f"  SL Multiplier: {SL_MULT}x ATR")
    print(f"  RSI Thresholds: {RSI_OVERSOLD}/{RSI_OVERBOUGHT}")
    print(f"  Cooldown: {COOLDOWN_BARS} bars")
    print(f"  Execution: REALISTIC (Next Candle Open)")
    print()
    
    symbols = get_top_symbols(50)
    print(f"Testing {len(symbols)} symbols over {DAYS} days...")
    
    all_trades = []
    for i, sym in enumerate(symbols):
        print(f"  [{i+1}/{len(symbols)}] {sym}...", end=" ", flush=True)
        df = fetch_data(sym)
        if len(df) < 100:
            print("skip")
            continue
        df = calc_indicators(df)
        if len(df) < 100:
            print("skip")
            continue
        
        trades = run_strategy_realistic(df)
        for t in trades: t['symbol'] = sym
        all_trades.extend(trades)
        print(f"{len(trades)} trades")
    
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    if not all_trades:
        print("No trades found!")
        return
    
    total_r = sum(t['r'] for t in all_trades)
    wins = sum(1 for t in all_trades if t['win'])
    wr = wins / len(all_trades) * 100
    avg_r = total_r / len(all_trades)
    
    print(f"Total Trades: {len(all_trades)}")
    print(f"Win Rate: {wr:.1f}%")
    print(f"Total R: {total_r:+.1f}R")
    print(f"Avg R/Trade: {avg_r:+.3f}R")
    
    # Side breakdown
    longs = [t for t in all_trades if t['side'] == 'long']
    shorts = [t for t in all_trades if t['side'] == 'short']
    
    print()
    if longs:
        wr_l = sum(1 for t in longs if t['win']) / len(longs) * 100
        r_l = sum(t['r'] for t in longs)
        print(f"📈 LONG: {len(longs)} trades | WR={wr_l:.1f}% | R={r_l:+.1f}R")
    
    if shorts:
        wr_s = sum(1 for t in shorts if t['win']) / len(shorts) * 100
        r_s = sum(t['r'] for t in shorts)
        print(f"📉 SHORT: {len(shorts)} trades | WR={wr_s:.1f}% | R={r_s:+.1f}R")
    
    print()
    if avg_r > 0:
        print("✅ PROFITABLE STRATEGY")
    else:
        print("❌ LOSING STRATEGY")
    print("=" * 60)

if __name__ == "__main__":
    main()
