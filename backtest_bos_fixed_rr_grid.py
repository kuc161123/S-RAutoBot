#!/usr/bin/env python3
"""
BOS DIVERGENCE BACKTEST - FIXED (NO LOOKAHEAD)
===============================================
Based on original backtest_all_divergences_bos.py but with:
1. NO look-ahead bias - entry at candle AFTER BOS confirmation
2. Wait up to 10 candles for BOS (current bot config)
3. Test multiple R:R ratios (1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
4. 1H timeframe (original backtest timeframe)
"""

import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# Config
TIMEFRAME = '60'  # 1H (original)
DATA_DAYS = 90
NUM_SYMBOLS = 100
MAX_WAIT_CANDLES = 10  # Wait up to 10 candles for BOS
RR_RATIOS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Costs
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

# RSI Settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5
SL_ATR_MULT = 0.8  # Current bot config

BASE_URL = "https://api.bybit.com"

def get_symbols(limit):
    resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear", timeout=10)
    tickers = resp.json().get('result', {}).get('list', [])
    usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in usdt[:limit]]

def fetch_klines(symbol, interval, days):
    end_ts = int(datetime.now().timestamp() * 1000)
    candles_needed = days * 24 * 60 // int(interval)
    all_candles = []
    current_end = end_ts
    
    while len(all_candles) < candles_needed:
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 
                  'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
            data = resp.json().get('result', {}).get('list', [])
            if not data: break
            all_candles.extend(data)
            current_end = int(data[-1][0]) - 1
            if len(data) < 1000: break
        except: break
    
    if not all_candles: return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    return df

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def find_pivots(data, left=3, right=3):
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    for i in range(left, n - right):
        is_high = all(data[j] < data[i] for j in range(i - left, i + right + 1) if j != i)
        is_low = all(data[j] > data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_high: pivot_highs[i] = data[i]
        if is_low: pivot_lows[i] = data[i]
    return pivot_highs, pivot_lows

def detect_divergences(df):
    if len(df) < 100: return []
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    for i in range(30, n - 15):
        # Find pivot lows for bullish divergence
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
        # Find pivot highs for bearish divergence
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        # Regular Bullish: Lower price low, higher RSI low
        if curr_pl and prev_pl:
            if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                if rsi[i] < RSI_OVERSOLD + 15:
                    swing_high = max(high[max(0, i-LOOKBACK_BARS):i+1])
                    signals.append({'idx': i, 'type': 'regular_bullish', 'side': 'long', 'swing_high': swing_high})
                    continue
        
        # Regular Bearish: Higher price high, lower RSI high
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                if rsi[i] > RSI_OVERBOUGHT - 15:
                    swing_low = min(low[max(0, i-LOOKBACK_BARS):i+1])
                    signals.append({'idx': i, 'type': 'regular_bearish', 'side': 'short', 'swing_low': swing_low})
                    continue
    
    return signals

def run_backtest(df, signals, rr):
    """Run backtest with NO look-ahead bias."""
    rows = list(df.itertuples())
    atr = df['atr'].values
    trades = []
    
    for sig in signals:
        div_idx = sig['idx']
        side = sig['side']
        
        # Wait for BOS (up to MAX_WAIT_CANDLES)
        bos_idx = None
        for j in range(div_idx + 1, min(div_idx + 1 + MAX_WAIT_CANDLES, len(rows) - 10)):
            if side == 'long':
                # BOS: Close breaks above swing high
                if rows[j].close > sig['swing_high']:
                    bos_idx = j
                    break
            else:
                # BOS: Close breaks below swing low
                if rows[j].close < sig['swing_low']:
                    bos_idx = j
                    break
        
        if bos_idx is None:
            continue
        
        # Entry at NEXT candle open (no look-ahead)
        entry_idx = bos_idx + 1
        if entry_idx >= len(rows) - 50:
            continue
        
        entry_row = rows[entry_idx]
        entry_price = entry_row.open
        
        # Apply slippage
        if side == 'long':
            entry_price *= (1 + SLIPPAGE_PCT)
        else:
            entry_price *= (1 - SLIPPAGE_PCT)
        
        # SL/TP
        entry_atr = atr[entry_idx - 1] if entry_idx > 0 else atr[entry_idx]
        sl_dist = entry_atr * SL_ATR_MULT
        
        if side == 'long':
            sl = entry_price - sl_dist
            tp = entry_price + (sl_dist * rr)
        else:
            sl = entry_price + sl_dist
            tp = entry_price - (sl_dist * rr)
        
        # Simulate trade
        result = None
        for k in range(entry_idx, min(entry_idx + 200, len(rows))):
            row = rows[k]
            if side == 'long':
                if row.low <= sl:
                    result = 'loss'
                    break
                if row.high >= tp:
                    result = 'win'
                    break
            else:
                if row.high >= sl:
                    result = 'loss'
                    break
                if row.low <= tp:
                    result = 'win'
                    break
        
        if result:
            # Calculate R with fees
            risk_pct = sl_dist / entry_price
            fee_cost = (FEE_PCT * 2 + SLIPPAGE_PCT) / risk_pct
            
            if result == 'win':
                r = rr - fee_cost
            else:
                r = -1.0 - fee_cost
            
            trades.append({
                'side': side,
                'type': sig['type'],
                'result': result,
                'r': r
            })
    
    return trades

def main():
    print("="*70)
    print("BOS DIVERGENCE BACKTEST - FIXED (NO LOOKAHEAD)")
    print("="*70)
    print(f"Timeframe: {TIMEFRAME}m (1H)")
    print(f"Data: {DATA_DAYS} days, {NUM_SYMBOLS} symbols")
    print(f"BOS Wait: Up to {MAX_WAIT_CANDLES} candles")
    print(f"SL: {SL_ATR_MULT}x ATR")
    print(f"Entry: Next candle OPEN after BOS (realistic)")
    print(f"Testing R:R ratios: {RR_RATIOS}")
    print("="*70)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\nFetching data for {len(symbols)} symbols...")
    
    # Fetch all data first
    symbol_data = {}
    for idx, sym in enumerate(symbols):
        df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
        if df.empty or len(df) < 200:
            continue
        
        df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
        hl = df['high'] - df['low']
        hc = abs(df['high'] - df['close'].shift())
        lc = abs(df['low'] - df['close'].shift())
        df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        df = df.dropna()
        
        if len(df) >= 100:
            signals = detect_divergences(df)
            if signals:
                symbol_data[sym] = (df, signals)
        
        if (idx + 1) % 20 == 0:
            print(f"  Fetched {idx+1}/{len(symbols)} symbols...")
    
    print(f"\nSymbols with signals: {len(symbol_data)}")
    
    # Test each R:R
    print("\n" + "="*70)
    print("RESULTS BY R:R RATIO")
    print("="*70)
    print(f"{'RR':<6} | {'Trades':<7} | {'WR':<8} | {'Avg R':<10} | {'Total R':<10} | {'Status':<8}")
    print("-"*70)
    
    best_rr = None
    best_avg_r = -999
    
    for rr in RR_RATIOS:
        all_trades = []
        
        for sym, (df, signals) in symbol_data.items():
            trades = run_backtest(df, signals, rr)
            all_trades.extend(trades)
        
        if all_trades:
            wins = sum(1 for t in all_trades if t['result'] == 'win')
            wr = wins / len(all_trades) * 100
            total_r = sum(t['r'] for t in all_trades)
            avg_r = total_r / len(all_trades)
            
            status = "✅ PROFIT" if avg_r > 0 else "❌ LOSS"
            
            print(f"{rr:<6.1f} | {len(all_trades):<7} | {wr:>6.1f}% | {avg_r:>+8.4f}R | {total_r:>+9.1f}R | {status}")
            
            if avg_r > best_avg_r:
                best_avg_r = avg_r
                best_rr = rr
        else:
            print(f"{rr:<6.1f} | {'--':<7} | {'--':<8} | {'--':<10} | {'--':<10} | NO TRADES")
    
    print("="*70)
    
    if best_rr and best_avg_r > 0:
        print(f"\n🏆 BEST CONFIG: R:R = {best_rr}:1 with {best_avg_r:+.4f}R avg per trade")
    else:
        print(f"\n❌ NO PROFITABLE R:R FOUND (Best was {best_rr}:1 with {best_avg_r:+.4f}R)")
    
    # Breakdown by divergence type for best RR
    if best_rr:
        print(f"\n--- Breakdown for R:R {best_rr}:1 ---")
        type_stats = defaultdict(lambda: {'w': 0, 'l': 0, 'r': 0})
        
        for sym, (df, signals) in symbol_data.items():
            trades = run_backtest(df, signals, best_rr)
            for t in trades:
                dtype = t['type']
                if t['result'] == 'win':
                    type_stats[dtype]['w'] += 1
                else:
                    type_stats[dtype]['l'] += 1
                type_stats[dtype]['r'] += t['r']
        
        for dtype in ['regular_bullish', 'regular_bearish']:
            s = type_stats[dtype]
            n = s['w'] + s['l']
            if n > 0:
                wr = s['w'] / n * 100
                avg_r = s['r'] / n
                print(f"  {dtype}: {n} trades, {wr:.1f}% WR, {avg_r:+.4f}R avg, {s['r']:+.1f}R total")

if __name__ == "__main__":
    main()
