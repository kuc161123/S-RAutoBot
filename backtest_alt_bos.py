#!/usr/bin/env python3
"""
ALTERNATIVE BOS DEFINITION TESTING
===================================
Testing different Break of Structure definitions:
1. CLOSE BOS (current): Close breaks swing
2. BODY BOS: Candle body (not wick) breaks level
3. DOUBLE CLOSE: Two consecutive closes beyond level
4. STRONG BOS: Break by at least 0.3 ATR
5. ENGULFING BOS: BOS candle engulfs previous candle
6. HIGH/LOW BOS: High (long) or Low (short) breaks level
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Config
TIMEFRAME = '5'
DATA_DAYS = 60
NUM_SYMBOLS = 40
MAX_WAIT_CANDLES = 10
SL_ATR_MULT = 0.8

# Test R:R
RR_RATIOS = [1.0, 1.5, 2.0]

# Costs
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

RSI_PERIOD = 14
LOOKBACK_BARS = 10
MIN_PIVOT_DISTANCE = 3

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

def add_indicators(df):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    # Body and direction
    df['body_top'] = df[['open', 'close']].max(axis=1)
    df['body_bottom'] = df[['open', 'close']].min(axis=1)
    df['bullish'] = df['close'] > df['open']
    df['bearish'] = df['close'] < df['open']
    df['body_size'] = abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    
    return df.dropna()

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
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        if curr_pl and prev_pl:
            if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                if rsi[i] < 40:
                    swing_high = max(high[max(0, i-LOOKBACK_BARS):i+1])
                    signals.append({'idx': i, 'side': 'long', 'swing': swing_high})
                    continue
        
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                if rsi[i] > 60:
                    swing_low = min(low[max(0, i-LOOKBACK_BARS):i+1])
                    signals.append({'idx': i, 'side': 'short', 'swing': swing_low})
    
    return signals

def check_bos(rows, j, side, swing, atr, bos_type, prev_close=None):
    """Check if BOS condition is met based on type."""
    row = rows[j]
    
    if bos_type == 'close':
        # Standard: Close breaks swing
        if side == 'long':
            return row.close > swing
        else:
            return row.close < swing
    
    elif bos_type == 'body':
        # Body BOS: Candle body (not wick) breaks level
        if side == 'long':
            return row.body_bottom > swing  # Entire body above swing
        else:
            return row.body_top < swing  # Entire body below swing
    
    elif bos_type == 'strong':
        # Strong BOS: Break by at least 0.3 ATR
        if side == 'long':
            return row.close > swing and (row.close - swing) > atr * 0.3
        else:
            return row.close < swing and (swing - row.close) > atr * 0.3
    
    elif bos_type == 'highlow':
        # High/Low BOS: High (long) or Low (short) breaks level
        if side == 'long':
            return row.high > swing
        else:
            return row.low < swing
    
    elif bos_type == 'engulfing':
        # Engulfing BOS: BOS candle engulfs previous candle
        if j < 1: return False
        prev = rows[j-1]
        if side == 'long':
            engulf = row.body_bottom < prev.body_bottom and row.body_top > prev.body_top
            return row.close > swing and engulf and row.bullish
        else:
            engulf = row.body_top > prev.body_top and row.body_bottom < prev.body_bottom
            return row.close < swing and engulf and row.bearish
    
    elif bos_type == 'double':
        # Double Close: Two consecutive closes beyond level
        if prev_close is None: return False
        if side == 'long':
            return row.close > swing and prev_close > swing
        else:
            return row.close < swing and prev_close < swing
    
    return False

def run_backtest(df, signals, rr, bos_type):
    rows = list(df.itertuples())
    atr = df['atr'].values
    close_arr = df['close'].values
    
    trades = []
    
    for sig in signals:
        div_idx = sig['idx']
        side = sig['side']
        
        bos_idx = None
        prev_close = None
        
        for j in range(div_idx + 1, min(div_idx + 1 + MAX_WAIT_CANDLES, len(rows) - 10)):
            current_atr = atr[j]
            
            if check_bos(rows, j, side, sig['swing'], current_atr, bos_type, prev_close):
                bos_idx = j
                break
            
            prev_close = close_arr[j]
        
        if bos_idx is None:
            continue
        
        entry_idx = bos_idx + 1
        if entry_idx >= len(rows) - 50:
            continue
        
        entry_price = rows[entry_idx].open
        if side == 'long':
            entry_price *= (1 + SLIPPAGE_PCT)
        else:
            entry_price *= (1 - SLIPPAGE_PCT)
        
        entry_atr = atr[entry_idx - 1]
        sl_dist = entry_atr * SL_ATR_MULT
        
        if side == 'long':
            sl = entry_price - sl_dist
            tp = entry_price + (sl_dist * rr)
        else:
            sl = entry_price + sl_dist
            tp = entry_price - (sl_dist * rr)
        
        result = None
        for k in range(entry_idx, min(entry_idx + 200, len(rows))):
            row = rows[k]
            if side == 'long':
                if row.low <= sl: result = 'loss'; break
                if row.high >= tp: result = 'win'; break
            else:
                if row.high >= sl: result = 'loss'; break
                if row.low <= tp: result = 'win'; break
        
        if result:
            risk_pct = sl_dist / entry_price
            fee_cost = (FEE_PCT * 2 + SLIPPAGE_PCT) / risk_pct
            r = (rr - fee_cost) if result == 'win' else (-1.0 - fee_cost)
            trades.append({'result': result, 'r': r})
    
    return trades

def main():
    print("="*80)
    print("ALTERNATIVE BOS DEFINITION TESTING - 5M")
    print("="*80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"Fetching data for {len(symbols)} symbols...")
    
    symbol_data = {}
    for idx, sym in enumerate(symbols):
        df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
        if df.empty or len(df) < 500:
            continue
        df = add_indicators(df)
        if len(df) < 100:
            continue
        signals = detect_divergences(df)
        if signals:
            symbol_data[sym] = (df, signals)
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(symbols)} symbols...")
    
    print(f"\nSymbols with signals: {len(symbol_data)}")
    
    bos_types = [
        ('close', 'Close BOS (standard)'),
        ('body', 'Body BOS (body > level)'),
        ('strong', 'Strong BOS (0.3 ATR)'),
        ('highlow', 'High/Low BOS'),
        ('engulfing', 'Engulfing BOS'),
        ('double', 'Double Close BOS'),
    ]
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    results = []
    
    for rr in RR_RATIOS:
        print(f"\n--- R:R = {rr}:1 ---")
        print(f"{'BOS Type':<25} | {'N':<6} | {'WR':<7} | {'Avg R':<10} | {'Status'}")
        print("-"*70)
        
        for bos_type, bos_name in bos_types:
            all_trades = []
            for sym, (df, signals) in symbol_data.items():
                trades = run_backtest(df, signals, rr, bos_type)
                all_trades.extend(trades)
            
            if all_trades:
                wins = sum(1 for t in all_trades if t['result'] == 'win')
                n = len(all_trades)
                wr = wins / n * 100
                total_r = sum(t['r'] for t in all_trades)
                avg_r = total_r / n
                status = "✅ PROFIT" if avg_r > 0 else "❌"
                print(f"{bos_name:<25} | {n:<6} | {wr:>5.1f}% | {avg_r:>+8.4f}R | {status}")
                results.append({'bos': bos_name, 'rr': rr, 'n': n, 'wr': wr, 'avg_r': avg_r})
            else:
                print(f"{bos_name:<25} | {'0':<6} | {'--':<7} | {'--':<10} | NO TRADES")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    profitable = [r for r in results if r['avg_r'] > 0]
    if profitable:
        print(f"\n🏆 FOUND {len(profitable)} PROFITABLE CONFIGURATIONS!")
        profitable.sort(key=lambda x: x['avg_r'], reverse=True)
        for r in profitable[:5]:
            print(f"  ✅ {r['bos']} | RR={r['rr']} | {r['wr']:.1f}% WR | {r['avg_r']:+.4f}R ({r['n']} trades)")
    else:
        print("\n❌ NO PROFITABLE BOS DEFINITIONS")
        # Show best performers
        if results:
            results.sort(key=lambda x: x['avg_r'], reverse=True)
            print("\nClosest to profitable:")
            for r in results[:3]:
                print(f"  {r['bos']} | RR={r['rr']} | {r['wr']:.1f}% WR | {r['avg_r']:+.4f}R")

if __name__ == "__main__":
    main()
