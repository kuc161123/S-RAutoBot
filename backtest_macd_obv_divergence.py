#!/usr/bin/env python3
"""
MACD & OBV DIVERGENCE + BOS - 5M TIMEFRAME
==========================================
Testing divergences on MACD (Histogram and Line) and OBV (Volume) 
with BOS (High/Low break) confirmation on 5M timeframe.

Divergence Types:
1. MACD Histogram (momentum exhaustion)
2. MACD Line (momentum trend)
3. OBV (volume/price discrepancy)

BOS Type:
- High/Low break (found to be most promising in previous tests)
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
NUM_SYMBOLS = 50
MAX_WAIT_CANDLES = 12  # Slightly longer wait for momentum
SL_ATR_MULT = 1.0     # Slightly wider SL for 5m noise

# Test R:R
RR_RATIOS = [1.0, 1.5, 2.0]

# Costs
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

# Pivot Settings
LOOKBACK_BARS = 12
MIN_PIVOT_DISTANCE = 4

BASE_URL = "https://api.bybit.com"

def get_symbols(limit):
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear", timeout=10)
        tickers = resp.json().get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        return [t['symbol'] for t in usdt[:limit]]
    except:
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT", "LINKUSDT", "ADAUSDT", "DOTUSDT", "MATICUSDT"]

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
    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['hist'] = df['macd'] - df['signal']
    
    # OBV
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # ATR
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    # EMAs
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
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

def detect_divergences(df, column):
    if len(df) < 100: return []
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    osc = df[column].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    osc_ph, osc_pl = find_pivots(osc, 3, 3)
    
    signals = []
    
    for i in range(30, n - 15):
        # Bullish Divergence
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
        if curr_pl and prev_pl:
            # Check for oscillator pivot at similar indices
            curr_osc_pl = prev_osc_pl = None
            for j in range(curr_pli - 2, curr_pli + 3):
                if j >= 0 and j < len(osc_pl) and not np.isnan(osc_pl[j]):
                    curr_osc_pl = osc_pl[j]
                    break
            for j in range(prev_pli - 2, prev_pli + 3):
                if j >= 0 and j < len(osc_pl) and not np.isnan(osc_pl[j]):
                    prev_osc_pl = osc_pl[j]
                    break
            
            if curr_osc_pl is not None and prev_osc_pl is not None:
                if curr_pl < prev_pl and curr_osc_pl > prev_osc_pl:
                    swing_high = max(high[max(0, i-LOOKBACK_BARS):i+1])
                    signals.append({'idx': i, 'side': 'long', 'swing': swing_high, 'type': column})
        
        # Bearish Divergence
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        if curr_ph and prev_ph:
            curr_osc_ph = prev_osc_ph = None
            for j in range(curr_phi - 2, curr_phi + 3):
                if j >= 0 and j < len(osc_ph) and not np.isnan(osc_ph[j]):
                    curr_osc_ph = osc_ph[j]
                    break
            for j in range(prev_phi - 2, prev_phi + 3):
                if j >= 0 and j < len(osc_ph) and not np.isnan(osc_ph[j]):
                    prev_osc_ph = osc_ph[j]
                    break
            
            if curr_osc_ph is not None and prev_osc_ph is not None:
                if curr_ph > prev_ph and curr_osc_ph < prev_osc_ph:
                    swing_low = min(low[max(0, i-LOOKBACK_BARS):i+1])
                    signals.append({'idx': i, 'side': 'short', 'swing': swing_low, 'type': column})
    
    return signals

def run_backtest(df, signals, rr, use_trend=False):
    rows = list(df.itertuples())
    atr = df['atr'].values
    ema200 = df['ema200'].values
    trades = []
    
    for sig in signals:
        div_idx = sig['idx']
        side = sig['side']
        
        # Trend filter
        if use_trend:
            if side == 'long' and rows[div_idx].close < ema200[div_idx]:
                continue
            if side == 'short' and rows[div_idx].close > ema200[div_idx]:
                continue
        
        # BOS: High/Low break
        bos_idx = None
        for j in range(div_idx + 1, min(div_idx + 1 + MAX_WAIT_CANDLES, len(rows) - 10)):
            if side == 'long':
                if rows[j].high > sig['swing']:
                    bos_idx = j
                    break
            else:
                if rows[j].low < sig['swing']:
                    bos_idx = j
                    break
        
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
            trades.append({'result': result, 'r': r, 'type': sig['type']})
    
    return trades

def main():
    print("="*80)
    print("MACD & OBV DIVERGENCE + BOS (5M)")
    print("="*80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"Fetching data for {len(symbols)} symbols...")
    
    symbol_data = {}
    for idx, sym in enumerate(symbols):
        df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
        if df.empty or len(df) < 500: continue
        df = add_indicators(df)
        if len(df) < 200: continue
        
        # Detect for each type
        hist_sigs = detect_divergences(df, 'hist')
        macd_sigs = detect_divergences(df, 'macd')
        obv_sigs = detect_divergences(df, 'obv')
        
        all_sigs = hist_sigs + macd_sigs + obv_sigs
        if all_sigs:
            symbol_data[sym] = (df, all_sigs)
        
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(symbols)} symbols...")
    
    print(f"\nSymbols with signals: {len(symbol_data)}")
    
    types = ['hist', 'macd', 'obv', 'combined']
    
    configs = [
        {'name': 'No Filter', 'trend': False},
        {'name': 'With Trend', 'trend': True},
    ]
    
    for rr in RR_RATIOS:
        print(f"\n{'='*80}")
        print(f"R:R = {rr}:1")
        print(f"{'='*80}")
        
        for cfg in configs:
            print(f"\n--- {cfg['name']} ---")
            print(f"{'Type':<15} | {'N':<6} | {'WR':<7} | {'Avg R':<10} | {'Status'}")
            print("-"*60)
            
            for dtype in types:
                all_trades = []
                for sym, (df, signals) in symbol_data.items():
                    if dtype == 'combined':
                        filtered_sigs = signals
                    else:
                        filtered_sigs = [s for s in signals if s['type'] == dtype]
                    
                    trades = run_backtest(df, filtered_sigs, rr, use_trend=cfg['trend'])
                    all_trades.extend(trades)
                
                if all_trades:
                    wins = sum(1 for t in all_trades if t['result'] == 'win')
                    n = len(all_trades)
                    wr = wins / n * 100
                    total_r = sum(t['r'] for t in all_trades)
                    avg_r = total_r / n
                    status = "✅ PROFIT" if avg_r > 0 else "❌"
                    print(f"{dtype:<15} | {n:<6} | {wr:>5.1f}% | {avg_r:>+8.4f}R | {status}")

if __name__ == "__main__":
    main()
