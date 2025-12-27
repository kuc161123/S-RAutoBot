#!/usr/bin/env python3
"""
4H WFO WITH REGIME FILTERS
==========================
Adding regime filters to the 4H strategy to improve WFO robustness.
Filters to Test:
1. ADX > 20 (Avoid chop)
2. Daily Trend (Daily EMA200 alignment) - Uses 4H EMA1200 as proxy

Hypothesis: Divergences fail in strong trending markets (counter-trend) or ultra-low vol chop.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Config
TIMEFRAME = '240'  # 4H
DATA_DAYS = 1100   # ~3 Years
NUM_SYMBOLS = 40
MAX_WAIT_CANDLES = 6

# WFO Settings
TRAIN_MONTHS = 6
TEST_MONTHS = 3
SLIDE_MONTHS = 3

# Costs
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

# Indicators
RSI_PERIOD = 14
LOOKBACK_BARS = 10
MIN_PIVOT_DISTANCE = 3

BASE_URL = "https://api.bybit.com"

def get_symbols(limit):
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear", timeout=10)
        tickers = resp.json().get('result', {}).get('list', [])
        usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        return [t['symbol'] for t in usdt[:limit]]
    except:
        return []

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
    
    # ATR
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    # ADX
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr1 = pd.DataFrame(hl)
    tr2 = pd.DataFrame(hc)
    tr3 = pd.DataFrame(lc)
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(14).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (abs(minus_dm.ewm(alpha=1/14).mean()) / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['adx'] = dx.rolling(14).mean()
    
    # Daily EMA Proxy (6 * 200 = 1200 periods on 4H)
    df['ema_daily'] = df['close'].ewm(span=1200, adjust=False).mean()
    
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
    # Pivot "right" parameter is 3, so a pivot at 'j' is only confirmed at 'j+3'
    # We can only see pivots up to 'i-3'
    PIVOT_RIGHT = 3 
    
    for i in range(30, n - 15):
        # Bullish
        curr_pl = curr_pli = prev_pl = prev_pli = None
        # Start searching from i - PIVOT_RIGHT to avoid lookahead
        for j in range(i - PIVOT_RIGHT, max(i - LOOKBACK_BARS - PIVOT_RIGHT, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        if curr_pl and prev_pl:
            if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                swing_high = max(high[max(0, i-LOOKBACK_BARS):i+1])
                signals.append({'idx': i, 'side': 'long', 'swing': swing_high, 'ts': df.index[i]})
        # Bearish
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i - PIVOT_RIGHT, max(i - LOOKBACK_BARS - PIVOT_RIGHT, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_ph[j], j
                    break
        if curr_ph and prev_ph:

            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                swing_low = min(low[max(0, i-LOOKBACK_BARS):i+1])
                signals.append({'idx': i, 'side': 'short', 'swing': swing_low, 'ts': df.index[i]})
    return signals

def run_backtest_window(df, signals, rr, sl_mult, filter_type, start_ts, end_ts):
    window_signals = [s for s in signals if start_ts <= s['ts'] < end_ts]
    if not window_signals: return []
    rows = df.loc[start_ts:end_ts]
    if rows.empty: return []
    row_list = list(rows.itertuples())
    
    trades = []
    
    for sig in window_signals:
        div_idx = rows.index.get_loc(sig['ts'])
        side = sig['side']
        current_row = row_list[div_idx]
        
        # APPLY FILTERS
        if filter_type == 'adx':
            if current_row.adx < 20: continue
            
        elif filter_type == 'daily_trend':
            # Only trade WITH daily trend
            if side == 'long' and current_row.close < current_row.ema_daily: continue
            if side == 'short' and current_row.close > current_row.ema_daily: continue
            
        elif filter_type == 'adx_trend':
            if current_row.adx < 20: continue
            if side == 'long' and current_row.close < current_row.ema_daily: continue
            if side == 'short' and current_row.close > current_row.ema_daily: continue
        
        bos_idx = None
        for j in range(div_idx + 1, min(div_idx + 1 + MAX_WAIT_CANDLES, len(row_list) - 10)):
            curr_row = row_list[j]
            if side == 'long':
                if curr_row.close > sig['swing']:
                    bos_idx = j
                    break
            else:
                if curr_row.close < sig['swing']:
                    bos_idx = j
                    break
        
        if bos_idx is None: continue
        entry_idx = bos_idx + 1
        if entry_idx >= len(row_list) - 50: continue
        
        entry_row = row_list[entry_idx]
        entry_price = entry_row.open
        if side == 'long': entry_price *= (1 + SLIPPAGE_PCT)
        else: entry_price *= (1 - SLIPPAGE_PCT)
        
        entry_atr = entry_row.atr
        sl_dist = entry_atr * sl_mult
        
        if side == 'long':
            sl = entry_price - sl_dist
            tp = entry_price + (sl_dist * rr)
        else:
            sl = entry_price + sl_dist
            tp = entry_price - (sl_dist * rr)
        
        result = None
        for k in range(entry_idx, min(entry_idx + 200, len(row_list))):
            curr_row = row_list[k]
            if side == 'long':
                if curr_row.low <= sl: result = 'loss'; break
                if curr_row.high >= tp: result = 'win'; break
            else:
                if curr_row.high >= sl: result = 'loss'; break
                if curr_row.low <= tp: result = 'win'; break
        
        if result:
            risk_pct = sl_dist / entry_price
            fee_cost = (FEE_PCT * 2 + SLIPPAGE_PCT) / risk_pct
            r = (rr - fee_cost) if result == 'win' else (-1.0 - fee_cost)
            trades.append({'result': result, 'r': r})
    return trades

def main():
    print("="*80)
    print("4H WFO WITH REGIME FILTERS")
    print("="*80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"Fetching data for {len(symbols)} symbols...")
    
    symbol_data = {}
    all_dates = []
    for idx, sym in enumerate(symbols):
        df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
        if df.empty or len(df) < 500: continue
        df = add_indicators(df)
        if len(df) < 200: continue
        signals = detect_divergences(df)
        if signals:
            symbol_data[sym] = (df, signals)
            all_dates.extend(df.index)
        if (idx + 1) % 10 == 0: print(f"  {idx+1}/{len(symbols)} symbols...")
            
    if not all_dates: return
    
    start_date = min(all_dates)
    end_date = max(all_dates)
    train_months_pd = pd.DateOffset(months=TRAIN_MONTHS)
    test_months_pd = pd.DateOffset(months=TEST_MONTHS)
    slide_months_pd = pd.DateOffset(months=SLIDE_MONTHS)
    
    filters = ['none', 'adx', 'daily_trend', 'adx_trend']
    results_summary = {f: {'total_r': 0, 'wins': 0, 'total': 0} for f in filters}
    
    # We will just run the WFO loop and accumulate results for each filter type
    # Using fixed parameters RR=4.0 SL=1.0 (Middle ground) to test filter impact
    RR = 4.0
    SL = 1.0
    
    print(f"\nTesting Filters (RR={RR}, SL={SL})...")
    print(f"{'Period':<20} | {'None':<8} | {'ADX':<8} | {'Trend':<8} | {'Both':<8}")
    print("-"*70)
    
    current_start = start_date
    while current_start + train_months_pd + test_months_pd < end_date:
        train_end = current_start + train_months_pd
        test_end = train_end + test_months_pd
        
        period_res = {}
        for f in filters:
            r_sum = 0
            for sym, (df, signals) in symbol_data.items():
                trades = run_backtest_window(df, signals, RR, SL, f, train_end, test_end)
                r_sum += sum(t['r'] for t in trades)
                
                results_summary[f]['total_r'] += sum(t['r'] for t in trades)
                results_summary[f]['wins'] += sum(1 for t in trades if t['result'] == 'win')
                results_summary[f]['total'] += len(trades)
            period_res[f] = r_sum
            
        period_str = f"{train_end.strftime('%y-%m')}->{test_end.strftime('%y-%m')}"
        print(f"{period_str:<20} | {period_res['none']:>8.1f} | {period_res['adx']:>8.1f} | {period_res['daily_trend']:>8.1f} | {period_res['adx_trend']:>8.1f}")
        
        current_start += slide_months_pd

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"{'Filter':<15} | {'Total R':<10} | {'Avg R':<10} | {'Trades':<8} | {'WR':<6}")
    print("-"*60)
    
    for f in filters:
        res = results_summary[f]
        avg_r = res['total_r'] / res['total'] if res['total'] > 0 else 0
        wr = res['wins'] / res['total'] * 100 if res['total'] > 0 else 0
        print(f"{f:<15} | {res['total_r']:>+8.1f}R | {avg_r:>+8.4f}R | {res['total']:<8} | {wr:>4.1f}%")

if __name__ == "__main__":
    main()
