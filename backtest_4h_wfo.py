#!/usr/bin/env python3
"""
4H WAL FORWARD OPTIMIZATION (WFO)
=================================
Validating the robustness of the 4H Divergence strategy.
- Strategy: RSI Divergence + Close BOS (No filters, as per findings)
- Windows: 6-month Train / 3-month Test windows
- Optimization Target: Total R
- Parameter Grid: 
    - R:R: [3.0, 4.0, 5.0, 6.0]
    - SL Multiplier: [0.8, 1.0, 1.2, 1.5]
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Config
TIMEFRAME = '240'  # 4H
DATA_DAYS = 1100   # ~3 Years (Jan 2021 - Present)
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
    
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
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
        # Bullish
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
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
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
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

def run_backtest_window(df, signals, rr, sl_mult, start_ts, end_ts):
    # Filter signals in window
    window_signals = [s for s in signals if start_ts <= s['ts'] < end_ts]
    if not window_signals: return []
    
    rows = df.loc[start_ts:end_ts]
    if rows.empty: return []
    
    row_list = list(rows.itertuples())
    # Need ATR map for quick lookup - simplified for WFO speed
    # We'll just grab ATR from the row itself
    
    trades = []
    
    for sig in window_signals:
        div_idx = rows.index.get_loc(sig['ts'])
        side = sig['side']
        
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
        
        if bos_idx is None:
            continue
        
        entry_idx = bos_idx + 1
        if entry_idx >= len(row_list) - 50:
            continue
        
        entry_row = row_list[entry_idx]
        entry_price = entry_row.open
        if side == 'long':
            entry_price *= (1 + SLIPPAGE_PCT)
        else:
            entry_price *= (1 - SLIPPAGE_PCT)
        
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
            trades.append({'result': result, 'r': r, 'ts': entry_row.Index})
    
    return trades

def main():
    print("="*80)
    print("4H WALK-FORWARD OPTIMIZATION (WFO)")
    print("="*80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"Fetching data for {len(symbols)} symbols...")
    
    symbol_data = {}
    total_signals = 0
    all_dates = []
    
    for idx, sym in enumerate(symbols):
        df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
        if df.empty or len(df) < 500: continue
        df = add_indicators(df)
        if len(df) < 200: continue
        signals = detect_divergences(df)
        if signals:
            symbol_data[sym] = (df, signals)
            total_signals += len(signals)
            all_dates.extend(df.index)
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(symbols)} symbols...")
            
    print(f"\nSymbols: {len(symbol_data)}, Total Signals: {total_signals}")
    
    if not all_dates:
        print("No data found.")
        return
        
    start_date = min(all_dates)
    end_date = max(all_dates)
    print(f"Data Range: {start_date} to {end_date}")
    
    # WFO Loop
    train_months_pd = pd.DateOffset(months=TRAIN_MONTHS)
    test_months_pd = pd.DateOffset(months=TEST_MONTHS)
    slide_months_pd = pd.DateOffset(months=SLIDE_MONTHS)
    
    current_start = start_date
    wfo_results = []
    
    configs = []
    for rr in [3.0, 4.0, 5.0, 6.0]:
        for sl in [0.8, 1.0, 1.2, 1.5]:
            configs.append({'rr': rr, 'sl': sl})
    
    print(f"\nRunning WFO Steps (Train {TRAIN_MONTHS}m / Test {TEST_MONTHS}m)...")
    print(f"{'Period':<25} | {'Best Config (Train)':<20} | {'Train R':<8} | {'Test R':<8} | {'Test Trades':<11} | {'Status'}")
    print("-"*100)
    
    step = 1
    total_test_r = 0
    total_test_trades = 0
    
    while current_start + train_months_pd + test_months_pd < end_date:
        train_end = current_start + train_months_pd
        test_end = train_end + test_months_pd
        
        # Optimization (In-Sample)
        best_cfg = None
        best_train_r = -9999
        
        for cfg in configs:
            train_r = 0
            for sym, (df, signals) in symbol_data.items():
                trades = run_backtest_window(df, signals, cfg['rr'], cfg['sl'], current_start, train_end)
                train_r += sum(t['r'] for t in trades)
            
            if train_r > best_train_r:
                best_train_r = train_r
                best_cfg = cfg
        
        # Testing (Out-of-Sample)
        test_r = 0
        test_n = 0
        
        if best_cfg:
            for sym, (df, signals) in symbol_data.items():
                trades = run_backtest_window(df, signals, best_cfg['rr'], best_cfg['sl'], train_end, test_end)
                test_r += sum(t['r'] for t in trades)
                test_n += len(trades)
        
        period_str = f"{train_end.strftime('%Y-%m')} -> {test_end.strftime('%Y-%m')}"
        cfg_str = f"RR={best_cfg['rr']} SL={best_cfg['sl']}" if best_cfg else "None"
        status = "✅" if test_r > 0 else "❌"
        
        print(f"{period_str:<25} | {cfg_str:<20} | {best_train_r:>8.1f} | {test_r:>8.1f} | {test_n:<11} | {status}")
        
        wfo_results.append({'period': period_str, 'train_r': best_train_r, 'test_r': test_r, 'test_n': test_n})
        total_test_r += test_r
        total_test_trades += test_n
        
        current_start += slide_months_pd
        step += 1
    
    print("\n" + "="*80)
    print("WFO SUMMARY")
    print("="*80)
    
    profitable_folds = sum(1 for r in wfo_results if r['test_r'] > 0)
    total_folds = len(wfo_results)
    avg_r_per_trade = total_test_r / total_test_trades if total_test_trades > 0 else 0
    
    print(f"Total Test R: {total_test_r:.1f}R")
    print(f"Total Trades: {total_test_trades}")
    print(f"Avg R/Trade:  {avg_r_per_trade:+.4f}R")
    print(f"Robustness:   {profitable_folds}/{total_folds} folds profitable ({profitable_folds/total_folds*100:.1f}%)")
    
    if profitable_folds / total_folds >= 0.6 and avg_r_per_trade > 0.05:
        print("\n✅ STRATEGY PASSED WALK-FORWARD VALIDATION!")
    else:
        print("\n❌ STRATEGY FAILED ROBUSTNESS CRITERIA")

if __name__ == "__main__":
    main()
