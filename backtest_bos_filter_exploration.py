#!/usr/bin/env python3
"""
BOS DIVERGENCE - FILTER EXPLORATION
===================================
Testing multiple filters to find profitable configuration:
1. Volume Filter (volume > X * MA)
2. Trend Filter (with EMA200)
3. Extreme RSI Filter (only trade at extremes)
4. ADX Filter (trending market)
5. Low Volatility Filter (reduced ATR)
6. Combinations
"""

import requests
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
import itertools
import warnings
warnings.filterwarnings('ignore')

# Config
TIMEFRAME = '60'
DATA_DAYS = 120
NUM_SYMBOLS = 80
MAX_WAIT_CANDLES = 10
RR = 1.5  # Best WR was at 1.5:1
SL_ATR_MULT = 0.8

# Costs
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

# RSI Settings
RSI_PERIOD = 14
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5

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
    # RSI
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
    
    # EMAs
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    
    # Volume MA
    df['vol_ma'] = df['volume'].rolling(20).mean()
    
    # ADX
    plus_dm = df['high'].diff()
    minus_dm = df['low'].diff().abs() * -1
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm < 0), 0).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / (atr14 + 1e-9))
    minus_di = 100 * (minus_dm.rolling(14).mean() / (atr14 + 1e-9))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    df['adx'] = dx.rolling(14).mean()
    
    # ATR percentile (volatility regime)
    df['atr_pct'] = df['atr'] / df['close']
    df['atr_rank'] = df['atr_pct'].rolling(100).rank(pct=True)
    
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
        row_data = {
            'rsi': rsi[i],
            'ema50': df['ema50'].iloc[i],
            'ema200': df['ema200'].iloc[i],
            'close': close[i],
            'volume': df['volume'].iloc[i],
            'vol_ma': df['vol_ma'].iloc[i],
            'adx': df['adx'].iloc[i],
            'atr_rank': df['atr_rank'].iloc[i],
        }
        
        # Find pivot lows
        curr_pl = curr_pli = prev_pl = prev_pli = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pl[j]):
                if curr_pl is None: curr_pl, curr_pli = price_pl[j], j
                elif prev_pl is None and j < curr_pli - MIN_PIVOT_DISTANCE:
                    prev_pl, prev_pli = price_pl[j], j
                    break
        
        # Find pivot highs
        curr_ph = curr_phi = prev_ph = prev_phi = None
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE:
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        # Regular Bullish
        if curr_pl and prev_pl:
            if curr_pl < prev_pl and rsi[curr_pli] > rsi[prev_pli]:
                swing_high = max(high[max(0, i-LOOKBACK_BARS):i+1])
                signals.append({'idx': i, 'side': 'long', 'swing': swing_high, **row_data})
                continue
        
        # Regular Bearish
        if curr_ph and prev_ph:
            if curr_ph > prev_ph and rsi[curr_phi] < rsi[prev_phi]:
                swing_low = min(low[max(0, i-LOOKBACK_BARS):i+1])
                signals.append({'idx': i, 'side': 'short', 'swing': swing_low, **row_data})
    
    return signals

def apply_filters(signals, filters):
    """Apply filter combination to signals."""
    filtered = []
    
    for sig in signals:
        passed = True
        
        # Volume Filter
        if 'vol_mult' in filters:
            if sig['volume'] < sig['vol_ma'] * filters['vol_mult']:
                passed = False
        
        # Trend Filter (trade with trend only)
        if filters.get('trend_filter'):
            if sig['side'] == 'long' and sig['close'] < sig['ema200']:
                passed = False
            elif sig['side'] == 'short' and sig['close'] > sig['ema200']:
                passed = False
        
        # Counter-Trend Filter (trade against trend - mean reversion)
        if filters.get('counter_trend'):
            if sig['side'] == 'long' and sig['close'] > sig['ema200']:
                passed = False
            elif sig['side'] == 'short' and sig['close'] < sig['ema200']:
                passed = False
        
        # RSI Extreme Filter
        if 'rsi_extreme' in filters:
            thresh = filters['rsi_extreme']
            if sig['side'] == 'long' and sig['rsi'] > thresh:
                passed = False
            elif sig['side'] == 'short' and sig['rsi'] < (100 - thresh):
                passed = False
        
        # ADX Filter (trending)
        if 'adx_min' in filters:
            if sig['adx'] < filters['adx_min']:
                passed = False
        
        # Low Volatility Filter
        if filters.get('low_vol'):
            if sig['atr_rank'] > 0.5:  # Only trade in lower 50% volatility
                passed = False
        
        if passed:
            filtered.append(sig)
    
    return filtered

def run_backtest(df, signals):
    rows = list(df.itertuples())
    atr = df['atr'].values
    trades = []
    
    for sig in signals:
        div_idx = sig['idx']
        side = sig['side']
        
        # Wait for BOS
        bos_idx = None
        for j in range(div_idx + 1, min(div_idx + 1 + MAX_WAIT_CANDLES, len(rows) - 10)):
            if side == 'long':
                if rows[j].close > sig['swing']:
                    bos_idx = j
                    break
            else:
                if rows[j].close < sig['swing']:
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
        
        entry_atr = atr[entry_idx - 1] if entry_idx > 0 else atr[entry_idx]
        sl_dist = entry_atr * SL_ATR_MULT
        
        if side == 'long':
            sl = entry_price - sl_dist
            tp = entry_price + (sl_dist * RR)
        else:
            sl = entry_price + sl_dist
            tp = entry_price - (sl_dist * RR)
        
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
            r = (RR - fee_cost) if result == 'win' else (-1.0 - fee_cost)
            trades.append({'result': result, 'r': r})
    
    return trades

def main():
    print("="*70)
    print("BOS DIVERGENCE - FILTER EXPLORATION")
    print("="*70)
    print(f"Base: 1H, 120 days, {NUM_SYMBOLS} symbols, R:R={RR}")
    print("="*70)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\nFetching data for {len(symbols)} symbols...")
    
    # Fetch all data
    symbol_data = {}
    for idx, sym in enumerate(symbols):
        df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
        if df.empty or len(df) < 300:
            continue
        
        df = add_indicators(df)
        if len(df) < 100:
            continue
        
        signals = detect_divergences(df)
        if signals:
            symbol_data[sym] = (df, signals)
        
        if (idx + 1) % 20 == 0:
            print(f"  Fetched {idx+1}/{len(symbols)} symbols...")
    
    print(f"\nSymbols with signals: {len(symbol_data)}")
    total_signals = sum(len(sig) for df, sig in symbol_data.values())
    print(f"Total raw signals: {total_signals}")
    
    # Define filter combinations to test
    filter_configs = [
        {'name': 'No Filter', 'filters': {}},
        {'name': 'Volume > 1.0x MA', 'filters': {'vol_mult': 1.0}},
        {'name': 'Volume > 1.5x MA', 'filters': {'vol_mult': 1.5}},
        {'name': 'With Trend (EMA200)', 'filters': {'trend_filter': True}},
        {'name': 'Counter Trend (EMA200)', 'filters': {'counter_trend': True}},
        {'name': 'RSI Extreme (<25/>75)', 'filters': {'rsi_extreme': 25}},
        {'name': 'RSI Extreme (<20/>80)', 'filters': {'rsi_extreme': 20}},
        {'name': 'ADX > 20', 'filters': {'adx_min': 20}},
        {'name': 'ADX > 25', 'filters': {'adx_min': 25}},
        {'name': 'Low Volatility', 'filters': {'low_vol': True}},
        {'name': 'Vol 1.5x + Trend', 'filters': {'vol_mult': 1.5, 'trend_filter': True}},
        {'name': 'Vol 1.5x + RSI Ext 25', 'filters': {'vol_mult': 1.5, 'rsi_extreme': 25}},
        {'name': 'Trend + ADX 25', 'filters': {'trend_filter': True, 'adx_min': 25}},
        {'name': 'Counter Trend + Low Vol', 'filters': {'counter_trend': True, 'low_vol': True}},
        {'name': 'Vol + Trend + RSI', 'filters': {'vol_mult': 1.2, 'trend_filter': True, 'rsi_extreme': 30}},
        {'name': 'COMBO: Vol + ADX + Low Vol', 'filters': {'vol_mult': 1.3, 'adx_min': 20, 'low_vol': True}},
    ]
    
    print("\n" + "="*70)
    print("FILTER RESULTS")
    print("="*70)
    print(f"{'Filter':<30} | {'Trades':<7} | {'WR':<8} | {'Avg R':<10} | {'Total R':<9} | {'Status'}")
    print("-"*95)
    
    results = []
    
    for config in filter_configs:
        all_trades = []
        
        for sym, (df, signals) in symbol_data.items():
            filtered_signals = apply_filters(signals, config['filters'])
            trades = run_backtest(df, filtered_signals)
            all_trades.extend(trades)
        
        if all_trades:
            wins = sum(1 for t in all_trades if t['result'] == 'win')
            n = len(all_trades)
            wr = wins / n * 100
            total_r = sum(t['r'] for t in all_trades)
            avg_r = total_r / n
            
            status = "✅ PROFIT" if avg_r > 0 else "❌ LOSS"
            
            print(f"{config['name']:<30} | {n:<7} | {wr:>6.1f}% | {avg_r:>+8.4f}R | {total_r:>+8.1f}R | {status}")
            
            results.append({
                'name': config['name'],
                'trades': n,
                'wr': wr,
                'avg_r': avg_r,
                'total_r': total_r,
                'profitable': avg_r > 0
            })
        else:
            print(f"{config['name']:<30} | {'0':<7} | {'--':<8} | {'--':<10} | {'--':<9} | NO TRADES")
    
    print("="*95)
    
    # Summary
    profitable = [r for r in results if r['profitable']]
    if profitable:
        profitable.sort(key=lambda x: x['avg_r'], reverse=True)
        print(f"\n🏆 FOUND {len(profitable)} PROFITABLE CONFIGURATIONS!")
        print("\nTop 3 by Avg R/Trade:")
        for r in profitable[:3]:
            print(f"  ✅ {r['name']}: {r['wr']:.1f}% WR, {r['avg_r']:+.4f}R avg, {r['total_r']:+.1f}R total ({r['trades']} trades)")
    else:
        print("\n❌ NO PROFITABLE FILTER COMBINATIONS FOUND")
        
        # Show closest to profitable
        if results:
            results.sort(key=lambda x: x['avg_r'], reverse=True)
            print("\nClosest to profitable:")
            for r in results[:3]:
                print(f"  {r['name']}: {r['wr']:.1f}% WR, {r['avg_r']:+.4f}R avg")

if __name__ == "__main__":
    main()
