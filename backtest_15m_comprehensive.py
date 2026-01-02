#!/usr/bin/env python3
"""
15M TIMEFRAME COMPREHENSIVE BACKTEST
====================================
Goal: Increase trade frequency by testing 15-minute timeframe

Test Matrix:
- Timeframe: 15 minutes
- Data: 3 years per symbol
- R:R Ratios: 2, 3, 4, 5, 6, 7, 8, 9, 10
- ATR Multipliers: 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
- Symbols: All 63 validated symbols

Validation:
1. Initial filter: WR > 45%, N > 50, Total R > 20
2. Top performers go through robust validation
3. Present final results for deployment decision
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yaml
import warnings
import itertools
from typing import List, Dict, Tuple
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
TIMEFRAME = '15'  # 15 minutes
DATA_DAYS = 1095  # 3 years
MAX_WAIT_CANDLES = 6

# Parameter grids
RR_RATIOS = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
ATR_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

# Costs
SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006

BASE_URL = "https://api.bybit.com"
RSI_PERIOD = 14
EMA_PERIOD = 200  # Same EMA200 on 15m

# Load symbols
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

SYMBOLS = [s for s, props in config.get('symbols', {}).items() if props.get('enabled', False)]

print(f"Testing {len(SYMBOLS)} symbols with {len(RR_RATIOS)} R:R ratios × {len(ATR_MULTS)} ATR mults = {len(RR_RATIOS) * len(ATR_MULTS)} configs per symbol")
print(f"Total combinations: {len(SYMBOLS) * len(RR_RATIOS) * len(ATR_MULTS)}")

def fetch_klines(symbol, interval, days):
    """Fetch klines with pagination for 3 years"""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_end = end_ts
    max_iterations = 50  # More iterations for 3 years
    
    while current_end > start_ts and max_iterations > 0:
        max_iterations -= 1
        params = {
            'category': 'linear', 
            'symbol': symbol, 
            'interval': interval, 
            'limit': 1000, 
            'end': current_end
        }
        
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=15)
            data = resp.json().get('result', {}).get('list', [])
            
            if not data:
                break
                
            all_candles.extend(data)
            oldest = int(data[-1][0])
            current_end = oldest - 1
            
            if len(data) < 1000:
                break
                
            time.sleep(0.12)  # Rate limit
            
        except Exception as e:
            time.sleep(0.5)
            continue
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles, columns=['start', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
    df['start'] = pd.to_datetime(df['start'].astype(int), unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
        
    df.set_index('start', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    
    return df

def prepare_data(df):
    """Calculate indicators - NO LOOK-AHEAD"""
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
    
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    return df.dropna()

def find_pivots(data, left=3, right=3):
    """Find pivot highs/lows - NO LOOK-AHEAD"""
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    for i in range(left, n - right):
        window = data[i-left : i+right+1]
        center = data[i]
        
        if len(window) != (left + right + 1): 
            continue
        
        if center == max(window) and list(window).count(center) == 1:
            pivot_highs[i] = center
            
        if center == min(window) and list(window).count(center) == 1:
            pivot_lows[i] = center
            
    return pivot_highs, pivot_lows

def detect_divergences(df):
    """Detect divergences - NO LOOK-AHEAD"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    ema = df['ema'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    # Scan from 30 to n-15 (leave room for BOS)
    for i in range(30, n - 15):
        curr_price = close[i]
        curr_ema = ema[i]
        
        # BULLISH
        p_lows = []
        for j in range(i-3, max(0, i-50), -1):  # i-3 for NO LOOK-AHEAD
            if not np.isnan(price_pl[j]):
                p_lows.append((j, price_pl[j]))
                if len(p_lows) >= 2: 
                    break
        
        if len(p_lows) == 2 and curr_price > curr_ema:  # Trend filter
            curr_idx, curr_val = p_lows[0]
            prev_idx, prev_val = p_lows[1]
            
            if (i - curr_idx) <= 10:
                if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                    swing_high = max(high[curr_idx:i+1])
                    signals.append({
                        'idx': curr_idx,
                        'conf_idx': i,
                        'side': 'long',
                        'swing': swing_high,
                        'price': curr_val
                    })

        # BEARISH
        p_highs = [] 
        for j in range(i-3, max(0, i-50), -1):
            if not np.isnan(price_ph[j]):
                p_highs.append((j, price_ph[j]))
                if len(p_highs) >= 2: 
                    break
        
        if len(p_highs) == 2 and curr_price < curr_ema:  # Trend filter
            curr_idx, curr_val = p_highs[0]
            prev_idx, prev_val = p_highs[1]
            
            if (i - curr_idx) <= 10:
                if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                    swing_low = min(low[curr_idx:i+1])
                    signals.append({
                        'idx': curr_idx,
                        'conf_idx': i,
                        'side': 'short',
                        'swing': swing_low,
                        'price': curr_val
                    })
                            
    return signals

def backtest_symbol(df, signals, rr, atr_mult):
    """Backtest with specific R:R and ATR multiplier"""
    rows = list(df.itertuples())
    trades = []
    
    for sig in signals:
        start_idx = sig['conf_idx']
        side = sig['side']
        
        if start_idx >= len(rows):
            continue
            
        # Check for BOS within max wait
        entry_idx = None
        for j in range(start_idx + 1, min(start_idx + 1 + MAX_WAIT_CANDLES, len(rows))):
            row = rows[j]
            if side == 'long':
                if row.close > sig['swing']:
                    entry_idx = j + 1
                    break
            else:
                if row.close < sig['swing']:
                    entry_idx = j + 1
                    break
        
        if not entry_idx or entry_idx >= len(rows): 
            continue
        
        entry_row = rows[entry_idx]
        entry_price = entry_row.open
        atr = entry_row.atr
        sl_dist = atr * atr_mult
        
        if side == 'long':
            entry_price *= (1 + SLIPPAGE_PCT)
            tp_price = entry_price + (sl_dist * rr)
            sl_price = entry_price - sl_dist
        else:
            entry_price *= (1 - SLIPPAGE_PCT)
            tp_price = entry_price - (sl_dist * rr)
            sl_price = entry_price + sl_dist
            
        result = None
        
        for k in range(entry_idx, min(entry_idx + 400, len(rows))):  # Longer timeout for 15m
            row = rows[k]
            if side == 'long':
                if row.low <= sl_price:
                    result = -1.0
                    break
                if row.high >= tp_price:
                    result = rr
                    break
            else:
                if row.high >= sl_price:
                    result = -1.0
                    break
                if row.low <= tp_price:
                    result = rr
                    break
                    
        if result is None:
            result = -0.5
            
        risk_pct = abs(entry_price - sl_price) / entry_price
        if risk_pct == 0: 
            risk_pct = 0.01
        total_fee_cost = (FEE_PCT * 2) / risk_pct
        final_r = result - total_fee_cost
        trades.append(final_r)
        
    return trades

def test_symbol(symbol):
    """Test all parameter combinations for a symbol"""
    print(f"\n{'='*60}")
    print(f"Testing {symbol}")
    print(f"{'='*60}")
    
    df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
    
    if len(df) < 5000:  # Need substantial data for 3 years on 15m
        print(f"  ❌ Insufficient data ({len(df)} candles)")
        return []
    
    print(f"  ✓ Fetched {len(df)} candles ({DATA_DAYS} days)")
    
    df = prepare_data(df)
    signals = detect_divergences(df)
    
    print(f"  ✓ Detected {len(signals)} divergences")
    
    if len(signals) < 10:
        print(f"  ❌ Too few signals")
        return []
    
    results = []
    
    # Test all R:R × ATR combinations
    for rr, atr_mult in itertools.product(RR_RATIOS, ATR_MULTS):
        trades = backtest_symbol(df, signals, rr, atr_mult)
        
        if len(trades) < 10:
            continue
            
        total_r = sum(trades)
        wins = sum(1 for t in trades if t > 0)
        wr = (wins / len(trades) * 100) if trades else 0
        avg_r = total_r / len(trades) if trades else 0
        
        results.append({
            'symbol': symbol,
            'rr': rr,
            'atr_mult': atr_mult,
            'total_r': round(total_r, 2),
            'wr': round(wr, 1),
            'avg_r': round(avg_r, 3),
            'trades': len(trades),
            'wins': wins
        })
    
    # Show best config for this symbol
    if results:
        best = max(results, key=lambda x: x['total_r'])
        print(f"  ✅ Best: {best['rr']}:1 R:R, {best['atr_mult']}× ATR → {best['total_r']:+.1f}R ({best['wr']:.0f}% WR, N={best['trades']})")
    
    return results

def main():
    print("="*70)
    print("15M TIMEFRAME COMPREHENSIVE BACKTEST")
    print("="*70)
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Data Period: {DATA_DAYS} days (3 years)")
    print(f"R:R Ratios: {RR_RATIOS}")
    print(f"ATR Multipliers: {ATR_MULTS}")
    print("="*70)
    
    all_results = []
    
    for i, sym in enumerate(SYMBOLS):
        print(f"\n[{i+1}/{len(SYMBOLS)}] {sym}")
        results = test_symbol(sym)
        all_results.extend(results)
        time.sleep(0.2)  # Rate limit
    
    # Save all results
    df_all = pd.DataFrame(all_results)
    df_all.to_csv('15m_backtest_results_full.csv', index=False)
    print(f"\n💾 Full results saved: 15m_backtest_results_full.csv ({len(all_results)} configs)")
    
    # Filter for quality candidates
    df_filtered = df_all[
        (df_all['wr'] > 45) & 
        (df_all['trades'] > 50) & 
        (df_all['total_r'] > 20)
    ].sort_values('total_r', ascending=False)
    
    df_filtered.to_csv('15m_backtest_candidates.csv', index=False)
    print(f"💎 Quality candidates: {len(df_filtered)} configs")
    
    print(f"\n{'='*70}")
    print("TOP 20 CONFIGURATIONS")
    print(f"{'='*70}")
    for idx, row in df_filtered.head(20).iterrows():
        print(f"{row['symbol']:15} | {row['rr']:.0f}:1 R:R | {row['atr_mult']:.2f}× ATR | {row['total_r']:+7.1f}R | WR: {row['wr']:5.1f}% | N: {row['trades']:3}")

if __name__ == "__main__":
    main()
