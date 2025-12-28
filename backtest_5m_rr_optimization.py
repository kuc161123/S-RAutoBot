#!/usr/bin/env python3
"""
5M PER-SYMBOL R:R OPTIMIZATION
==============================
Objective: Find if ANY of the 24 "4H Winners" can also be profitable on 5M.

Key Differences from 4H:
- Timeframe: 5 minutes
- Data: 60 days (~17,280 candles per symbol)
- EMA Period: 200 (standard)
- R:R Range: 1.5 to 5.0 (shorter trades)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import yaml
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
TIMEFRAME = '5'  # 5 Minutes
DATA_DAYS = 60   # 2 months of data
MAX_WAIT_CANDLES = 6  # 30 mins to wait for BOS
SL_MULT = 1.0

# RR Ratios to test (smaller for 5M)
RR_RATIOS = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# Costs (higher for 5M due to more trades)
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

# Use the 24 winners from 4H as test subjects
SYMBOLS = [
    "ADAUSDT", "ARBUSDT", "AVAXUSDT", "BCHUSDT", "DASHUSDT", "DOGEUSDT",
    "DOTUSDT", "ENAUSDT", "ETCUSDT", "FLOWUSDT", "GALAUSDT", "GASUSDT",
    "LINKUSDT", "LTCUSDT", "NEARUSDT", "NTRNUSDT", "OPUSDT", "PAXGUSDT",
    "SEIUSDT", "STRKUSDT", "TRUUSDT", "TRXUSDT", "VIRTUALUSDT", "XLMUSDT"
]

BASE_URL = "https://api.bybit.com"

# === INDICATOR LOGIC ===
RSI_PERIOD = 14
LOOKBACK_BARS = 10
MIN_PIVOT_DISTANCE = 3
PIVOT_RIGHT = 3
EMA_PERIOD = 200  # Standard 200 EMA for 5M

def fetch_klines(symbol, interval, days):
    """Fetch 5M klines with pagination"""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_end = end_ts
    
    # 60 days of 5M = 60 * 24 * 12 = 17,280 candles = ~18 API calls
    max_iterations = 20
    
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
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=10)
            data = resp.json().get('result', {}).get('list', [])
            
            if not data:
                break
                
            all_candles.extend(data)
            oldest = int(data[-1][0])
            current_end = oldest - 1
            
            if len(data) < 1000:
                break
                
            time.sleep(0.15)  # Be nice to rate limits
            
        except Exception as e:
            print(f"    Fetch error: {e}")
            time.sleep(1)
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
    """Calculate indicators for 5M"""
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
    
    # 200 EMA (standard for 5M)
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    return df.dropna()

def find_pivots(data, left=3, right=3):
    """Find pivots without look-ahead"""
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    for i in range(left, n - right):
        window = data[i-left : i+right+1]
        center = data[i]
        
        if len(window) != (left + right + 1): continue
        
        if center == max(window) and list(window).count(center) == 1:
            pivot_highs[i] = center
            
        if center == min(window) and list(window).count(center) == 1:
            pivot_lows[i] = center
            
    return pivot_highs, pivot_lows

def detect_divergences(df):
    """Detect RSI divergences"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    
    signals = []
    
    for i in range(50, n):
        # FIND LAST 2 PIVOT LOWS
        p_lows = []
        for j in range(i-3, max(0, i-50), -1):
            if not np.isnan(price_pl[j]):
                p_lows.append((j, price_pl[j]))
                if len(p_lows) >= 2: break
        
        # BULLISH DIV
        if len(p_lows) == 2:
            curr_idx, curr_val = p_lows[0]
            prev_idx, prev_val = p_lows[1]
            
            if (i - curr_idx) <= 10:
                if curr_val < prev_val and rsi[curr_idx] > rsi[prev_idx]:
                    is_new = all(s['idx'] != curr_idx or s['side'] != 'long' for s in signals)
                    if is_new:
                        swing_high = max(high[curr_idx:i+1])
                        signals.append({
                            'idx': curr_idx,
                            'conf_idx': i,
                            'side': 'long',
                            'swing': swing_high,
                            'price': curr_val
                        })

        # FIND LAST 2 PIVOT HIGHS
        p_highs = [] 
        for j in range(i-3, max(0, i-50), -1):
            if not np.isnan(price_ph[j]):
                p_highs.append((j, price_ph[j]))
                if len(p_highs) >= 2: break
        
        # BEARISH DIV
        if len(p_highs) == 2:
            curr_idx, curr_val = p_highs[0]
            prev_idx, prev_val = p_highs[1]
            
            if (i - curr_idx) <= 10:
                if curr_val > prev_val and rsi[curr_idx] < rsi[prev_idx]:
                    is_new = all(s['idx'] != curr_idx or s['side'] != 'short' for s in signals)
                    if is_new:
                        swing_low = min(low[curr_idx:i+1])
                        signals.append({
                            'idx': curr_idx,
                            'conf_idx': i,
                            'side': 'short',
                            'swing': swing_low,
                            'price': curr_val
                        })
                            
    return signals

def backtest_symbol(df, signals, rr):
    """Run backtest for a single symbol and R:R"""
    rows = list(df.itertuples())
    trades = []
    
    for sig in signals:
        start_idx = sig['conf_idx']
        side = sig['side']
        
        # Trend Filter
        curr_price = rows[start_idx].close
        ema = rows[start_idx].ema
        
        if side == 'long' and curr_price < ema: continue
        if side == 'short' and curr_price > ema: continue
        
        # Look for BOS
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
        
        if not entry_idx or entry_idx >= len(rows): continue
        
        # EXECUTE
        entry_row = rows[entry_idx]
        entry_price = entry_row.open
        atr = entry_row.atr
        sl_dist = atr * SL_MULT
        
        # Slippage
        if side == 'long':
            entry_price *= (1 + SLIPPAGE_PCT)
            tp_price = entry_price + (sl_dist * rr)
            sl_price = entry_price - sl_dist
        else:
            entry_price *= (1 - SLIPPAGE_PCT)
            tp_price = entry_price - (sl_dist * rr)
            sl_price = entry_price + sl_dist
            
        # Outcome (max 100 candles = ~8 hours)
        result = None
        
        for k in range(entry_idx, min(entry_idx + 100, len(rows))):
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
            result = -0.2  # Timeout = small loss
            
        # Fees
        risk_pct = abs(entry_price - sl_price) / entry_price
        if risk_pct == 0: risk_pct = 0.01
        
        total_fee_cost = (FEE_PCT * 2) / risk_pct
        
        final_r = result - total_fee_cost
        trades.append(final_r)
        
    return trades

def main():
    print("="*80)
    print("5M PER-SYMBOL R:R OPTIMIZATION")
    print("Testing if 4H Winners can also be 5M Winners")
    print("="*80)
    
    results = []
    output_file = '5m_optimization_results.csv'
    
    for i, sym in enumerate(SYMBOLS):
        print(f"\n[{i+1}/{len(SYMBOLS)}] Processing {sym}...")
        
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            print(f"    Fetched {len(df)} candles")
            
            if len(df) < 2000:
                print(f"    -> Insufficient data")
                continue
                
            df = prepare_data(df)
            signals = detect_divergences(df)
            
            print(f"    Detected {len(signals)} divergence signals")
            
            if len(signals) < 10:
                print(f"    -> Not enough signals")
                continue
                
            # Optimize R:R
            best_r = -999
            best_rr = 0
            best_wr = 0
            best_n = 0
            
            for rr in RR_RATIOS:
                trades = backtest_symbol(df, signals, rr)
                if not trades: continue
                
                n = len(trades)
                total_r = sum(trades)
                wr = len([t for t in trades if t > 0]) / n
                
                if total_r > best_r:
                    best_r = total_r
                    best_rr = rr
                    best_wr = wr
                    best_n = n
            
            status = "✅ WINNER" if best_r > 0 else "❌ LOSER"
            print(f"    {status}: R:R={best_rr}:1 | Total R: {best_r:+.1f} | WR: {best_wr*100:.0f}% | N: {best_n}")
            
            results.append({
                'symbol': sym,
                'best_rr': best_rr,
                'total_r': round(best_r, 2),
                'win_rate': round(best_wr, 3),
                'trades': best_n,
                'avg_r': round(best_r / best_n, 3) if best_n > 0 else 0,
                'profitable': best_r > 0
            })
                
        except Exception as e:
            print(f"    -> Error: {e}")
            
    # Save results
    df_res = pd.DataFrame(results)
    df_res.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    winners = df_res[df_res['profitable'] == True]
    losers = df_res[df_res['profitable'] == False]
    
    print(f"\n🏆 PROFITABLE ON 5M: {len(winners)}/{len(df_res)}")
    if len(winners) > 0:
        print(winners[['symbol', 'best_rr', 'total_r', 'win_rate', 'trades']].to_string(index=False))
    
    print(f"\n❌ NOT PROFITABLE ON 5M: {len(losers)}")
    if len(losers) > 0:
        print(losers[['symbol', 'best_rr', 'total_r']].to_string(index=False))
    
    print(f"\n📊 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
