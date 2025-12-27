#!/usr/bin/env python3
"""
FULL MARKET R:R OPTIMIZATION (400+ SYMBOLS)
===========================================
1. Fetch ALL USDT Perpetual symbols from Bybit.
2. Filter by minimum liquidity (Turnover > $5M).
3. Backtest 4H Divergence + BOS Strategy on each.
4. Optimize R:R for each symbol individually.
5. Save results to CSV and generate config.
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
TIMEFRAME = '240'  # 4H
DATA_DAYS = 365    # 1 Year of data
MIN_TURNOVER = 5_000_000  # $5M daily volume min
MAX_WAIT_CANDLES = 6
SL_MULT = 1.0

# RR Ratios to test
RR_RATIOS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]

# Optimistic Costs (Limit Orders)
SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006

BASE_URL = "https://api.bybit.com"

# === INDICATOR LOGIC ===
RSI_PERIOD = 14
LOOKBACK_BARS = 10
MIN_PIVOT_DISTANCE = 3
PIVOT_RIGHT = 3

def get_all_usdt_symbols():
    """Fetch all USDT perpetuals sorted by volume"""
    try:
        print("Fetching symbol list from Bybit...")
        resp = requests.get(f"{BASE_URL}/v5/market/tickers?category=linear", timeout=10)
        tickers = resp.json().get('result', {}).get('list', [])
        
        usdt = []
        for t in tickers:
            if t['symbol'].endswith('USDT'):
                turnover = float(t.get('turnover24h', 0))
                if turnover >= MIN_TURNOVER:
                    usdt.append(t)
        
        usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
        symbols = [t['symbol'] for t in usdt]
        print(f"Found {len(symbols)} liquid USDT pairs (Turnover > ${MIN_TURNOVER/1e6}M)")
        return symbols
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return []

def fetch_klines(symbol, interval, days):
    """Fetch 4H klines with retry logic"""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_end = end_ts
    
    # We need ~1 year of 4H candles = 365 * 6 = 2190 candles
    # Bybit limit is 1000. So ~3 calls.
    
    retries = 3
    while current_end > start_ts and retries > 0:
        params = {
            'category': 'linear', 
            'symbol': symbol, 
            'interval': interval, 
            'limit': 1000, 
            'end': current_end
        }
        
        try:
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=5)
            data = resp.json().get('result', {}).get('list', [])
            
            if not data:
                break
                
            all_candles.extend(data)
            oldest = int(data[-1][0])
            current_end = oldest - 1
            
            if len(data) < 1000:
                break
                
            time.sleep(0.1)  # Rate limit nice-ness
            
        except:
            retries -= 1
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
    
    # Remove duplicates if any
    df = df[~df.index.duplicated(keep='first')]
    
    return df

def prepare_data(df):
    """Calculate indicators"""
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
    
    # Daily EMA (approx 6 * 200 = 1200 bars on 4H)
    df['ema_daily'] = df['close'].ewm(span=1200, adjust=False).mean()
    
    return df.dropna()

def find_pivots(data, left=3, right=3):
    """Find pivots without look-ahead (confirmed at i+right)"""
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    # We iterate such that at index i, we check if i-right was a pivot
    # But effectively we just scan all valid windows
    for i in range(left, n - right):
        window = data[i-left : i+right+1]
        center = data[i]
        
        if len(window) != (left + right + 1): continue
        
        # Check High
        if center == max(window) and list(window).count(center) == 1:
            pivot_highs[i] = center
            
        # Check Low
        if center == min(window) and list(window).count(center) == 1:
            pivot_lows[i] = center
            
    return pivot_highs, pivot_lows

def detect_divergences(df):
    """Detect RSI divergences using pivot logic"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    # Note: Pivot at index i is ONLY known at i+3.
    # So we can only use pivots where i+3 <= current_candle
    
    signals = []
    
    # Scan history
    for i in range(50, n):
        # Current Candle = i
        # We look for pivots that confirmed BEFORE or AT i
        # Latest confirmed pivot index <= i - 3
        
        # FIND LAST 2 PIVOT LOWS
        p_lows = [] # (index, value)
        for j in range(i-3, max(0, i-50), -1):
            if not np.isnan(price_pl[j]):
                p_lows.append((j, price_pl[j]))
                if len(p_lows) >= 2: break
        
        # BULLISH DIV
        if len(p_lows) == 2:
            curr_idx, curr_val = p_lows[0]
            prev_idx, prev_val = p_lows[1]
            
            # Check proximity of current pivot to now (must be fresh)
            if (i - curr_idx) <= 10:
                if curr_val < prev_val: # Lower Low Price
                    if rsi[curr_idx] > rsi[prev_idx]: # Higher Low RSI
                        # Overlap check
                        is_new = True
                        for s in signals:
                            if s['idx'] == curr_idx and s['side'] == 'long':
                                is_new = False
                        if is_new:
                            swing_high = max(high[curr_idx:i+1])
                            signals.append({
                                'idx': curr_idx, # Pivot index
                                'conf_idx': i,   # Detection index
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
                if curr_val > prev_val: # Higher High Price
                    if rsi[curr_idx] < rsi[prev_idx]: # Lower High RSI
                         # Overlap check
                        is_new = True
                        for s in signals:
                            if s['idx'] == curr_idx and s['side'] == 'short':
                                is_new = False
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
        # Signal discovered at conf_idx
        start_idx = sig['conf_idx']
        side = sig['side']
        
        # Trend Filter check at discovery
        curr_price = rows[start_idx].close
        ema = rows[start_idx].ema_daily
        
        if side == 'long' and curr_price < ema: continue
        if side == 'short' and curr_price > ema: continue
        
        # Look for BOS
        entry_idx = None
        
        # Look ahead up to MAX_WAIT_CANDLES
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
            
        # Outcome
        result = None
        exit_bar = None
        
        for k in range(entry_idx, min(entry_idx + 300, len(rows))):
            row = rows[k]
            if side == 'long':
                if row.low <= sl_price:
                    result = -1.0
                    exit_bar = k
                    break
                if row.high >= tp_price:
                    result = rr
                    exit_bar = k
                    break
            else:
                if row.high >= sl_price:
                    result = -1.0
                    exit_bar = k
                    break
                if row.low <= tp_price:
                    result = rr
                    exit_bar = k
                    break
                    
        # Force close
        if result is None:
            # Mark as break-even/small loss for timeout
            result = -0.1
            
        # Fees
        # 2x Fee + Slippage cost relative to SL distance
        risk_pct = abs(entry_price - sl_price) / entry_price
        if risk_pct == 0: risk_pct = 0.01
        
        total_fee_cost = (FEE_PCT * 2) / risk_pct
        
        final_r = result - total_fee_cost
        trades.append(final_r)
        
    return trades

def main():
    print("="*80)
    print("MEGA-OPTIMIZATION: 400+ SYMBOLS")
    print("="*80)
    
    symbols = get_all_usdt_symbols()
    
    # Resume support
    processed_symbols = set()
    results = []
    
    if os.path.exists('optimization_results.csv'):
        try:
            existing = pd.read_csv('optimization_results.csv')
            processed_symbols = set(existing['symbol'].unique())
            print(f"Resuming... {len(processed_symbols)} symbols already done.")
        except:
            pass
            
    print(f"Starting processing of {len(symbols)} symbols...")
    
    batch_size = 10
    
    for i, sym in enumerate(symbols):
        if sym in processed_symbols:
            continue
            
        print(f"Processing {sym} ({i+1}/{len(symbols)})...")
        
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if len(df) < 500:
                print(f"  -> Insufficient data ({len(df)} candles)")
                continue
                
            df = prepare_data(df)
            signals = detect_divergences(df)
            
            if len(signals) < 5:
                print(f"  -> Not enough signals ({len(signals)})")
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
            
            if best_rr > 0:
                print(f"  -> Best: {best_rr}:1 | R: {best_r:.1f} | WR: {best_wr*100:.1f}% | N: {best_n}")
                
                # Append to results
                res = {
                    'symbol': sym,
                    'best_rr': best_rr,
                    'total_r': round(best_r, 2),
                    'win_rate': round(best_wr, 3),
                    'trades': best_n,
                    'avg_r': round(best_r / best_n, 3)
                }
                
                # Save immediately to CSV (append mode)
                df_res = pd.DataFrame([res])
                hdr = not os.path.exists('optimization_results.csv')
                df_res.to_csv('optimization_results.csv', mode='a', header=hdr, index=False)
                
        except Exception as e:
            print(f"  -> Error: {e}")
            
    print("\nOptimization Complete!")
    
    # Now Generate Config
    print("Generating updated configuration...")
    try:
        final_df = pd.read_csv('optimization_results.csv')
        
        # Filter Logic
        # 1. Total R > 10R per year (modest)
        # 2. Trades > 12 (at least 1 per month)
        # 3. Avg R > 0.20
        
        passing = final_df[
            (final_df['total_r'] >= 10.0) & 
            (final_df['trades'] >= 12) & 
            (final_df['avg_r'] >= 0.20)
        ].copy()
        
        passing = passing.sort_values(by='total_r', ascending=False)
        
        print(f"Qualified Symbols: {len(passing)}")
        print(passing[['symbol', 'best_rr', 'total_r', 'trades']].head(10))
        
        # Write to symbol_rr_config.yaml
        config_data = {}
        for _, row in passing.iterrows():
            config_data[row['symbol']] = {
                'rr': float(row['best_rr']),
                'enabled': True
            }
            
        with open('symbol_rr_config.yaml', 'w') as f:
            yaml.dump(config_data, f, default_flow_style=None)
            
        print("✅ Updated symbol_rr_config.yaml")

        # Update bot config.yaml (optional, but good to keep in sync)
        # Check current config
        with open('config.yaml', 'r') as f:
            full_config = yaml.safe_load(f)
            
        # Update strategy symbols list
        full_config['symbols'] = config_data
        
        with open('config.yaml', 'w') as f:
            yaml.dump(full_config, f, default_flow_style=None)
            
        print("✅ Updated config.yaml")
        
    except Exception as e:
        print(f"Error generating config: {e}")

if __name__ == "__main__":
    main()
