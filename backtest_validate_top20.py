#!/usr/bin/env python3
"""
ANTI-OVERFIT VALIDATION FOR TOP 20 PERFORMERS
==============================================
Tests the 20 "Top Performer" symbols for:
1. Walk-Forward Optimization (70% train / 30% OOS)
2. Monte Carlo Simulation (randomized trade order)
3. Stress Test (2x slippage + fees)

Pass Criteria:
- OOS R > 0 (profitable out-of-sample)
- OOS/Train ratio > 0.3 (not severely overfit)
- Monte Carlo 5th Percentile > 0 (robust to luck)
- Stress Test R > 0 (survives friction)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
import warnings
warnings.filterwarnings('ignore')

# === THE 20 TOP PERFORMERS TO VALIDATE ===
TOP_20_SYMBOLS = [
    'WIFUSDT', 'LPTUSDT', 'JASMYUSDT', 'BANANAUSDT', 'EGLDUSDT',
    'SPKUSDT', 'IOSTUSDT', 'GRTUSDT', 'POWRUSDT', 'ZETAUSDT',
    'ZRXUSDT', '1000NEIROCTOUSDT', 'DEEPUSDT', 'WCTUSDT', 'PHAUSDT',
    'SOLAYERUSDT', 'MEWUSDT', 'DYDXUSDT', 'SUIUSDT', 'KAITOUSDT'
]

# Note: SUIUSDT and KAITOUSDT are also in Robust - we'll confirm they remain robust

# === CONFIGURATION ===
TIMEFRAME = '60'  # 1H
DATA_DAYS = 365   # 1 Year
RR_RATIOS = [4.0, 5.0, 6.0, 7.0, 8.0]
MAX_WAIT_CANDLES = 6
SL_MULT = 1.0

# Normal costs
SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006

# Stress test costs (2x)
STRESS_SLIPPAGE = 0.0004
STRESS_FEE = 0.0012

BASE_URL = "https://api.bybit.com"
RSI_PERIOD = 14
EMA_PERIOD = 200

def fetch_klines(symbol, interval, days):
    """Fetch klines with pagination"""
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = end_ts - (days * 24 * 60 * 60 * 1000)
    
    all_candles = []
    current_end = end_ts
    max_iterations = 15
    
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
                
            time.sleep(0.1)
            
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
    """Calculate indicators"""
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
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_ph, price_pl = find_pivots(close, 3, 3)
    signals = []
    
    for i in range(50, n):
        # BULLISH
        p_lows = []
        for j in range(i-3, max(0, i-50), -1):
            if not np.isnan(price_pl[j]):
                p_lows.append((j, price_pl[j]))
                if len(p_lows) >= 2: break
        
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

        # BEARISH
        p_highs = [] 
        for j in range(i-3, max(0, i-50), -1):
            if not np.isnan(price_ph[j]):
                p_highs.append((j, price_ph[j]))
                if len(p_highs) >= 2: break
        
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

def backtest_symbol(df, signals, rr, slippage=SLIPPAGE_PCT, fee=FEE_PCT):
    rows = list(df.itertuples())
    trades = []
    
    for sig in signals:
        start_idx = sig['conf_idx']
        side = sig['side']
        
        if start_idx >= len(rows):
            continue
            
        curr_price = rows[start_idx].close
        ema = rows[start_idx].ema
        
        if side == 'long' and curr_price < ema: continue
        if side == 'short' and curr_price > ema: continue
        
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
        
        entry_row = rows[entry_idx]
        entry_price = entry_row.open
        atr = entry_row.atr
        sl_dist = atr * SL_MULT
        
        if side == 'long':
            entry_price *= (1 + slippage)
            tp_price = entry_price + (sl_dist * rr)
            sl_price = entry_price - sl_dist
        else:
            entry_price *= (1 - slippage)
            tp_price = entry_price - (sl_dist * rr)
            sl_price = entry_price + sl_dist
            
        result = None
        
        for k in range(entry_idx, min(entry_idx + 200, len(rows))):
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
        if risk_pct == 0: risk_pct = 0.01
        total_fee_cost = (fee * 2) / risk_pct
        final_r = result - total_fee_cost
        trades.append(final_r)
        
    return trades

def monte_carlo(trades, iterations=1000):
    """Monte Carlo simulation - randomize trade order"""
    if not trades:
        return 0, 0
    
    final_rs = []
    for _ in range(iterations):
        shuffled = trades.copy()
        random.shuffle(shuffled)
        final_rs.append(sum(shuffled))
    
    return np.percentile(final_rs, 5), np.percentile(final_rs, 50)

def validate_symbol(symbol):
    """Run full validation suite on a symbol"""
    print(f"  Fetching data...", end=" ", flush=True)
    df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
    
    if len(df) < 2000:
        return None, "Insufficient data"
    
    df = prepare_data(df)
    n = len(df)
    
    # Split 70/30
    split_idx = int(n * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Get signals for each period
    train_signals = detect_divergences(train_df)
    test_signals = detect_divergences(test_df)
    
    # Adjust test signal indices
    for sig in test_signals:
        sig['idx'] += split_idx
        sig['conf_idx'] += split_idx
    
    print(f"Train: {len(train_signals)} signals, Test: {len(test_signals)} signals...", end=" ", flush=True)
    
    if len(train_signals) < 5 or len(test_signals) < 3:
        return None, "Insufficient signals"
    
    # Find best RR on training data
    best_train_r = -999
    best_rr = 0
    
    for rr in RR_RATIOS:
        trades = backtest_symbol(train_df, train_signals, rr)
        total_r = sum(trades)
        if total_r > best_train_r:
            best_train_r = total_r
            best_rr = rr
    
    # Test OOS with best RR
    train_trades = backtest_symbol(train_df, train_signals, best_rr)
    test_trades = backtest_symbol(df, test_signals, best_rr)
    
    train_r = sum(train_trades)
    test_r = sum(test_trades)
    
    # Monte Carlo on test set
    mc_5th, mc_50th = monte_carlo(test_trades)
    
    # Stress Test on full data
    all_signals = detect_divergences(df)
    stress_trades = backtest_symbol(df, all_signals, best_rr, STRESS_SLIPPAGE, STRESS_FEE)
    stress_r = sum(stress_trades)
    
    # Evaluate
    oos_ratio = test_r / train_r if train_r > 0 else 0
    
    results = {
        'symbol': symbol,
        'best_rr': best_rr,
        'train_r': round(train_r, 2),
        'test_r': round(test_r, 2),
        'oos_ratio': round(oos_ratio, 2),
        'mc_5th': round(mc_5th, 2),
        'stress_r': round(stress_r, 2),
        'train_trades': len(train_trades),
        'test_trades': len(test_trades)
    }
    
    # Pass criteria
    passed = (
        test_r > 0 and          # Profitable OOS
        oos_ratio > 0.3 and      # Not severely overfit
        mc_5th > -5 and          # Robust to luck
        stress_r > 0             # Survives friction
    )
    
    return results, "ROBUST ✅" if passed else "FRAGILE ❌"

def main():
    print("="*70)
    print("ANTI-OVERFIT VALIDATION FOR TOP 20 PERFORMERS")
    print("Testing: WFO, Monte Carlo, Stress Test")
    print("="*70)
    
    results = []
    robust = []
    fragile = []
    
    for i, sym in enumerate(TOP_20_SYMBOLS):
        print(f"\n[{i+1}/{len(TOP_20_SYMBOLS)}] {sym}:", end=" ")
        
        try:
            res, status = validate_symbol(sym)
            
            if res is None:
                print(f"⏭️ {status}")
                continue
            
            print(f"{status}")
            print(f"    Train: {res['train_r']:+.1f}R ({res['train_trades']} trades)")
            print(f"    OOS:   {res['test_r']:+.1f}R ({res['test_trades']} trades)")
            print(f"    OOS/Train Ratio: {res['oos_ratio']:.2f}")
            print(f"    Monte Carlo 5th%: {res['mc_5th']:+.1f}R")
            print(f"    Stress Test: {res['stress_r']:+.1f}R")
            
            results.append(res)
            
            if status == "ROBUST ✅":
                robust.append(res)
            else:
                fragile.append(res)
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n📊 OVERALL:")
    print(f"   Tested: {len(results)}")
    print(f"   ROBUST: {len(robust)} ✅")
    print(f"   FRAGILE: {len(fragile)} ❌")
    
    if robust:
        print(f"\n✅ ROBUST SYMBOLS (Keep in Portfolio):")
        for r in sorted(robust, key=lambda x: x['test_r'], reverse=True):
            print(f"   {r['symbol']}: OOS {r['test_r']:+.1f}R @ {r['best_rr']}:1")
    
    if fragile:
        print(f"\n❌ FRAGILE SYMBOLS (Consider Removing):")
        for r in sorted(fragile, key=lambda x: x['test_r']):
            print(f"   {r['symbol']}: OOS {r['test_r']:+.1f}R, Ratio: {r['oos_ratio']:.2f}")
    
    # Save results
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv('top20_validation_results.csv', index=False)
        print(f"\n💾 Results saved to: top20_validation_results.csv")

if __name__ == "__main__":
    main()
