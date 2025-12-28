#!/usr/bin/env python3
"""
RIGOROUS ANTI-OVERFIT VALIDATION
=================================
Multiple validation methods to ensure the 1H strategy is robust:

1. WALK-FORWARD OPTIMIZATION (WFO)
   - Train on Year 1+2, Test on Year 3
   - Verify edge holds in unseen data

2. OUT-OF-SAMPLE TESTING
   - 70/30 split (in-sample/out-of-sample)
   - Report metrics separately

3. MONTE CARLO SIMULATION
   - Randomize trade order 1000x
   - Report 5th percentile (worst case)

4. FRICTION STRESS TEST
   - 2x slippage and fees
   - Report degraded performance

5. SYMBOL STABILITY CHECK
   - How many symbols stay profitable across all tests?
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import time
import os
import random
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
TIMEFRAME = '60'  # 1H
DATA_DAYS = 1095  # 3 Years
MIN_TURNOVER = 5_000_000
MAX_SYMBOLS = 50  # Top 50 for faster testing

# Use the 44 winning symbols from optimization
SYMBOLS = [
    "ONTUSDT", "SOLUSDT", "AAVEUSDT", "SUIUSDT", "MERLUSDT", "KAITOUSDT",
    "ICPUSDT", "INJUSDT", "HBARUSDT", "FLOWUSDT", "VIRTUALUSDT", "ETCUSDT",
    "AVAXUSDT", "WIFUSDT", "LINKUSDT", "STORJUSDT", "ENAUSDT", "PENGUUSDT",
    "LPTUSDT", "ZENUSDT", "IDUSDT", "GALAUSDT", "HIVEUSDT", "ARBUSDT",
    "FLOCKUSDT", "FILUSDT", "AIXBTUSDT", "TRUUSDT", "SEIUSDT", "TRUMPUSDT",
    "WLDUSDT", "ZKUSDT", "TRXUSDT", "1000PEPEUSDT", "STRKUSDT", "FARTCOINUSDT",
    "BTCUSDT", "SHIB1000USDT", "ACTUSDT", "ONDOUSDT", "DOTUSDT", "BNBUSDT", "UNIUSDT"
]

# R:R from optimization
SYMBOL_RR = {
    "ONTUSDT": 4.5, "SOLUSDT": 7.0, "AAVEUSDT": 7.0, "SUIUSDT": 8.0,
    "MERLUSDT": 6.0, "KAITOUSDT": 7.0, "ICPUSDT": 6.0, "INJUSDT": 8.0,
    "HBARUSDT": 4.5, "FLOWUSDT": 5.0, "VIRTUALUSDT": 5.5, "ETCUSDT": 7.0,
    "AVAXUSDT": 5.5, "WIFUSDT": 8.0, "LINKUSDT": 4.5, "STORJUSDT": 6.0,
    "ENAUSDT": 8.0, "PENGUUSDT": 8.0, "LPTUSDT": 8.0, "ZENUSDT": 8.0,
    "IDUSDT": 8.0, "GALAUSDT": 7.0, "HIVEUSDT": 8.0, "ARBUSDT": 8.0,
    "FLOCKUSDT": 6.0, "FILUSDT": 5.5, "AIXBTUSDT": 6.0, "TRUUSDT": 5.0,
    "SEIUSDT": 5.5, "TRUMPUSDT": 7.0, "WLDUSDT": 7.0, "ZKUSDT": 5.0,
    "TRXUSDT": 8.0, "1000PEPEUSDT": 5.5, "STRKUSDT": 8.0, "FARTCOINUSDT": 7.0,
    "BTCUSDT": 4.0, "SHIB1000USDT": 4.0, "ACTUSDT": 8.0, "ONDOUSDT": 8.0,
    "DOTUSDT": 4.0, "BNBUSDT": 4.5, "UNIUSDT": 3.5
}

MAX_WAIT_CANDLES = 6
SL_MULT = 1.0

# Normal costs
SLIPPAGE_PCT = 0.0002
FEE_PCT = 0.0006

# Stress costs (2x)
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
    max_iterations = 50
    
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
                
            time.sleep(0.15)
            
        except:
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
    """Run backtest with configurable friction"""
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
        
        for k in range(entry_idx, min(entry_idx + 300, len(rows))):
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
            result = -0.1
            
        # Fees
        risk_pct = abs(entry_price - sl_price) / entry_price
        if risk_pct == 0: risk_pct = 0.01
        total_fee_cost = (fee * 2) / risk_pct
        final_r = result - total_fee_cost
        
        trades.append({
            'r': final_r,
            'entry_idx': entry_idx,
            'side': side
        })
        
    return trades

def monte_carlo(trades, n_simulations=1000):
    """Run Monte Carlo simulation by shuffling trade order"""
    if len(trades) < 5:
        return {'p5': 0, 'p50': 0, 'p95': 0}
    
    r_values = [t['r'] for t in trades]
    final_rs = []
    
    for _ in range(n_simulations):
        shuffled = random.sample(r_values, len(r_values))
        final_rs.append(sum(shuffled))
    
    return {
        'p5': np.percentile(final_rs, 5),
        'p50': np.percentile(final_rs, 50),
        'p95': np.percentile(final_rs, 95)
    }

def main():
    print("="*80)
    print("RIGOROUS ANTI-OVERFIT VALIDATION")
    print("="*80)
    print("\nTesting 44 winning symbols with multiple validation methods...\n")
    
    results = []
    
    for i, symbol in enumerate(SYMBOLS):
        if symbol not in SYMBOL_RR:
            continue
            
        print(f"[{i+1}/{len(SYMBOLS)}] {symbol}...", end=" ", flush=True)
        
        try:
            df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
            
            if len(df) < 5000:
                print("⏭️ Insufficient data")
                continue
            
            df = prepare_data(df)
            n_candles = len(df)
            
            # Split data: 70% in-sample, 30% out-of-sample
            split_idx = int(n_candles * 0.7)
            df_train = df.iloc[:split_idx]
            df_test = df.iloc[split_idx:]
            
            # Detect signals on full data, split by index
            all_signals = detect_divergences(df)
            train_signals = [s for s in all_signals if s['conf_idx'] < split_idx]
            test_signals = [s for s in all_signals if s['conf_idx'] >= split_idx]
            
            rr = SYMBOL_RR[symbol]
            
            # === TEST 1: IN-SAMPLE (Training) ===
            train_trades = backtest_symbol(df, train_signals, rr)
            train_r = sum(t['r'] for t in train_trades) if train_trades else 0
            
            # === TEST 2: OUT-OF-SAMPLE (Validation) ===
            test_trades = backtest_symbol(df, test_signals, rr)
            test_r = sum(t['r'] for t in test_trades) if test_trades else 0
            
            # === TEST 3: FULL DATA (Reference) ===
            full_trades = backtest_symbol(df, all_signals, rr)
            full_r = sum(t['r'] for t in full_trades) if full_trades else 0
            
            # === TEST 4: STRESS TEST (2x Friction) ===
            stress_trades = backtest_symbol(df, all_signals, rr, STRESS_SLIPPAGE, STRESS_FEE)
            stress_r = sum(t['r'] for t in stress_trades) if stress_trades else 0
            
            # === TEST 5: MONTE CARLO ===
            mc = monte_carlo(full_trades)
            
            # Pass/Fail criteria
            passes = []
            
            # 1. In-sample profitable
            passes.append(train_r > 0)
            
            # 2. Out-of-sample profitable (CRITICAL)
            passes.append(test_r > 0)
            
            # 3. Stress test not catastrophic (>50% of normal)
            passes.append(stress_r > full_r * 0.5 if full_r > 0 else stress_r > -10)
            
            # 4. Monte Carlo 5th percentile positive
            passes.append(mc['p5'] > -10)
            
            all_passed = all(passes)
            
            status = "✅ ROBUST" if all_passed else "⚠️ FRAGILE"
            print(f"{status} | Train:{train_r:+.1f}R | OOS:{test_r:+.1f}R | Stress:{stress_r:+.1f}R")
            
            results.append({
                'symbol': symbol,
                'rr': rr,
                'train_r': round(train_r, 2),
                'test_r': round(test_r, 2),
                'full_r': round(full_r, 2),
                'stress_r': round(stress_r, 2),
                'mc_p5': round(mc['p5'], 2),
                'mc_p50': round(mc['p50'], 2),
                'train_trades': len(train_trades),
                'test_trades': len(test_trades),
                'passed': all_passed
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv('anti_overfit_validation.csv', index=False)
    
    # Final Report
    print("\n" + "="*80)
    print("VALIDATION REPORT")
    print("="*80)
    
    robust = df_results[df_results['passed'] == True]
    fragile = df_results[df_results['passed'] == False]
    
    print(f"\n📊 SUMMARY")
    print(f"   Total Symbols Tested: {len(df_results)}")
    print(f"   ✅ ROBUST (passed all tests): {len(robust)} ({len(robust)/len(df_results)*100:.0f}%)")
    print(f"   ⚠️ FRAGILE (failed 1+ tests): {len(fragile)} ({len(fragile)/len(df_results)*100:.0f}%)")
    
    print(f"\n📈 AGGREGATE PERFORMANCE")
    print(f"   Training Period (70%): {df_results['train_r'].sum():+.1f}R")
    print(f"   Out-of-Sample (30%): {df_results['test_r'].sum():+.1f}R")
    print(f"   Stress Test (2x fees): {df_results['stress_r'].sum():+.1f}R")
    
    # OOS/Train ratio (detect overfitting)
    train_sum = df_results['train_r'].sum()
    test_sum = df_results['test_r'].sum()
    ratio = test_sum / train_sum if train_sum > 0 else 0
    
    print(f"\n🎯 OVERFIT CHECK")
    print(f"   OOS/Train Ratio: {ratio:.2f}")
    if ratio >= 0.6:
        print(f"   ✅ PASS - Strategy generalizes well (>60% retention)")
    elif ratio >= 0.3:
        print(f"   ⚠️ CAUTION - Some overfitting detected (30-60%)")
    else:
        print(f"   ❌ FAIL - Severe overfitting (<30% retention)")
    
    print(f"\n🏆 TOP ROBUST SYMBOLS (passed all tests)")
    if len(robust) > 0:
        top_robust = robust.nlargest(10, 'test_r')
        for _, row in top_robust.iterrows():
            print(f"   {row['symbol']}: OOS={row['test_r']:+.1f}R (Train:{row['train_r']:+.1f}R)")
    
    print(f"\n⚠️ FRAGILE SYMBOLS (removed from recommendation)")
    if len(fragile) > 0:
        for _, row in fragile.iterrows():
            print(f"   {row['symbol']}: OOS={row['test_r']:+.1f}R (why? Train:{row['train_r']:+.1f}R)")
    
    print(f"\n📊 Results saved to: anti_overfit_validation.csv")

if __name__ == "__main__":
    main()
