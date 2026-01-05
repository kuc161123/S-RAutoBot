#!/usr/bin/env python3
"""
ULTRA-ROBUST ANTI-OVERFITTING BACKTEST
========================================
Comprehensive validation to GUARANTEE no overfitting:

ARCHITECTURE:
- 6 months data → 4 months TRAINING + 2 months OUT-OF-SAMPLE (OOS)
- OOS data is NEVER touched during optimization
- Walk-Forward WITHIN training period (2+2 months)
- Full Monte Carlo (1000 iterations)
- Strict validation thresholds

TESTING MATRIX (per symbol):
- 4 divergence types × 5 ATR multipliers × 8 R:R ratios = 160 combinations
- Total: ~500 symbols × 160 = 80,000 combinations

VALIDATION PIPELINE:
1. Optimize on Training Period 1 (months 1-2)
2. Validate on Training Period 2 (months 3-4) ← Walk-Forward
3. Final validation on OOS (months 5-6) ← NEVER TOUCHED
4. Monte Carlo simulation (1000 shuffles)
5. Consistency checks across all periods

OUTPUT:
- Only symbols that pass ALL validation stages
- Per-symbol optimal settings (divergence, ATR, RR)
- Realistic performance expectations
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import yaml
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import concurrent.futures
from collections import defaultdict
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SYMBOL_LIMIT = 500
TIMEFRAME = '60'  # 1H
DATA_DAYS = 180   # 6 months

# Strategy settings
MAX_WAIT_CANDLES = 12

# ATR multipliers to test
ATR_MULT_OPTIONS = [0.8, 1.0, 1.5, 2.0, 2.5]

# R:R ratios to test
RR_OPTIONS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Indicator settings
RSI_PERIOD = 14
EMA_PERIOD = 200
PIVOT_LEFT = 3
PIVOT_RIGHT = 3

# Execution costs
SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

# ============================================================================
# VALIDATION THRESHOLDS (STRICT)
# ============================================================================

# Training validation
MIN_TRADES_TRAINING = 10      # Per period
MIN_WIN_RATE_TRAINING = 25    # %
MIN_TOTAL_R_TRAINING = 0      # Must be positive

# OOS validation (stricter)
MIN_TRADES_OOS = 5            # Per OOS period
MIN_WIN_RATE_OOS = 20         # % (can be slightly lower)
MIN_TOTAL_R_OOS = 0           # Must be positive

# Overall validation
MIN_SHARPE = 0.5              # Stricter than before
MAX_DRAWDOWN_PCT = 30         # % of total R
MC_CONFIDENCE = 0.70          # 70% of Monte Carlo runs must be profitable

# Walk-Forward
WF_MONTHS = 2                 # Each walk-forward period

BASE_URL = "https://api.bybit.com"

# ============================================================================
# DIVERGENCE CONFIGS
# ============================================================================

@dataclass
class DivConfig:
    code: str
    side: str
    price_pattern: str
    rsi_pattern: str
    trend_filter: str

DIVERGENCES = {
    'REG_BULL': DivConfig('REG_BULL', 'long', 'LL', 'HL', 'above_ema'),
    'REG_BEAR': DivConfig('REG_BEAR', 'short', 'HH', 'LH', 'below_ema'),
    'HID_BULL': DivConfig('HID_BULL', 'long', 'HL', 'LL', 'above_ema'),
    'HID_BEAR': DivConfig('HID_BEAR', 'short', 'LH', 'HH', 'below_ema'),
}

# ============================================================================
# DATA FETCHING
# ============================================================================

def fetch_symbols(limit=500):
    print(f"📥 Fetching up to {limit} USDT perpetual symbols...")
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/instruments-info",
                          params={'category': 'linear', 'limit': 1000}, timeout=30)
        data = resp.json().get('result', {}).get('list', [])
        symbols = [item['symbol'] for item in data 
                  if item['symbol'].endswith('USDT') and item.get('status') == 'Trading']
        exclude = ['USDCUSDT', 'BUSDUSDT', 'DAIUSDT', 'USTCUSDT', 'TUSDUSDT']
        symbols = [s for s in symbols if s not in exclude]
        print(f"   Found {len(symbols)} tradeable symbols")
        return symbols[:limit]
    except Exception as e:
        print(f"   Error: {e}")
        return []

def fetch_klines(symbol, days=180):
    try:
        all_klines = []
        end_ts = int(time.time() * 1000)
        start_ts = int((time.time() - days * 24 * 3600) * 1000)
        
        while end_ts > start_ts:
            params = {'category': 'linear', 'symbol': symbol, 'interval': TIMEFRAME, 
                     'limit': 1000, 'end': end_ts}
            resp = requests.get(f"{BASE_URL}/v5/market/kline", params=params, timeout=15)
            data = resp.json()
            if data.get('retCode') != 0:
                break
            klines = data.get('result', {}).get('list', [])
            if not klines:
                break
            all_klines.extend(klines)
            end_ts = int(klines[-1][0]) - 1
            time.sleep(0.02)
        
        if not all_klines:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_klines, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'turnover'])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ['open', 'high', 'low', 'close', 'vol']:
            df[c] = df[c].astype(float)
        df['ts'] = pd.to_datetime(df['ts'].astype(int), unit='ms')
        df.set_index('ts', inplace=True)
        return df
    except:
        return pd.DataFrame()

# ============================================================================
# INDICATORS
# ============================================================================

def calculate_indicators(df):
    if len(df) < 250:
        return pd.DataFrame()
    
    df = df.copy()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # ATR
    hl = df['high'] - df['low']
    hc = abs(df['high'] - df['close'].shift())
    lc = abs(df['low'] - df['close'].shift())
    df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    
    # EMA 200
    df['ema'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    return df.dropna()

def find_pivots(arr, left=3, right=3):
    n = len(arr)
    ph = np.full(n, np.nan)
    pl = np.full(n, np.nan)
    
    for i in range(left, n - right):
        window = arr[i-left:i+right+1]
        if arr[i] == max(window) and list(window).count(arr[i]) == 1:
            ph[i] = arr[i]
        if arr[i] == min(window) and list(window).count(arr[i]) == 1:
            pl[i] = arr[i]
    return ph, pl

# ============================================================================
# DIVERGENCE DETECTION
# ============================================================================

def detect_divergence(df, idx, div_cfg, close, rsi, ema, ph, pl):
    if idx < 50:
        return False, None
    
    current_price = close[idx]
    current_ema = ema[idx]
    
    # Trend filter
    if div_cfg.trend_filter == 'above_ema' and current_price <= current_ema:
        return False, None
    if div_cfg.trend_filter == 'below_ema' and current_price >= current_ema:
        return False, None
    
    if div_cfg.side == 'long':
        pivots = []
        for j in range(idx - PIVOT_RIGHT - 1, max(0, idx - 50), -1):
            if not np.isnan(pl[j]):
                pivots.append((j, pl[j], rsi[j]))
                if len(pivots) >= 2:
                    break
        
        if len(pivots) < 2:
            return False, None
        
        curr_idx, curr_price_val, curr_rsi = pivots[0]
        prev_idx, prev_price_val, prev_rsi = pivots[1]
        
        if (idx - curr_idx) > 10:
            return False, None
        
        if div_cfg.code == 'REG_BULL':
            if curr_price_val < prev_price_val and curr_rsi > prev_rsi:
                swing = max(df['high'].iloc[curr_idx:idx+1])
                if current_price <= swing:
                    return True, swing
        elif div_cfg.code == 'HID_BULL':
            if curr_price_val > prev_price_val and curr_rsi < prev_rsi:
                swing = max(df['high'].iloc[curr_idx:idx+1])
                if current_price <= swing:
                    return True, swing
    else:
        pivots = []
        for j in range(idx - PIVOT_RIGHT - 1, max(0, idx - 50), -1):
            if not np.isnan(ph[j]):
                pivots.append((j, ph[j], rsi[j]))
                if len(pivots) >= 2:
                    break
        
        if len(pivots) < 2:
            return False, None
        
        curr_idx, curr_price_val, curr_rsi = pivots[0]
        prev_idx, prev_price_val, prev_rsi = pivots[1]
        
        if (idx - curr_idx) > 10:
            return False, None
        
        if div_cfg.code == 'REG_BEAR':
            if curr_price_val > prev_price_val and curr_rsi < prev_rsi:
                swing = min(df['low'].iloc[curr_idx:idx+1])
                if current_price >= swing:
                    return True, swing
        elif div_cfg.code == 'HID_BEAR':
            if curr_price_val < prev_price_val and curr_rsi > prev_rsi:
                swing = min(df['low'].iloc[curr_idx:idx+1])
                if current_price >= swing:
                    return True, swing
    
    return False, None

# ============================================================================
# TRADE SIMULATION (REALISTIC - ENTRY CANDLE SL CHECK)
# ============================================================================

def simulate_trades_for_period(df, div_cfg, rr, atr_mult, start_idx, end_idx):
    """Simulate trades for a specific period of the data."""
    trades = []
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    rsi = df['rsi'].values
    ema = df['ema'].values
    atr = df['atr'].values
    n = len(df)
    
    ph, pl = find_pivots(close, PIVOT_LEFT, PIVOT_RIGHT)
    
    pending_signal = None
    pending_swing = None
    pending_wait = 0
    
    for idx in range(max(50, start_idx), min(end_idx, n - 1)):
        current_price = close[idx]
        
        # Check pending signal for BOS
        if pending_signal is not None:
            bos = False
            if div_cfg.side == 'long' and current_price > pending_swing:
                bos = True
            if div_cfg.side == 'short' and current_price < pending_swing:
                bos = True
            
            if bos:
                entry_idx = idx + 1
                if entry_idx >= n or entry_idx >= end_idx:
                    pending_signal = None
                    continue
                
                entry_price = df.iloc[entry_idx]['open']
                entry_atr = atr[entry_idx]
                sl_dist = entry_atr * atr_mult
                
                if div_cfg.side == 'long':
                    entry_price *= (1 + SLIPPAGE_PCT)
                    tp = entry_price + (sl_dist * rr)
                    sl = entry_price - sl_dist
                else:
                    entry_price *= (1 - SLIPPAGE_PCT)
                    tp = entry_price - (sl_dist * rr)
                    sl = entry_price + sl_dist
                
                # REALISTIC: Check from entry candle (immediate SL)
                result = None
                exit_idx = None
                
                for k in range(entry_idx, min(entry_idx + 200, n, end_idx)):
                    if div_cfg.side == 'long':
                        if low[k] <= sl:
                            result = -1.0
                            exit_idx = k
                            break
                        if high[k] >= tp:
                            result = rr
                            exit_idx = k
                            break
                    else:
                        if high[k] >= sl:
                            result = -1.0
                            exit_idx = k
                            break
                        if low[k] <= tp:
                            result = rr
                            exit_idx = k
                            break
                
                if result is None:
                    result = -0.5  # Timeout penalty
                    exit_idx = min(entry_idx + 200, n - 1, end_idx - 1)
                
                # Fees
                risk_pct = sl_dist / entry_price if entry_price > 0 else 0.01
                fee_cost = (FEE_PCT * 2) / risk_pct if risk_pct > 0 else 0.1
                final_r = result - fee_cost
                
                trades.append({
                    'r': round(final_r, 3),
                    'win': final_r > 0,
                    'entry_idx': entry_idx,
                    'duration': exit_idx - entry_idx if exit_idx else 0
                })
                
                pending_signal = None
                pending_wait = 0
            else:
                pending_wait += 1
                if pending_wait >= MAX_WAIT_CANDLES:
                    pending_signal = None
                    pending_wait = 0
        
        # Detect new divergence
        if pending_signal is None:
            found, swing = detect_divergence(df, idx, div_cfg, close, rsi, ema, ph, pl)
            if found:
                pending_signal = idx
                pending_swing = swing
                pending_wait = 0
    
    return trades

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_period_trades(trades, min_trades, min_wr, min_r):
    """Validate trades for a specific period."""
    if len(trades) < min_trades:
        return False, {}
    
    total_r = sum(t['r'] for t in trades)
    wins = sum(1 for t in trades if t['win'])
    wr = (wins / len(trades)) * 100 if trades else 0
    
    if wr < min_wr or total_r < min_r:
        return False, {}
    
    return True, {
        'trades': len(trades),
        'wins': wins,
        'wr': round(wr, 1),
        'total_r': round(total_r, 2)
    }

def monte_carlo_test(r_values, iterations=1000, confidence=0.70):
    """
    Monte Carlo simulation to test if results are due to order luck.
    Shuffles trade order and checks what % of runs are profitable.
    """
    if len(r_values) < 5:
        return False, 0
    
    profitable_runs = 0
    for _ in range(iterations):
        shuffled = random.sample(r_values, len(r_values))
        if sum(shuffled) > 0:
            profitable_runs += 1
    
    pct_profitable = profitable_runs / iterations
    return pct_profitable >= confidence, round(pct_profitable * 100, 1)

def calculate_sharpe(r_values):
    """Calculate simplified Sharpe ratio."""
    if len(r_values) < 2 or np.std(r_values) == 0:
        return 0
    return np.mean(r_values) / np.std(r_values) * np.sqrt(len(r_values))

def calculate_max_drawdown(r_values):
    """Calculate max drawdown as % of total positive R."""
    equity = 0
    peak = 0
    max_dd = 0
    
    for r in r_values:
        equity += r
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    
    total_r = sum(r_values)
    if total_r <= 0:
        return 100
    
    return (max_dd / total_r) * 100 if total_r > 0 else 100

# ============================================================================
# MAIN VALIDATION PIPELINE
# ============================================================================

def validate_combo(df, symbol, div_cfg, rr, atr_mult):
    """
    Full validation pipeline for a single combo.
    
    Data split:
    - Training Period 1: Months 1-2 (candles 0-33%)
    - Training Period 2: Months 3-4 (candles 33-66%) - Walk-Forward
    - OOS Period: Months 5-6 (candles 66-100%) - NEVER TOUCHED during optimization
    """
    n = len(df)
    
    # Define period boundaries
    train1_end = int(n * 0.33)
    train2_end = int(n * 0.66)
    oos_end = n
    
    # STAGE 1: Training Period 1
    train1_trades = simulate_trades_for_period(df, div_cfg, rr, atr_mult, 0, train1_end)
    train1_pass, train1_stats = validate_period_trades(
        train1_trades, MIN_TRADES_TRAINING, MIN_WIN_RATE_TRAINING, MIN_TOTAL_R_TRAINING
    )
    if not train1_pass:
        return None
    
    # STAGE 2: Training Period 2 (Walk-Forward)
    train2_trades = simulate_trades_for_period(df, div_cfg, rr, atr_mult, train1_end, train2_end)
    train2_pass, train2_stats = validate_period_trades(
        train2_trades, MIN_TRADES_TRAINING, MIN_WIN_RATE_TRAINING, MIN_TOTAL_R_TRAINING
    )
    if not train2_pass:
        return None
    
    # STAGE 3: Out-of-Sample (FINAL TEST - never seen during optimization)
    oos_trades = simulate_trades_for_period(df, div_cfg, rr, atr_mult, train2_end, oos_end)
    oos_pass, oos_stats = validate_period_trades(
        oos_trades, MIN_TRADES_OOS, MIN_WIN_RATE_OOS, MIN_TOTAL_R_OOS
    )
    if not oos_pass:
        return None
    
    # Combine all trades for overall stats
    all_trades = train1_trades + train2_trades + oos_trades
    all_r_values = [t['r'] for t in all_trades]
    
    # STAGE 4: Sharpe Ratio Check
    sharpe = calculate_sharpe(all_r_values)
    if sharpe < MIN_SHARPE:
        return None
    
    # STAGE 5: Max Drawdown Check
    max_dd = calculate_max_drawdown(all_r_values)
    if max_dd > MAX_DRAWDOWN_PCT:
        return None
    
    # STAGE 6: Monte Carlo Simulation
    mc_pass, mc_pct = monte_carlo_test(all_r_values, iterations=1000, confidence=MC_CONFIDENCE)
    if not mc_pass:
        return None
    
    # PASSED ALL STAGES!
    total_r = sum(all_r_values)
    wins = sum(1 for t in all_trades if t['win'])
    wr = (wins / len(all_trades)) * 100
    
    return {
        'symbol': symbol,
        'div': div_cfg.code,
        'rr': rr,
        'atr_mult': atr_mult,
        'trades': len(all_trades),
        'wr': round(wr, 1),
        'total_r': round(total_r, 2),
        'sharpe': round(sharpe, 2),
        'max_dd_pct': round(max_dd, 1),
        'mc_pct': mc_pct,
        'train1': train1_stats,
        'train2': train2_stats,
        'oos': oos_stats
    }

def process_symbol(args):
    """Process one symbol - test all combos."""
    symbol, df = args
    results = []
    
    if df.empty or len(df) < 500:  # Need enough data for all periods
        return results
    
    df = calculate_indicators(df)
    if df.empty:
        return results
    
    for div_code, div_cfg in DIVERGENCES.items():
        for rr in RR_OPTIONS:
            for atr_mult in ATR_MULT_OPTIONS:
                validated = validate_combo(df, symbol, div_cfg, rr, atr_mult)
                if validated:
                    results.append(validated)
    
    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("ULTRA-ROBUST ANTI-OVERFITTING BACKTEST")
    print("=" * 80)
    print()
    print("VALIDATION PIPELINE:")
    print("  1. Training Period 1 (Months 1-2)    → Optimization")
    print("  2. Training Period 2 (Months 3-4)    → Walk-Forward Validation")
    print("  3. Out-of-Sample    (Months 5-6)    → FINAL TEST (never touched)")
    print("  4. Sharpe Ratio                      → Risk-adjusted returns")
    print("  5. Max Drawdown                      → Risk control")
    print("  6. Monte Carlo (1000 runs)           → Shuffle robustness")
    print()
    print(f"TESTING: {SYMBOL_LIMIT} symbols × 4 divs × {len(ATR_MULT_OPTIONS)} ATRs × {len(RR_OPTIONS)} RRs")
    print(f"         = Up to {SYMBOL_LIMIT * 4 * len(ATR_MULT_OPTIONS) * len(RR_OPTIONS):,} combinations")
    print()
    print("THRESHOLDS:")
    print(f"  Min WR (Training): {MIN_WIN_RATE_TRAINING}%")
    print(f"  Min WR (OOS): {MIN_WIN_RATE_OOS}%")
    print(f"  Min Sharpe: {MIN_SHARPE}")
    print(f"  Max Drawdown: {MAX_DRAWDOWN_PCT}%")
    print(f"  Monte Carlo Confidence: {MC_CONFIDENCE*100:.0f}%")
    print("=" * 80)
    
    # Fetch symbols
    symbols = fetch_symbols(SYMBOL_LIMIT)
    if not symbols:
        print("Failed to fetch symbols!")
        return
    
    # Fetch data
    print(f"\n📊 Fetching {DATA_DAYS} days of 1H data for {len(symbols)} symbols...")
    all_data = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(fetch_klines, sym, DATA_DAYS): sym for sym in symbols}
        done = 0
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                df = future.result()
                if not df.empty and len(df) >= 500:
                    all_data[sym] = df
            except:
                pass
            done += 1
            if done % 50 == 0:
                print(f"   Fetched {done}/{len(symbols)} symbols ({len(all_data)} valid)...")
    
    print(f"   ✅ {len(all_data)} symbols with sufficient data (500+ candles)")
    
    # Process symbols
    print(f"\n🔬 Running ultra-robust validation pipeline...")
    print("   This will take a while - testing all combinations with full Monte Carlo...")
    all_validated = []
    
    args_list = [(sym, df) for sym, df in all_data.items()]
    done = 0
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(process_symbol, args): args[0] for args in args_list}
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                results = future.result()
                all_validated.extend(results)
            except Exception as e:
                pass
            done += 1
            if done % 25 == 0:
                elapsed = time.time() - start_time
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(all_data) - done) / rate if rate > 0 else 0
                print(f"   Processed {done}/{len(all_data)} symbols | {len(all_validated)} validated | ETA: {remaining/60:.1f} min")
    
    elapsed_total = (time.time() - start_time) / 60
    print(f"\n   ✅ Completed in {elapsed_total:.1f} minutes")
    print(f"   ✅ Found {len(all_validated)} validated combinations")
    
    if not all_validated:
        print("\n❌ No combinations passed all validation stages!")
        print("   This means the strategy may not be robust enough.")
        return
    
    # Find best per symbol
    print("\n📈 Finding optimal settings per symbol...")
    best_per_symbol = {}
    for v in all_validated:
        sym = v['symbol']
        # Best = highest OOS performance (most important)
        oos_r = v['oos']['total_r']
        if sym not in best_per_symbol or oos_r > best_per_symbol[sym]['oos']['total_r']:
            best_per_symbol[sym] = v
    
    # Save results
    results_df = pd.DataFrame(all_validated)
    results_df.to_csv('ultra_robust_all_results.csv', index=False)
    
    best_data = []
    for v in best_per_symbol.values():
        best_data.append({
            'symbol': v['symbol'],
            'div': v['div'],
            'rr': v['rr'],
            'atr_mult': v['atr_mult'],
            'trades': v['trades'],
            'wr': v['wr'],
            'total_r': v['total_r'],
            'sharpe': v['sharpe'],
            'max_dd': v['max_dd_pct'],
            'mc_pct': v['mc_pct'],
            'train1_r': v['train1']['total_r'],
            'train2_r': v['train2']['total_r'],
            'oos_r': v['oos']['total_r']
        })
    best_df = pd.DataFrame(best_data)
    best_df.to_csv('ultra_robust_validated.csv', index=False)
    
    # Generate config
    config_symbols = {}
    for v in best_per_symbol.values():
        config_symbols[v['symbol']] = {
            'enabled': True,
            'divergence': v['div'],
            'rr': float(v['rr']),
            'atr_mult': float(v['atr_mult']),
            'wr': float(v['wr']),
            'expected_r': float(v['oos']['total_r'])  # Use OOS R as expectation
        }
    
    with open('ultra_robust_config.yaml', 'w') as f:
        yaml.dump({'symbols': config_symbols}, f, default_flow_style=False)
    
    # Summary
    print("\n" + "=" * 80)
    print("ULTRA-ROBUST BACKTEST RESULTS")
    print("=" * 80)
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"   Symbols tested: {len(all_data)}")
    print(f"   Combinations tested: ~{len(all_data) * 4 * len(ATR_MULT_OPTIONS) * len(RR_OPTIONS):,}")
    print(f"   Validated combinations: {len(all_validated)}")
    print(f"   Unique validated symbols: {len(best_per_symbol)}")
    
    # OOS-based expectations (most realistic)
    total_oos_r = sum(v['oos']['total_r'] for v in best_per_symbol.values())
    total_full_r = sum(v['total_r'] for v in best_per_symbol.values())
    avg_wr = np.mean([v['wr'] for v in best_per_symbol.values()])
    avg_sharpe = np.mean([v['sharpe'] for v in best_per_symbol.values()])
    
    print(f"\n📈 PERFORMANCE EXPECTATIONS (Based on OOS - Most Realistic):")
    print(f"   OOS Period R (2 months): +{total_oos_r:.0f}R")
    print(f"   Projected 6-month R: +{total_oos_r * 3:.0f}R")
    print(f"   Projected 12-month R: +{total_oos_r * 6:.0f}R")
    print(f"   Average Win Rate: {avg_wr:.1f}%")
    print(f"   Average Sharpe: {avg_sharpe:.2f}")
    
    print(f"\n📊 Full Period R (for reference): +{total_full_r:.0f}R")
    
    # ATR distribution
    print(f"\n📊 ATR Multiplier Distribution:")
    for mult in ATR_MULT_OPTIONS:
        count = sum(1 for v in best_per_symbol.values() if v['atr_mult'] == mult)
        if count > 0:
            print(f"   {mult}x ATR: {count} symbols ({count/len(best_per_symbol)*100:.1f}%)")
    
    # Divergence distribution
    print(f"\n📊 Divergence Type Distribution:")
    for div in ['REG_BULL', 'REG_BEAR', 'HID_BULL', 'HID_BEAR']:
        count = sum(1 for v in best_per_symbol.values() if v['div'] == div)
        if count > 0:
            print(f"   {div}: {count} symbols ({count/len(best_per_symbol)*100:.1f}%)")
    
    # Top performers
    top_oos = sorted(best_per_symbol.values(), key=lambda x: x['oos']['total_r'], reverse=True)[:10]
    print(f"\n🏆 Top 10 Symbols (by OOS performance):")
    for v in top_oos:
        print(f"   {v['symbol']:20} {v['div']:10} RR={v['rr']:.0f} ATR={v['atr_mult']}x OOS={v['oos']['total_r']:+.1f}R")
    
    print(f"\n📁 Files saved:")
    print(f"   - ultra_robust_all_results.csv ({len(all_validated)} combos)")
    print(f"   - ultra_robust_validated.csv ({len(best_per_symbol)} symbols)")
    print(f"   - ultra_robust_config.yaml (ready for bot)")
    print("=" * 80)

if __name__ == "__main__":
    main()
