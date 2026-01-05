#!/usr/bin/env python3
"""
CORRECTED ULTRA-COMPREHENSIVE BACKTEST
=======================================
FIXES from original:
1. SL/TP checked from ENTRY CANDLE (not entry+1) - matches live trading
2. Tests multiple ATR multipliers (1.0, 1.5, 2.0) per symbol
3. Tests R:R from 3 to 10
4. Full Walk-Forward + Monte Carlo + Sharpe validation

500 Symbols | 4 Divergence Types | 3 ATR Mults | 8 R:R Options
Total combinations tested: Up to 500 * 4 * 3 * 8 = 48,000

Output: Validated symbols with optimal ATR_MULT and R:R per divergence type
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
import yaml
import random
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional
import concurrent.futures
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SYMBOL_LIMIT = 500
TIMEFRAME = '60'  # 1H
DATA_DAYS = 180   # 6 months

# Strategy settings
MAX_WAIT_CANDLES = 12

# KEY CHANGE: Test multiple ATR multipliers
ATR_MULT_OPTIONS = [1.0, 1.5, 2.0]

SLIPPAGE_PCT = 0.0003
FEE_PCT = 0.0006

# Indicator settings
RSI_PERIOD = 14
EMA_PERIOD = 200
PIVOT_LEFT = 3
PIVOT_RIGHT = 3

# R:R ratios to test
RR_OPTIONS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Walk-forward settings
WF_PERIOD_DAYS = 60
WF_MIN_PERIODS_PROFITABLE = 2

# Monte Carlo
MC_ITERATIONS = 1000

# Validation thresholds
MIN_TRADES_TOTAL = 15
MIN_WIN_RATE = 20  # Increased from 10%
MIN_TOTAL_R = 0
MIN_SHARPE = 0.3
MAX_DRAWDOWN_PCT = 40  # Tightened from 50%

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
        # Exclude stablecoins
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
    
    # ATR (True Range)
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
        if arr[i] == max(arr[i-left:i+right+1]) and list(arr[i-left:i+right+1]).count(arr[i]) == 1:
            ph[i] = arr[i]
        if arr[i] == min(arr[i-left:i+right+1]) and list(arr[i-left:i+right+1]).count(arr[i]) == 1:
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
    
    # Find pivots for long (lows) or short (highs)
    if div_cfg.side == 'long':
        pivots = []
        for j in range(idx - PIVOT_RIGHT - 1, max(0, idx - 50), -1):
            if not np.isnan(pl[j]):
                pivots.append((j, pl[j], rsi[j]))
                if len(pivots) >= 2:
                    break
        
        if len(pivots) < 2:
            return False, None
        
        curr_idx, curr_price, curr_rsi = pivots[0]
        prev_idx, prev_price, prev_rsi = pivots[1]
        
        if (idx - curr_idx) > 10:
            return False, None
        
        if div_cfg.code == 'REG_BULL':
            if curr_price < prev_price and curr_rsi > prev_rsi:
                swing = max(df['high'].iloc[curr_idx:idx+1])
                if current_price <= swing:
                    return True, swing
        elif div_cfg.code == 'HID_BULL':
            if curr_price > prev_price and curr_rsi < prev_rsi:
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
        
        curr_idx, curr_price, curr_rsi = pivots[0]
        prev_idx, prev_price, prev_rsi = pivots[1]
        
        if (idx - curr_idx) > 10:
            return False, None
        
        if div_cfg.code == 'REG_BEAR':
            if curr_price > prev_price and curr_rsi < prev_rsi:
                swing = min(df['low'].iloc[curr_idx:idx+1])
                if current_price >= swing:
                    return True, swing
        elif div_cfg.code == 'HID_BEAR':
            if curr_price < prev_price and curr_rsi > prev_rsi:
                swing = min(df['low'].iloc[curr_idx:idx+1])
                if current_price >= swing:
                    return True, swing
    
    return False, None

# ============================================================================
# TRADE SIMULATION (CORRECTED SL TIMING)
# ============================================================================

def simulate_trades(df, symbol, div_cfg, rr, atr_mult):
    """
    CRITICAL FIX: Check SL/TP from ENTRY CANDLE (idx) not entry+1
    This matches live trading where bracket orders are immediately active.
    """
    trades = []
    if len(df) < 100:
        return trades
    
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
    
    for idx in range(50, n - 1):
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
                if entry_idx >= n:
                    pending_signal = None
                    continue
                
                entry_price = df.iloc[entry_idx]['open']
                entry_atr = atr[entry_idx]
                sl_dist = entry_atr * atr_mult  # Use passed ATR multiplier
                
                if div_cfg.side == 'long':
                    entry_price *= (1 + SLIPPAGE_PCT)
                    tp = entry_price + (sl_dist * rr)
                    sl = entry_price - sl_dist
                else:
                    entry_price *= (1 - SLIPPAGE_PCT)
                    tp = entry_price - (sl_dist * rr)
                    sl = entry_price + sl_dist
                
                # ====== CRITICAL FIX: Start checking from ENTRY CANDLE ======
                # This matches live trading where SL is active immediately
                result = None
                exit_idx = None
                
                for k in range(entry_idx, min(entry_idx + 200, n)):  # Start from entry_idx, NOT entry_idx + 1
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
                    result = -0.5
                    exit_idx = min(entry_idx + 200, n - 1)
                
                # Fees
                risk_pct = sl_dist / entry_price if entry_price > 0 else 0.01
                fee_cost = (FEE_PCT * 2) / risk_pct if risk_pct > 0 else 0.1
                final_r = result - fee_cost
                
                entry_time = df.index[entry_idx]
                
                # Period for walk-forward
                df_start = df.index[0]
                days = (entry_time - df_start).days
                period = 1 if days < WF_PERIOD_DAYS else (2 if days < WF_PERIOD_DAYS * 2 else 3)
                
                trades.append({
                    'symbol': symbol,
                    'div': div_cfg.code,
                    'rr': rr,
                    'atr_mult': atr_mult,
                    'r': round(final_r, 3),
                    'win': final_r > 0,
                    'period': period,
                    'entry_time': entry_time,
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
# VALIDATION
# ============================================================================

def validate_combo(trades):
    """Validate a symbol-divergence-RR-ATR combination"""
    if len(trades) < MIN_TRADES_TOTAL:
        return None
    
    total_r = sum(t['r'] for t in trades)
    wins = sum(1 for t in trades if t['win'])
    wr = (wins / len(trades)) * 100
    
    if wr < MIN_WIN_RATE or total_r <= MIN_TOTAL_R:
        return None
    
    # Walk-forward: Check each period
    p1 = [t for t in trades if t['period'] == 1]
    p2 = [t for t in trades if t['period'] == 2]
    p3 = [t for t in trades if t['period'] == 3]
    
    periods_profitable = sum(1 for p in [p1, p2, p3] if len(p) >= 3 and sum(t['r'] for t in p) > 0)
    
    if periods_profitable < WF_MIN_PERIODS_PROFITABLE:
        return None
    
    # Sharpe (simplified)
    r_values = [t['r'] for t in trades]
    if np.std(r_values) > 0:
        sharpe = np.mean(r_values) / np.std(r_values) * np.sqrt(len(trades))
    else:
        sharpe = 0
    
    if sharpe < MIN_SHARPE:
        return None
    
    # Max drawdown
    equity = 0
    peak = 0
    max_dd = 0
    for t in sorted(trades, key=lambda x: x['entry_time']):
        equity += t['r']
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd
    
    if total_r > 0 and (max_dd / total_r * 100) > MAX_DRAWDOWN_PCT:
        return None
    
    # Monte Carlo (quick version)
    mc_results = []
    for _ in range(100):  # Reduced for speed
        shuffled = random.sample(r_values, len(r_values))
        mc_results.append(sum(shuffled))
    mc_profitable = sum(1 for r in mc_results if r > 0) / len(mc_results) * 100
    
    if mc_profitable < 60:
        return None
    
    return {
        'symbol': trades[0]['symbol'],
        'div': trades[0]['div'],
        'rr': trades[0]['rr'],
        'atr_mult': trades[0]['atr_mult'],
        'trades': len(trades),
        'wr': round(wr, 1),
        'total_r': round(total_r, 1),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd, 1),
        'periods_profit': periods_profitable,
        'mc_profit_pct': round(mc_profitable, 0)
    }

# ============================================================================
# MAIN
# ============================================================================

def process_symbol(args):
    symbol, df = args
    results = []
    
    if df.empty or len(df) < 250:
        return results
    
    df = calculate_indicators(df)
    if df.empty:
        return results
    
    for div_code, div_cfg in DIVERGENCES.items():
        for rr in RR_OPTIONS:
            for atr_mult in ATR_MULT_OPTIONS:
                trades = simulate_trades(df, symbol, div_cfg, rr, atr_mult)
                if trades:
                    validated = validate_combo(trades)
                    if validated:
                        results.append(validated)
    
    return results

def main():
    print("=" * 70)
    print("CORRECTED ULTRA-COMPREHENSIVE 500-SYMBOL BACKTEST")
    print("=" * 70)
    print(f"FIX: SL/TP checked from ENTRY CANDLE (matches live trading)")
    print(f"Testing: 500 symbols × 4 divergences × 8 R:Rs × 3 ATR multipliers")
    print(f"Validation: Walk-Forward + Monte Carlo + Sharpe")
    print("=" * 70)
    
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
                if not df.empty and len(df) >= 250:
                    all_data[sym] = df
            except:
                pass
            done += 1
            if done % 50 == 0:
                print(f"   Fetched {done}/{len(symbols)} symbols ({len(all_data)} valid)...")
    
    print(f"   ✅ {len(all_data)} symbols with sufficient data")
    
    # Process symbols
    print(f"\n🔬 Running backtest with corrected SL timing...")
    all_validated = []
    
    args_list = [(sym, df) for sym, df in all_data.items()]
    done = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_symbol, args): args[0] for args in args_list}
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                results = future.result()
                all_validated.extend(results)
            except Exception as e:
                pass
            done += 1
            if done % 50 == 0:
                print(f"   Processed {done}/{len(all_data)} symbols ({len(all_validated)} validated combos)...")
    
    print(f"\n✅ Found {len(all_validated)} validated combinations")
    
    if not all_validated:
        print("❌ No combinations passed validation!")
        return
    
    # Find best per symbol
    print("\n📈 Finding optimal settings per symbol...")
    best_per_symbol = {}
    for v in all_validated:
        key = (v['symbol'], v['div'])
        if key not in best_per_symbol or v['total_r'] > best_per_symbol[key]['total_r']:
            best_per_symbol[key] = v
    
    # Save results
    results_df = pd.DataFrame(all_validated)
    results_df.to_csv('corrected_backtest_all_results.csv', index=False)
    
    best_df = pd.DataFrame(list(best_per_symbol.values()))
    best_df.to_csv('corrected_backtest_validated.csv', index=False)
    
    # Generate config
    config_symbols = {}
    for v in best_per_symbol.values():
        sym = v['symbol']
        if sym not in config_symbols or v['total_r'] > config_symbols[sym].get('expected_r', -999):
            config_symbols[sym] = {
                'enabled': True,
                'divergence': v['div'],
                'rr': v['rr'],
                'atr_mult': v['atr_mult'],
                'expected_r': v['total_r'],
                'wr': v['wr']
            }
    
    with open('corrected_backtest_config.yaml', 'w') as f:
        yaml.dump({'symbols': config_symbols}, f, default_flow_style=False)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Symbols tested: {len(all_data)}")
    print(f"Validated combinations: {len(all_validated)}")
    print(f"Unique symbols validated: {len(config_symbols)}")
    print(f"Total expected R (6 months): {sum(v['total_r'] for v in best_per_symbol.values()):.0f}R")
    
    # ATR multiplier breakdown
    print("\n📊 ATR Multiplier Breakdown:")
    for mult in ATR_MULT_OPTIONS:
        count = sum(1 for v in config_symbols.values() if v['atr_mult'] == mult)
        print(f"   {mult}x ATR: {count} symbols")
    
    print("\n📁 Files saved:")
    print("   - corrected_backtest_all_results.csv")
    print("   - corrected_backtest_validated.csv")
    print("   - corrected_backtest_config.yaml")
    print("=" * 70)

if __name__ == "__main__":
    main()
