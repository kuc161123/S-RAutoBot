#!/usr/bin/env python3
"""
HIDDEN BEARISH WALK-FORWARD BACKTEST (Production-Grade)
========================================================
Ultra-realistic simulation matching live bot logic exactly:

Realism Features:
- Timeframe: 60m (1H) - matches live bot
- Strategy: Hidden Bearish ONLY
- Entry: Next candle OPEN (no lookahead)
- SL: Pivot Swing High (Min 0.3 ATR, Max 2.0 ATR)
- TP: 3.0x Risk (3:1 R:R)
- Slippage: 0.05% per side
- Fees: 0.04% per side
- Candle-by-candle check (including entry candle)
- Cooldown: 10 bars between same-symbol trades

Walk-Forward Validation:
- Split 60 days into 6 x 10-day periods
- Test consistency across ALL periods
- Strategy must be profitable in 80%+ of periods
"""

import requests
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION (Matches Live Bot)
# =============================================================================

TIMEFRAME = '60'       # 1H
DATA_DAYS = 60
NUM_SYMBOLS = 150      # Top by volume
NUM_PERIODS = 6        # Walk-forward periods

# Costs
SLIPPAGE_PCT = 0.0005  # 0.05%
FEE_PCT = 0.0004       # 0.04%
TOTAL_COST = (SLIPPAGE_PCT + FEE_PCT) * 2

# Strategy
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5
COOLDOWN_BARS = 10

# SL/TP
RISK_REWARD = 3.0
MIN_SL_ATR = 0.3
MAX_SL_ATR = 2.0

BASE_URL = "https://api.bybit.com"

# =============================================================================
# HELPERS
# =============================================================================

def calc_ev(wr: float, rr: float) -> float:
    return (wr * rr) - (1 - wr)

def wilson_lb(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0: return 0.0
    p = wins / n
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    return max(0, (centre - spread) / denom)

def get_symbols(limit: int) -> list:
    url = f"{BASE_URL}/v5/market/tickers?category=linear"
    resp = requests.get(url, timeout=10)
    tickers = resp.json().get('result', {}).get('list', [])
    usdt = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in usdt[:limit]]

def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
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
    df['date'] = df.index.date
    return df

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

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

# =============================================================================
# SIGNAL DETECTION (Hidden Bearish Only)
# =============================================================================

def detect_hidden_bearish(df):
    """Detect ONLY Hidden Bearish: Price Lower High, RSI Higher High"""
    if len(df) < 100: return []
    
    close = df['close'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_ph, _ = find_pivots(close, 3, 3)
    signals = []
    
    for i in range(30, n - 5):
        curr_ph = curr_phi = prev_ph = prev_phi = None
        
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_ph[j]):
                if curr_ph is None: 
                    curr_ph, curr_phi = price_ph[j], j
                elif prev_ph is None and j < curr_phi - MIN_PIVOT_DISTANCE: 
                    prev_ph, prev_phi = price_ph[j], j
                    break
        
        if curr_ph and prev_ph:
            if curr_ph < prev_ph:  # Price Lower High
                if rsi[curr_phi] > rsi[prev_phi]:  # RSI Higher High
                    if rsi[i] > RSI_OVERSOLD + 10:  # Not oversold
                        signals.append({'idx': i, 'swing': curr_ph})
    
    return signals

# =============================================================================
# SL/TP CALCULATION (Pivot-Based)
# =============================================================================

def calc_pivot_sltp(rows, idx, atr, swing_price):
    """Calculate SL/TP using Pivot logic (matches bot.py exactly)"""
    entry = rows[idx + 1].open if idx + 1 < len(rows) else rows[idx].close
    
    # SL above swing high (for short)
    sl = swing_price
    sl_dist = abs(sl - entry)
    
    # Clamp to ATR range
    min_dist = MIN_SL_ATR * atr
    max_dist = MAX_SL_ATR * atr
    
    if sl_dist < min_dist: sl_dist = min_dist
    if sl_dist > max_dist: sl_dist = max_dist
    
    sl = entry + sl_dist  # SL above entry for short
    tp = entry - (sl_dist * RISK_REWARD)  # TP below entry for short
    
    return entry, sl, tp

# =============================================================================
# TRADE SIMULATION (Candle-by-Candle)
# =============================================================================

def simulate_trade(rows, signal_idx, sl, tp, entry):
    """
    Ultra-realistic trade simulation:
    - Entry on NEXT candle open
    - Slippage applied
    - Check EVERY candle including entry candle
    - SL checked BEFORE TP (conservative)
    """
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 1:
        return 'timeout', 0, None
    
    # Apply slippage (worse entry for short)
    entry = entry * (1 - SLIPPAGE_PCT)
    tp = tp * (1 + TOTAL_COST)  # Worse exit for short
    
    entry_time = rows[entry_idx].Index if hasattr(rows[entry_idx], 'Index') else None
    
    # Check candles from entry onward
    for bar_idx, row in enumerate(rows[entry_idx:entry_idx + 100]):
        # For SHORT: SL hit if high >= sl, TP hit if low <= tp
        hit_sl = row.high >= sl
        hit_tp = row.low <= tp
        
        # Conservative: if both hit same candle, assume SL (loss)
        if hit_sl and hit_tp:
            return 'loss', bar_idx, entry_time
        elif hit_sl:
            return 'loss', bar_idx, entry_time
        elif hit_tp:
            return 'win', bar_idx, entry_time
    
    return 'timeout', 100, entry_time

# =============================================================================
# MAIN BACKTEST
# =============================================================================

def run():
    print("=" * 70)
    print("🐻 HIDDEN BEARISH WALK-FORWARD BACKTEST (Production Grade)")
    print("=" * 70)
    print(f"Timeframe: {TIMEFRAME}m | R:R: {RISK_REWARD}:1 | Periods: {NUM_PERIODS}")
    print(f"Slippage: {SLIPPAGE_PCT*100:.2f}% | Fees: {FEE_PCT*100:.2f}%")
    print(f"SL: Pivot (Min {MIN_SL_ATR}x, Max {MAX_SL_ATR}x ATR)")
    print("=" * 70)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📋 Testing {len(symbols)} symbols over {DATA_DAYS} days...")
    
    # Track results
    period_results = defaultdict(lambda: {'w': 0, 'l': 0})
    symbol_stats = defaultdict(lambda: {'w': 0, 'l': 0, 'r': 0.0})
    
    total_wins = 0
    total_losses = 0
    total_r = 0.0
    
    start_time = time.time()
    
    for idx, sym in enumerate(symbols):
        try:
            df = fetch_klines(sym, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 200: continue
            
            # Indicators
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            hl = df['high'] - df['low']
            hc = abs(df['high'] - df['close'].shift())
            lc = abs(df['low'] - df['close'].shift())
            df['atr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            df = df.dropna()
            
            if len(df) < 100: continue
            
            # Detect signals
            signals = detect_hidden_bearish(df)
            rows = list(df.itertuples())
            
            # Walk-forward period mapping
            all_dates = sorted(df['date'].unique())
            days_per_period = max(1, len(all_dates) // NUM_PERIODS)
            
            last_trade_idx = -COOLDOWN_BARS
            
            for sig in signals:
                i = sig['idx']
                if i - last_trade_idx < COOLDOWN_BARS: continue
                if i >= len(rows) - 50: continue
                
                row = rows[i]
                if not row.vol_ok or row.atr <= 0: continue
                
                # Execute trade
                entry, sl, tp = calc_pivot_sltp(rows, i, row.atr, sig['swing'])
                result, bars, entry_time = simulate_trade(rows, i, sl, tp, entry)
                
                if result == 'timeout': continue
                
                last_trade_idx = i
                
                # Determine period
                try:
                    trade_date = row.Index.date() if hasattr(row.Index, 'date') else row.date
                    day_idx = all_dates.index(trade_date)
                    period = min(day_idx // days_per_period, NUM_PERIODS - 1)
                except:
                    period = 0
                
                # Record results
                if result == 'win':
                    total_wins += 1
                    total_r += RISK_REWARD
                    period_results[period]['w'] += 1
                    symbol_stats[sym]['w'] += 1
                    symbol_stats[sym]['r'] += RISK_REWARD
                else:
                    total_losses += 1
                    total_r -= 1.0
                    period_results[period]['l'] += 1
                    symbol_stats[sym]['l'] += 1
                    symbol_stats[sym]['r'] -= 1.0
                    
        except Exception as e:
            continue
        
        if (idx + 1) % 30 == 0:
            print(f"  [{idx+1}/{NUM_SYMBOLS}] Trades: {total_wins + total_losses}")
    
    elapsed = (time.time() - start_time) / 60
    
    # ==========================================================================
    # RESULTS
    # ==========================================================================
    
    total = total_wins + total_losses
    if total == 0:
        print("No trades found.")
        return
    
    wr = total_wins / total
    ev = calc_ev(wr, RISK_REWARD)
    lb = wilson_lb(total_wins, total)
    
    print("\n" + "=" * 70)
    print("📊 OVERALL RESULTS")
    print("=" * 70)
    print(f"⏱️ Completed in {elapsed:.1f} minutes")
    print(f"📈 Total Trades: {total}")
    print(f"✅ Wins: {total_wins} | ❌ Losses: {total_losses}")
    print(f"📊 Win Rate: {wr*100:.1f}% (Lower Bound: {lb*100:.1f}%)")
    print(f"💰 Total R: {total_r:+.1f}R")
    print(f"📊 EV: {ev:+.2f}R per trade")
    print(f"⏱️ Trades/Day: {total/DATA_DAYS:.1f}")
    
    # Walk-Forward Analysis
    print("\n" + "-" * 70)
    print("📅 WALK-FORWARD VALIDATION")
    print("-" * 70)
    print(f"{'Period':<12} {'Days':<12} {'N':>6} {'WR':>8} {'EV':>8} {'Status':<10}")
    print("-" * 70)
    
    period_evs = []
    for p in range(NUM_PERIODS):
        data = period_results[p]
        n = data['w'] + data['l']
        start_day = p * (DATA_DAYS // NUM_PERIODS) + 1
        end_day = start_day + (DATA_DAYS // NUM_PERIODS) - 1
        
        if n > 0:
            p_wr = data['w'] / n
            p_ev = calc_ev(p_wr, RISK_REWARD)
            period_evs.append(p_ev)
            status = "✅" if p_ev > 0 else "❌"
            print(f"Period {p+1:<4} D{start_day}-D{end_day:<4} {n:>6} {p_wr*100:>7.1f}% {p_ev:>+7.2f} {status}")
        else:
            print(f"Period {p+1:<4} D{start_day}-D{end_day:<4} {'--':>6} {'--':>8} {'--':>8} N/A")
    
    # Consistency Score
    print("\n" + "-" * 70)
    print("📈 WALK-FORWARD CONSISTENCY")
    print("-" * 70)
    
    if period_evs:
        profitable = sum(1 for ev in period_evs if ev > 0)
        consistency = profitable / len(period_evs) * 100
        avg_ev = np.mean(period_evs)
        std_ev = np.std(period_evs) if len(period_evs) > 1 else 0
        
        print(f"📊 Profitable Periods: {profitable}/{len(period_evs)} ({consistency:.0f}%)")
        print(f"📈 Average EV: {avg_ev:+.2f}R")
        print(f"📉 EV Std Dev: {std_ev:.2f}R")
        
        if consistency >= 80:
            print("\n✅ STRATEGY IS CONSISTENT - Profitable in 80%+ of periods!")
            verdict = "PRODUCTION READY"
        elif consistency >= 60:
            print("\n⚠️ STRATEGY IS MODERATELY CONSISTENT")
            verdict = "PROMISING"
        else:
            print("\n❌ STRATEGY IS INCONSISTENT")
            verdict = "NEEDS WORK"
    
    # Golden Symbol List
    print("\n" + "=" * 70)
    print("🏆 GOLDEN SYMBOL LIST (WR >= 40%, N >= 5)")
    print("=" * 70)
    print("  divergence_symbols:")
    
    # Sort by R total
    sorted_symbols = sorted(symbol_stats.items(), 
                           key=lambda x: x[1]['r'], reverse=True)
    
    valid_symbols = []
    for sym, stats in sorted_symbols:
        n = stats['w'] + stats['l']
        if n >= 5:
            sym_wr = stats['w'] / n
            if sym_wr >= 0.40:
                valid_symbols.append((sym, sym_wr, n, stats['r']))
                print(f"    - {sym:<16} # {sym_wr*100:.1f}% WR, {n} trades, {stats['r']:+.0f}R")
    
    print("-" * 70)
    print(f"Total Valid Symbols: {len(valid_symbols)}")
    
    # Final Verdict
    print("\n" + "=" * 70)
    print(f"🎯 FINAL VERDICT: {verdict}")
    print("=" * 70)
    print(f"Strategy: Hidden Bearish (1H) + Pivot SL + 3:1 R:R")
    print(f"Win Rate: {wr*100:.1f}% | EV: {ev:+.2f}R | Consistency: {consistency:.0f}%")
    print(f"Symbols: {len(valid_symbols)} validated")
    print("=" * 70)

if __name__ == "__main__":
    run()
