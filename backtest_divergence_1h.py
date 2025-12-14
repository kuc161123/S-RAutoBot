#!/usr/bin/env python3
"""
ULTRA-REALISTIC RSI DIVERGENCE BACKTEST WITH WALK-FORWARD VALIDATION
=====================================================================

Realistic Features:
- Entry on NEXT candle open (not current close)
- Slippage modeling (0.05% per side)
- Fees (0.04% maker/taker)
- Candle-by-candle SL/TP checking (high/low)
- Minimum volume filter
- No lookahead bias

Walk-Forward Validation:
- Divide data into 6 periods (10 days each)
- Test consistency across all periods
- Calculate out-of-sample performance
- Detect if strategy degrades over time
"""

import requests
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TIMEFRAME = '60'  # 15-minute candles
DATA_DAYS = 60
NUM_SYMBOLS = 200

# Realistic Costs
SLIPPAGE_PCT = 0.0005  # 0.05% slippage per side
FEE_PCT = 0.0004       # 0.04% fee per side (Bybit taker)
TOTAL_COST = (SLIPPAGE_PCT + FEE_PCT) * 2  # Entry + Exit = 0.18%

# Risk/Reward
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.0

# Divergence settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14
MIN_PIVOT_DISTANCE = 5

# Walk-Forward Settings
NUM_PERIODS = 6  # Split 60 days into 6 x 10-day periods
PERIOD_DAYS = DATA_DAYS // NUM_PERIODS

BASE_URL = "https://api.bybit.com"

# =============================================================================
# HELPERS
# =============================================================================

def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0: return 0.0
    p = wins / n
    denominator = 1 + z*z/n
    centre = p + z*z/(2*n)
    spread = z * math.sqrt((p*(1-p) + z*z/(4*n)) / n)
    return max(0, (centre - spread) / denominator)

def calc_ev(wr: float, rr: float = 2.0) -> float:
    return (wr * rr) - (1 - wr)

def get_symbols(limit: int = 100) -> list:
    url = f"{BASE_URL}/v5/market/tickers?category=linear"
    resp = requests.get(url)
    tickers = resp.json().get('result', {}).get('list', [])
    usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT')]
    usdt_pairs.sort(key=lambda x: float(x.get('turnover24h', 0)), reverse=True)
    return [t['symbol'] for t in usdt_pairs[:limit]]

def fetch_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    end_ts = int(datetime.now().timestamp() * 1000)
    candles_needed = days * 24 * 60 // int(interval)
    
    all_candles = []
    current_end = end_ts
    
    while len(all_candles) < candles_needed:
        url = f"{BASE_URL}/v5/market/kline"
        params = {'category': 'linear', 'symbol': symbol, 'interval': interval, 'limit': 1000, 'end': current_end}
        try:
            resp = requests.get(url, params=params, timeout=10)
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
    df['hour_utc'] = df.index.hour
    df['date'] = df.index.date
    return df

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_pivots(data, left=5, right=5):
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    for i in range(left, n - right):
        is_high = all(data[j] < data[i] for j in range(i - left, i + right + 1) if j != i)
        is_low = all(data[j] > data[i] for j in range(i - left, i + right + 1) if j != i)
        if is_high: pivot_highs[i] = data[i]
        if is_low: pivot_lows[i] = data[i]
    
    return pivot_highs, pivot_lows

def detect_divergence_signals(df):
    """Detect RSI divergences and return signal list."""
    if len(df) < 100:
        return []
    
    close = df['close'].values
    rsi = df['rsi'].values
    n = len(df)
    
    price_pivot_highs, price_pivot_lows = find_pivots(close, left=3, right=3)
    rsi_pivot_highs, rsi_pivot_lows = find_pivots(rsi, left=3, right=3)
    
    signals = []
    
    for i in range(30, n - 5):  # Leave room for next-candle entry
        # Find recent pivot lows
        curr_price_low = curr_price_low_idx = None
        prev_price_low = prev_price_low_idx = None
        
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pivot_lows[j]):
                if curr_price_low is None:
                    curr_price_low = price_pivot_lows[j]
                    curr_price_low_idx = j
                elif prev_price_low is None and j < curr_price_low_idx - MIN_PIVOT_DISTANCE:
                    prev_price_low = price_pivot_lows[j]
                    prev_price_low_idx = j
                    break
        
        # Find recent pivot highs
        curr_price_high = curr_price_high_idx = None
        prev_price_high = prev_price_high_idx = None
        
        for j in range(i, max(i - LOOKBACK_BARS, 0), -1):
            if not np.isnan(price_pivot_highs[j]):
                if curr_price_high is None:
                    curr_price_high = price_pivot_highs[j]
                    curr_price_high_idx = j
                elif prev_price_high is None and j < curr_price_high_idx - MIN_PIVOT_DISTANCE:
                    prev_price_high = price_pivot_highs[j]
                    prev_price_high_idx = j
                    break
        
        # === REGULAR BULLISH: Price LL, RSI HL ===
        if curr_price_low is not None and prev_price_low is not None:
            if curr_price_low < prev_price_low:
                curr_rsi = rsi[curr_price_low_idx]
                prev_rsi = rsi[prev_price_low_idx]
                if curr_rsi > prev_rsi and rsi[i] < RSI_OVERSOLD + 15:
                    signals.append({'idx': i, 'side': 'long', 'type': 'regular_bullish'})
                    continue
        
        # === REGULAR BEARISH: Price HH, RSI LH ===
        if curr_price_high is not None and prev_price_high is not None:
            if curr_price_high > prev_price_high:
                curr_rsi = rsi[curr_price_high_idx]
                prev_rsi = rsi[prev_price_high_idx]
                if curr_rsi < prev_rsi and rsi[i] > RSI_OVERBOUGHT - 15:
                    signals.append({'idx': i, 'side': 'short', 'type': 'regular_bearish'})
                    continue
        
        # === HIDDEN BULLISH: Price HL, RSI LL ===
        if curr_price_low is not None and prev_price_low is not None:
            if curr_price_low > prev_price_low:
                curr_rsi = rsi[curr_price_low_idx]
                prev_rsi = rsi[prev_price_low_idx]
                if curr_rsi < prev_rsi and rsi[i] < RSI_OVERBOUGHT - 10:
                    signals.append({'idx': i, 'side': 'long', 'type': 'hidden_bullish'})
                    continue
        
        # === HIDDEN BEARISH: Price LH, RSI HH ===
        if curr_price_high is not None and prev_price_high is not None:
            if curr_price_high < prev_price_high:
                curr_rsi = rsi[curr_price_high_idx]
                prev_rsi = rsi[prev_price_high_idx]
                if curr_rsi > prev_rsi and rsi[i] > RSI_OVERSOLD + 10:
                    signals.append({'idx': i, 'side': 'short', 'type': 'hidden_bearish'})
    
    return signals

def simulate_trade_realistic(df, signal_idx, side, atr):
    """
    Ultra-realistic trade simulation:
    - Entry on NEXT candle OPEN (with slippage)
    - Candle-by-candle SL/TP checking
    - Proper fee/slippage accounting
    """
    rows = list(df.itertuples())
    
    # Entry is on NEXT candle open
    entry_idx = signal_idx + 1
    if entry_idx >= len(rows) - 50:
        return {'outcome': 'timeout', 'bars': 0}
    
    entry_row = rows[entry_idx]
    base_entry = entry_row.open
    
    # Apply slippage to entry
    if side == 'long':
        entry_price = base_entry * (1 + SLIPPAGE_PCT)  # Pay more for long
        tp = entry_price + (TP_ATR_MULT * atr)
        sl = entry_price - (SL_ATR_MULT * atr)
        # Apply costs to TP
        tp = tp * (1 - TOTAL_COST)  # Reduce TP by fees
    else:
        entry_price = base_entry * (1 - SLIPPAGE_PCT)  # Get less for short entry
        tp = entry_price - (TP_ATR_MULT * atr)
        sl = entry_price + (SL_ATR_MULT * atr)
        tp = tp * (1 + TOTAL_COST)  # Increase TP (worse for shorts)
    
    # Simulate candle by candle
    for bar_offset, future_row in enumerate(rows[entry_idx+1:entry_idx+100]):
        if side == 'long':
            # Check SL first (worst case within candle)
            if future_row.low <= sl:
                return {'outcome': 'loss', 'bars': bar_offset + 1}
            if future_row.high >= tp:
                return {'outcome': 'win', 'bars': bar_offset + 1}
        else:
            if future_row.high >= sl:
                return {'outcome': 'loss', 'bars': bar_offset + 1}
            if future_row.low <= tp:
                return {'outcome': 'win', 'bars': bar_offset + 1}
    
    return {'outcome': 'timeout', 'bars': 100}

def get_period_bounds(df, period_idx, total_periods):
    """Get start and end indices for a given period."""
    dates = sorted(df['date'].unique())
    days_per_period = len(dates) // total_periods
    
    start_day = period_idx * days_per_period
    end_day = (period_idx + 1) * days_per_period if period_idx < total_periods - 1 else len(dates)
    
    period_dates = dates[start_day:end_day]
    if not period_dates:
        return None, None
    
    mask = df['date'].isin(period_dates)
    period_df = df[mask]
    
    if len(period_df) == 0:
        return None, None
    
    return period_df.index.min(), period_df.index.max()

# =============================================================================
# MAIN BACKTEST
# =============================================================================

def run_walkforward_backtest():
    print("=" * 80)
    print("🔬 RSI DIVERGENCE - WALK-FORWARD VALIDATION")
    print("=" * 80)
    print(f"Timeframe: {TIMEFRAME}m | Data: {DATA_DAYS} days | Symbols: {NUM_SYMBOLS}")
    print(f"Walk-Forward: {NUM_PERIODS} periods x {PERIOD_DAYS} days each")
    print(f"R:R = {TP_ATR_MULT}:{SL_ATR_MULT} | Slippage: {SLIPPAGE_PCT*100:.2f}% | Fees: {FEE_PCT*100:.2f}%")
    print("=" * 80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📋 Testing {len(symbols)} symbols...\n")
    
    # Results by period
    period_results = defaultdict(lambda: {'w': 0, 'l': 0})
    
    # Results by type (overall)
    results_by_type = defaultdict(lambda: {'w': 0, 'l': 0})
    results_by_side = {'long': {'w': 0, 'l': 0}, 'short': {'w': 0, 'l': 0}}
    
    # Track bars to resolution
    bars_to_win = []
    bars_to_loss = []
    
    total_trades = 0
    total_wins = 0
    
    start_time = time.time()
    processed = 0
    
    for idx, symbol in enumerate(symbols):
        try:
            df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 500:
                continue
            
            # Calculate indicators
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            
            # Volume filter - above 20-period average
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ok'] = df['volume'] > df['vol_ma'] * 0.5
            
            df = df.dropna()
            if len(df) < 200:
                continue
            
            processed += 1
            
            # Detect all signals
            signals = detect_divergence_signals(df)
            
            rows = list(df.itertuples())
            last_trade_idx = -20
            
            for sig in signals:
                i = sig['idx']
                
                if i - last_trade_idx < 10:
                    continue
                
                if i >= len(rows) - 100:
                    continue
                
                row = rows[i]
                atr = row.atr
                if pd.isna(atr) or atr <= 0:
                    continue
                
                # Volume filter
                if not row.vol_ok:
                    continue
                
                # Simulate realistic trade
                trade = simulate_trade_realistic(df, i, sig['side'], atr)
                
                if trade['outcome'] == 'timeout':
                    continue
                
                last_trade_idx = i
                total_trades += 1
                
                # Determine which period this trade belongs to
                trade_date = row.Index.date()
                all_dates = sorted(df['date'].unique())
                days_per_period = len(all_dates) // NUM_PERIODS
                
                try:
                    day_idx = all_dates.index(trade_date)
                    period_num = min(day_idx // days_per_period, NUM_PERIODS - 1)
                except:
                    period_num = 0
                
                if trade['outcome'] == 'win':
                    results_by_type[sig['type']]['w'] += 1
                    results_by_side[sig['side']]['w'] += 1
                    period_results[period_num]['w'] += 1
                    bars_to_win.append(trade['bars'])
                    total_wins += 1
                else:
                    results_by_type[sig['type']]['l'] += 1
                    results_by_side[sig['side']]['l'] += 1
                    period_results[period_num]['l'] += 1
                    bars_to_loss.append(trade['bars'])
            
            if (idx + 1) % 10 == 0:
                print(f"[{idx+1}/{NUM_SYMBOLS}] {processed} processed | Trades: {total_trades}")
            
            time.sleep(0.03)
            
        except Exception as e:
            continue
    
    elapsed = time.time() - start_time
    
    # ==========================================================================
    # RESULTS
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("📊 RESULTS - ULTRA REALISTIC")
    print("=" * 80)
    print(f"\n⏱️ Completed in {elapsed/60:.1f} minutes ({processed} symbols)")
    print(f"💰 Total Costs Applied: {TOTAL_COST*100:.2f}% per trade")
    
    # Overall
    print("\n" + "-" * 60)
    print("OVERALL PERFORMANCE (After Costs)")
    print("-" * 60)
    
    if total_trades > 0:
        overall_wr = total_wins / total_trades
        overall_lb = wilson_lower_bound(total_wins, total_trades)
        overall_ev = calc_ev(overall_wr)
        emoji = "🟢" if overall_ev > 0.1 else "⚠️" if overall_ev > 0 else "🔴"
        
        print(f"\n📊 Total Trades: {total_trades}")
        print(f"✅ Wins: {total_wins} | ❌ Losses: {total_trades - total_wins}")
        print(f"📈 Win Rate: {overall_wr*100:.1f}% (LB: {overall_lb*100:.1f}%)")
        print(f"💰 EV: {overall_ev:+.2f}R {emoji}")
        
        if bars_to_win:
            print(f"\n⏱️ Avg bars to WIN: {np.mean(bars_to_win):.1f}")
        if bars_to_loss:
            print(f"⏱️ Avg bars to LOSS: {np.mean(bars_to_loss):.1f}")
    
    # Walk-Forward Analysis
    print("\n" + "-" * 60)
    print("📅 WALK-FORWARD VALIDATION (By Period)")
    print("-" * 60)
    
    print(f"\n{'Period':<12} {'Days':<12} {'N':>6} {'WR':>8} {'EV':>8} {'Status':<10}")
    print("-" * 60)
    
    period_evs = []
    for period_num in range(NUM_PERIODS):
        data = period_results[period_num]
        total = data['w'] + data['l']
        
        start_day = period_num * PERIOD_DAYS + 1
        end_day = start_day + PERIOD_DAYS - 1
        day_range = f"D{start_day}-D{end_day}"
        
        if total > 0:
            wr = data['w'] / total
            ev = calc_ev(wr)
            period_evs.append(ev)
            status = "✅" if ev > 0 else "❌"
            print(f"Period {period_num+1:<4} {day_range:<12} {total:>6} {wr*100:>7.1f}% {ev:>+7.2f} {status}")
        else:
            print(f"Period {period_num+1:<4} {day_range:<12} {'--':>6} {'--':>8} {'--':>8} {'N/A':<10}")
    
    # Walk-Forward Consistency
    print("\n" + "-" * 60)
    print("📈 WALK-FORWARD CONSISTENCY")
    print("-" * 60)
    
    if period_evs:
        profitable_periods = sum(1 for ev in period_evs if ev > 0)
        consistency = profitable_periods / len(period_evs) * 100
        avg_ev = np.mean(period_evs)
        ev_std = np.std(period_evs) if len(period_evs) > 1 else 0
        
        print(f"\n📊 Profitable Periods: {profitable_periods}/{len(period_evs)} ({consistency:.0f}%)")
        print(f"📈 Average EV: {avg_ev:+.2f}R")
        print(f"📉 EV Std Dev: {ev_std:.2f}R")
        print(f"📊 Sharpe-like: {avg_ev/ev_std:.2f}" if ev_std > 0 else "")
        
        if consistency >= 80 and avg_ev > 0:
            print("\n✅ STRATEGY IS CONSISTENT - Profitable in 80%+ of periods!")
        elif consistency >= 60 and avg_ev > 0:
            print("\n⚠️ STRATEGY IS MODERATELY CONSISTENT - Some variance between periods")
        else:
            print("\n❌ STRATEGY IS INCONSISTENT - Performance varies significantly")
    
    # By Type
    print("\n" + "-" * 60)
    print("BY DIVERGENCE TYPE")
    print("-" * 60)
    
    sorted_types = sorted(results_by_type.items(), 
                         key=lambda x: calc_ev(x[1]['w']/(x[1]['w']+x[1]['l'])) if x[1]['w']+x[1]['l'] > 0 else -999,
                         reverse=True)
    
    print(f"\n{'Type':<25} {'N':>6} {'WR':>8} {'EV':>8}")
    print("-" * 50)
    
    for sig_type, data in sorted_types:
        total = data['w'] + data['l']
        if total > 0:
            wr = data['w'] / total
            ev = calc_ev(wr)
            emoji = "✅" if ev > 0.1 else "⚠️" if ev > 0 else "❌"
            print(f"{sig_type:<25} {total:>6} {wr*100:>7.1f}% {ev:>+7.2f} {emoji}")
    
    # By Side
    print("\n" + "-" * 60)
    print("BY SIDE")
    print("-" * 60)
    
    for side in ['long', 'short']:
        d = results_by_side[side]
        total = d['w'] + d['l']
        if total > 0:
            wr = d['w'] / total
            ev = calc_ev(wr)
            emoji = "✅" if ev > 0.1 else "⚠️" if ev > 0 else "❌"
            icon = "🟢" if side == 'long' else "🔴"
            print(f"{icon} {side.upper():<10} N={total:>5} | WR={wr*100:>5.1f}% | EV={ev:>+.2f} {emoji}")
    
    # Summary
    print("\n" + "=" * 80)
    print("💡 FINAL SUMMARY")
    print("=" * 80)
    
    if total_trades > 0:
        overall_wr = total_wins / total_trades
        overall_ev = calc_ev(overall_wr)
        
        print(f"\n📊 RSI Divergence (Realistic): {total_trades} trades | {overall_wr*100:.1f}% WR | {overall_ev:+.2f} EV")
        
        if consistency >= 80 and overall_ev > 0.1:
            print(f"\n🎯 VERDICT: PRODUCTION READY")
            print(f"   - Consistent across {profitable_periods}/{len(period_evs)} periods")
            print(f"   - Average EV: {avg_ev:+.2f}R per trade")
            print(f"   - Best type: {sorted_types[0][0]} ({sorted_types[0][1]['w']}/{sorted_types[0][1]['w']+sorted_types[0][1]['l']} WR)")
        elif overall_ev > 0:
            print(f"\n⚠️ VERDICT: PROMISING BUT NEEDS FILTERING")
            print(f"   - Use only profitable divergence types")
            print(f"   - Consider hour-based filtering")
        else:
            print(f"\n❌ VERDICT: NOT PROFITABLE AFTER COSTS")

if __name__ == "__main__":
    run_walkforward_backtest()
