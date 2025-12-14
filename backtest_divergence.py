#!/usr/bin/env python3
"""
RSI DIVERGENCE STRATEGY BACKTEST
================================
Divergence is one of the most reliable reversal signals.

Types of Divergence:
1. REGULAR (Reversal signals):
   - Bullish: Price makes LOWER LOW, RSI makes HIGHER LOW → Long
   - Bearish: Price makes HIGHER HIGH, RSI makes LOWER HIGH → Short

2. HIDDEN (Trend continuation):
   - Hidden Bullish: Price makes HIGHER LOW, RSI makes LOWER LOW → Long (trend continues)
   - Hidden Bearish: Price makes LOWER HIGH, RSI makes HIGHER HIGH → Short (trend continues)

Additional filters:
- RSI oversold/overbought zones for confirmation
- Volume confirmation
- Trend filter (optional)
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
# CONFIGURATION
# =============================================================================

TIMEFRAME = '15'  # 15-minute candles (good for divergence)
DATA_DAYS = 60
NUM_SYMBOLS = 100

# Risk/Reward - testing both
TP_ATR_MULT = 2.0   # 2:1 R:R
SL_ATR_MULT = 1.0
TOTAL_FEES = 0.001

# Divergence settings
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
LOOKBACK_BARS = 14  # How far back to look for divergence
MIN_PIVOT_DISTANCE = 5  # Minimum bars between pivots

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
    return df

def calculate_rsi(close, period=14):
    """Calculate RSI"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_pivots(data, left=5, right=5):
    """Find pivot highs and lows"""
    n = len(data)
    pivot_highs = np.full(n, np.nan)
    pivot_lows = np.full(n, np.nan)
    
    for i in range(left, n - right):
        # Check for pivot high
        is_high = True
        for j in range(i - left, i + right + 1):
            if j != i and data[j] >= data[i]:
                is_high = False
                break
        if is_high:
            pivot_highs[i] = data[i]
        
        # Check for pivot low
        is_low = True
        for j in range(i - left, i + right + 1):
            if j != i and data[j] <= data[i]:
                is_low = False
                break
        if is_low:
            pivot_lows[i] = data[i]
    
    return pivot_highs, pivot_lows

def detect_divergence(df, lookback=14, min_distance=5):
    """
    Detect RSI divergences.
    
    Returns dict with divergence type at each bar.
    """
    close = df['close'].values
    rsi = df['rsi'].values
    low = df['low'].values
    high = df['high'].values
    n = len(df)
    
    # Find pivots on price and RSI
    price_pivot_highs, price_pivot_lows = find_pivots(close, left=3, right=3)
    rsi_pivot_highs, rsi_pivot_lows = find_pivots(rsi, left=3, right=3)
    
    divergences = []
    
    for i in range(lookback + 10, n):
        signal = None
        signal_type = None
        
        # === REGULAR BULLISH DIVERGENCE ===
        # Price: Lower Low, RSI: Higher Low
        # Look for recent price pivot low
        current_price_low = None
        current_price_low_idx = None
        prev_price_low = None
        prev_price_low_idx = None
        
        # Find most recent pivot lows
        for j in range(i, max(i - lookback, 0), -1):
            if not np.isnan(price_pivot_lows[j]):
                if current_price_low is None:
                    current_price_low = price_pivot_lows[j]
                    current_price_low_idx = j
                elif prev_price_low is None and j < current_price_low_idx - min_distance:
                    prev_price_low = price_pivot_lows[j]
                    prev_price_low_idx = j
                    break
        
        if current_price_low is not None and prev_price_low is not None:
            # Price made lower low
            if current_price_low < prev_price_low:
                # Check RSI for higher low
                current_rsi_low = None
                prev_rsi_low = None
                
                for j in range(current_price_low_idx, max(current_price_low_idx - 3, 0), -1):
                    if not np.isnan(rsi_pivot_lows[j]):
                        current_rsi_low = rsi_pivot_lows[j]
                        break
                
                for j in range(prev_price_low_idx, max(prev_price_low_idx - 3, 0), -1):
                    if not np.isnan(rsi_pivot_lows[j]):
                        prev_rsi_low = rsi_pivot_lows[j]
                        break
                
                # Use approximation if no exact pivot
                if current_rsi_low is None:
                    current_rsi_low = rsi[current_price_low_idx]
                if prev_rsi_low is None:
                    prev_rsi_low = rsi[prev_price_low_idx]
                
                # Regular bullish: Price LL, RSI HL
                if current_rsi_low > prev_rsi_low and rsi[i] < RSI_OVERSOLD + 10:
                    signal = 'long'
                    signal_type = 'regular_bullish'
        
        # === REGULAR BEARISH DIVERGENCE ===
        # Price: Higher High, RSI: Lower High
        current_price_high = None
        current_price_high_idx = None
        prev_price_high = None
        prev_price_high_idx = None
        
        for j in range(i, max(i - lookback, 0), -1):
            if not np.isnan(price_pivot_highs[j]):
                if current_price_high is None:
                    current_price_high = price_pivot_highs[j]
                    current_price_high_idx = j
                elif prev_price_high is None and j < current_price_high_idx - min_distance:
                    prev_price_high = price_pivot_highs[j]
                    prev_price_high_idx = j
                    break
        
        if current_price_high is not None and prev_price_high is not None:
            # Price made higher high
            if current_price_high > prev_price_high:
                # Check RSI for lower high
                current_rsi_high = None
                prev_rsi_high = None
                
                for j in range(current_price_high_idx, max(current_price_high_idx - 3, 0), -1):
                    if not np.isnan(rsi_pivot_highs[j]):
                        current_rsi_high = rsi_pivot_highs[j]
                        break
                
                for j in range(prev_price_high_idx, max(prev_price_high_idx - 3, 0), -1):
                    if not np.isnan(rsi_pivot_highs[j]):
                        prev_rsi_high = rsi_pivot_highs[j]
                        break
                
                if current_rsi_high is None:
                    current_rsi_high = rsi[current_price_high_idx]
                if prev_rsi_high is None:
                    prev_rsi_high = rsi[prev_price_high_idx]
                
                # Regular bearish: Price HH, RSI LH
                if current_rsi_high < prev_rsi_high and rsi[i] > RSI_OVERBOUGHT - 10:
                    signal = 'short'
                    signal_type = 'regular_bearish'
        
        # === HIDDEN BULLISH DIVERGENCE (Trend Continuation) ===
        # Price: Higher Low, RSI: Lower Low
        if current_price_low is not None and prev_price_low is not None:
            if current_price_low > prev_price_low:  # Higher low in price
                current_rsi_low = rsi[current_price_low_idx] if current_price_low_idx else None
                prev_rsi_low = rsi[prev_price_low_idx] if prev_price_low_idx else None
                
                if current_rsi_low is not None and prev_rsi_low is not None:
                    if current_rsi_low < prev_rsi_low:  # Lower low in RSI
                        # Only if RSI is not overbought
                        if rsi[i] < RSI_OVERBOUGHT - 10:
                            signal = 'long'
                            signal_type = 'hidden_bullish'
        
        # === HIDDEN BEARISH DIVERGENCE (Trend Continuation) ===
        # Price: Lower High, RSI: Higher High
        if current_price_high is not None and prev_price_high is not None:
            if current_price_high < prev_price_high:  # Lower high in price
                current_rsi_high = rsi[current_price_high_idx] if current_price_high_idx else None
                prev_rsi_high = rsi[prev_price_high_idx] if prev_price_high_idx else None
                
                if current_rsi_high is not None and prev_rsi_high is not None:
                    if current_rsi_high > prev_rsi_high:  # Higher high in RSI
                        # Only if RSI is not oversold
                        if rsi[i] > RSI_OVERSOLD + 10:
                            signal = 'short'
                            signal_type = 'hidden_bearish'
        
        divergences.append({
            'idx': i,
            'signal': signal,
            'type': signal_type
        })
    
    return divergences

def simulate_trade(df, entry_idx, side, entry_price, atr) -> dict:
    if side == 'long':
        tp = entry_price + (TP_ATR_MULT * atr)
        sl = entry_price - (SL_ATR_MULT * atr)
    else:
        tp = entry_price - (TP_ATR_MULT * atr)
        sl = entry_price + (SL_ATR_MULT * atr)
    
    if side == 'long': tp = tp * (1 + TOTAL_FEES)
    else: tp = tp * (1 - TOTAL_FEES)
    
    rows = list(df.iloc[entry_idx+1:entry_idx+101].itertuples())
    
    for future_row in rows:
        if side == 'long':
            if future_row.low <= sl: return {'outcome': 'loss'}
            if future_row.high >= tp: return {'outcome': 'win'}
        else:
            if future_row.high >= sl: return {'outcome': 'loss'}
            if future_row.low <= tp: return {'outcome': 'win'}
    
    return {'outcome': 'timeout'}

# =============================================================================
# MAIN BACKTEST
# =============================================================================

def run_divergence_backtest():
    print("=" * 80)
    print("🔬 RSI DIVERGENCE STRATEGY BACKTEST")
    print("=" * 80)
    print(f"Timeframe: {TIMEFRAME}m | Data: {DATA_DAYS} days | Symbols: {NUM_SYMBOLS}")
    print(f"RSI Period: {RSI_PERIOD} | Oversold: {RSI_OVERSOLD} | Overbought: {RSI_OVERBOUGHT}")
    print(f"R:R = {TP_ATR_MULT}:{SL_ATR_MULT} ({int(TP_ATR_MULT)}:1)")
    print("=" * 80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📋 Testing {len(symbols)} symbols...\n")
    
    # Results
    results_by_type = defaultdict(lambda: {'w': 0, 'l': 0})
    results_by_side = {'long': {'w': 0, 'l': 0}, 'short': {'w': 0, 'l': 0}}
    results_by_hour = defaultdict(lambda: {'w': 0, 'l': 0})
    
    total_trades = 0
    total_wins = 0
    
    start_time = time.time()
    processed = 0
    
    for idx, symbol in enumerate(symbols):
        try:
            df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 500:
                continue
            
            # Calculate RSI
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
            
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = true_range.rolling(14).mean()
            
            df = df.dropna()
            if len(df) < 200:
                continue
            
            processed += 1
            
            # Detect divergences
            divergences = detect_divergence(df, lookback=LOOKBACK_BARS, min_distance=MIN_PIVOT_DISTANCE)
            
            rows = list(df.itertuples())
            last_trade_idx = -20  # Cooldown
            
            for div in divergences:
                i = div['idx']
                signal = div['signal']
                sig_type = div['type']
                
                if signal is None:
                    continue
                
                if i - last_trade_idx < 10:  # 10-bar cooldown
                    continue
                
                if i >= len(rows) - 100:
                    continue
                
                row = rows[i]
                atr = row.atr
                if pd.isna(atr) or atr <= 0:
                    continue
                
                # Simulate trade
                trade = simulate_trade(df, i, signal, row.close, atr)
                
                if trade['outcome'] == 'timeout':
                    continue
                
                last_trade_idx = i
                total_trades += 1
                
                if trade['outcome'] == 'win':
                    results_by_type[sig_type]['w'] += 1
                    results_by_side[signal]['w'] += 1
                    results_by_hour[row.hour_utc]['w'] += 1
                    total_wins += 1
                else:
                    results_by_type[sig_type]['l'] += 1
                    results_by_side[signal]['l'] += 1
                    results_by_hour[row.hour_utc]['l'] += 1
            
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
    print("📊 RESULTS")
    print("=" * 80)
    print(f"\n⏱️ Completed in {elapsed/60:.1f} minutes ({processed} symbols)")
    
    # Overall
    print("\n" + "-" * 60)
    print("OVERALL PERFORMANCE")
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
    
    # By Divergence Type
    print("\n" + "-" * 60)
    print("BY DIVERGENCE TYPE")
    print("-" * 60)
    
    print(f"\n{'Type':<25} {'N':>6} {'WR':>8} {'EV':>8}")
    print("-" * 50)
    
    sorted_types = sorted(results_by_type.items(), 
                         key=lambda x: calc_ev(x[1]['w']/(x[1]['w']+x[1]['l'])) if x[1]['w']+x[1]['l'] > 0 else -999,
                         reverse=True)
    
    for sig_type, data in sorted_types:
        total = data['w'] + data['l']
        if total > 0:
            wr = data['w'] / total
            ev = calc_ev(wr)
            emoji = "🟢" if ev > 0.1 else "⚠️" if ev > 0 else "🔴"
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
            emoji = "🟢" if ev > 0.1 else "⚠️" if ev > 0 else "🔴"
            icon = "🟢" if side == 'long' else "🔴"
            print(f"{icon} {side.upper():<10} N={total:>5} | WR={wr*100:>5.1f}% | EV={ev:>+.2f} {emoji}")
    
    # By Hour (Best 5)
    print("\n" + "-" * 60)
    print("BY HOUR (Best Hours)")
    print("-" * 60)
    
    sorted_hours = sorted(results_by_hour.items(), 
                          key=lambda x: calc_ev(x[1]['w']/(x[1]['w']+x[1]['l'])) if x[1]['w']+x[1]['l'] > 10 else -999,
                          reverse=True)
    
    for hour, d in sorted_hours[:5]:
        total = d['w'] + d['l']
        if total >= 10:
            wr = d['w'] / total
            ev = calc_ev(wr)
            emoji = "🟢" if ev > 0 else "🔴"
            print(f"{hour:02d}:00 UTC | N={total:>4} | WR={wr*100:>5.1f}% | EV={ev:>+.2f} {emoji}")
    
    # Summary
    print("\n" + "=" * 80)
    print("💡 SUMMARY")
    print("=" * 80)
    
    if total_trades > 0:
        overall_wr = total_wins / total_trades
        overall_ev = calc_ev(overall_wr)
        
        print(f"\n📊 RSI Divergence: {total_trades} trades | {overall_wr*100:.1f}% WR | {overall_ev:+.2f} EV")
        
        if overall_ev > 0:
            print(f"\n✅ Strategy is PROFITABLE at {TP_ATR_MULT}:1 R:R!")
        else:
            print(f"\n❌ Strategy needs filtering to be profitable")
        
        print("\n🎯 Divergence Type Ranking:")
        for sig_type, data in sorted_types:
            total = data['w'] + data['l']
            if total >= 5:
                wr = data['w'] / total
                ev = calc_ev(wr)
                status = "✅ Profitable" if ev > 0 else "❌ Unprofitable"
                print(f"   {sig_type}: {wr*100:.0f}% WR | {ev:+.2f} EV | {status}")

if __name__ == "__main__":
    run_divergence_backtest()
