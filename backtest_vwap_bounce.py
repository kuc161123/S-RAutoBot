#!/usr/bin/env python3
"""
VWAP BOUNCE STRATEGY BACKTEST
=============================
Testing VWAP as support/resistance (bounce) vs current cross strategy.

VWAP Bounce Logic:
- LONG: Price dips to touch VWAP from above, then bounces up (close > VWAP, low near VWAP)
- SHORT: Price rises to touch VWAP from below, then bounces down (close < VWAP, high near VWAP)

Key difference from Cross:
- Cross: Price moves THROUGH VWAP
- Bounce: Price touches VWAP and REVERSES
"""

import requests
import pandas as pd
import numpy as np
import yaml
import math
from collections import defaultdict
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

TIMEFRAME = '3'
DATA_DAYS = 60  # 60 days for robust testing
NUM_SYMBOLS = 50  # Top 50 by volume

# Risk:Reward
TP_ATR_MULT = 2.0
SL_ATR_MULT = 1.0

# Realistic execution
TOTAL_FEES = 0.001  # 0.1% (taker fee + slippage)

# Walk-forward validation
TRAIN_RATIO = 0.7

# Thresholds
MIN_TRADES = 15
TARGET_LB_WR = 38.0  # Breakeven at 2:1 is 33%

# VWAP bounce sensitivity - what counts as "near VWAP"
VWAP_TOUCH_THRESHOLD = 0.002  # 0.2% - price must be within this % of VWAP to count as "touch"

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

def get_symbols(limit: int = 50) -> list:
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

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 50: return pd.DataFrame()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Fib levels (50-period)
    df['roll_high'] = df['high'].rolling(50).max()
    df['roll_low'] = df['low'].rolling(50).min()
    
    # VWAP (24-hour rolling for crypto)
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (tp * df['volume']).rolling(480).sum() / df['volume'].rolling(480).sum()
    
    # Previous values for bounce detection
    df['prev_close'] = df['close'].shift(1)
    df['prev_vwap'] = df['vwap'].shift(1)
    
    return df.dropna()

# =============================================================================
# SIGNAL DETECTION
# =============================================================================

def check_vwap_cross(row, prev_row) -> str:
    """Original VWAP Cross strategy (current bot logic)"""
    if prev_row is None: return None
    
    # LONG: Candle touches VWAP and closes above
    if row['low'] <= row['vwap'] and row['close'] > row['vwap']:
        return 'long'
    
    # SHORT: Candle touches VWAP and closes below
    if row['high'] >= row['vwap'] and row['close'] < row['vwap']:
        return 'short'
    
    return None

def check_vwap_bounce(row, prev_row, threshold=VWAP_TOUCH_THRESHOLD) -> str:
    """NEW VWAP Bounce strategy - price bounces off VWAP"""
    if prev_row is None: return None
    
    vwap = row['vwap']
    close = row['close']
    low = row['low']
    high = row['high']
    prev_close = prev_row['close']
    
    # Calculate touch distance as percentage
    touch_distance_low = abs(low - vwap) / vwap
    touch_distance_high = abs(high - vwap) / vwap
    
    # LONG BOUNCE CONDITIONS:
    # 1. Previous close was above VWAP (price coming from above)
    # 2. Current candle's low touched/came near VWAP (within threshold)
    # 3. Current candle closed ABOVE VWAP (bounced up)
    # 4. Current close > current open (bullish candle)
    if (prev_close > prev_row['vwap'] and 
        touch_distance_low <= threshold and
        close > vwap and
        close > row['open']):
        return 'long'
    
    # SHORT BOUNCE CONDITIONS:
    # 1. Previous close was below VWAP (price coming from below)
    # 2. Current candle's high touched/came near VWAP (within threshold)
    # 3. Current candle closed BELOW VWAP (bounced down)
    # 4. Current close < current open (bearish candle)
    if (prev_close < prev_row['vwap'] and 
        touch_distance_high <= threshold and
        close < vwap and
        close < row['open']):
        return 'short'
    
    return None

def get_combo(row) -> str:
    rsi = row['rsi']
    if rsi < 40: r_bin = 'oversold'
    elif rsi > 60: r_bin = 'overbought'
    else: r_bin = 'neutral'
    
    m_bin = 'bull' if row['macd'] > row['macd_signal'] else 'bear'
    
    high, low, close = row['roll_high'], row['roll_low'], row['close']
    if high == low: f_bin = 'low'
    else:
        fib = (high - close) / (high - low) * 100
        if fib < 38: f_bin = 'low'
        elif fib < 62: f_bin = 'mid'
        else: f_bin = 'high'
    
    return f"RSI:{r_bin} MACD:{m_bin} Fib:{f_bin}"

def simulate_trade(df, entry_idx, side, entry_price, atr) -> dict:
    if side == 'long':
        tp = entry_price + (TP_ATR_MULT * atr)
        sl = entry_price - (SL_ATR_MULT * atr)
    else:
        tp = entry_price - (TP_ATR_MULT * atr)
        sl = entry_price + (SL_ATR_MULT * atr)
    
    # Add fees to TP (makes it harder to hit)
    if side == 'long': tp = tp * (1 + TOTAL_FEES)
    else: tp = tp * (1 - TOTAL_FEES)
    
    # Simulate forward 100 candles
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

def run_comparison_backtest():
    print("=" * 80)
    print("🔬 VWAP BOUNCE vs VWAP CROSS BACKTEST")
    print("=" * 80)
    print(f"Timeframe: {TIMEFRAME}m | Data: {DATA_DAYS} days | Symbols: {NUM_SYMBOLS}")
    print(f"Bounce threshold: {VWAP_TOUCH_THRESHOLD*100:.2f}% of VWAP")
    print(f"R:R = {TP_ATR_MULT}:{SL_ATR_MULT} (2:1)")
    print("=" * 80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📋 Testing {len(symbols)} symbols...\n")
    
    # Results tracking
    results = {
        'cross': {'all': {'w': 0, 'n': 0}, 'long': {'w': 0, 'n': 0}, 'short': {'w': 0, 'n': 0}},
        'bounce': {'all': {'w': 0, 'n': 0}, 'long': {'w': 0, 'n': 0}, 'short': {'w': 0, 'n': 0}}
    }
    
    # Hour tracking for both strategies
    cross_by_hour = defaultdict(lambda: {'w': 0, 'n': 0})
    bounce_by_hour = defaultdict(lambda: {'w': 0, 'n': 0})
    
    # Combo tracking
    cross_by_combo = defaultdict(lambda: {'w': 0, 'n': 0})
    bounce_by_combo = defaultdict(lambda: {'w': 0, 'n': 0})
    
    start_time = time.time()
    processed = 0
    
    for idx, symbol in enumerate(symbols):
        try:
            df = fetch_klines(symbol, TIMEFRAME, DATA_DAYS)
            if df.empty or len(df) < 1000:
                continue
            
            df = calculate_indicators(df)
            if df.empty:
                continue
            
            processed += 1
            rows = list(df.itertuples())
            
            for i in range(1, len(rows) - 100):
                row = rows[i]
                prev_row = rows[i-1]
                
                row_dict = {
                    'low': row.low, 'high': row.high, 'close': row.close, 
                    'open': row.open, 'vwap': row.vwap, 'rsi': row.rsi,
                    'macd': row.macd, 'macd_signal': row.macd_signal,
                    'roll_high': row.roll_high, 'roll_low': row.roll_low
                }
                prev_dict = {
                    'close': prev_row.close, 'vwap': prev_row.vwap
                }
                
                atr = row.atr
                if pd.isna(atr) or atr <= 0:
                    continue
                
                hour = row.hour_utc
                combo = get_combo(row_dict)
                
                # Test CROSS strategy
                cross_side = check_vwap_cross(row_dict, prev_dict)
                if cross_side:
                    trade = simulate_trade(df, i, cross_side, row.close, atr)
                    if trade['outcome'] != 'timeout':
                        win = 1 if trade['outcome'] == 'win' else 0
                        results['cross']['all']['n'] += 1
                        results['cross']['all']['w'] += win
                        results['cross'][cross_side]['n'] += 1
                        results['cross'][cross_side]['w'] += win
                        cross_by_hour[hour]['n'] += 1
                        cross_by_hour[hour]['w'] += win
                        cross_by_combo[combo]['n'] += 1
                        cross_by_combo[combo]['w'] += win
                
                # Test BOUNCE strategy
                bounce_side = check_vwap_bounce(row_dict, prev_dict)
                if bounce_side:
                    trade = simulate_trade(df, i, bounce_side, row.close, atr)
                    if trade['outcome'] != 'timeout':
                        win = 1 if trade['outcome'] == 'win' else 0
                        results['bounce']['all']['n'] += 1
                        results['bounce']['all']['w'] += win
                        results['bounce'][bounce_side]['n'] += 1
                        results['bounce'][bounce_side]['w'] += win
                        bounce_by_hour[hour]['n'] += 1
                        bounce_by_hour[hour]['w'] += win
                        bounce_by_combo[combo]['n'] += 1
                        bounce_by_combo[combo]['w'] += win
            
            # Progress
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"[{idx+1}/{NUM_SYMBOLS}] {processed} processed | Cross: {results['cross']['all']['n']} trades | Bounce: {results['bounce']['all']['n']} trades")
            
            time.sleep(0.03)
            
        except Exception as e:
            continue
    
    elapsed = time.time() - start_time
    
    # ==========================================================================
    # RESULTS
    # ==========================================================================
    
    print("\n" + "=" * 80)
    print("📊 COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\n⏱️ Completed in {elapsed/60:.1f} minutes ({processed} symbols)")
    
    print("\n" + "-" * 60)
    print("OVERALL COMPARISON")
    print("-" * 60)
    
    print(f"\n{'Strategy':<20} {'Trades':>8} {'WR':>8} {'LB WR':>8} {'EV':>8}")
    print("-" * 55)
    
    for strat in ['cross', 'bounce']:
        d = results[strat]['all']
        if d['n'] > 0:
            wr = d['w'] / d['n']
            lb = wilson_lower_bound(d['w'], d['n'])
            ev = calc_ev(wr)
            emoji = "🟢" if ev > 0.1 else "⚠️" if ev > 0 else "🔴"
            name = "VWAP Cross" if strat == 'cross' else "VWAP Bounce"
            print(f"{name:<20} {d['n']:>8} {wr*100:>7.1f}% {lb*100:>7.1f}% {ev:>+7.2f} {emoji}")
    
    print("\n" + "-" * 60)
    print("BY SIDE")
    print("-" * 60)
    
    for strat in ['cross', 'bounce']:
        name = "CROSS" if strat == 'cross' else "BOUNCE"
        print(f"\n{name}:")
        for side in ['long', 'short']:
            d = results[strat][side]
            if d['n'] > 0:
                wr = d['w'] / d['n']
                ev = calc_ev(wr)
                emoji = "🟢" if ev > 0.1 else "⚠️" if ev > 0 else "🔴"
                print(f"   {side.upper():<10} N={d['n']:>5} | WR={wr*100:>5.1f}% | EV={ev:>+.2f} {emoji}")
    
    # Best hours comparison
    print("\n" + "-" * 60)
    print("TOP 5 HOURS BY STRATEGY")
    print("-" * 60)
    
    for strat, hour_dict in [('CROSS', cross_by_hour), ('BOUNCE', bounce_by_hour)]:
        print(f"\n{strat}:")
        sorted_hours = sorted(hour_dict.items(), key=lambda x: calc_ev(x[1]['w']/x[1]['n']) if x[1]['n'] > 20 else -999, reverse=True)
        for hour, d in sorted_hours[:5]:
            if d['n'] >= 20:
                wr = d['w'] / d['n']
                ev = calc_ev(wr)
                print(f"   {hour:02d}:00 UTC | N={d['n']:>4} | WR={wr*100:>5.1f}% | EV={ev:>+.2f}")
    
    # Best combos comparison
    print("\n" + "-" * 60)
    print("TOP 5 COMBOS BY STRATEGY")
    print("-" * 60)
    
    for strat, combo_dict in [('CROSS', cross_by_combo), ('BOUNCE', bounce_by_combo)]:
        print(f"\n{strat}:")
        sorted_combos = sorted(combo_dict.items(), key=lambda x: calc_ev(x[1]['w']/x[1]['n']) if x[1]['n'] > 20 else -999, reverse=True)
        for combo, d in sorted_combos[:5]:
            if d['n'] >= 20:
                wr = d['w'] / d['n']
                ev = calc_ev(wr)
                print(f"   {combo:<40} | N={d['n']:>4} | WR={wr*100:>5.1f}% | EV={ev:>+.2f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("💡 SUMMARY")
    print("=" * 80)
    
    cross_all = results['cross']['all']
    bounce_all = results['bounce']['all']
    
    if cross_all['n'] > 0 and bounce_all['n'] > 0:
        cross_wr = cross_all['w'] / cross_all['n']
        bounce_wr = bounce_all['w'] / bounce_all['n']
        cross_ev = calc_ev(cross_wr)
        bounce_ev = calc_ev(bounce_wr)
        
        print(f"\n📊 VWAP Cross:  {cross_all['n']:>5} trades | {cross_wr*100:.1f}% WR | {cross_ev:+.2f} EV")
        print(f"📊 VWAP Bounce: {bounce_all['n']:>5} trades | {bounce_wr*100:.1f}% WR | {bounce_ev:+.2f} EV")
        
        if cross_ev > bounce_ev:
            diff = cross_ev - bounce_ev
            print(f"\n✅ VWAP CROSS is better by {diff:.2f} EV")
        elif bounce_ev > cross_ev:
            diff = bounce_ev - cross_ev
            print(f"\n✅ VWAP BOUNCE is better by {diff:.2f} EV")
        else:
            print(f"\n⚖️ Both strategies perform similarly")
        
        # Trade frequency
        print(f"\n📈 Trade Frequency:")
        print(f"   Cross:  {cross_all['n']} trades ({cross_all['n']/processed:.0f} per symbol)")
        print(f"   Bounce: {bounce_all['n']} trades ({bounce_all['n']/processed:.0f} per symbol)")
    
    return results

if __name__ == "__main__":
    run_comparison_backtest()
