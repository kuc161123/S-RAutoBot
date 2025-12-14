#!/usr/bin/env python3
"""
50 EMA + VWAP STRATEGY BACKTEST
================================
Strategy based on professional trader approach:

1. Clean Break of 50 EMA: Enter with confidence
2. Failure to Hold 50 EMA: Anticipate VWAP magnet pull
3. VWAP Holds/Bounces: Go long
4. VWAP Breaks Down: Go short (continuation)

Rules:
- Skip first hour (volatile/choppy)
- 2:1 R:R for realistic execution
- SL based on ATR
- Test 100 symbols
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

TIMEFRAME = '3'  # 3-minute candles
DATA_DAYS = 60
NUM_SYMBOLS = 100

# Risk/Reward
TP_ATR_MULT = 2.0   # Take profit at 2 ATR
SL_ATR_MULT = 1.0   # Stop loss at 1 ATR
TOTAL_FEES = 0.001  # 0.1% total (entry + exit)

# Strategy settings
EMA_PERIOD = 50
SKIP_FIRST_HOUR = True  # Skip first hour of each day
VWAP_TOUCH_THRESHOLD = 0.002  # 0.2% from VWAP for touch detection

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

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < EMA_PERIOD + 50: return pd.DataFrame()
    
    # 50 EMA
    df['ema50'] = df['close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    # ATR for SL/TP
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Rolling VWAP (24-hour rolling)
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (tp * df['volume']).rolling(480).sum() / df['volume'].rolling(480).sum()
    
    # Price position relative to EMA and VWAP
    df['above_ema'] = df['close'] > df['ema50']
    df['above_vwap'] = df['close'] > df['vwap']
    
    # EMA distance (for breakout detection)
    df['ema_dist'] = (df['close'] - df['ema50']) / df['ema50'] * 100  # percentage
    
    # VWAP distance
    df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap'] * 100  # percentage
    
    # Previous positions for cross detection
    df['prev_above_ema'] = df['above_ema'].shift(1)
    df['prev_above_vwap'] = df['above_vwap'].shift(1)
    
    return df.dropna()

def check_ema_vwap_signal(row, prev_row) -> dict:
    """
    Strategy Logic:
    
    LONG signals:
    1. EMA Break Up: Price crosses above 50 EMA with momentum
    2. VWAP Bounce: Price touches VWAP and bounces up
    3. EMA Hold + VWAP Above: Price holds EMA as support, VWAP above attracts
    
    SHORT signals:
    1. EMA Break Down: Price crosses below 50 EMA with momentum
    2. VWAP Break Down: Price fails to hold VWAP
    3. EMA Rejection: Price fails at EMA resistance
    """
    
    close = row['close']
    ema = row['ema50']
    vwap = row['vwap']
    low = row['low']
    high = row['high']
    
    prev_above_ema = prev_row['above_ema']
    prev_above_vwap = prev_row['above_vwap']
    
    signal = None
    signal_type = None
    
    # === LONG SIGNALS ===
    
    # 1. Clean EMA Break Up: Was below, now above with strong close
    if not prev_above_ema and close > ema:
        # Confirm with VWAP position (stronger if above VWAP too)
        if close > vwap:
            signal = 'long'
            signal_type = 'ema_break_up_strong'
        else:
            signal = 'long'
            signal_type = 'ema_break_up'
    
    # 2. VWAP Bounce: Price touched VWAP (within threshold) and closed above
    elif low <= vwap * (1 + VWAP_TOUCH_THRESHOLD) and close > vwap:
        if prev_row['above_vwap']:  # Was above, touched VWAP, bounced
            signal = 'long'
            signal_type = 'vwap_bounce'
    
    # 3. EMA Support Hold: Price dipped to EMA and bounced
    elif low <= ema * 1.002 and close > ema and prev_above_ema:
        signal = 'long'
        signal_type = 'ema_support_hold'
    
    # === SHORT SIGNALS ===
    
    # 1. Clean EMA Break Down: Was above, now below with strong close
    elif prev_above_ema and close < ema:
        # Confirm with VWAP position (stronger if below VWAP too)
        if close < vwap:
            signal = 'short'
            signal_type = 'ema_break_down_strong'
        else:
            signal = 'short'
            signal_type = 'ema_break_down'
    
    # 2. VWAP Break Down: Price touched VWAP (within threshold) and closed below
    elif high >= vwap * (1 - VWAP_TOUCH_THRESHOLD) and close < vwap:
        if not prev_row['above_vwap']:  # Was below, tested VWAP, rejected
            signal = 'short'
            signal_type = 'vwap_rejection'
    
    # 3. EMA Resistance Rejection: Price tested EMA from below and failed
    elif high >= ema * 0.998 and close < ema and not prev_above_ema:
        signal = 'short'
        signal_type = 'ema_resistance_rejection'
    
    return {'signal': signal, 'type': signal_type}

def is_first_hour_of_day(row) -> bool:
    """Check if this candle is in the first hour of the trading day (00:00-01:00 UTC)"""
    hour = row['hour_utc']
    return hour < 1  # Skip 00:xx hour

def simulate_trade(df, entry_idx, side, entry_price, atr) -> dict:
    """Simulate trade with candle-by-candle SL/TP checking"""
    
    if side == 'long':
        tp = entry_price + (TP_ATR_MULT * atr)
        sl = entry_price - (SL_ATR_MULT * atr)
    else:
        tp = entry_price - (TP_ATR_MULT * atr)
        sl = entry_price + (SL_ATR_MULT * atr)
    
    # Apply fees to TP
    if side == 'long': tp = tp * (1 + TOTAL_FEES)
    else: tp = tp * (1 - TOTAL_FEES)
    
    # Look ahead up to 100 candles (5 hours at 3min)
    rows = list(df.iloc[entry_idx+1:entry_idx+101].itertuples())
    
    max_r = 0
    for future_row in rows:
        if side == 'long':
            # Check for SL hit first (worst case on same candle)
            if future_row.low <= sl: 
                return {'outcome': 'loss', 'max_r': max_r}
            # Check for TP hit
            if future_row.high >= tp: 
                return {'outcome': 'win', 'max_r': 2.0}
            # Track max R reached
            profit = future_row.high - entry_price
            r_reached = profit / (entry_price - sl) if sl != entry_price else 0
            max_r = max(max_r, r_reached)
        else:
            # Check for SL hit first
            if future_row.high >= sl: 
                return {'outcome': 'loss', 'max_r': max_r}
            # Check for TP hit
            if future_row.low <= tp: 
                return {'outcome': 'win', 'max_r': 2.0}
            # Track max R reached
            profit = entry_price - future_row.low
            r_reached = profit / (sl - entry_price) if sl != entry_price else 0
            max_r = max(max_r, r_reached)
    
    return {'outcome': 'timeout', 'max_r': max_r}

# =============================================================================
# MAIN BACKTEST
# =============================================================================

def run_ema_vwap_backtest():
    print("=" * 80)
    print("🔬 50 EMA + VWAP STRATEGY BACKTEST")
    print("=" * 80)
    print(f"Timeframe: {TIMEFRAME}m | Data: {DATA_DAYS} days | Symbols: {NUM_SYMBOLS}")
    print(f"50 EMA + Rolling VWAP (24h)")
    print(f"R:R = {TP_ATR_MULT}:{SL_ATR_MULT} ({int(TP_ATR_MULT)}:1)")
    print(f"Skip first hour: {SKIP_FIRST_HOUR}")
    print("=" * 80)
    
    symbols = get_symbols(NUM_SYMBOLS)
    print(f"\n📋 Testing {len(symbols)} symbols...\n")
    
    # Results by signal type
    results = defaultdict(lambda: {'w': 0, 'l': 0, 'timeout': 0})
    results_by_side = {'long': {'w': 0, 'l': 0}, 'short': {'w': 0, 'l': 0}}
    results_by_hour = defaultdict(lambda: {'w': 0, 'l': 0})
    
    total_trades = 0
    total_wins = 0
    
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
            
            last_trade_idx = -100  # Cooldown between trades
            
            for i in range(1, len(rows) - 100):
                row = rows[i]
                prev_row = rows[i-1]
                
                # Skip first hour if enabled
                if SKIP_FIRST_HOUR and is_first_hour_of_day(row):
                    continue
                
                # Cooldown: No trade if we just traded
                if i - last_trade_idx < 10:  # 30 min cooldown at 3min candles
                    continue
                
                atr = row.atr
                if pd.isna(atr) or atr <= 0:
                    continue
                
                # Check for signal
                signal_result = check_ema_vwap_signal(
                    {'close': row.close, 'high': row.high, 'low': row.low, 
                     'ema50': row.ema50, 'vwap': row.vwap, 'above_ema': row.above_ema,
                     'above_vwap': row.above_vwap, 'hour_utc': row.hour_utc},
                    {'above_ema': prev_row.above_ema, 'above_vwap': prev_row.above_vwap}
                )
                
                signal = signal_result['signal']
                signal_type = signal_result['type']
                
                if signal is None:
                    continue
                
                # Simulate trade
                trade = simulate_trade(df, i, signal, row.close, atr)
                
                if trade['outcome'] == 'timeout':
                    results[signal_type]['timeout'] += 1
                    continue
                
                last_trade_idx = i
                total_trades += 1
                
                if trade['outcome'] == 'win':
                    results[signal_type]['w'] += 1
                    results_by_side[signal]['w'] += 1
                    results_by_hour[row.hour_utc]['w'] += 1
                    total_wins += 1
                else:
                    results[signal_type]['l'] += 1
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
    
    # By Signal Type
    print("\n" + "-" * 60)
    print("BY SIGNAL TYPE")
    print("-" * 60)
    
    print(f"\n{'Signal Type':<35} {'N':>6} {'WR':>8} {'EV':>8}")
    print("-" * 60)
    
    sorted_types = sorted(results.items(), key=lambda x: x[1]['w'] + x[1]['l'], reverse=True)
    for signal_type, data in sorted_types:
        total = data['w'] + data['l']
        if total > 0:
            wr = data['w'] / total
            ev = calc_ev(wr)
            emoji = "🟢" if ev > 0.1 else "⚠️" if ev > 0 else "🔴"
            print(f"{signal_type:<35} {total:>6} {wr*100:>7.1f}% {ev:>+7.2f} {emoji}")
    
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
    
    # By Hour
    print("\n" + "-" * 60)
    print("BY HOUR (Best Hours)")
    print("-" * 60)
    
    sorted_hours = sorted(results_by_hour.items(), 
                          key=lambda x: calc_ev(x[1]['w']/(x[1]['w']+x[1]['l'])) if x[1]['w']+x[1]['l'] > 20 else -999,
                          reverse=True)
    
    for hour, d in sorted_hours[:5]:
        total = d['w'] + d['l']
        if total >= 20:
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
        
        print(f"\n📊 50 EMA + VWAP Strategy: {total_trades} trades | {overall_wr*100:.1f}% WR | {overall_ev:+.2f} EV")
        
        if overall_ev > 0:
            print(f"\n✅ Strategy is PROFITABLE at 2:1 R:R!")
        else:
            print(f"\n❌ Strategy needs filtering to be profitable at 2:1 R:R")
        
        # Best signal types
        print("\n🎯 Best Signal Types:")
        for signal_type, data in sorted_types[:3]:
            total = data['w'] + data['l']
            if total >= 10:
                wr = data['w'] / total
                ev = calc_ev(wr)
                if ev > 0:
                    print(f"   ✅ {signal_type}: {wr*100:.0f}% WR, {ev:+.2f} EV (N={total})")

if __name__ == "__main__":
    run_ema_vwap_backtest()
